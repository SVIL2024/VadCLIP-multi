"""
XD-Violence Multi-Scale Training Script

多尺度视频异常检测训练脚本

核心功能：
1. 多尺度模型训练（支持8/16/32帧尺度）
2. Baseline模式对比训练
3. 详细日志记录
4. 训练时间统计

维度说明：
- 输入: [batch_size, sequence_length, feature_dim] - CLIP特征
- 输出: 多尺度融合后的异常分数

Author: VadCLIP-Multi-Seg
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import time
import os
import logging
from datetime import datetime
from copy import deepcopy

# 导入多尺度模型和数据集
from model_seg import CLIPVADSeg
from xd_test_seg import test
from utils.dataset_seg import XDDatasetSeg
from utils.tools import get_prompt_text, get_batch_label
import xd_option_seg


def CLASM(logits, labels, lengths, device):
    """
    Multiple Instance Learning Loss (CLASM)
    
    用于弱监督异常检测的MIL损失函数
    对每个bag（视频），选取top-k高置信度的实例进行分类
    
    Args:
        logits: 模型输出 [batch_size, seq_len, num_classes]
        labels: 视频级标签 [batch_size, num_classes]
        lengths: 实际序列长度
        device: 计算设备
    """
    instance_logits = torch.zeros(0).to(device)
    # 归一化标签
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    # 选取top-k实例的均值作为bag的表示
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    # 计算交叉熵损失
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    """
    Binary Classification Loss (CLAS2)
    
    二分类损失，用于异常/正常的二分类
    对top-k实例取平均后进行二分类
    
    Args:
        logits: 分类器输出 [batch_size, seq_len, 1]
        labels: 二分类标签 [batch_size, 1]
        lengths: 序列长度
        device: 设备
    """
    instance_logits = torch.zeros(0).to(device)
    # 二分类：1 - 正常概率 = 异常概率
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def setup_logger(args):
    """
    设置训练日志记录器
    
    输出到控制台和文件
    """
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('VadCLIP-Seg')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件Handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'xd_train_seg_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def train(model, train_loader, test_loader, args, label_map: dict, device, logger, checkpoint_path):
    """
    训练函数
    
    Args:
        model: 多尺度模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        args: 配置参数
        label_map: 标签映射
        device: 计算设备
        logger: 日志记录器
    """
    model.to(device)

    # 加载ground truth
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    
    # 获取提示文本
    prompt_text = get_prompt_text(label_map)
    
    # 记录初始best AP和AUC
    ap_best = 0
    auc_best = 0
    epoch = 0
    
    # 训练开始时间
    total_start_time = time.time()
    
    logger.info("="*80)
    logger.info("VadCLIP-Seg Training Configuration")
    logger.info("="*80)
    logger.info(f"Scales: {args.scales}")
    logger.info(f"Fusion Type: {args.fusion_type}")
    logger.info(f"Visual Length: {args.visual_length}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Max Epoch: {args.max_epoch}")
    logger.info(f"Use Baseline Mode: {args.use_baseline_mode}")
    logger.info(f"Use Scheme 1: {args.use_scheme1}")
    logger.info("="*80)

    # 加载checkpoint（如果有）
    if args.use_checkpoint == True:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint.get('ap', checkpoint.get('AP', 0))
        auc_best = checkpoint.get('auc', checkpoint.get('AUC', 0))
        logger.info(f"Loaded checkpoint: epoch {epoch+1}, AP: {ap_best:.4f}, AUC: {auc_best:.4f}")

    # 训练循环
    for e in range(args.max_epoch):
        epoch_start_time = time.time()
        
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_scale = 0
        num_batches = 0
        
        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # 前向传播
            if args.use_scheme1 and not args.use_baseline_mode:
                text_features, logits1, logits2, multi_scale_logits1, multi_scale_logits2 = model(
                    visual_feat, None, prompt_text, feat_lengths, 
                    use_baseline=False,
                    use_scheme1=True
                )
                
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device)
                
            else:
                text_features, logits1, logits2 = model(
                    visual_feat, None, prompt_text, feat_lengths, 
                    use_baseline=args.use_baseline_mode
                )
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            
            loss_total1 += loss1.item()
            loss_total2 += loss2.item()

            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            text_normal = text_features_norm[:, 0]
            text_others = text_features_norm[:, 1:]
            loss3 = torch.abs(text_normal @ text_others.transpose(-2, -1)).mean()
            loss3 = loss3 * 1e-4
            loss_total3 += loss3.item()

            # 总损失
            loss = loss1 + loss2 + loss3 * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            num_batches += 1
            
            # 定期打印训练信息
            if i % args.log_interval == 0 and i != 0:
                avg_loss1 = loss_total1 / num_batches
                avg_loss2 = loss_total2 / num_batches
                avg_loss3 = loss_total3 / num_batches
                scheme1_info = f", Loss1_scale: {loss_total_scale/num_batches:.4f}" if args.use_scheme1 else ""
                logger.info(
                    f"Epoch [{e+1}/{args.max_epoch}] "
                    f"Step [{i}/{len(train_loader)}] "
                    f"Loss1: {avg_loss1:.4f} | "
                    f"Loss2: {avg_loss2:.4f} | "
                    f"Loss3: {avg_loss3:.4f}{scheme1_info}"
                )
                
        scheduler.step()
        
        # 测试评估（每个epoch测试，可改为(e+1) % test_interval == 0）
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        logger.info("-"*60)
        logger.info(f"Epoch {e+1} completed in {epoch_time:.2f}s")
        
        test_interval = 1
        if (e + 1) % test_interval == 0:
            ROC1, AP1, ROC2, AP2, mAP = test(model, test_loader, args.visual_length, prompt_text, 
                                gt, gtsegments, gtlabels, device, logger, args.use_baseline_mode, args.use_scheme1)
            AUC = max(ROC1, ROC2)
            AP = max(AP1, AP2)
            if AUC > 0:
                logger.info(f"Test Result -> AP1: {AP1:.4f}, AUC1: {ROC1:.4f} | AP2: {AP2:.4f}, AUC2: {ROC2:.4f}")
        else:
            AUC, AP = 0, 0
            ROC1, AP1, ROC2, AP2 = 0, 0, 0, 0

        if AP > ap_best:
            ap_best = AP
            auc_best = AUC
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best,
                'auc': auc_best,
                'config': {
                    'scales': args.scales,
                    'fusion_type': args.fusion_type,
                    'visual_length': args.visual_length,
                    'use_scheme1': args.use_scheme1
                }
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"New best model saved! AP: {ap_best:.4f}, AUC: {auc_best:.4f}")
        else:
            if AUC > 0:
                logger.info(f"Current Best -> AP: {ap_best:.4f}, AUC: {auc_best:.4f}")

        # 加载最佳模型进行下一轮训练
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 保存最终模型
    total_time = time.time() - total_start_time
    checkpoint = torch.load(checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    
    logger.info("="*80)
    logger.info(f"Training completed!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best AP: {ap_best:.4f}, Best AUC: {auc_best:.4f}")
    logger.info("="*80)
    logger.info(f"Model saved to: {args.model_path}")
    logger.info("="*60)
    
    return ap_best


def train_single_fusion(model_class, fusion_type, train_loader, test_loader, args, label_map, device, logger):
    """
    训练单个融合方法
    
    Args:
        model_class: 模型类 (CLIPVADSeg)
        fusion_type: 融合类型 ('attention', 'concat', 'weighted')
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        args: 配置参数
        label_map: 标签映射
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        ap_best: 最佳AP
        auc_best: 最佳AUC
        model: 训练好的模型
    """
    # 创建模型，使用指定的融合类型
    from copy import deepcopy
    args_copy = deepcopy(args)
    args_copy.fusion_type = fusion_type
    
    # 为每个融合方法使用不同的模型路径
    original_model_path = args_copy.model_path
    args_copy.model_path = original_model_path.replace('.pth', f'_{fusion_type}.pth')
    checkpoint_path = args_copy.model_path.replace('.pth', '_checkpoint.pth')
    
    # 创建模型
    model = model_class(
        num_class=args_copy.classes_num,
        embed_dim=args_copy.embed_dim,
        visual_length=args_copy.visual_length,
        visual_width=args_copy.visual_width,
        visual_head=args_copy.visual_head,
        visual_layers=args_copy.visual_layers,
        attn_window=args_copy.attn_window,
        prompt_prefix=args_copy.prompt_prefix,
        prompt_postfix=args_copy.prompt_postfix,
        device=device,
        scales=args_copy.scales,
        fusion_type=fusion_type,
        use_baseline_init=args_copy.use_baseline_init
    )
    
    # 训练
    ap_best = train(model, train_loader, test_loader, args_copy, label_map, device, logger, checkpoint_path)
    
    # 获取最佳AUC (从checkpoint加载)
    checkpoint = torch.load(checkpoint_path)
    auc_best = checkpoint.get('auc', checkpoint.get('AUC', 0))
    
    return ap_best, auc_best, model


def setup_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # 解析参数
    args = xd_option_seg.parser.parse_args()
    
    # checkpoint自动在model_path后加_checkpoint后缀
    checkpoint_path = args.model_path.replace('.pth', '_checkpoint.pth')

    # 自动创建必要的目录
    os.makedirs(os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else 'model', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 设置标签映射 (XD-Violence)
    label_map = dict({
        'A': 'normal', 
        'B1': 'fighting', 
        'B2': 'shooting', 
        'B4': 'riot', 
        'B5': 'abuse', 
        'B6': 'car accident', 
        'G': 'explosion'
    })
    
    # 创建日志记录器
    logger, log_file = setup_logger(args)
    logger.info(f"Log file: {log_file}")
    
    # 加载数据集
    logger.info("Loading datasets...")
    train_dataset = XDDatasetSeg(args.visual_length, args.train_list, False, label_map, args.scales)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = XDDatasetSeg(args.visual_length, args.test_list, True, label_map, args.scales)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Support fusion-type all - run all fusion methods and compare
    if args.fusion_type == "all":
        logger.info("="*80)
        logger.info("Running ALL fusion methods for comparison")
        logger.info("Fusion types: attention, concat, weighted")
        logger.info("="*80)
        
        fusion_results = {}
        
        # 1. Attention fusion
        print("")
        print("="*60)
        print("Training with ATTENTION fusion...")
        print("="*60)
        ap_attn, auc_attn, _ = train_single_fusion(
            CLIPVADSeg, "attention", train_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["attention"] = {"AP": ap_attn, "AUC": auc_attn}
        
        # 2. Concat fusion
        print("")
        print("="*60)
        print("Training with CONCAT fusion...")
        print("="*60)
        ap_concat, auc_concat, _ = train_single_fusion(
            CLIPVADSeg, "concat", train_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["concat"] = {"AP": ap_concat, "AUC": auc_concat}
        
        # 3. Weighted fusion
        print("")
        print("="*60)
        print("Training with WEIGHTED fusion...")
        print("="*60)
        ap_weight, auc_weight, _ = train_single_fusion(
            CLIPVADSeg, "weighted", train_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["weighted"] = {"AP": ap_weight, "AUC": auc_weight}
        
        # Print comparison results
        logger.info("")
        logger.info("FUSION METHODS COMPARISON RESULTS")
        logger.info("="*80)
        logger.info(f"{'Fusion Type':<15} {'AP':>10} {'AUC':>10}")
        logger.info("-"*40)
        for ft, res in fusion_results.items():
            logger.info(f"{ft:<15} {res['AP']:>10.4f} {res['AUC']:>10.4f}")
        
        # Find best method
        best_ft = max(fusion_results.keys(), key=lambda x: fusion_results[x]["AP"])
        logger.info("-"*40)
        logger.info(f"Best method: {best_ft} (AP: {fusion_results[best_ft]['AP']:.4f}, AUC: {fusion_results[best_ft]['AUC']:.4f})")
        logger.info("="*80)
        
        print("")
        print("="*60)
        print("All fusion methods comparison completed!")
        print("="*60)
        exit(0)
    
    # 创建多尺度模型
    logger.info("Creating multi-scale model...")
    model = CLIPVADSeg(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        prompt_prefix=args.prompt_prefix,
        prompt_postfix=args.prompt_postfix,
        device=device,
        scales=args.scales,
        fusion_type=args.fusion_type,
        use_baseline_init=args.use_baseline_init
    )
    
    # 训练
    ap_best = train(model, train_loader, test_loader, args, label_map, device, logger, checkpoint_path)
    
    print(f"\nTraining completed. Best AP: {ap_best:.4f}")
    print(f"Log saved to: {log_file}")
