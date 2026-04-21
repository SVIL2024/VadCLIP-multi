"""
UCF-Crime Multi-Scale Training Script

多尺度视频异常检测训练脚本

核心功能：
1. 多尺度模型训练（支持8/16/32帧尺度）
2. Normal/Anomaly分离训练（UCF特色）
3. Baseline模式对比训练
4. 详细日志记录
5. 训练时间统计

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
from ucf_test_seg import test
from utils.dataset_seg import UCFDatasetSeg
from utils.tools import get_prompt_text, get_batch_label
import ucf_option_seg


def CLASM(logits, labels, lengths, device):
    """
    Multiple Instance Learning Loss (CLASM)
    
    用于弱监督异常检测的MIL损失函数
    """
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    """
    Binary Classification Loss (CLAS2)
    
    二分类损失，用于异常/正常的二分类
    """
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp), dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def setup_logger(args):
    """设置训练日志记录器"""
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = logging.getLogger('VadCLIP-Seg-UCF')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'ucf_train_seg_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device, logger, checkpoint_path):
    """
    UCF训练函数
    """
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    auc_best = 0
    epoch = 0
    
    total_start_time = time.time()
    
    logger.info("="*60)
    logger.info("VadCLIP-Seg Training Configuration (UCF-Crime)")
    logger.info("="*60)
    logger.info(f"Scales: {args.scales}")
    logger.info(f"Fusion Type: {args.fusion_type}")
    logger.info(f"Visual Length: {args.visual_length}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Max Epoch: {args.max_epoch}")
    logger.info(f"Use Baseline Mode: {args.use_baseline_mode}")
    logger.info(f"Use Scheme 1: {args.use_scheme1}")
    logger.info("="*60)

    if args.use_checkpoint == True:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint.get('ap', checkpoint.get('AP', 0))
        auc_best = checkpoint.get('auc', checkpoint.get('AUC', 0))
        logger.info(f"Loaded checkpoint: epoch {epoch+1}, AP: {ap_best:.4f}, AUC: {auc_best:.4f}")

    step = 0
    
    for e in range(args.max_epoch):
        epoch_start_time = time.time()
        
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        
        # UCF使用交替的normal和anomaly样本训练
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            # 加载normal和anomaly样本
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            # 合并batch
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # 前向传播
            if args.use_scheme1 and not args.use_baseline_mode:
                text_features, logits1, logits2 = model(
                    visual_features, None, prompt_text, feat_lengths,
                    use_baseline=False,
                    use_scheme1=True
                )
                
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device)
                
            else:
                text_features, logits1, logits2 = model(
                    visual_features, None, prompt_text, feat_lengths,
                    use_baseline=args.use_baseline_mode
                )
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            
            loss_total1 += loss1.item()
            loss_total2 += loss2.item()

            # 文本特征正则化损失
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1
            loss_total3 += loss3.item()

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 定期打印训练信息
            if i % args.log_interval == 0 and i != 0:
                logger.info(
                    f"Epoch [{e+1}/{args.max_epoch}] "
                    f"Step [{i}] "
                    f"Loss1: {loss_total1/(i+1):.4f} | "
                    f"Loss2: {loss_total2/(i+1):.4f} | "
                    f"Loss3: {loss_total3/(i+1):.4f}"
                )

        scheduler.step()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        logger.info("-"*60)
        logger.info(f"Epoch {e+1} completed in {epoch_time:.2f}s")
        
        test_interval = 1
        if (e + 1) % test_interval == 0:
            ROC1, AP1, ROC2, AP2 = test(model, test_loader, args.visual_length, prompt_text, 
                           gt, gtsegments, gtlabels, device, logger, args.use_baseline_mode, args.use_scheme1)
            AUC = max(ROC1, ROC2)
            AP = max(AP1, AP2)
            if AUC > 0:
                logger.info(f"Test Result -> AP1: {AP1:.4f}, AUC1: {ROC1:.4f} | AP2: {AP2:.4f}, AUC2: {ROC2:.4f}")
        else:
            AUC, AP = 0, 0
            ROC1, AP1, ROC2, AP2 = 0, 0, 0, 0

        if AUC > auc_best:
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
                    'visual_length': args.visual_length
                }
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"New best model saved! AP: {ap_best:.4f}, AUC: {auc_best:.4f}")
        else:
            if AUC > 0:
                logger.info(f"Current Best -> AP: {ap_best:.4f}, AUC: {auc_best:.4f}")

    total_time = time.time() - total_start_time
    checkpoint = torch.load(checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    
    logger.info("="*60)
    logger.info(f"Training completed!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best AP: {ap_best:.4f}, Best AUC: {auc_best:.4f}")
    logger.info(f"Model saved to: {args.model_path}")
    logger.info("="*60)
    
    return ap_best


# Training function for single fusion type
def train_single_fusion(model_class, fusion_type, normal_loader, anomaly_loader, test_loader, args_base, label_map, device, logger):
    args = deepcopy(args_base)
    args.fusion_type = fusion_type
    
    logger.info("="*60)
    logger.info(f"Training with fusion_type: {fusion_type}")
    logger.info("="*60)
    
    # Create model
    model = model_class(
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
        use_baseline_init=args.use_baseline_init,
    )
    
    # Modify checkpoint path
    checkpoint_path = args.model_path.replace(".pth", f"_{fusion_type}_checkpoint.pth")
    
    # Train (UCF uses normal/anomaly separate loaders)
    ap_best = train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device, logger, checkpoint_path)
    
    # Load best model and return results
    checkpoint = torch.load(checkpoint_path)
    return ap_best, checkpoint.get('auc', checkpoint.get('AUC', 0)), checkpoint_path


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = ucf_option_seg.parser.parse_args()
    
    # checkpoint自动在model_path后加_checkpoint后缀
    checkpoint_path = args.model_path.replace('.pth', '_checkpoint.pth')

    # 自动创建必要的目录
    os.makedirs(os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else 'model', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    setup_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # UCF-Crime标签映射
    label_map = dict({
        'Normal': 'normal', 
        'Abuse': 'abuse', 
        'Arrest': 'arrest', 
        'Arson': 'arson', 
        'Assault': 'assault', 
        'Burglary': 'burglary', 
        'Explosion': 'explosion', 
        'Fighting': 'fighting', 
        'RoadAccidents': 'roadAccidents', 
        'Robbery': 'robbery', 
        'Shooting': 'shooting', 
        'Shoplifting': 'shoplifting', 
        'Stealing': 'stealing', 
        'Vandalism': 'vandalism'
    })
    
    logger, log_file = setup_logger(args)
    logger.info(f"Log file: {log_file}")
    
    # 加载数据集 - UCF使用normal/anomaly分离
    logger.info("Loading datasets...")
    normal_dataset = UCFDatasetSeg(args.visual_length, args.train_list, False, label_map, True, args.scales)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    anomaly_dataset = UCFDatasetSeg(args.visual_length, args.train_list, False, label_map, False, args.scales)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    test_dataset = UCFDatasetSeg(args.visual_length, args.test_list, True, label_map, False, args.scales)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    logger.info(f"Normal samples: {len(normal_dataset)}")
    logger.info(f"Anomaly samples: {len(anomaly_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
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
        use_baseline_init=args.use_baseline_init,
    )
    
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
            CLIPVADSeg, "attention", normal_loader, anomaly_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["attention"] = {"AP": ap_attn, "AUC": auc_attn}
        
        # 2. Concat fusion
        print("")
        print("="*60)
        print("Training with CONCAT fusion...")
        print("="*60)
        ap_concat, auc_concat, _ = train_single_fusion(
            CLIPVADSeg, "concat", normal_loader, anomaly_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["concat"] = {"AP": ap_concat, "AUC": auc_concat}
        
        # 3. Weighted fusion
        print("")
        print("="*60)
        print("Training with WEIGHTED fusion...")
        print("="*60)
        ap_weight, auc_weight, _ = train_single_fusion(
            CLIPVADSeg, "weighted", normal_loader, anomaly_loader, test_loader, args, 
            label_map, device, logger
        )
        fusion_results["weighted"] = {"AP": ap_weight, "AUC": auc_weight}
        
        # Print comparison results
        logger.info("")
        logger.info("="*80)
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

    # 训练
    ap_best = train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device, logger, checkpoint_path)
    
    print(f"\nTraining completed. Best AP: {ap_best:.4f}")
    print(f"Log saved to: {log_file}")
