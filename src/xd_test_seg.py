"""
XD-Violence Multi-Scale Test Script

多尺度视频异常检测测试脚本

功能：
1. 加载多尺度训练好的模型
2. 在测试集上评估性能
3. 计算AUC、AP等指标
4. 支持baseline模式对比

Author: VadCLIP-Multi-Seg
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time

from model_seg import CLIPVADSeg
from utils.dataset_seg import XDDatasetSeg
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option_seg


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, logger=None, use_baseline=False, use_scheme1=False):
    """
    测试函数
    
    Args:
        model: 训练好的模型
        testdataloader: 测试数据加载器
        maxlen: 最大序列长度
        prompt_text: 提示文本列表
        gt: ground truth
        gtsegments: 标注片段
        gtlabels: 标注标签
        device: 计算设备
        logger: 日志记录器（可选）
        use_baseline: 是否使用baseline模式
    """
    model.to(device)
    model.eval()

    element_logits2_stack = []
    
    test_start_time = time.time()

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            
            # Padding处理
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            # 处理长视频：分段预测
            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            
            # 前向传播（可选择baseline模式）
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths, use_baseline=use_baseline, use_scheme1=use_scheme1)
            
            # 维度调整
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            
            # 计算概率
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    test_time = time.time() - test_start_time
    
    # 计算指标
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    # 计算mAP
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        averageMAP += dmap[i]
    averageMAP = averageMAP / 5
    
    # 记录日志
    mode_str = "Baseline" if use_baseline else "Multi-Scale"
    if logger:
        logger.info(f"[{mode_str}] Test Results:")
        logger.info(f"  AUC1: {ROC1:.4f} | AP1: {AP1:.4f}")
        logger.info(f"  AUC2: {ROC2:.4f} | AP2: {AP2:.4f}")
        logger.info(f"  Average mAP: {averageMAP*100:.2f}%")
        logger.info(f"  Test time: {test_time:.2f}s")
    else:
        print(f"[{mode_str}] AUC1: {ROC1:.4f}, AP1: {AP1:.4f}")
        print(f"[{mode_str}] AUC2: {ROC2:.4f}, AP2: {AP2:.4f}")
        print(f"[{mode_str}] Average MAP: {averageMAP*100:.2f}%")

    return ROC1, AP1, ROC2, AP2, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option_seg.parser.parse_args()

    # XD-Violence标签映射
    label_map = dict({
        'A': 'normal', 
        'B1': 'fighting', 
        'B2': 'shooting', 
        'B4': 'riot', 
        'B5': 'abuse', 
        'B6': 'car accident', 
        'G': 'explosion'
    })

    # 加载测试数据集
    test_dataset = XDDatasetSeg(args.visual_length, args.test_list, True, label_map, args.scales)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 创建模型
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
    
    # 加载权重
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)
    
    print(f"Loaded model from: {args.model_path}")
    print(f"Scales: {args.scales}, Fusion: {args.fusion_type}")
    
    # 测试
    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, None, args.use_baseline_mode)
