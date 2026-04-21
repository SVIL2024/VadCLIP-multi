"""
UCF-Crime Multi-Scale Configuration

多尺度视频异常检测配置文件
包含所有可调参数，支持与baseline对比

核心参数：
- scales: 多尺度列表，如[8, 16, 32]
- fusion_type: 融合方式 ('attention', 'concat', 'weighted')
- use_baseline: 是否同时训练baseline模式进行对比

Author: VadCLIP-Multi-Seg
"""

import argparse

parser = argparse.ArgumentParser(description='VadCLIP-Seg for UCF-Crime')

# ==================== 随机种子 ====================
parser.add_argument('--seed', default=999999, type=int)

# ==================== 模型架构参数 ====================
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)  # 输入序列长度
parser.add_argument('--visual-width', default=512, type=int)  # 特征维度
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)  # UCF-Crime: 2类

# ==================== 多尺度参数（核心） ====================
parser.add_argument('--scales', nargs='+', default=[1, 8, 16], type=int,
                    help='Multi-scale segment sizes, e.g., --scales 1 8 16')
parser.add_argument('--fusion-type', default='attention', type=str,
                    choices=['attention', 'concat', 'weighted', 'all'],
                    help='Multi-scale fusion method: attention/concat/weighted, or all to compare all methods')

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    return False

parser.add_argument('--use-baseline-init', default=False, type=str_to_bool,
                    help='Initialize from baseline model')
parser.add_argument('--use-baseline-mode', default=False, type=str_to_bool,
                    help='Use baseline (single-scale) mode for comparison')
parser.add_argument('--use-scheme1', default=False, type=str_to_bool,
                    help='Use Scheme 1: per-scale vision-text interaction')
parser.add_argument('--use-gcn', default=True, type=str_to_bool,
                    help='use gcn or not')
# parser.add_argument('--use-enhanced-fusion', default=True, type=str_to_bool,
#                     help='Use enhanced multi-scale fusion with cross-scale attention')

# ==================== 训练参数 ====================
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='../model/model_ucf_seg.pth')
parser.add_argument('--use-checkpoint', default=False, type=str_to_bool)
parser.add_argument('--batch-size', default=48, type=int)
parser.add_argument('--train-list', default='../list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='../list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='../list/gt_label_ucf.npy')
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-type', default='multistep', choices=['multistep', 'cosine'],
                    help='Learning rate scheduler type: multistep or cosine')
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[2, 4])

# ==================== 日志参数 ====================
parser.add_argument('--log-dir', default='logs/', type=str,
                    help='Directory for saving training logs')
parser.add_argument('--log-interval', default=100, type=int,
                    help='Logging interval in steps')
