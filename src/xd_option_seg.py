"""
XD-Violence Multi-Scale Configuration

多尺度视频异常检测配置文件
包含所有可调参数，支持与baseline对比

核心参数：
- scales: 多尺度列表，如[8, 16, 32]
- fusion_type: 融合方式 ('attention', 'concat', 'weighted')
- use_baseline: 是否同时训练baseline模式进行对比

Author: VadCLIP-Multi-Seg
"""

import argparse

parser = argparse.ArgumentParser(description='VadCLIP-Seg for XD-Violence')

# ==================== 随机种子 ====================
parser.add_argument('--seed', default=234, type=int)

# ==================== 模型架构参数 ====================
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)  # 输入序列长度
parser.add_argument('--visual-width', default=512, type=int)  # 特征维度
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

# ==================== 多尺度参数（核心） ====================
# scales: 多尺度列表，每个元素表示一个尺度的帧数
# 8帧：适合检测短期异常（shooting, car accident）
# 16帧：中等时长异常（fighting）
# 32帧：长期异常（riot, explosion）
parser.add_argument('--scales', nargs='+', default=[1, 8, 16], type=int,
                    help='Multi-scale segment sizes, e.g., --scales 8 16 32')

# fusion_type: 多尺度特征融合方式
# attention: 注意力融合（推荐）
# concat: 拼接融合
# weighted: 加权融合
parser.add_argument('--fusion-type', default='attention', type=str,
                    choices=['attention', 'concat', 'weighted', 'all'],
                    help='Multi-scale fusion method: attention/concat/weighted, or all to compare all methods')

# use_baseline_init: 是否使用baseline预训练权重初始化
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

# use_baseline_mode: 是否使用baseline模式（单尺度）进行对比训练
parser.add_argument('--use-baseline-mode', default=False, type=str_to_bool,
                    help='Use baseline (single-scale) mode for comparison')

# use_scheme1: 是否使用Scheme 1（每尺度独立视觉-文本交互）
parser.add_argument('--use-scheme1', default=False, type=str_to_bool,
                    help='Use Scheme 1: per-scale vision-text interaction')
parser.add_argument('--use-enhanced-fusion', default=False, type=str_to_bool,
                    help='Use enhanced multi-scale fusion with cross-scale attention')

# ==================== 训练参数 ====================
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='../model/model_xd_seg.pth')
parser.add_argument('--use-checkpoint', default=False, type=str_to_bool)
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--train-list', default='../list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../list/gt.npy')
parser.add_argument('--gt-segment-path', default='../list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='../list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[2, 6, 10])

# ==================== 日志参数 ====================
parser.add_argument('--log-dir', default='logs/', type=str,
                    help='Directory for saving training logs')
parser.add_argument('--log-interval', default=100, type=int,
                    help='Logging interval in steps')
