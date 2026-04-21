"""
Multi-Scale Dataset Classes for XD-Violence and UCF-Crime

支持多尺度处理的Dataset类：
- 在__getitem__中返回多个尺度的特征
- 保持与baseline相同的接口以便对比

维度说明：
- 输入: 原始CLIP特征 [T, D]
- 输出: 保持原始长度，由模型进行多尺度划分

Author: VadCLIP-Multi-Seg
"""

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools


class UCFDatasetSeg(data.Dataset):
    """
    UCF-Crime多尺度数据集
    
    与UCFDataset接口相同，内部处理逻辑一致
    返回的feature由模型进行多尺度划分
    """
    
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, 
                 normal: bool = False, scales: list = [8, 16, 32]):
        """
        Args:
            clip_dim: 期望的特征长度（帧数）
            file_path: CSV文件路径
            test_mode: 是否为测试模式
            label_map: 标签映射
            normal: 是否只包含正常样本
            scales: 多尺度列表
        """
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.scales = scales
        
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
        print(f"[UCFDatasetSeg] Loaded {len(self.df)} samples, scales: {scales}")
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """获取样本"""
        clip_feature = np.load(self.df.loc[index]['path'])
        
        if self.test_mode == False:
            # 训练模式：统一采样到固定长度
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            # 测试模式：分割为多个测试片段
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        
        # 返回多尺度信息（在训练时使用）
        return clip_feature, clip_label, clip_length


class XDDatasetSeg(data.Dataset):
    """
    XD-Violence多尺度数据集
    
    与XDDataset接口相同，内部处理逻辑一致
    返回的feature由模型进行多尺度划分
    """
    
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 scales: list = [8, 16, 32]):
        """
        Args:
            clip_dim: 期望的特征长度（帧数）
            file_path: CSV文件路径
            test_mode: 是否为测试模式
            label_map: 标签映射
            scales: 多尺度列表
        """
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.scales = scales
        
        print(f"[XDDatasetSeg] Loaded {len(self.df)} samples, scales: {scales}")
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """获取样本"""
        clip_feature = np.load(self.df.loc[index]['path'])
        
        if self.test_mode == False:
            # 训练模式：统一采样到固定长度
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            # 测试模式：分割为多个测试片段
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        
        return clip_feature, clip_label, clip_length
