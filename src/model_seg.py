"""
VadCLIP-Seg: Multi-Scale Video Anomaly Detection

核心思想：
- 多尺度时序建模：同时处理8帧、16帧、32帧的segment
- 注意力融合：学习不同尺度特征的融合权重
- 模块化设计：支持baseline模式（单尺度）进行对比

维度变化说明：
- 输入: [batch_size, sequence_length, feature_dim] - frame-level CLIP features
- 多尺度分割: 将sequence按不同尺度划分为segments
- 每尺度编码: [batch_size, num_segments, scale, feature_dim]
- 融合后: [batch_size, sequence_length, feature_dim]

Author: VadCLIP-Multi-Seg
"""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):
    """LayerNorm with automatic dtype preservation"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU activation function - faster than standard GELU"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer residual attention block with windowed attention"""
    
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    """Transformer encoder for temporal modeling"""
    
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class MultiScaleTemporalEncoder(nn.Module):
    """
    多尺度时序编码器
    
    对每个尺度（8帧、16帧、32帧）分别进行时序建模
    维度变化：
    - 输入: [batch_size * num_segments, scale, feature_dim]
    - 位置嵌入: 添加时序位置信息
    - Transformer输出: [batch_size * num_segments, scale, feature_dim]
    """
    
    def __init__(self, scale: int, visual_width: int, visual_head: int, visual_layers: int, device):
        super().__init__()
        self.scale = scale
        self.visual_width = visual_width
        
        # 为每个尺度创建独立的Transformer
        # 窗口大小设为scale，实现segment内全连接注意力
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(scale)
        )
        
        # 位置嵌入：为每个时序位置学习位置表示
        self.frame_position_embeddings = nn.Embedding(scale, visual_width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        """
        构建注意力掩码，支持窗口内注意力
        维度: [attn_window, attn_window]
        """
        mask = torch.empty(attn_window, attn_window)
        mask.fill_(float('-inf'))
        # 窗口内全连接，窗口间无连接
        mask[:attn_window, :attn_window] = 0
        return mask

    def forward(self, x):
        """
        Args:
            x: [batch_size * num_segments, scale, feature_dim]
        Returns:
            encoded: [batch_size * num_segments, scale, feature_dim]
        """
        batch_segments = x.shape[0]
        
        # 生成位置索引 [0, 1, 2, ..., scale-1]
        position_ids = torch.arange(self.scale, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_segments, -1)
        
        # 位置嵌入: [batch_segments, scale, feature_dim]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        
        # 添加位置信息: [batch_segments, scale, feature_dim]
        x = x + frame_position_embeddings
        
        # 维度调整: Transformer期望 [scale, batch, feature]
        x = x.permute(1, 0, 2)
        
        # 时序建模: [scale, batch_segments, feature_dim]
        x, _ = self.temporal((x, None))
        
        # 维度恢复: [batch_segments, scale, feature_dim]
        x = x.permute(1, 0, 2)
        
        return x


class AttentionFusion(nn.Module):
    """
    注意力融合模块
    
    核心思想：学习不同尺度特征的融合权重
    维度变化：
    - 输入: [batch_size, num_scales, num_segments, feature_dim]
    - 注意力权重: [batch_size, num_scales, num_segments, 1]
    - 输出: [batch_size, num_segments, feature_dim]
    
    实现方式：
    1. 对每个segment的各尺度特征计算注意力权重
    2. 加权求和得到融合特征
    """
    
    def __init__(self, num_scales: int, feature_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # 可学习的注意力权重参数
        # 初始化为均匀分布，让模型自动学习最优权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 可选的投影层，对融合后的特征进行变换
        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            QuickGELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: [batch_size, num_scales, num_segments, feature_dim]
        Returns:
            fused: [batch_size, num_segments, feature_dim]
        """
        batch_size = multi_scale_features.shape[0]
        num_segments = multi_scale_features.shape[2]
        
        # 计算注意力权重（softmax归一化）
        # 维度: [num_scales] -> [1, num_scales, 1, 1]
        weights = F.softmax(self.scale_weights, dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # 应用注意力权重
        # 维度: [batch_size, num_scales, num_segments, feature_dim] * [1, num_scales, 1, 1]
        weighted_features = multi_scale_features * weights
        
        # 沿尺度维度求和: [batch_size, num_segments, feature_dim]
        fused = torch.sum(weighted_features, dim=1)
        
        # 残差连接和投影
        # 直接对多尺度特征取平均作为残差
        avg_features = torch.mean(multi_scale_features, dim=1)
        fused = fused + self.fusion_projection(avg_features)
        
        return fused


class GraphAggregation(nn.Module):
    """
    图卷积聚合模块
    
    继承自baseline的GCN设计，使用两种邻接矩阵：
    1. 距离-based邻接矩阵：基于特征相似度构建
    2. 全连接邻接矩阵：学习得到的关系
    
    维度变化：
    - 输入: [batch_size, sequence_length, feature_dim]
    - GCN输出: [batch_size, sequence_length, feature_dim * 2]
    """
    
    def __init__(self, visual_width: int):
        super().__init__()
        
        width = int(visual_width / 2)
        
        # 双图卷积分支
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

    def adj4(self, x, seq_len):
        """基于特征相似度构建邻接矩阵"""
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def forward(self, x, lengths):
        """
        Args:
            x: [batch_size, sequence_length, feature_dim]
            lengths: 实际序列长度
        Returns:
            output: [batch_size, sequence_length, feature_dim]
        """
        # 构建邻接矩阵
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        
        # 双分支图卷积
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        # 拼接并变换
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        
        return x


class CLIPVADSeg(nn.Module):
    """
    VadCLIP-Seg: 多尺度视频异常检测模型
    
    核心设计：
    1. 多尺度segment划分：将视频按不同尺度(8,16,32帧)划分为segments
    2. 独立编码：每个尺度使用独立的Transformer进行时序建模
    3. 注意力融合：学习不同尺度特征的融合权重
    4. 图卷积聚合：捕捉帧间关系（继承自baseline）
    5. Baseline模式：支持单尺度模式与原始VadCLIP对比
    
    维度流程：
    输入 [B, T, D] -> 多尺度分割 [B, S, scale, D] -> 编码 [B, S, scale, D] 
    -> 融合 [B, S*scale, D] -> GCN [B, T, D] -> 输出
    
    Args:
        num_class: 异常类别数量
        embed_dim: 文本嵌入维度
        visual_length: 输入序列长度（帧数）
        visual_width: 视觉特征维度
        visual_head: 注意力头数
        visual_layers: Transformer层数
        attn_window: 注意力窗口大小
        prompt_prefix: 提示前缀长度
        prompt_postfix: 提示后缀长度
        device: 设备
        scales: 多尺度列表，如[8, 16, 32]
        fusion_type: 融合方式 ('attention', 'concat', 'weighted')
        use_baseline_init: 是否使用baseline初始化
    """
    
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 scales: list = [8, 16, 32],
                 fusion_type: str = 'attention',
                 use_baseline_init: bool = False):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.scales = scales
        self.num_scales = len(scales)
        self.fusion_type = fusion_type
        self.use_baseline_init = use_baseline_init
        
        # 验证scales是否合法
        assert all(s <= visual_length for s in scales), "Scale must be <= visual_length"
        
        # print(f"[VadCLIP-Seg] Initializing with scales: {scales}")
        # print(f"[VadCLIP-Seg] Fusion type: {fusion_type}")
        # print(f"[VadCLIP-Seg] Visual length: {visual_length}")

        # ==================== 多尺度时序编码器 ====================
        # 为每个尺度创建独立的时序编码器
        self.scale_encoders = nn.ModuleList([
            MultiScaleTemporalEncoder(
                scale=scale,
                visual_width=visual_width,
                visual_head=visual_head,
                visual_layers=visual_layers,
                device=device
            ) for scale in scales
        ])
        
        # ==================== 融合模块 ====================
        if fusion_type == 'attention':
            # 注意力融合：学习融合权重
            self.fusion = AttentionFusion(num_scales=self.num_scales, feature_dim=visual_width)
        elif fusion_type == 'concat':
            # 拼接融合：直接拼接多尺度特征
            self.concat_projection = nn.Linear(visual_width * self.num_scales, visual_width)
        elif fusion_type == 'weighted':
            # 加权融合：简单加权平均
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
            self.fusion_projection = nn.Linear(visual_width, visual_width)
        
        # ==================== 图卷积聚合（继承自baseline） ====================
        self.graph_aggregation = GraphAggregation(visual_width)
        
        # ==================== Baseline时序编码器（用于对比） ====================
        # 如果需要与baseline对比，创建原始的Transformer
        self.baseline_temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_baseline_attention_mask(visual_length)
        )
        self.baseline_position_embeddings = nn.Embedding(visual_length, visual_width)

        # ==================== 分类器 ====================
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        # ==================== CLIP模型 ====================
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.baseline_position_embeddings.weight, std=0.01)

    def build_baseline_attention_mask(self, attn_window):
        """构建baseline模式的注意力掩码"""
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def segment_video(self, features, scale):
        """
        将视频特征划分为多个segments
        
        维度变化：
        - 输入: [batch_size, sequence_length, feature_dim]
        - 输出: [batch_size, num_segments, scale, feature_dim]
        
        例如：256帧，scale=8 -> 32个segments
        """
        # 修正：features.shape[0]是batch_size，shape[1]是sequence_length
        batch_size = features.shape[0]
        sequence_length = features.shape[1]
        feature_dim = features.shape[2]
        
        # 计算segment数量
        num_segments = sequence_length // scale
        if sequence_length % scale != 0:
            num_segments += 1
        
        # 划分segments
        # 每个segment取scale帧，不足则padding
        segments = []
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = min(start_idx + scale, sequence_length)
            
            # 提取segment特征
            segment = features[:, start_idx:end_idx, :]
            
            # Padding到固定scale长度
            if segment.shape[1] < scale:
                padding = torch.zeros(batch_size, scale - segment.shape[1], feature_dim).to(features.device)
                segment = torch.cat([segment, padding], dim=1)
            
            segments.append(segment)
        
        # 堆叠: [batch_size, num_segments, scale, feature_dim]
        return torch.stack(segments, dim=1)

    def fuse_multi_scale(self, multi_scale_features):
        """
        融合多尺度特征
        
        维度变化（attention模式）：
        - 输入: [batch_size, num_scales, num_segments_i, scale_i, feature_dim]
        - 插值后: [batch_size, num_scales, max_segments, feature_dim]
        - 融合: [batch_size, max_segments, feature_dim]
        
        这里简化处理：对每尺度取池化后拼接
        """
        batch_size = multi_scale_features[0].shape[0]
        
        if self.fusion_type == 'attention':
            # 获取最大segment数量
            max_segments = max(feat.shape[1] for feat in multi_scale_features)
            
            # 对齐到最大segment数量
            pooled_features = []
            for feat, scale in zip(multi_scale_features, self.scales):
                pooled = feat.mean(dim=2)
                
                # 插值对齐到max_segments
                if pooled.shape[1] != max_segments:
                    pooled = F.interpolate(
                        pooled.permute(0, 2, 1),  # [B, D, S]
                        size=max_segments,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)  # [B, max_segments, D]
                
                pooled_features.append(pooled)
            
            # 堆叠: [batch_size, num_scales, max_segments, feature_dim]
            stacked = torch.stack(pooled_features, dim=1)
            
            # 融合: [batch_size, max_segments, feature_dim]
            fused = self.fusion(stacked)
            return fused
            
        elif self.fusion_type == 'concat':
            # 获取最大segment数量并对齐
            max_segments = max(feat.shape[1] for feat in multi_scale_features)
            
            pooled_features = []
            for feat, scale in zip(multi_scale_features, self.scales):
                pooled = feat.mean(dim=2)
                
                # 插值对齐
                if pooled.shape[1] != max_segments:
                    pooled = F.interpolate(
                        pooled.permute(0, 2, 1),
                        size=max_segments,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)
                pooled_features.append(pooled)
            
            # 拼接: [batch_size, num_scales * max_segments, feature_dim]
            concat = torch.cat(pooled_features, dim=1)
            
            # Reshape for projection: [B, num_scales*max_segments, D] -> [B*max_segments, num_scales*D]
            batch_size = concat.shape[0]
            concat_reshaped = concat.view(batch_size * max_segments, self.num_scales * self.visual_width)
            
            # Project each timestep: [B*S, num_scales*D] -> [B*S, D]
            fused_reshaped = self.concat_projection(concat_reshaped)
            
            # Reshape back: [B*S, D] -> [B, S, D]
            fused = fused_reshaped.view(batch_size, max_segments, self.visual_width)
            
            # Interpolate to original sequence length
            if fused.shape[1] != self.visual_length:
                fused = F.interpolate(
                    fused.permute(0, 2, 1),
                    size=self.visual_length,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
            
            return fused
            
        elif self.fusion_type == 'weighted':
            # 获取最大segment数量并对齐
            max_segments = max(feat.shape[1] for feat in multi_scale_features)
            weights = F.softmax(self.scale_weights, dim=0)
            
            pooled_features = []
            for feat, scale, w in zip(multi_scale_features, self.scales, weights):
                pooled = feat.mean(dim=2) * w
                
                # 插值对齐
                if pooled.shape[1] != max_segments:
                    pooled = F.interpolate(
                        pooled.permute(0, 2, 1),
                        size=max_segments,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)
                pooled_features.append(pooled)
            
            # 加权求和
            fused = sum(pooled_features)
            fused = fused + self.fusion_projection(fused)
            return fused

    def encode_video_multi_scale(self, images, padding_mask, lengths, 
                                 return_multi_scale=False, use_gcn=True):
        """
        多尺度视频编码
        
        维度流程：
        1. 对每个尺度分别划分segments
        2. 每尺度独立时序编码
        3. 注意力融合多尺度特征
        4. 图卷积聚合
        
        Args:
            images: [batch_size, sequence_length, feature_dim]
            padding_mask: 填充掩码
            lengths: 实际长度
            return_multi_scale: 是否返回每尺度的编码结果（用于Scheme 1）
            
        Returns:
            encoded: [batch_size, sequence_length, feature_dim]
            (如果return_multi_scale=True): 额外返回 multi_scale_pooled: list of [B, S, D]
        """
        batch_size = images.shape[0]
        sequence_length = images.shape[1]
        
        # ==================== 多尺度编码 ====================
        multi_scale_encoded = []
        multi_scale_pooled = []  # 每尺度池化后的特征
        
        for scale, encoder in zip(self.scales, self.scale_encoders):
            # 划分segments: [B, T, D] -> [B, S, scale, D]
            segments = self.segment_video(images, scale)
            
            # 编码每个segment
            # segments shape: [B, S, scale, D]
            batch_size = segments.shape[0]
            num_segments = segments.shape[1]
            scale_size = segments.shape[2]
            feature_dim = segments.shape[3]
            
            # 展平为: [B * S, scale, D]
            segments_flat = segments.view(batch_size * num_segments, scale_size, feature_dim)
            encoded_segs = encoder(segments_flat)
            
            # 恢复维度: [B, S, scale, D]
            encoded_segs = encoded_segs.view(batch_size, num_segments, scale_size, feature_dim)
            multi_scale_encoded.append(encoded_segs)
            
            pooled = encoded_segs.mean(dim=2)
            multi_scale_pooled.append(pooled)
        
        # 如果需要返回每尺度特征（Scheme 1）
        if return_multi_scale:
            return multi_scale_pooled
        
        # ==================== 融合多尺度特征 ====================
        fused_features = self.fuse_multi_scale(multi_scale_encoded)
        
        # 将融合后的特征恢复到原始序列长度
        # 使用插值对齐到原始sequence_length
        if fused_features.shape[1] != sequence_length:
            fused_features = F.interpolate(
                fused_features.permute(0, 2, 1),  # [B, D, S] -> [B, D, T]
                size=sequence_length,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [B, T, D]
        
        # ==================== 图卷积聚合 ====================
        if use_gcn:
            graph_features = self.graph_aggregation(fused_features, lengths)
        else:
            graph_features = fused_features
        return graph_features

    def encode_video_baseline(self, images, padding_mask, lengths):
        """
        Baseline模式编码（与原始VadCLIP相同）
        
        维度变化：
        - 输入: [batch_size, sequence_length, feature_dim]
        - 位置嵌入: [sequence_length, batch_size, feature_dim]
        - Transformer: [sequence_length, batch_size, feature_dim]
        - GCN: [batch_size, sequence_length, feature_dim]
        """
        images = images.to(torch.float)
        
        # 位置嵌入
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.baseline_position_embeddings(position_ids)
        
        # 添加位置信息
        images = images.permute(1, 0, 2) + frame_position_embeddings.permute(1, 0, 2)
        
        # 时序建模
        x, _ = self.baseline_temporal((images, None))
        x = x.permute(1, 0, 2)
        
        # 图卷积聚合
        graph_features = self.graph_aggregation(x, lengths)
        
        return graph_features

    def encode_video(self, images, padding_mask, lengths, 
                     use_baseline=False, use_gcn=True):
        """
        视频编码主函数
        
        Args:
            images: 视觉特征
            padding_mask: 填充掩码
            lengths: 序列长度
            use_baseline: 是否使用baseline模式
        """
        if use_baseline:
            return self.encode_video_baseline(images, padding_mask, lengths)
        else:
            return self.encode_video_multi_scale(images, padding_mask, lengths,use_gcn=use_gcn)

    def encode_textprompt(self, text):
        """
        文本提示编码（与baseline相同）
        
        Args:
            text: 文本列表
        Returns:
            text_features: 文本特征
        """
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, visual, padding_mask, text, lengths, 
                use_baseline=False, use_scheme1=False, use_gcn=True):
        """
        前向传播
        
        Args:
            visual: 视觉特征 [batch_size, sequence_length, feature_dim]
            padding_mask: 填充掩码
            text: 文本提示
            lengths: 序列长度
            use_baseline: 是否使用 baseline 模式进行对比
            use_scheme1: 是否使用 Scheme 1（多尺度编码）
        
        Returns:
            text_features: 处理后的文本特征 (融合了视觉信息)
            logits1: 分类器输出
            logits2: 视觉 - 文本相似度
        """
        # Step 1: 编码视觉特征（多尺度 or 单尺度）
        if use_scheme1 and not use_baseline:
            visual_features = self.encode_video_multi_scale(
                visual, padding_mask, lengths, return_multi_scale=False, use_gcn=use_gcn
            )
        else:
            visual_features = self.encode_video(visual, padding_mask, lengths, use_baseline)
        
        # Step 2: Cross-attention (同 baseline)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))
        
        text_features_ori = self.encode_textprompt(text)
        
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)
        
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
        
        return text_features_ori, logits1, logits2
