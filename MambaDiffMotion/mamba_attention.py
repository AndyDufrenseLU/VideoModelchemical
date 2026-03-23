"""
混合时空模块 (Hybrid Space-Time Module)
- 时序方向: Mamba SSM (O(N) 线性复杂度，替代 Temporal Attention)
- 空间方向: Multi-Head Self-Attention (与 TimeSformer 相同)

显存优化:
- Temporal Mamba 使用 gradient checkpointing（梯度检查点）
  正向传播不保留中间激活(dA/dBu/h)，反向传播时重新计算
  牺牲约 33% 计算时间，显存从 ~85GB 降到可接受范围

设计原理:
  原始 TimeSformer:   Temporal Attn(O(T²)) → Spatial Attn(O(P²)) → MLP
  本模块:             Temporal Mamba(O(T))  → Spatial Attn(O(P²)) → MLP
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from einops import rearrange
from mamba_block import MambaBlock


# ============================================================
# 多头自注意力（空间方向，与 TimeStransformer1/attention.py 相同）
# ============================================================

class MultiHeadAttention(nn.Module):
    """标准多头自注意力"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # PyTorch 内置 FlashAttention（若 CUDA 支持）
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# 时序 Mamba 模块（替代 TemporalAttention）
# ============================================================

class TemporalMambaBlock(nn.Module):
    """
    时序 Mamba 模块:
    对同一空间位置的 T 帧 token 做 Mamba SSM 扫描

    原来: Reshape → (B*P, T, D) → Temporal Attention → Reshape
    现在: Reshape → (B*P, T, D) → Mamba SSM → Reshape

    优势:
      - O(T) 线性复杂度（vs Attention O(T²)）
      - 帧数增多时，显存不爆炸
    """

    def __init__(self, dim, num_frames=10, d_state=16, d_conv=4, expand=2,
                 use_checkpoint=True):
        super().__init__()
        self.num_frames = num_frames
        self.use_checkpoint = use_checkpoint
        self.mamba = MambaBlock(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x, num_patches_per_frame):
        """
        Args:
            x: (B, 1+T*P, D)
        Returns:
            x: (B, 1+T*P, D)

        显存说明:
          Mamba SSM 在时序维度扫描时，每步产生 (B*P, D_inner, N) 的中间张量。
          对于 B=16, P=196, D_inner=1536, N=16, 12层: 约 85 GB 累积激活。
          use_checkpoint=True 时启用梯度检查点：正向传播不保留这些中间张量，
          反向传播时重新计算，显存从 85GB 大幅降低。
        """
        B, N, D = x.shape

        # 拆出 CLS token
        cls_token = x[:, 0:1, :]    # (B, 1, D)
        x_patches = x[:, 1:, :]     # (B, T*P, D)

        # 重排: 同一空间位置的 T 帧放在一起
        # (B, T*P, D) → (B*P, T, D)
        x_patches = rearrange(x_patches, 'b (t p) d -> (b p) t d', t=self.num_frames)

        # Mamba SSM 在时序维度扫描
        # ✅ use_reentrant=False 是 PyTorch 推荐的新 API，避免部分 autograd 问题
        if self.use_checkpoint and self.training:
            x_patches = ckpt.checkpoint(self.mamba, x_patches, use_reentrant=True)
        else:
            x_patches = self.mamba(x_patches)  # (B*P, T, D)

        # 还原
        x_patches = rearrange(x_patches, '(b p) t d -> b (t p) d', b=B)

        # 拼上 CLS
        x = torch.cat([cls_token, x_patches], dim=1)
        return x


# ============================================================
# 空间 Transformer 注意力（保留，与 TimeSformer1 相同）
# ============================================================

class SpatialAttention(nn.Module):
    """
    空间注意力: 在同一帧内做 patch 级自注意力
    与 TimeStransformer1 的 SpatialAttention 完全相同
    """

    def __init__(self, dim, num_heads=8, num_frames=10, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.attention = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, x, num_patches_per_frame):
        B, N, D = x.shape

        cls_token = x[:, 0:1, :]   # (B, 1, D)
        x_patches = x[:, 1:, :]    # (B, T*P, D)

        # 重排为 (B, T, P, D)
        x_patches = rearrange(x_patches, 'b (t p) d -> b t p d', t=self.num_frames)

        # CLS 扩展并加入每帧做 spatial attention
        cls_expanded = cls_token.unsqueeze(1).expand(-1, self.num_frames, -1, -1)  # (B, T, 1, D)
        x_with_cls = torch.cat([cls_expanded, x_patches], dim=2)                  # (B, T, 1+P, D)

        x_with_cls = rearrange(x_with_cls, 'b t n d -> (b t) n d')
        x_with_cls = self.attention(x_with_cls)                                    # (B*T, 1+P, D)
        x_with_cls = rearrange(x_with_cls, '(b t) n d -> b t n d', b=B)

        # 聚合所有帧的 CLS
        new_cls = x_with_cls[:, :, 0, :].mean(dim=1, keepdim=True)                # (B, 1, D)

        x_patches = x_with_cls[:, :, 1:, :]                                       # (B, T, P, D)
        x_patches = rearrange(x_patches, 'b t p d -> b (t p) d')

        return torch.cat([new_cls, x_patches], dim=1)


# ============================================================
# 混合时空模块（Temporal Mamba + Spatial Attention）
# ============================================================

class HybridSpaceTimeMamba(nn.Module):
    """
    混合时空模块:
      1. 时序方向: Temporal Mamba SSM + 残差 + LayerNorm
      2. 空间方向: Spatial Attention + 残差 + LayerNorm

    与 DividedSpaceTimeAttention 接口完全兼容（可无缝替换）
    """

    def __init__(self, dim, num_heads, num_frames,
                 d_state=16, d_conv=4, mamba_expand=2,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 use_checkpoint=True):
        super().__init__()

        # 时序: Mamba SSM（默认启用梯度检查点节省显存）
        self.temporal_mamba = TemporalMambaBlock(
            dim=dim,
            num_frames=num_frames,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            use_checkpoint=use_checkpoint,
        )
        self.temporal_norm = nn.LayerNorm(dim)

        # 空间: Multi-Head Attention（与 TimeSformer1 一致）
        self.spatial_attn = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            num_frames=num_frames,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.spatial_norm = nn.LayerNorm(dim)

    def forward(self, x, num_patches_per_frame):
        """
        Args:
            x: (B, 1+T*P, D)
        Returns:
            x: (B, 1+T*P, D)
        """
        # 时序 Mamba（内部已有 Pre-Norm + 残差）
        x = self.temporal_mamba(self.temporal_norm(x), num_patches_per_frame) + x

        # 空间 Attention（Pre-Norm + 残差）
        x = self.spatial_attn(self.spatial_norm(x), num_patches_per_frame) + x

        return x
