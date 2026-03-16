"""
Divided Space-Time Attention module for TimeSformer
Implements separate temporal and spatial attention mechanisms
"""

import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
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
        """
        Args:
            x: (B, N, D) - input tokens
            mask: Optional attention mask
        Returns:
            x: (B, N, D) - output after attention
        """
        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用PyTorch内置的高效attention (支持FlashAttention)
        # 比手动matmul节省4-10倍显存
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


class TemporalAttention(nn.Module):
    """
    Temporal attention: attend across time at the same spatial location
    """
    
    def __init__(self, dim, num_heads=8, num_frames=10, qkv_bias=False, 
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.attention = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        
    def forward(self, x, num_patches_per_frame):
        """
        Args:
            x: (B, 1+T*P, D) where T=num_frames, P=patches per frame
            num_patches_per_frame: P
        Returns:
            x: (B, 1+T*P, D)
        """
        B, N, D = x.shape
        
        # Extract CLS token
        cls_token = x[:, 0:1, :]  # (B, 1, D)
        x = x[:, 1:, :]  # (B, T*P, D)
        
        # Reshape to (B, T, P, D)
        x = rearrange(x, 'b (t p) d -> (b p) t d', t=self.num_frames)
        
        # Apply temporal attention across T dimension
        x = self.attention(x)
        
        # Reshape back
        x = rearrange(x, '(b p) t d -> b (t p) d', b=B)
        
        # Concatenate CLS token back
        x = torch.cat([cls_token, x], dim=1)
        
        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention: attend across spatial patches within the same frame
    """
    
    def __init__(self, dim, num_heads=8, num_frames=10, qkv_bias=False, 
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.attention = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        
    def forward(self, x, num_patches_per_frame):
        """
        Args:
            x: (B, 1+T*P, D) where T=num_frames, P=patches per frame
            num_patches_per_frame: P
        Returns:
            x: (B, 1+T*P, D)
        """
        B, N, D = x.shape
        
        # Extract CLS token
        cls_token = x[:, 0:1, :]  # (B, 1, D)
        x_patches = x[:, 1:, :]  # (B, T*P, D)
        
        # Reshape to (B, T, P, D)
        x_patches = rearrange(x_patches, 'b (t p) d -> b t p d', t=self.num_frames)
        
        # CLS参与每一帧的spatial attention (原始TimeSformer论文)
        # 将CLS token扩展并拼接到每帧前面
        cls_expanded = cls_token.unsqueeze(1).expand(-1, self.num_frames, -1, -1)  # (B, T, 1, D)
        x_with_cls = torch.cat([cls_expanded, x_patches], dim=2)  # (B, T, 1+P, D)
        
        # Reshape for attention: (B*T, 1+P, D)
        x_with_cls = rearrange(x_with_cls, 'b t n d -> (b t) n d')
        
        # Apply spatial attention
        x_with_cls = self.attention(x_with_cls)  # (B*T, 1+P, D)
        
        # Reshape back: (B, T, 1+P, D)
        x_with_cls = rearrange(x_with_cls, '(b t) n d -> b t n d', b=B)
        
        # 取所有帧CLS的平均作为新CLS
        new_cls = x_with_cls[:, :, 0, :].mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # 取patch tokens
        x_patches = x_with_cls[:, :, 1:, :]  # (B, T, P, D)
        x_patches = rearrange(x_patches, 'b t p d -> b (t p) d')
        
        # Concatenate
        x = torch.cat([new_cls, x_patches], dim=1)
        
        return x


class DividedSpaceTimeAttention(nn.Module):
    """
    Divided Space-Time Attention: 
    Apply temporal attention first, then spatial attention
    """
    
    def __init__(self, dim, num_heads=8, num_frames=10, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        
        self.temporal_attn = TemporalAttention(
            dim, num_heads, num_frames, qkv_bias, attn_drop, proj_drop
        )
        
        self.spatial_attn = SpatialAttention(
            dim, num_heads, num_frames, qkv_bias, attn_drop, proj_drop
        )
        
        # Layer normalization
        self.temporal_norm = nn.LayerNorm(dim)
        self.spatial_norm = nn.LayerNorm(dim)
        
    def forward(self, x, num_patches_per_frame):
        """
        Args:
            x: (B, 1+T*P, D)
            num_patches_per_frame: P
        Returns:
            x: (B, 1+T*P, D)
        """
        # Temporal attention with residual
        x = x + self.temporal_attn(self.temporal_norm(x), num_patches_per_frame)
        
        # Spatial attention with residual
        x = x + self.spatial_attn(self.spatial_norm(x), num_patches_per_frame)
        
        return x

