"""
Patch Embedding module for TimeSformer
Converts video frames into patch tokens with positional encoding
Supports rectangular input (separate height and width)
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """
    Video to Patch Embedding
    Splits each frame into patches and projects them to embedding dimension
    """

    def __init__(self, img_height=448, img_width=448, patch_size=16,
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.patches_h = img_height // patch_size
        self.patches_w = img_width // patch_size
        self.num_patches_per_frame = self.patches_h * self.patches_w

        # Patch projection using Conv2d
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
        Returns:
            x: (B, N, D) - patch embeddings
               N = T * patches_h * patches_w
        """
        B, T, C, H, W = x.shape

        # Reshape to process all frames together
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        # Apply patch projection
        x = self.proj(x)  # (B*T, D, H/P, W/P)

        # Flatten spatial dimensions
        x = rearrange(x, '(b t) d h w -> b (t h w) d', b=B, t=T)

        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for space and time
    """

    def __init__(self, num_patches, num_frames, embed_dim, dropout=0.0):
        super().__init__()
        self.num_patches = num_patches  # patches per frame
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        # Total tokens = 1 (CLS) + num_frames * num_patches
        total_tokens = 1 + num_frames * num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=dropout)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) - patch embeddings
        Returns:
            x: (B, 1+N, D) - with CLS token and positional encoding
        """
        B = x.shape[0]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x


class VideoEmbedding(nn.Module):
    """
    Complete video embedding: patch embedding + positional encoding
    """

    def __init__(self, img_height=448, img_width=448, patch_size=16,
                 num_frames=16, in_channels=3, embed_dim=768, dropout=0.0):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_height, img_width, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches_per_frame

        self.pos_encoding = PositionalEncoding(
            num_patches, num_frames, embed_dim, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
        Returns:
            x: (B, 1+T*P, D) - embedded tokens with CLS token
        """
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        return x
