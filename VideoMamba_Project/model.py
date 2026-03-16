"""
VideoMamba For Fluidized Bed dual-task classification
"""

import torch
import torch.nn as nn
from videomamba import videomamba_tiny, videomamba_small, videomamba_middle

class VideoMambaMultiTask(nn.Module):
    """
    VideoMamba wrapped for Fluidized Bed dual-task classification.
    """
    def __init__(self, config):
        super().__init__()
        
        # Choose backbone
        if config.model_size == "tiny":
            self.backbone = videomamba_tiny(num_frames=config.num_frames, pretrained=False)
        elif config.model_size == "small":
            self.backbone = videomamba_small(num_frames=config.num_frames, pretrained=False)
        elif config.model_size == "middle":
            self.backbone = videomamba_middle(num_frames=config.num_frames, pretrained=False)
        else:
            raise ValueError(f"Unknown model_size: {config.model_size}")
            
        embed_dim = self.backbone.num_features
        # Replace the original classification head
        self.backbone.head = nn.Identity()
        
        # Dropout
        self.head_dropout = nn.Dropout(p=config.head_dropout)
        
        # Dual Classification heads
        self.head_gas = nn.Linear(embed_dim, config.num_classes_gas)
        self.head_particle = nn.Linear(embed_dim, config.num_classes_particle)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) 
        Returns:
            gas_logits: (B, num_classes_gas)
            particle_logits: (B, num_classes_particle)
        """
        # VideoMamba input expectation: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # Forward through backbones (returning features because head is Identity)
        features = self.backbone(x)
        
        # Apply dropout
        features = self.head_dropout(features)
        
        # Multi-task heads
        gas_logits = self.head_gas(features)
        particle_logits = self.head_particle(features)
        
        return gas_logits, particle_logits

def create_videomamba(config):
    return VideoMambaMultiTask(config)
