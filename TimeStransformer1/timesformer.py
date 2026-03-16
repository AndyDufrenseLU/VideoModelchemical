"""
TimeSformer模型 - 修复版
修复项:
1. depth=12 (使用ViT-B/16全部12层)
2. 分类头前加dropout (0.4)
3. load_pretrained_vit支持12层
"""

import os
import torch
import torch.nn as nn
from patch_embedding import VideoEmbedding
from attention import DividedSpaceTimeAttention


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with divided space-time attention + MLP"""
    def __init__(self, dim, num_heads, num_frames, mlp_ratio=4.0, 
                 qkv_bias=True, dropout=0.0, attn_drop=0.0):
        super().__init__()
        
        self.attention = DividedSpaceTimeAttention(
            dim, num_heads, num_frames, qkv_bias, attn_drop, dropout
        )
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x, num_patches_per_frame):
        # Attention (already has residual + norm inside DividedSpaceTimeAttention)
        x = self.attention(x, num_patches_per_frame)
        # MLP with residual
        x = x + self.mlp(self.norm(x))
        return x


class TimeSformer(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_frames = config.num_frames
        
        # Video embedding (patch + positional)
        self.embedding = VideoEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            in_channels=3,
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )
        
        self.num_patches_per_frame = (config.img_size // config.patch_size) ** 2
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                num_frames=config.num_frames,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                dropout=config.dropout,
                attn_drop=config.attention_dropout
            )
            for _ in range(config.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Dropout before classification heads (降低过强dropout)
        self.head_dropout = nn.Dropout(p=0.0)
        
        # Classification heads (gas velocity + particle type)
        self.head_gas = nn.Linear(config.embed_dim, config.num_classes_gas)
        self.head_particle = nn.Linear(config.embed_dim, config.num_classes_particle)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video frames
        Returns:
            gas_logits: (B, num_classes_gas)
            particle_logits: (B, num_classes_particle)
        """
        # Embed patches
        x = self.embedding(x)  # (B, 1+T*P, D)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, self.num_patches_per_frame)
        
        # Final norm
        x = self.norm(x)
        
        # Use CLS token for classification
        cls = x[:, 0]  # (B, D)
        
        # Dropout before heads
        cls = self.head_dropout(cls)
        
        # Dual classification heads
        gas_logits = self.head_gas(cls)
        particle_logits = self.head_particle(cls)
        
        return gas_logits, particle_logits


def create_timesformer(config):
    """Create TimeSformer model from config"""
    model = TimeSformer(config)
    return model


def load_pretrained_vit(model, pretrained_path="vit_b_16-c867db91.pth"):
    """
    Load pretrained ViT-B/16 weights into TimeSformer model.
    Maps ViT weights to TimeSformer's divided space-time attention.
    """
    if not os.path.exists(pretrained_path):
        print(f"  ⚠️ 预训练权重文件不存在: {pretrained_path}")
        return model
    
    print(f"  📥 加载预训练权重: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Handle nested state_dict
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    loaded = 0
    skipped = 0
    
    # ---- 1. Patch embedding ----
    try:
        model.embedding.patch_embed.proj.weight.data.copy_(
            state_dict['conv_proj.weight'])
        model.embedding.patch_embed.proj.bias.data.copy_(
            state_dict['conv_proj.bias'])
        loaded += 2
        print("    ✅ Patch embedding")
    except Exception as e:
        print(f"    ⚠️ Patch embedding失败: {e}")
        skipped += 2
    
    # ---- 2. CLS token ----
    try:
        model.embedding.pos_encoding.cls_token.data.copy_(
            state_dict['class_token'])
        loaded += 1
        print("    ✅ CLS token")
    except Exception as e:
        print(f"    ⚠️ CLS token失败: {e}")
        skipped += 1
    
    # ---- 3. Positional embedding ----
    try:
        vit_pos = state_dict['encoder.pos_embedding']  # (1, 1+196, 768)
        num_frames = model.num_frames
        
        # ViT pos: (1, 1+196, 768) = CLS + 196 spatial patches
        cls_pos = vit_pos[:, 0:1, :]    # (1, 1, 768)
        spatial_pos = vit_pos[:, 1:, :]  # (1, 196, 768)
        
        # Repeat spatial pos for each frame
        repeated_pos = spatial_pos.repeat(1, num_frames, 1)  # (1, 196*T, 768)
        full_pos = torch.cat([cls_pos, repeated_pos], dim=1)  # (1, 1+196*T, 768)
        
        model.embedding.pos_encoding.pos_embed.data.copy_(full_pos)
        loaded += 1
        print(f"    ✅ Positional embedding (空间pos重复{num_frames}次)")
    except Exception as e:
        print(f"    ⚠️ Positional embedding失败: {e}")
        skipped += 1
    
    # ---- 4. Transformer blocks ----
    num_model_blocks = len(model.blocks)
    
    for i in range(num_model_blocks):
        vit_prefix = f"encoder.layers.encoder_layer_{i}"
        
        # Check if this layer exists in pretrained weights
        test_key = f"{vit_prefix}.ln_1.weight"
        if test_key not in state_dict:
            print(f"    ⚠️ Block {i}: ViT层不存在，跳过")
            skipped += 1
            continue
        
        try:
            # ---- 4a. Spatial Attention (copy from ViT attention) ----
            sa = model.blocks[i].attention.spatial_attn.attention
            sa.qkv.weight.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.in_proj_weight"])
            sa.qkv.bias.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.in_proj_bias"])
            sa.proj.weight.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.out_proj.weight"])
            sa.proj.bias.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.out_proj.bias"])
            
            # ---- 4b. Temporal Attention (initialize from same ViT attention) ----
            ta = model.blocks[i].attention.temporal_attn.attention
            ta.qkv.weight.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.in_proj_weight"])
            ta.qkv.bias.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.in_proj_bias"])
            ta.proj.weight.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.out_proj.weight"])
            ta.proj.bias.data.copy_(
                state_dict[f"{vit_prefix}.self_attention.out_proj.bias"])
            
            # ---- 4c. MLP ----
            # Detect key format: torchvision新版: mlp.linear_1/linear_2, 旧版: mlp.0/mlp.3
            mlp_fc1_key = f"{vit_prefix}.mlp.linear_1.weight"
            if mlp_fc1_key not in state_dict:
                mlp_fc1_key = f"{vit_prefix}.mlp.0.weight"
            
            # Get prefix for fc1 and fc2
            fc1_name = mlp_fc1_key.rsplit('.weight', 1)[0]
            if 'linear_1' in fc1_name:
                fc2_name = fc1_name.replace('linear_1', 'linear_2')
            else:
                fc2_name = fc1_name.replace('.0', '.3')
            
            model.blocks[i].mlp.fc1.weight.data.copy_(
                state_dict[f"{fc1_name}.weight"])
            model.blocks[i].mlp.fc1.bias.data.copy_(
                state_dict[f"{fc1_name}.bias"])
            model.blocks[i].mlp.fc2.weight.data.copy_(
                state_dict[f"{fc2_name}.weight"])
            model.blocks[i].mlp.fc2.bias.data.copy_(
                state_dict[f"{fc2_name}.bias"])
            
            # ---- 4d. Layer Norms ----
            # Temporal norm (from ViT ln_1)
            model.blocks[i].attention.temporal_norm.weight.data.copy_(
                state_dict[f"{vit_prefix}.ln_1.weight"])
            model.blocks[i].attention.temporal_norm.bias.data.copy_(
                state_dict[f"{vit_prefix}.ln_1.bias"])
            
            # Spatial norm (from ViT ln_1 also)
            model.blocks[i].attention.spatial_norm.weight.data.copy_(
                state_dict[f"{vit_prefix}.ln_1.weight"])
            model.blocks[i].attention.spatial_norm.bias.data.copy_(
                state_dict[f"{vit_prefix}.ln_1.bias"])
            
            # MLP norm (from ViT ln_2)
            model.blocks[i].norm.weight.data.copy_(
                state_dict[f"{vit_prefix}.ln_2.weight"])
            model.blocks[i].norm.bias.data.copy_(
                state_dict[f"{vit_prefix}.ln_2.bias"])
            
            loaded += 19
            print(f"    ✅ Block {i}: spatial+temporal attention + MLP")
            
        except Exception as e:
            print(f"    ⚠️ Block {i} 失败: {e}")
            skipped += 1
    
    # ---- 5. Final LayerNorm ----
    try:
        model.norm.weight.data.copy_(state_dict['encoder.ln.weight'])
        model.norm.bias.data.copy_(state_dict['encoder.ln.bias'])
        loaded += 2
        print("    ✅ Final LayerNorm")
    except Exception as e:
        print(f"    ⚠️ Final LayerNorm失败: {e}")
        skipped += 2
    
    # Skip classification head (task-specific)
    print("    ⏭️  分类头: 跳过(使用随机初始化，匹配我们的任务)")
    
    print(f"\n  📊 加载结果: {loaded}个参数组已加载, {skipped}个跳过")
    print(f"  🎯 预训练权重已就绪！模型已知道基础视觉特征。")
    
    return model