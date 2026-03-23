"""
MambaSformer 主模型
结合 TimeSformer 架构与 Mamba SSM 时序建模

架构:
  [Video Frames (B,T,C,H,W)]
       ↓
  [VideoEmbedding]  → (B, 1+T*P, D)  CLS + patch tokens
       ↓
  [MambaSformerBlock × depth]
       ├─ Temporal Mamba SSM   (O(T) 线性，替代 Temporal Attention)
       ├─ Spatial Attention    (保留 ViT 预训练特征)
       └─ MLP (FFN)
       ↓
  [LayerNorm]
       ↓
  [CLS Token]  → (B, D)
       ↓
  [head_gas]     → (B, 19)   气速分类
  [head_particle] → (B, 3)   粒子分类
"""

import os
import torch
import torch.nn as nn
from patch_embedding import VideoEmbedding
from mamba_attention import HybridSpaceTimeMamba


# ============================================================
# FFN (前馈网络，与 TimeSformer1 的 MLP 相同)
# ============================================================

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ============================================================
# MambaSformer Block
# ============================================================

class MambaSformerBlock(nn.Module):
    """
    单个 MambaSformer Block:
      HybridSpaceTimeMamba (Temporal Mamba + Spatial Attention) + MLP + LayerNorm
    """

    def __init__(self, dim, num_heads, num_frames,
                 d_state=16, d_conv=4, mamba_expand=2,
                 mlp_ratio=4.0, qkv_bias=True, dropout=0.0, attn_drop=0.0,
                 use_checkpoint=True):
        super().__init__()

        # 混合时空注意力（Temporal=Mamba, Spatial=Attention）
        self.hybrid_attn = HybridSpaceTimeMamba(
            dim=dim,
            num_heads=num_heads,
            num_frames=num_frames,
            d_state=d_state,
            d_conv=d_conv,
            mamba_expand=mamba_expand,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout,
            use_checkpoint=use_checkpoint,
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, num_patches_per_frame):
        # 混合时空处理（内部已含 Pre-Norm + 残差）
        x = self.hybrid_attn(x, num_patches_per_frame)
        # MLP + 残差
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ============================================================
# MambaSformer 主模型
# ============================================================

class MambaSformer(nn.Module):
    """
    MambaSformer: TimeSformer + Mamba SSM 时序建模

    与 TimeSformer1 接口完全兼容（同样的输入格式、双任务输出）
    主要改进:
      - 时序 Attention → Mamba SSM (O(T) 复杂度)
      - 帧数增多时计算量线性增长，而非平方
      - 空间方向保留 ViT 预训练权重
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_frames = config.num_frames

        # Patch embedding（与 TimeSformer1 完全相同）
        self.embedding = VideoEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            in_channels=3,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )

        self.num_patches_per_frame = (config.img_size // config.patch_size) ** 2

        # MambaSformer blocks
        self.blocks = nn.ModuleList([
            MambaSformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                num_frames=config.num_frames,
                d_state=config.d_state,
                d_conv=config.d_conv,
                mamba_expand=config.mamba_expand,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                dropout=config.dropout,
                attn_drop=config.attention_dropout,
                use_checkpoint=getattr(config, 'use_checkpoint', True),
            )
            for _ in range(config.depth)
        ])

        # 最终 LayerNorm
        self.norm = nn.LayerNorm(config.embed_dim)

        # Dropout before classification heads
        self.head_dropout = nn.Dropout(p=config.head_dropout)

        # 双任务分类头（与 TimeSformer1 相同）
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
            x: (B, T, C, H, W)  视频帧序列
        Returns:
            gas_logits:      (B, num_classes_gas)
            particle_logits: (B, num_classes_particle)
        """
        # Patch embedding
        x = self.embedding(x)         # (B, 1+T*P, D)

        # MambaSformer blocks
        for block in self.blocks:
            x = block(x, self.num_patches_per_frame)

        # Final norm
        x = self.norm(x)

        # CLS token → 分类
        cls = x[:, 0]                 # (B, D)
        cls = self.head_dropout(cls)

        gas_logits = self.head_gas(cls)
        particle_logits = self.head_particle(cls)

        return gas_logits, particle_logits


# ============================================================
# 工厂函数
# ============================================================

def create_mambasformer(config):
    """创建 MambaSformer 模型"""
    return MambaSformer(config)


# ============================================================
# 权重加载工具函数
# ============================================================

def _unwrap_state_dict(checkpoint):
    """从 Facebook .pyth checkpoint 中提取 state_dict"""
    for key in ('model_state', 'model', 'state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            print(f"    解包 checkpoint key: '{key}'")
            return checkpoint[key]
    first = next(iter(checkpoint), None)
    if first and any(k in first for k in ('blocks', 'patch_embed')):
        return checkpoint
    return checkpoint


def _strip_prefix(state_dict):
    """去掉 PySlowFast 保存时附加的 'model.' 前缀"""
    if any(k.startswith('model.') for k in state_dict):
        return {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def _safe_copy(dst, src, name=""):
    """形状匹配则复制权重，否则打印警告"""
    if dst.shape != src.shape:
        print(f"      ⚠️ 形状不匹配 {name}: dst={tuple(dst.shape)} src={tuple(src.shape)}")
        return False
    dst.data.copy_(src)
    return True


def load_pretrained_timesformer(model, pretrained_path, config=None):
    """
    加载 Facebook Research TimeSformer 预训练权重 (.pyth 格式)
    支持: TimeSformer_divST_16x16_448_K400.pyth 等变体

    为什么 TimeSformer 比 ViT 更好？
      - ViT 只在 ImageNet（静态图片）上预训练 → 不理解时序关系
      - TimeSformer 在 Kinetics-400（视频）上预训练 → 空间特征更适合视频任务
      - 对于流化床（连续帧）任务，TimeSformer 空间特征质量更高

    映射规则 (FB checkpoint → MambaSformer):
      patch_embed.proj.*              → embedding.patch_embed.proj.*
      cls_token                       → embedding.pos_encoding.cls_token
      pos_embed + time_embed          → embedding.pos_encoding.pos_embed
      blocks.{i}.norm1.*             → blocks[i].hybrid_attn.spatial_norm.*
      blocks.{i}.attn.qkv.*         → blocks[i].hybrid_attn.spatial_attn.attention.qkv.*
      blocks.{i}.attn.proj.*        → blocks[i].hybrid_attn.spatial_attn.attention.proj.*
      blocks.{i}.temporal_norm1.*   → blocks[i].hybrid_attn.temporal_norm.*  (初始化用)
      blocks.{i}.temporal_attn.*    → 跳过 (替换为 Mamba，无需加载)
      blocks.{i}.norm2.*            → blocks[i].norm_mlp.*
      blocks.{i}.mlp.*              → blocks[i].mlp.*
      norm.*                         → norm.*
      head.*                         → 跳过 (任务特定，随机初始化)
    """
    import torch.nn.functional as F

    if not os.path.exists(pretrained_path):
        print(f"  ⚠️ 预训练权重不存在: {pretrained_path}")
        print("     将使用随机初始化（训练效果较差）")
        return model

    cfg = config if config is not None else model.config
    print(f"\n  📥 加载 TimeSformer 预训练权重: {pretrained_path}")
    print(f"     文件大小: {os.path.getsize(pretrained_path) / 1e6:.1f} MB")

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

    state_dict = _unwrap_state_dict(checkpoint)
    state_dict = _strip_prefix(state_dict)
    print(f"     checkpoint 共 {len(state_dict)} 个张量")

    loaded = skipped = 0

    # ---- 1. Patch embedding ----
    print("  [1] Patch embedding")
    for s in ('weight', 'bias'):
        key = f'patch_embed.proj.{s}'
        dst = getattr(model.embedding.patch_embed.proj, s)
        if key in state_dict and _safe_copy(dst, state_dict[key], key):
            loaded += 1; print(f"      ✅ patch_embed.proj.{s}")
        else:
            skipped += 1

    # ---- 2. CLS token ----
    print("  [2] CLS token")
    if 'cls_token' in state_dict:
        src = state_dict['cls_token']
        if src.dim() == 2: src = src.unsqueeze(0)
        dst = model.embedding.pos_encoding.cls_token
        if _safe_copy(dst, src, 'cls_token'):
            loaded += 1; print(f"      ✅ cls_token")
    else:
        skipped += 1

    # ---- 3. Positional embedding（空间 + 时序组合）----
    print("  [3] Positional embedding")
    P = model.num_patches_per_frame
    T = model.num_frames
    D = cfg.embed_dim

    if 'pos_embed' in state_dict:
        pre_pos    = state_dict['pos_embed']        # (1, 1+P_pre, D)
        cls_pos    = pre_pos[:, 0:1, :]
        spatial    = pre_pos[:, 1:, :]              # (1, P_pre, D)
        P_pre      = spatial.shape[1]
        side_pre   = int(P_pre ** 0.5)
        side_target = int(P ** 0.5)

        # 需要对空间 pos 做双三次插值（如分辨率不同）
        if P_pre != P and side_pre * side_pre == P_pre:
            sp = spatial.reshape(1, side_pre, side_pre, D).permute(0, 3, 1, 2)
            sp = F.interpolate(sp, size=(side_target, side_target),
                               mode='bicubic', align_corners=False)
            spatial = sp.permute(0, 2, 3, 1).reshape(1, P, D)
            print(f"      空间 PE: 双三次插值 {side_pre}×{side_pre} → {side_target}×{side_target}")
        else:
            print(f"      空间 PE: 精确匹配 {side_target}×{side_target} = {P} patches/帧")

        # 时序 embedding（可选），加到每帧的空间 pos 上
        time_embed = state_dict.get('time_embed', None)
        combined = torch.zeros(1, T * P, D, dtype=spatial.dtype)
        for t in range(T):
            if time_embed is not None:
                T_pre = time_embed.shape[1]
                te = time_embed.permute(0, 2, 1)
                te = F.interpolate(te, size=T, mode='linear', align_corners=False)
                time_embed = te.permute(0, 2, 1)
                combined[0, t*P:(t+1)*P] = spatial[0] + time_embed[0, t]
                time_embed = None  # 只处理一次
            else:
                combined[0, t*P:(t+1)*P] = spatial[0]

        full_pos = torch.cat([cls_pos, combined], dim=1)   # (1, 1+T*P, D)
        target   = model.embedding.pos_encoding.pos_embed
        if _safe_copy(target, full_pos, 'pos_embed'):
            loaded += 1; print(f"      ✅ pos_embed: shape={tuple(full_pos.shape)}")
    else:
        skipped += 1; print("      ⚠️ pos_embed 不存在，跳过")

    # ---- 4. Transformer blocks ----
    print("  [4] Transformer blocks")
    num_blocks = len(model.blocks)

    for i in range(num_blocks):
        if f'blocks.{i}.norm1.weight' not in state_dict:
            print(f"      Block {i}: checkpoint 中不存在，跳过")
            skipped += 1; continue

        bl = 0  # block-level loaded count

        sa = model.blocks[i].hybrid_attn.spatial_attn.attention

        # 空间注意力 norm (norm1)
        _safe_copy(model.blocks[i].hybrid_attn.spatial_norm.weight,
                   state_dict[f'blocks.{i}.norm1.weight'], f'b{i}.norm1.w') and (bl := bl+1)
        _safe_copy(model.blocks[i].hybrid_attn.spatial_norm.bias,
                   state_dict[f'blocks.{i}.norm1.bias'],   f'b{i}.norm1.b') and (bl := bl+1)

        # 空间注意力权重 (attn.qkv / attn.proj)
        _safe_copy(sa.qkv.weight,  state_dict.get(f'blocks.{i}.attn.qkv.weight',  sa.qkv.weight),
                   f'b{i}.attn.qkv.w') and (bl := bl+1)
        _safe_copy(sa.qkv.bias,    state_dict.get(f'blocks.{i}.attn.qkv.bias',    sa.qkv.bias),
                   f'b{i}.attn.qkv.b') and (bl := bl+1)
        _safe_copy(sa.proj.weight, state_dict.get(f'blocks.{i}.attn.proj.weight', sa.proj.weight),
                   f'b{i}.attn.proj.w') and (bl := bl+1)
        _safe_copy(sa.proj.bias,   state_dict.get(f'blocks.{i}.attn.proj.bias',   sa.proj.bias),
                   f'b{i}.attn.proj.b') and (bl := bl+1)

        # temporal_norm：用 norm1 初始化（合理默认值）
        tn_key = f'blocks.{i}.temporal_norm1.weight'
        if tn_key in state_dict:
            _safe_copy(model.blocks[i].hybrid_attn.temporal_norm.weight,
                       state_dict[tn_key], f'b{i}.tnorm.w') and (bl := bl+1)
            _safe_copy(model.blocks[i].hybrid_attn.temporal_norm.bias,
                       state_dict[f'blocks.{i}.temporal_norm1.bias'], f'b{i}.tnorm.b') and (bl := bl+1)
        else:
            # 用空间 norm 权重初始化时序 norm（合理 fallback）
            model.blocks[i].hybrid_attn.temporal_norm.weight.data.copy_(
                model.blocks[i].hybrid_attn.spatial_norm.weight.data)
            model.blocks[i].hybrid_attn.temporal_norm.bias.data.copy_(
                model.blocks[i].hybrid_attn.spatial_norm.bias.data)
            bl += 2

        # temporal_attn → 跳过！(已被 Mamba 取代，无需加载 FB 的时序 attention 权重)

        # MLP norm (norm2)
        _safe_copy(model.blocks[i].norm_mlp.weight,
                   state_dict[f'blocks.{i}.norm2.weight'], f'b{i}.norm2.w') and (bl := bl+1)
        _safe_copy(model.blocks[i].norm_mlp.bias,
                   state_dict[f'blocks.{i}.norm2.bias'],   f'b{i}.norm2.b') and (bl := bl+1)

        # MLP fc1/fc2
        _safe_copy(model.blocks[i].mlp.fc1.weight,
                   state_dict[f'blocks.{i}.mlp.fc1.weight'], f'b{i}.mlp.fc1.w') and (bl := bl+1)
        _safe_copy(model.blocks[i].mlp.fc1.bias,
                   state_dict[f'blocks.{i}.mlp.fc1.bias'],   f'b{i}.mlp.fc1.b') and (bl := bl+1)
        _safe_copy(model.blocks[i].mlp.fc2.weight,
                   state_dict[f'blocks.{i}.mlp.fc2.weight'], f'b{i}.mlp.fc2.w') and (bl := bl+1)
        _safe_copy(model.blocks[i].mlp.fc2.bias,
                   state_dict[f'blocks.{i}.mlp.fc2.bias'],   f'b{i}.mlp.fc2.b') and (bl := bl+1)

        loaded += bl
        print(f"      ✅ Block {i:2d}: 空间 Attention + MLP 已加载 ({bl} 个张量)"
              f" | Mamba 时序层跳过（随机初始化）")

    # ---- 5. Final LayerNorm ----
    print("  [5] Final LayerNorm")
    for s in ('weight', 'bias'):
        key = f'norm.{s}'
        if key in state_dict and _safe_copy(getattr(model.norm, s), state_dict[key], key):
            loaded += 1; print(f"      ✅ norm.{s}")
        else:
            skipped += 1

    # ---- 6. 分类头：跳过 ----
    print("  [6] 分类头: 跳过（任务特定，随机初始化）")

    print(f"\n  📊 TimeSformer 权重加载完成")
    print(f"     已加载: {loaded} 个张量")
    print(f"     跳过:   {skipped} 个张量")
    print(f"     🎯 空间注意力 + MLP 已从 TimeSformer(Kinetics-400) 迁移")
    print(f"     🌀 Mamba 时序层从头学习（无对应预训练权重）\n")

    return model


def load_pretrained_vit(model, pretrained_path="vit_b_16-c867db91.pth"):
    """
    降级方案：加载 ViT-B/16 预训练权重（精度不如 TimeSformer，但可用）
    优先使用 load_pretrained_timesformer()
    """
    if not os.path.exists(pretrained_path):
        print(f"  ⚠️ 预训练权重文件不存在: {pretrained_path}，使用随机初始化")
        return model

    print(f"  📥 加载 ViT-B/16 预训练权重: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    loaded = skipped = 0

    for s in ('weight', 'bias'):
        dst = getattr(model.embedding.patch_embed.proj, s)
        src_key = f'conv_proj.{s}'
        if src_key in state_dict and _safe_copy(dst, state_dict[src_key], src_key):
            loaded += 1
        else:
            skipped += 1

    if 'class_token' in state_dict:
        if _safe_copy(model.embedding.pos_encoding.cls_token,
                      state_dict['class_token'], 'class_token'):
            loaded += 1

    try:
        vit_pos    = state_dict['encoder.pos_embedding']
        cls_pos    = vit_pos[:, 0:1, :]
        spatial    = vit_pos[:, 1:, :]
        full_pos   = torch.cat([cls_pos, spatial.repeat(1, model.num_frames, 1)], dim=1)
        if _safe_copy(model.embedding.pos_encoding.pos_embed, full_pos, 'pos_embed'):
            loaded += 1
    except Exception as e:
        print(f"    ⚠️ pos_embed 失败: {e}"); skipped += 1

    for i in range(min(len(model.blocks), 12)):
        prefix = f"encoder.layers.encoder_layer_{i}"
        if f"{prefix}.ln_1.weight" not in state_dict:
            skipped += 1; continue
        try:
            sa = model.blocks[i].hybrid_attn.spatial_attn.attention
            _safe_copy(sa.qkv.weight,  state_dict[f"{prefix}.self_attention.in_proj_weight"])
            _safe_copy(sa.qkv.bias,    state_dict[f"{prefix}.self_attention.in_proj_bias"])
            _safe_copy(sa.proj.weight, state_dict[f"{prefix}.self_attention.out_proj.weight"])
            _safe_copy(sa.proj.bias,   state_dict[f"{prefix}.self_attention.out_proj.bias"])
            _safe_copy(model.blocks[i].hybrid_attn.spatial_norm.weight,
                       state_dict[f"{prefix}.ln_1.weight"])
            _safe_copy(model.blocks[i].hybrid_attn.spatial_norm.bias,
                       state_dict[f"{prefix}.ln_1.bias"])
            _safe_copy(model.blocks[i].norm_mlp.weight, state_dict[f"{prefix}.ln_2.weight"])
            _safe_copy(model.blocks[i].norm_mlp.bias,   state_dict[f"{prefix}.ln_2.bias"])
            loaded += 8
        except Exception as e:
            print(f"    ⚠️ Block {i}: {e}"); skipped += 1

    try:
        _safe_copy(model.norm.weight, state_dict['encoder.ln.weight'])
        _safe_copy(model.norm.bias,   state_dict['encoder.ln.bias'])
        loaded += 2
    except Exception:
        skipped += 2

    print(f"  📊 ViT-B/16 加载结果: {loaded} 已加载, {skipped} 跳过")
    return model
