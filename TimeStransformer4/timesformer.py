"""
TimeSformer model for TimeStransformer3
Divided Space-Time Attention for multi-task fluidized bed analysis.

Includes load_pretrained_timesformer() to load from Facebook Research
TimeSformer .pyth checkpoints (e.g. TimeSformer_divST_16x16_448_K400.pyth).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Single transformer block: divided space-time attention + MLP"""

    def __init__(self, dim, num_heads, num_frames, mlp_ratio=4.0,
                 qkv_bias=True, dropout=0.0, attn_drop=0.0):
        super().__init__()

        self.attention = DividedSpaceTimeAttention(
            dim, num_heads, num_frames, qkv_bias, attn_drop, dropout
        )

        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, num_patches_per_frame):
        # Attention with residuals (inside DividedSpaceTimeAttention)
        x = self.attention(x, num_patches_per_frame)
        # MLP with residual
        x = x + self.mlp(self.norm(x))
        return x


class TimeSformer(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
    Multi-task version: simultaneous gas velocity + particle type classification.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_frames = config.num_frames

        # Video embedding (patch + positional)
        self.embedding = VideoEmbedding(
            img_height=config.img_height,
            img_width=config.img_width,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            in_channels=3,
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )

        self.num_patches_per_frame = config.num_patches_per_frame

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

        # Dropout before classification heads
        self.head_dropout = nn.Dropout(p=config.head_dropout)

        # Classification heads
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
            gas_logits:      (B, num_classes_gas)
            particle_logits: (B, num_classes_particle)
        """
        x = self.embedding(x)  # (B, 1+T*P, D)

        for block in self.blocks:
            x = block(x, self.num_patches_per_frame)

        x = self.norm(x)

        cls = x[:, 0]           # (B, D) — CLS token
        cls = self.head_dropout(cls)

        gas_logits      = self.head_gas(cls)
        particle_logits = self.head_particle(cls)

        return gas_logits, particle_logits


def create_timesformer(config):
    """Create TimeSformer model from config."""
    return TimeSformer(config)


# ---------------------------------------------------------------------------
# Pretrained weight loading — Facebook Research TimeSformer .pyth format
# ---------------------------------------------------------------------------

def _unwrap_state_dict(checkpoint):
    """
    Extract the actual state_dict from the checkpoint dict.
    Facebook .pyth files typically wrap the weights under 'model_state'.
    """
    for wrapper_key in ('model_state', 'model', 'state_dict'):
        if wrapper_key in checkpoint and isinstance(checkpoint[wrapper_key], dict):
            print(f"    Unwrapped checkpoint via key: '{wrapper_key}'")
            return checkpoint[wrapper_key]

    # Maybe the checkpoint is the state_dict itself
    first_key = next(iter(checkpoint), None)
    if first_key and ('blocks' in first_key or 'patch_embed' in first_key
                      or 'model.blocks' in first_key):
        print("    Checkpoint appears to be a raw state_dict")
        return checkpoint

    print(f"    WARNING: Could not detect wrapper key. Top-level keys: "
          f"{list(checkpoint.keys())[:10]}")
    print("    Attempting to use checkpoint as-is...")
    return checkpoint


def _strip_model_prefix(state_dict):
    """Strip leading 'model.' prefix added by some PySlowFast saves."""
    if any(k.startswith('model.') for k in state_dict.keys()):
        stripped = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        print("    Stripped 'model.' prefix from keys")
        return stripped
    return state_dict


def _copy_weight(dst_tensor, src_tensor, name, loaded_count, skipped_count):
    """Safe weight copy with shape check. Returns (loaded, skipped) delta."""
    if dst_tensor.shape != src_tensor.shape:
        print(f"      SHAPE MISMATCH {name}: "
              f"dst={tuple(dst_tensor.shape)} src={tuple(src_tensor.shape)}")
        return 0, 1
    dst_tensor.data.copy_(src_tensor)
    return 1, 0


def _build_combined_pos_embed(state_dict, model, config):
    """
    Build the combined positional embedding for our model from the pretrained
    spatial pos_embed and (optional) time_embed.

    Pretrained (448 px, 28x28 patches):
        pos_embed  : (1, 1+784, 768)  — CLS + 784 spatial positions
        time_embed : (1, T_pre, 768)  — T_pre temporal positions (optional)

    Our model (448 px, 28x28 patches, 16 frames):
        pos_embed  : (1, 1+16*784, 768) = (1, 12545, 768)

    Strategy:
        combined[frame t, patch p] = spatial_pos[p] + time_embed[t]
    If time_embed is missing, fall back to pure spatial repetition.
    """
    patches_h = config.patches_h   # 28
    patches_w = config.patches_w   # 28
    P = config.num_patches_per_frame  # 784
    T = config.num_frames             # 16
    D = config.embed_dim              # 768

    target_shape = model.embedding.pos_encoding.pos_embed.shape  # (1, 12545, 768)

    if 'pos_embed' not in state_dict:
        print("    WARNING: 'pos_embed' not found in checkpoint — skipping PE")
        return False

    pre_pos = state_dict['pos_embed']  # (1, 1+P_pre, D)
    print(f"    pos_embed  shape in checkpoint: {tuple(pre_pos.shape)}")

    # Split CLS and spatial
    cls_pos     = pre_pos[:, 0:1, :]   # (1, 1, D)
    spatial_pos = pre_pos[:, 1:, :]    # (1, P_pre, D)
    P_pre = spatial_pos.shape[1]       # 784 for 448 resolution

    # ---- Spatial interpolation (only if sizes differ) ----
    side_pre = int(P_pre ** 0.5)
    if P_pre != P or side_pre * side_pre != P_pre:
        # Need interpolation; reshape to 2D grid
        if side_pre * side_pre == P_pre:
            sp = spatial_pos.reshape(1, side_pre, side_pre, D).permute(0, 3, 1, 2)  # (1,D,H,W)
            sp = F.interpolate(sp, size=(patches_h, patches_w),
                               mode='bicubic', align_corners=False)
            spatial_pos = sp.permute(0, 2, 3, 1).reshape(1, P, D)
            print(f"    Spatial PE: bicubic interpolated {side_pre}x{side_pre} -> "
                  f"{patches_h}x{patches_w}")
        else:
            print(f"    WARNING: Cannot interpret spatial PE shape {P_pre} as square grid. "
                  f"Repeating for each frame instead.")
            spatial_pos = spatial_pos[:, :P, :] if P_pre >= P else spatial_pos.repeat(
                1, (P // P_pre) + 1, 1)[:, :P, :]
    else:
        print(f"    Spatial PE: exact match {side_pre}x{side_pre} = {P} patches/frame, "
              f"no interpolation needed")

    # ---- Temporal embedding ----
    time_embed = state_dict.get('time_embed', None)
    if time_embed is not None:
        T_pre = time_embed.shape[1]
        print(f"    time_embed shape in checkpoint: {tuple(time_embed.shape)}")
        if T_pre != T:
            te = time_embed.permute(0, 2, 1)          # (1, D, T_pre)
            te = F.interpolate(te, size=T, mode='linear', align_corners=False)
            time_embed = te.permute(0, 2, 1)           # (1, T, D)
            print(f"    Temporal PE: interpolated {T_pre} -> {T} frames")
        else:
            print(f"    Temporal PE: exact match T={T}, no interpolation needed")
    else:
        print("    time_embed: not found in checkpoint, "
              "using pure spatial repetition (no temporal shift)")

    # ---- Combine: for each frame t, add spatial + temporal ----
    combined = torch.zeros(1, T * P, D, dtype=spatial_pos.dtype)
    for t in range(T):
        start = t * P
        end   = start + P
        if time_embed is not None:
            combined[0, start:end] = spatial_pos[0] + time_embed[0, t]
        else:
            combined[0, start:end] = spatial_pos[0]

    full_pos = torch.cat([cls_pos, combined], dim=1)  # (1, 1+T*P, D)

    if full_pos.shape != target_shape:
        print(f"    ERROR: built PE shape {tuple(full_pos.shape)} != "
              f"target {tuple(target_shape)}")
        return False

    model.embedding.pos_encoding.pos_embed.data.copy_(full_pos)
    print(f"    Positional embedding: OK  shape={tuple(full_pos.shape)}")
    return True


def load_pretrained_timesformer(model, pretrained_path, config=None):
    """
    Load Facebook Research TimeSformer pretrained weights (.pyth file).

    Supports:
        TimeSformer_divST_16x16_448_K400.pyth  (and similar variants)

    Key mapping  (FB checkpoint  ->  our model):
        patch_embed.proj.*                   -> embedding.patch_embed.proj.*
        cls_token                            -> embedding.pos_encoding.cls_token
        pos_embed + time_embed              -> embedding.pos_encoding.pos_embed
        blocks.{i}.norm1.*                  -> blocks.{i}.attention.spatial_norm.*
        blocks.{i}.attn.{qkv,proj}.*       -> blocks.{i}.attention.spatial_attn.attention.*
        blocks.{i}.temporal_norm1.*         -> blocks.{i}.attention.temporal_norm.*
        blocks.{i}.temporal_attn.{qkv,proj}.* -> blocks.{i}.attention.temporal_attn.attention.*
        blocks.{i}.norm2.*                  -> blocks.{i}.norm.*
        blocks.{i}.mlp.{fc1,fc2}.*         -> blocks.{i}.mlp.{fc1,fc2}.*
        norm.*                              -> norm.*
        head.*                              -> SKIP  (task-specific heads)
    """
    if not os.path.exists(pretrained_path):
        print(f"  WARNING: Pretrained weights not found: {pretrained_path}")
        print("  Model will use random initialization.")
        return model

    print(f"\n  Loading pretrained TimeSformer weights: {pretrained_path}")
    print(f"  File size: {os.path.getsize(pretrained_path) / 1e6:.1f} MB")

    # ---- Load checkpoint ----
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only argument
        checkpoint = torch.load(pretrained_path, map_location='cpu')

    print(f"  Checkpoint top-level keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

    state_dict = _unwrap_state_dict(checkpoint)
    state_dict = _strip_model_prefix(state_dict)

    print(f"  State dict: {len(state_dict)} keys total")
    print("  First 12 keys:")
    for k in list(state_dict.keys())[:12]:
        print(f"    {k}: {tuple(state_dict[k].shape)}")

    loaded = 0
    skipped = 0

    # ---- 1. Patch embedding ----
    print("\n  [1] Patch embedding")
    for suffix in ('weight', 'bias'):
        src_key = f'patch_embed.proj.{suffix}'
        dst     = getattr(model.embedding.patch_embed.proj, suffix)
        if src_key in state_dict:
            dl, sk = _copy_weight(dst, state_dict[src_key], src_key, 0, 0)
            loaded += dl; skipped += sk
            if dl:
                print(f"    patch_embed.proj.{suffix}: OK  {tuple(dst.shape)}")
        else:
            print(f"    WARNING: '{src_key}' not in checkpoint")
            skipped += 1

    # ---- 2. CLS token ----
    print("\n  [2] CLS token")
    if 'cls_token' in state_dict:
        src = state_dict['cls_token']  # (1, 1, 768) or (1, 768)
        dst = model.embedding.pos_encoding.cls_token
        if src.dim() == 2:
            src = src.unsqueeze(0)
        dl, sk = _copy_weight(dst, src, 'cls_token', 0, 0)
        loaded += dl; skipped += sk
        if dl:
            print(f"    cls_token: OK  {tuple(dst.shape)}")
    else:
        print("    WARNING: 'cls_token' not in checkpoint")
        skipped += 1

    # ---- 3. Positional embedding ----
    print("\n  [3] Positional embedding")
    cfg = config if config is not None else model.config
    ok = _build_combined_pos_embed(state_dict, model, cfg)
    if ok:
        loaded += 1
    else:
        skipped += 1

    # ---- 4. Transformer blocks ----
    print("\n  [4] Transformer blocks")
    num_blocks = len(model.blocks)

    for i in range(num_blocks):
        # Check block exists in checkpoint
        probe_key = f'blocks.{i}.norm1.weight'
        if probe_key not in state_dict:
            print(f"    Block {i}: not found in checkpoint (key '{probe_key}' missing)")
            skipped += 1
            continue

        block_loaded = 0
        block_skipped = 0

        def _load_block_param(src_key, dst_tensor):
            nonlocal block_loaded, block_skipped
            if src_key in state_dict:
                dl, sk = _copy_weight(dst_tensor, state_dict[src_key], src_key, 0, 0)
                block_loaded += dl
                block_skipped += sk
            else:
                block_skipped += 1

        sa = model.blocks[i].attention.spatial_attn.attention
        ta = model.blocks[i].attention.temporal_attn.attention

        # Spatial norm (norm1 in FB)
        _load_block_param(f'blocks.{i}.norm1.weight',
                          model.blocks[i].attention.spatial_norm.weight)
        _load_block_param(f'blocks.{i}.norm1.bias',
                          model.blocks[i].attention.spatial_norm.bias)

        # Spatial attention  (attn in FB)
        _load_block_param(f'blocks.{i}.attn.qkv.weight',  sa.qkv.weight)
        _load_block_param(f'blocks.{i}.attn.qkv.bias',    sa.qkv.bias)
        _load_block_param(f'blocks.{i}.attn.proj.weight', sa.proj.weight)
        _load_block_param(f'blocks.{i}.attn.proj.bias',   sa.proj.bias)

        # Temporal norm (temporal_norm1 in FB)
        tn1_key = f'blocks.{i}.temporal_norm1.weight'
        if tn1_key in state_dict:
            _load_block_param(f'blocks.{i}.temporal_norm1.weight',
                              model.blocks[i].attention.temporal_norm.weight)
            _load_block_param(f'blocks.{i}.temporal_norm1.bias',
                              model.blocks[i].attention.temporal_norm.bias)
        else:
            # Fall back: initialize temporal norm = spatial norm (reasonable default)
            model.blocks[i].attention.temporal_norm.weight.data.copy_(
                model.blocks[i].attention.spatial_norm.weight.data)
            model.blocks[i].attention.temporal_norm.bias.data.copy_(
                model.blocks[i].attention.spatial_norm.bias.data)
            print(f"    Block {i}: temporal_norm1 not found, "
                  f"initialized from spatial_norm")
            block_loaded += 2

        # Temporal attention  (temporal_attn in FB)
        if f'blocks.{i}.temporal_attn.qkv.weight' in state_dict:
            _load_block_param(f'blocks.{i}.temporal_attn.qkv.weight',  ta.qkv.weight)
            _load_block_param(f'blocks.{i}.temporal_attn.qkv.bias',    ta.qkv.bias)
            _load_block_param(f'blocks.{i}.temporal_attn.proj.weight', ta.proj.weight)
            _load_block_param(f'blocks.{i}.temporal_attn.proj.bias',   ta.proj.bias)
        else:
            # Fall back: initialize temporal attention = spatial attention weights
            ta.qkv.weight.data.copy_(sa.qkv.weight.data)
            ta.qkv.bias.data.copy_(sa.qkv.bias.data)
            ta.proj.weight.data.copy_(sa.proj.weight.data)
            ta.proj.bias.data.copy_(sa.proj.bias.data)
            print(f"    Block {i}: temporal_attn not found, "
                  f"initialized from spatial_attn")
            block_loaded += 4

        # MLP norm  (norm2 in FB)
        _load_block_param(f'blocks.{i}.norm2.weight', model.blocks[i].norm.weight)
        _load_block_param(f'blocks.{i}.norm2.bias',   model.blocks[i].norm.bias)

        # MLP fc1 / fc2
        _load_block_param(f'blocks.{i}.mlp.fc1.weight', model.blocks[i].mlp.fc1.weight)
        _load_block_param(f'blocks.{i}.mlp.fc1.bias',   model.blocks[i].mlp.fc1.bias)
        _load_block_param(f'blocks.{i}.mlp.fc2.weight', model.blocks[i].mlp.fc2.weight)
        _load_block_param(f'blocks.{i}.mlp.fc2.bias',   model.blocks[i].mlp.fc2.bias)

        loaded  += block_loaded
        skipped += block_skipped
        print(f"    Block {i:2d}: {block_loaded:2d} loaded, "
              f"{block_skipped} skipped")

    # ---- 5. Final LayerNorm ----
    print("\n  [5] Final LayerNorm")
    for suffix in ('weight', 'bias'):
        src_key = f'norm.{suffix}'
        dst = getattr(model.norm, suffix)
        if src_key in state_dict:
            dl, sk = _copy_weight(dst, state_dict[src_key], src_key, 0, 0)
            loaded += dl; skipped += sk
            if dl:
                print(f"    norm.{suffix}: OK  {tuple(dst.shape)}")
        else:
            print(f"    WARNING: '{src_key}' not in checkpoint")
            skipped += 1

    # ---- 6. Classification head — always skip ----
    print("\n  [6] Classification heads: SKIPPED "
          "(head_gas and head_particle use random init for our task)")

    print(f"\n  Pretrained weight loading complete.")
    print(f"  Loaded: {loaded} parameter tensors")
    print(f"  Skipped / mismatched: {skipped}")
    print(f"  Note: task-specific heads (gas velocity, particle type) "
          f"are randomly initialized and will be fine-tuned.\n")

    return model


# Alias for backward compatibility with any existing code that calls load_pretrained_vit
load_pretrained_vit = load_pretrained_timesformer
