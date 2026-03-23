import torch
import torch.nn as nn

from patch_embedding import VideoEmbedding
from mamba_attention import HybridSpaceTimeMamba
from mambasformer import load_pretrained_timesformer as _load_pretrained_timesformer
from mambasformer import load_pretrained_vit as _load_pretrained_vit


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MambaSformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_frames,
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_drop=0.0,
        use_checkpoint=True,
    ):
        super().__init__()
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
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, num_patches_per_frame):
        x = self.hybrid_attn(x, num_patches_per_frame)
        return x + self.mlp(self.norm_mlp(x))


class AppearanceEncoder(nn.Module):
    """Single heavy stream reused from the baseline model."""

    def __init__(self, config, input_scale=1.0):
        super().__init__()
        self.config = config
        self.num_frames = config.num_frames
        self.input_scale = input_scale

        self.embedding = VideoEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            in_channels=3,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )
        self.num_patches_per_frame = (config.img_size // config.patch_size) ** 2
        self.blocks = nn.ModuleList(
            [
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
                    use_checkpoint=getattr(config, "use_checkpoint", True),
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x * self.input_scale
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, self.num_patches_per_frame)
        return self.norm(x)


class MotionStem(nn.Module):
    """
    Lightweight motion branch:
    - patchify diff frames with a small projection
    - spatially pool each frame
    - run cheap temporal Conv1d over frame descriptors
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_scale = config.motion_input_scale
        self.patch_proj = nn.Conv2d(
            3,
            config.motion_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(
                config.motion_dim,
                config.motion_dim,
                kernel_size=config.motion_kernel_size,
                padding=config.motion_kernel_size // 2,
                groups=max(1, config.motion_dim // 32),
            ),
            nn.GELU(),
            nn.Conv1d(config.motion_dim, config.motion_dim, kernel_size=1),
            nn.GELU(),
        )
        self.temporal_pool = nn.Sequential(
            nn.LayerNorm(config.motion_dim),
            nn.Linear(config.motion_dim, config.motion_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
        )
        self.motion_proj = nn.Linear(config.motion_dim, config.embed_dim)

    def forward(self, diff_frames):
        bsz, frames, channels, height, width = diff_frames.shape
        x = diff_frames.reshape(bsz * frames, channels, height, width) * self.input_scale
        x = self.patch_proj(x)
        x = x.flatten(2).mean(-1)
        x = x.view(bsz, frames, -1).transpose(1, 2)
        x = self.temporal_mixer(x).transpose(1, 2)
        x = self.temporal_pool(x.mean(dim=1))
        return self.motion_proj(x)


class TemporalDiffFusion(nn.Module):
    """Single heavy appearance backbone plus a lightweight motion stem."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_frames = config.num_frames

        self.appearance_encoder = AppearanceEncoder(config, input_scale=1.0)
        self.motion_stem = MotionStem(config)

        self.appearance_dropout = nn.Dropout(p=config.head_dropout)
        self.motion_dropout = nn.Dropout(p=config.motion_head_dropout)
        self.residual_dropout = nn.Dropout(p=config.fusion_dropout)

        fusion_dim = config.embed_dim * 3
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Sigmoid(),
        )
        self.motion_residual = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
        )
        self.head_cls = nn.Linear(config.embed_dim, config.num_classes_combo)
        self.head_gas_aux = nn.Linear(config.embed_dim, config.num_classes_gas)

        self._init_head_weights()

    def _init_head_weights(self):
        for module in [
            self.motion_stem,
            self.fusion_gate,
            self.motion_residual,
            self.head_cls,
            self.head_gas_aux,
        ]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.trunc_normal_(submodule.weight, std=0.02)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def forward(self, frames, diff_frames):
        appearance_tokens = self.appearance_encoder(frames)

        appearance_cls = self.appearance_dropout(appearance_tokens[:, 0])
        motion_cls = self.motion_dropout(self.motion_stem(diff_frames))
        cross_delta = motion_cls - appearance_cls

        fusion_source = torch.cat([appearance_cls, motion_cls, cross_delta], dim=-1)
        gate = self.fusion_gate(fusion_source)
        residual = self.residual_dropout(self.motion_residual(cross_delta))
        fused = (
            gate * appearance_cls
            + (1.0 - gate) * motion_cls
            + self.config.motion_residual_weight * residual
        )

        combo_logits = self.head_cls(fused)
        gas_logits = self.head_gas_aux(fused)
        return combo_logits, gas_logits


def create_mambasformer_combo(config):
    return TemporalDiffFusion(config)


def load_pretrained_timesformer(model, pretrained_path, config=None):
    cfg = config if config is not None else model.config
    model.appearance_encoder = _load_pretrained_timesformer(model.appearance_encoder, pretrained_path, cfg)
    print("  已加载 appearance 主干预训练权重，motion stem 维持轻量随机初始化")
    return model


def load_pretrained_vit(model, pretrained_path="vit_b_16-c867db91.pth"):
    model.appearance_encoder = _load_pretrained_vit(model.appearance_encoder, pretrained_path)
    print("  已加载 appearance 主干 ViT 权重，motion stem 维持轻量随机初始化")
    return model


__all__ = [
    "TemporalDiffFusion",
    "create_mambasformer_combo",
    "load_pretrained_timesformer",
    "load_pretrained_vit",
]
