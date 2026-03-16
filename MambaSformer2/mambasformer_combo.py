import torch
import torch.nn as nn

from patch_embedding import VideoEmbedding
from mamba_attention import HybridSpaceTimeMamba
from mambasformer import load_pretrained_timesformer, load_pretrained_vit


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


class MambaSformerCombo(nn.Module):
    """38-class main head with an auxiliary 19-class gas head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_frames = config.num_frames

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
        self.head_dropout = nn.Dropout(p=config.head_dropout)
        self.head_cls = nn.Linear(config.embed_dim, config.num_classes_combo)
        self.head_gas_aux = nn.Linear(config.embed_dim, config.num_classes_gas)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, self.num_patches_per_frame)
        x = self.norm(x)
        cls = self.head_dropout(x[:, 0])
        combo_logits = self.head_cls(cls)
        gas_logits = self.head_gas_aux(cls)
        return combo_logits, gas_logits


def create_mambasformer_combo(config):
    return MambaSformerCombo(config)


__all__ = [
    "MambaSformerCombo",
    "create_mambasformer_combo",
    "load_pretrained_timesformer",
    "load_pretrained_vit",
]
