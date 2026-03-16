"""
Configuration file for TimeSformer model - TimeStransformer3
448x448 resolution, 16 frames, matching TimeSformer_divST_16x16_448_K400 pretrained weights
"""


class TimeSformerConfig:
    """Configuration for TimeSformer model"""

    def __init__(self):
        # ===== Image dimensions =====
        self.img_width = 448
        self.img_height = 448
        self.patch_size = 16
        self.num_frames = 16

        # Derived patch grid dimensions
        self.patches_w = self.img_width // self.patch_size    # 28
        self.patches_h = self.img_height // self.patch_size   # 28
        self.num_patches_per_frame = self.patches_h * self.patches_w  # 784

        # Backward compat for train_multigpu.py
        self.img_size = self.img_width

        # ===== Multi-task classification =====
        self.num_classes_gas = 19       # Gas velocity (1-19 mapped to 0-18)
        self.num_classes_particle = 3   # Particle types (0, 1, 2)

        # ===== Model architecture =====
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.mlp_ratio = 4.0

        # ===== Regularization =====
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.head_dropout = 0.5
        self.drop_path_rate = 0.1
        self.label_smoothing = 0.1
        self.use_class_weights = True

        # ===== Training =====
        self.batch_size = 4
        self.grad_accum_steps = 16      # Effective batch = 4 * 16 = 64
        self.backbone_lr = 5e-6
        self.head_lr = 5e-4
        self.weight_decay = 0.05
        self.num_epochs = 100
        self.warmup_epochs = 3

        # ===== Data loading =====
        self.num_workers = 2

        # ===== Attention type =====
        self.attention_type = "divided_space_time"

        # ===== Multi-GPU =====
        self.use_multi_gpu = True

        # ===== Early stopping =====
        self.patience = 20

        # ===== Paths =====
        self.data_root = "./data"
        self.train_split = "train"
        self.val_split = "val"

    def __repr__(self):
        return (f"TimeSformerConfig(img={self.img_height}x{self.img_width}, "
                f"frames={self.num_frames}, embed={self.embed_dim}, "
                f"depth={self.depth}, heads={self.num_heads}, "
                f"dropout={self.dropout}, drop_path={self.drop_path_rate}, "
                f"head_dropout={self.head_dropout}, "
                f"label_smoothing={self.label_smoothing})")

    def to_dict(self):
        """Export all parameters as a dict for saving."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

