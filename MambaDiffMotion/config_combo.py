class MambaSformerComboConfig:
    """Single-head 38-class configuration tuned for a 24 GB RTX A5000."""

    def __init__(self):
        # Input
        self.img_size = 224
        self.patch_size = 16
        self.num_frames = 16

        # Labels
        self.num_classes_combo = 38
        self.num_classes_particle = 3
        self.num_classes_gas = 19

        # Model
        self.embed_dim = 768
        self.depth = 8
        self.num_heads = 12
        self.mlp_ratio = 4.0
        self.d_state = 8
        self.d_conv = 4
        self.mamba_expand = 1

        # Runtime
        self.use_checkpoint = True
        self.compile_model = False

        # Regularization
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.head_dropout = 0.1
        self.motion_head_dropout = 0.15
        self.fusion_dropout = 0.1
        self.label_smoothing = 0.0
        self.use_class_weights = False
        self.weight_decay = 1e-4
        self.motion_input_scale = 3.0
        self.motion_residual_weight = 0.35
        self.motion_dim = 128
        self.motion_kernel_size = 3

        # Optimization
        self.batch_size = 8
        self.grad_accum_steps = 1
        self.backbone_lr = 5e-5
        self.mamba_lr = 1.5e-4
        self.head_lr = 2e-3
        self.combo_loss_weight = 1.0
        self.gas_aux_loss_weight = 0.3
        self.num_epochs = 60
        self.warmup_epochs = 5
        self.val_interval = 1

        # Data
        self.num_workers = 8
        self.train_ratio = 0.8
        self.max_images_per_group = None
        self.data_root = "./data"

        # Multi-GPU
        self.use_multi_gpu = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self):
        patches = (self.img_size // self.patch_size) ** 2
        return (
            f"MambaSformerComboConfig("
            f"img_size={self.img_size}(P={patches}), "
            f"frames={self.num_frames}, "
            f"depth={self.depth}, "
            f"batch={self.batch_size}x{self.grad_accum_steps}, "
            f"classes={self.num_classes_combo})"
        )
