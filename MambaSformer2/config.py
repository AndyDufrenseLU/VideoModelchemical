"""
MambaSformer 配置文件

速度优化说明 (69s/batch → 目标 <15s/batch):
  最大瓶颈是 Mamba SSM 处理的 batch 维度 = B × P
    img_size=224: P=196, B×P = 16×196 = 3136  (慢)
    img_size=160: P=100, B×P = 16×100 = 1600  (快一倍)
  第二大瓶颈是 depth (Block数量)
    depth=12 → depth=8: 节省 33% 计算
"""


class MambaSformerConfig:
    """MambaSformer 完整配置"""

    def __init__(self):
        # ===== 图像/视频参数 =====
        # ⚡ 方法一(最重要): img_size 决定 patch 数量
        #    224 → P=196, B×P=3136 → 慢
        #    160 → P=100, B×P=1600 → 快一倍，对准确率影响有限
        #    128 → P=64,  B×P=1024 → 更快，适合快速实验
        self.img_size = 224       # 🔥 从224改为160，速度提升~2x
        self.patch_size = 16      # 160/16=10，每帧10×10=100个patch
        self.num_frames = 8       # 每个样本8帧连续图片

        # ===== 双任务分类输出 =====
        self.num_classes_particle = 3    # 粒子类型: 0, 1, 2
        self.num_classes_gas = 19        # 气速: 1-19 → 映射 0-18

        # ===== 模型结构参数 =====
        self.embed_dim = 768
        # ⚡ 方法一: depth 越小越快
        #    12→8: 节省 33% 计算量
        #    12→6: 节省 50%（精度会有一定下降）
        self.depth = 8            # 🔥 从12改为8，节省33%计算
        self.num_heads = 12
        self.mlp_ratio = 4.0

        # ===== Mamba SSM 参数 =====
        self.d_state = 8          # SSM 状态向量维度 N
        self.d_conv = 4           # 因果卷积核大小
        self.mamba_expand = 1     # d_inner = 1×768

        # ===== 显存与速度优化 =====
        self.use_checkpoint = True
        # ⚠️ Windows 不支持 torch.compile！
        # 原因: torch.compile 的 inductor 后端需要 Triton，
        #       而 Triton 官方只支持 Linux，Windows 上报错:
        #       "RuntimeError: Cannot find a working triton installation"
        # 注意: torch.compile() 调用本身不报错，但第一次 forward 时崩溃
        self.compile_model = False   # Windows 必须 False

        # ===== 正则化 =====
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.head_dropout = 0.1
        self.label_smoothing = 0.0
        self.use_class_weights = False
        self.drop_path_rate = 0.0

        # ===== 训练参数 =====
        self.batch_size = 16
        self.num_epochs = 100
        self.warmup_epochs = 5
        self.val_interval = 2

        # 差异化学习率
        self.backbone_lr = 1e-4
        self.mamba_lr = 2e-4
        self.head_lr = 1e-3
        self.weight_decay = 1e-4
        self.gas_loss_weight = 1.5
        self.particle_loss_weight = 0.5

        # ===== 梯度累积 =====
        self.grad_accum_steps = 2

        # ===== 数据 =====
        self.num_workers = 6
        self.data_root = "./data"
        self.train_ratio = 0.75
        # ⚡ 方法四: 每个(粒子,气速)组合最多使用多少张图片
        #    None = 使用全部数据
        #    800  = 每组最多800张 → 38组×800张/8帧 = 3800样本，epoch更短
        #    推荐用 800 快速验证模型是否收敛，收敛后再用全部数据
        self.max_images_per_group = 800   # 🔥 新增：每组限800张

        # ===== 早停 =====
        self.patience = 12

        # ===== 多 GPU =====
        self.use_multi_gpu = False

    def __repr__(self):
        P = (self.img_size // self.patch_size) ** 2
        return (
            f"MambaSformerConfig("
            f"img_size={self.img_size}(P={P}), "
            f"num_frames={self.num_frames}, "
            f"embed_dim={self.embed_dim}, depth={self.depth}, "
            f"d_state={self.d_state}, mamba_expand={self.mamba_expand}, "
            f"compile={self.compile_model}, "
            f"max_imgs/group={self.max_images_per_group}, "
            f"val_every={self.val_interval})"
        )

    def to_dict(self):
        """将所有参数导出为字典，用于保存"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
