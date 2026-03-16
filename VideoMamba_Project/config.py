"""
Configuration file for VideoMamba model
Adapted for 16-frame fluidized bed time-series imagery
"""

class VideoMambaConfig:
    """Configuration for VideoMamba model"""
    
    def __init__(self):
        # Model architecture
        self.img_size = 224  # 图像尺寸
        self.patch_size = 16
        self.num_frames = 16  # 处理 16 帧 (可改为10帧)
        
        # Multi-task learning: two classification tasks
        self.num_classes_particle = 3   # Particle classes (0-2)
        self.num_classes_gas = 19       # Gas velocity classes (1-19 → mapped to 0-18)
        
        # VideoMamba settings (aligning with tiny)
        self.model_size = "tiny" # tiny, small, or middle
        
        # ===== 正则化参数 (anti-overfitting) =====
        self.head_dropout = 0.5       # 分类头前的强dropout
        self.label_smoothing = 0.1    # 防止模型过度自信
        self.use_class_weights = True # 启用类别权重平衡
        
        # Training settings
        self.batch_size = 16  # VRAM permitted
        self.weight_decay = 0.05  
        self.num_epochs = 100 
        self.warmup_epochs = 3
        
        # 差异化学习率
        self.backbone_lr = 5e-6  
        self.head_lr = 5e-4      
        
        # 梯度累积
        self.grad_accum_steps = 4  
        
        # Data loading
        self.num_workers = 2
        
        # Data paths
        self.data_root = "./data"
        self.train_split = "train"
        self.val_split = "val"
        
        # Early stopping
        self.patience = 20  
        self.use_multi_gpu = True
        
    def __repr__(self):
        return f"VideoMambaConfig(img_size={self.img_size}, num_frames={self.num_frames}, " \
               f"model_size={self.model_size}, " \
               f"head_dropout={self.head_dropout}, label_smoothing={self.label_smoothing})"
    
    def to_dict(self):
        """将所有参数导出为字典，用于保存"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
