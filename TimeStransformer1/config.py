"""
Configuration file for TimeSformer model
"""

class TimeSformerConfig:
    """Configuration for TimeSformer model"""
    
    def __init__(self):
        # Model architecture
        self.img_size = 224  # 图像尺寸
        self.patch_size = 16
        self.num_frames = 10 # 单帧版本排查时序问题
        
        # Multi-task learning: two classification tasks
        # 粒子标签: 0, 1, 2, 3 → 4类
        # 气速标签: 1-19 (不是每个粒子都有19个气速) → 映射到 0-18 共19类
        self.num_classes_particle = 3   # Particle classes (0-3)
        self.num_classes_gas = 19       # Gas velocity classes (1-19 → mapped to 0-18)
        self.embed_dim = 768  # Embedding dimension
        self.depth = 12  # 使用ViT-B/16全部12层
        self.num_heads = 12  # Number of attention heads
        self.mlp_ratio = 4.0  # MLP hidden dim ratio
        self.dropout = 0.0  # 先关闭dropout让模型起跳
        self.attention_dropout = 0.0  # 先关闭attention dropout
        
        # Training settings
        self.batch_size = 16  # 30帧时attention激活很大，16较安全
        self.learning_rate = 1e-4  # backbone LR
        self.weight_decay = 1e-4  # 参考ConvNeXt降低weight_decay
        self.num_epochs = 50
        self.warmup_epochs = 3
        
        # Multi-GPU settings
        self.use_multi_gpu = True  # Enable multi-GPU training
        self.num_workers = 4  # Number of data loading workers (降低以避免Windows多进程开销)
        
        # Data paths
        self.data_root = "./data"  # Root directory containing sequential images
        self.train_split = "train"
        self.val_split = "val"
        
        # Attention type
        self.attention_type = "divided_space_time"  # divided space-time attention
        
    def __repr__(self):
        return f"TimeSformerConfig(img_size={self.img_size}, num_frames={self.num_frames}, " \
               f"embed_dim={self.embed_dim}, depth={self.depth}, num_heads={self.num_heads})"
