import torch

# 检查是否有可用的GPU
def check_gpu():
    # 查看当前是否有可用的 GPU
    if torch.cuda.is_available():
        print(f"CUDA is available! Your GPU is: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")

# 测试一个简单的张量操作
def test_gpu_computation():
    # 如果有 GPU, 将 tensor 移动到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个简单的 tensor
    x = torch.rand(3, 3)
    print(f"Tensor on {device}: \n{x}")
    
    # 在 GPU 上进行计算
    x = x.to(device)
    print(f"After moving tensor to {device}, result is: \n{x}")
    
    # 执行一个简单的计算操作，看看是否能在 GPU 上运行
    result = x * 2
    print(f"Computation result on {device}: \n{result}")

if __name__ == "__main__":
    check_gpu()  # 检查 GPU 是否可用
    test_gpu_computation()  # 测试 GPU 计算

