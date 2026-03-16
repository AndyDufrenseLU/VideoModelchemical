# VideoMamba 项目使用说明 (Windows 远程服务器专用版)

该项目基于 `2403.06977v2` (VideoMamba)，已针对您的流化床识别任务 (16帧多任务分类: 气速和粒子标签) 进行了完全适配。

> [!WARNING]
> **Windows 环境安装核心难点**：
> `mamba-ssm` 官方严重依赖 Linux 下的 `triton` 加速库，而 `triton` 官方**不支持 Windows**。同时 `causal-conv1d` 在 Windows 下极易因为 CUDA/C++ 编译器版本不匹配（如您遇到的 `The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.4)`）导致编译失败。
> 
> 因此，在 Windows 下**绝对不能直接使用 `pip install`** 安装这两个库。必须严格遵循下方基于**预编译 (Prebuilt Wheel)** 的安装教程。

---

## 必须要做的准备：全新 Conda 环境搭建

为了杜绝版本冲突，强烈建议在您的**远程系统**上新建一个干净的 Conda 环境：

### 1. 创建并激活基础环境
```cmd
conda create -n videomamba python=3.10.13 -y
conda activate videomamba
```
*(注：推荐使用 Python 3.10，这是目前深度学习算子预编译包支持最广泛的版本)*

### 2. 安装严格匹配的 PyTorch 及其对应的 CUDA 运行库
由于之前报错显示您的 PyTorch 编译自 CUDA 12.4，为了确保最大兼容性，我们直接指明安装适配 CUDA 11.8 / 12.1 的版本（推荐 12.1）：
```cmd
 
```

### 3. 安装常规依赖包
```cmd
pip install timm==0.9.12 einops==0.7.0 packaging==23.2 class_registry==2.1.2
```

---

## 核心难点：Mamba 相关算子的 Windows 原生安装

**前方高能**：请**不要**直接 `pip install mamba-ssm causal-conv1d`，因为会触发源码编译和拉取 Linux 的 triton。请按照以下步骤利用社区大神预编译的 Windows Wheel 资源进行安装：

### 第一步：安装 Windows 版的 Triton
在 Windows 下，我们需要使用第三方的 Triton 预编译包。下载并安装 `triton-2.1.0-cp310-cp310-win_amd64.whl`。可以通过以下命令直接通过 URL 安装（适用于 python 3.10）：
```cmd
pip install https://github.com/vllm-project/vllm/releases/download/v0.3.0/triton-2.1.0-cp310-cp310-win_amd64.whl
```
*(如果该链接失效，您可以访问 GitHub 仓库 `https://github.com/CGQAQ/triton-windows` 寻找对应 python 和 cuda 版本的轮子文件拖入 cmd 安装)*

### 第二步：安装预编译的 causal-conv1d 和 mamba-ssm
同样，为了避免触发 Windows 上的 C++ CUDA 源码编译（`ninja` / `cl.exe` / `nvcc` 等环境变量大坑），请直接安装针对 CUDA 12.1 和 Python 3.10 编译好的 WHL 文件：

国内有许多研究者编译了适配 Windows 的版本包（参考仓库 [Mamba-Windows](https://github.com/yyy1128/Mamba-Windows) / [mamba-ssm-win](https://github.com/1-11-1-1/mamba-ssm-win)）。

由于网络环境限制，最优的办法是：
1. 打开浏览器访问：[https://github.com/1-11-1-1/mamba-ssm-win/releases](https://github.com/1-11-1-1/mamba-ssm-win/releases)
2. 在资源列表中，寻找与您刚安装的 Python (cp310) 和 CUDA (cu121) 以及 Torch (torch2.1.2) 完全匹配的两个文件下载到本地：
   - `causal_conv1d-1.1.3-cp310-cp310-win_amd64.whl` (具体版本号视 release 页面为准)
   - `mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl` (具体版本号视 release 页面为准)
3. 假设您下载到了 D 盘，打开命令提示符，执行本地安装：
   ```cmd
   pip install D:\causal_conv1d-xxx-win_amd64.whl
   pip install D:\mamba_ssm-xxx-win_amd64.whl
   ```

如果在安装期间报错缺少 `triton`，由于第一步已经安装，可以加上 `--no-deps` 忽略依赖检查强行安装 mamba：
```cmd
pip install D:\mamba_ssm-xxx-win_amd64.whl --no-deps
```

---

## 如何使用及修改

1. **准备数据集**：将之前项目中使用的 `alltrain_with_c.txt` 文件放入本 `VideoMamba_Project` 目录内，因为 `dataset_txt.py` 将直接从此文件读取时序图片地址和标签。

2. **更改帧数设置**:
   代码默认使用 **16帧** 进行截取。
   如果您要测试使用 **10帧**，只需要打开 `config.py`，然后将：
   `self.num_frames = 16` 改为 `self.num_frames = 10`

3. **运行单 GPU 训练**:
   ```cmd
   python train.py
   ```
   
4. **运行多 GPU 加速训练**:
   ```cmd
   python train_multigpu.py
   ```

系统会自动加载并且在 `VideoMamba_Project` 目录下记录 log 以及权重信息 `best_model.pth`。
