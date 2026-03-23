"""
纯 PyTorch 实现的 Mamba SSM 块（Windows 兼容，无需 mamba-ssm / triton）
参考: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)

核心思想:
  - 状态空间模型 (SSM): h_t = A*h_{t-1} + B*x_t,  y_t = C*h_t
  - "Selective" 机制: B, C, Δ 由输入 x_t 动态决定（不同于S4的固定参数）
  - 线性时序复杂度 O(L)，非 Transformer 的 O(L²)

Windows 兼容性说明:
  - 不依赖 mamba_ssm (需要 Linux CUDA Triton 编译)
  - 全部使用 torch.einsum / cumsum 实现 selective_scan
  - 性能略低于 CUDA 融合版，但功能完整
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_seq(u, delta, A, B, C, D, delta_bias=None, delta_softplus=True):
    """
    纯 PyTorch 实现的 Selective Scan（序列扫描版）
    对应 Mamba 论文 Algorithm 2

    Args:
        u     : (B, L, D)     输入序列
        delta : (B, L, D)     步长 Δ，控制"记忆衰减速度"
        A     : (D, N)        状态转移矩阵（对数形式，负实数）
        B     : (B, L, N)     输入映射矩阵（由 x 动态生成）
        C     : (B, L, N)     输出映射矩阵（由 x 动态生成）
        D     : (D,)          跳跃连接系数（skip connection）
        delta_bias  : (D,)    可选 bias for delta
        delta_softplus: bool  是否对 delta 做 softplus

    Returns:
        y: (B, L, D)  输出序列
    """
    B_batch, L, D_dim = u.shape
    N = A.shape[1]

    # 1. 处理 delta: softplus + bias
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = F.softplus(delta)  # (B, L, D)

    # ================================================================
    # 2. 分步计算 — 每步只创建 (B, D, N) 切片，彻底避免 OOM
    #
    # 旧方案 (OOM): 预算 deltaA = (B, L, D, N)
    #   batch=48, P=196 → B_inner=9408, L=8, D=768, N=8
    #   = 9408×8×768×8 × 4 bytes = 1.72 GB → OOM !
    #
    # 新方案: 每步只分配 (B, D, N) = 9408×768×8 × 4 bytes = 0.22 GB
    #   L=8 步循环，峰值显存仅 ~0.22 GB，batch=48 完全安全
    # ================================================================

    A_f32     = A.float()           # (D, N)
    delta_f32 = delta.float()       # (B, L, D)
    u_f32     = u.float()           # (B, L, D)
    B_f32     = B.float()           # (B, L, N)
    C_f32     = C.float()           # (B, L, N)

    h = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(L):
        d_i  = delta_f32[:, i]          # (B, D)
        u_i  = u_f32[:, i]              # (B, D)
        B_i  = B_f32[:, i]              # (B, N)
        C_i  = C_f32[:, i]              # (B, N)

        # 每步只创建 (B, D, N) 的中间张量 — 不再有 (B, L, D, N)
        dA_i  = torch.exp(torch.einsum('bd,dn->bdn', d_i, A_f32))   # (B, D, N)
        dBu_i = torch.einsum('bd,bn,bd->bdn', d_i, B_i, u_i)        # (B, D, N)

        h = dA_i * h + dBu_i                                         # (B, D, N)
        y_i = torch.einsum('bdn,bn->bd', h, C_i)                    # (B, D)
        ys.append(y_i)

    y = torch.stack(ys, dim=1)      # (B, L, D) float32

    # skip connection
    y = y + u_f32 * D.float().unsqueeze(0).unsqueeze(0)

    return y.to(u.dtype)


class MambaBlock(nn.Module):
    """
    单个 Mamba Block
    结构: LN → expand → SSM → gate → contract → 残差

    参数:
        d_model : 输入/输出 token 维度
        d_state : SSM 状态向量维度 N（默认 16）
        d_conv  : 卷积核大小（因果卷积，提取局部特征，默认 4）
        expand  : 内部维度扩展倍数（默认 2）
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)  # 内部维度

        # 输入投影: d_model → 2 * d_inner (一半做 SSM, 一半做 gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 因果深度卷积（提取局部时序特征）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,       # causal padding
            groups=self.d_inner,      # depth-wise
            bias=True
        )

        # SSM 输入投影: x → dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)

        # dt (delta) 投影
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM 参数
        # A: (d_inner, d_state) 对数形式（固定初始化，可学习）
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))  # 存对数，保证 A 负实数

        # D: skip connection 系数
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer Norm（Pre-Norm）
        self.norm = nn.LayerNorm(d_model)

        self._init_dt()

    def _init_dt(self):
        """初始化 dt 偏置（Mamba 论文附录 D）"""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)  输入序列（B=batch, L=序列长度, D=特征维度）
        Returns:
            out: (B, L, D)
        """
        residual = x
        x = self.norm(x)

        B, L, D = x.shape

        # 1. 输入投影，拆分为 z (gate) 和 x
        xz = self.in_proj(x)   # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 2. 因果深度卷积（在序列维度上滑动）
        x_conv = self.conv1d(x_in.transpose(1, 2))      # (B, d_inner, L + d_conv - 1)
        x_conv = x_conv[:, :, :L].transpose(1, 2)       # 去掉 padding → (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 3. SSM 参数生成
        # x_proj: (B, L, d_inner) → (B, L, d_state*2 + d_inner)
        x_dbl = self.x_proj(x_conv)
        B_ssm, C_ssm, dt_raw = torch.split(x_dbl, [self.d_state, self.d_state, self.d_inner], dim=-1)
        # B_ssm, C_ssm: (B, L, N),  dt_raw: (B, L, d_inner)

        dt = self.dt_proj(dt_raw)  # (B, L, d_inner)

        # 4. SSM 扫描
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)  确保 A < 0
        y = selective_scan_seq(
            u=x_conv,
            delta=dt,
            A=A,
            B=B_ssm,
            C=C_ssm,
            D=self.D.float(),
            delta_softplus=True,
        )  # (B, L, d_inner)

        # 5. 门控（SiLU gate）
        y = y * F.silu(z)  # (B, L, d_inner)

        # 6. 输出投影
        out = self.out_proj(y)  # (B, L, D)

        # 残差
        return out + residual
