"""
纯 PyTorch backward 实现，用于对比测试
"""

import torch
import torch.nn.functional as F


def sinkhorn_knopp_pytorch(inp: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Sinkhorn-Knopp normalization - 纯 PyTorch 实现（支持 autograd）

    Args:
        inp: [B, M, N]
        num_iters: 迭代次数
        eps: 数值稳定性

    Returns:
        out: [B, M, N] doubly stochastic matrix
    """
    B, M, N = inp.shape
    A = inp.clone()

    for _ in range(num_iters):
        # Row normalization
        row_sum = A.sum(dim=2, keepdim=True) + eps
        A = A / row_sum

        # Column normalization
        col_sum = A.sum(dim=1, keepdim=True) + eps
        A = A / col_sum

    return A


def stream_distribute_mix_add_pytorch(
    y_norm: torch.Tensor,      # [B, C]
    H_post_raw: torch.Tensor,  # [B, n]
    M: torch.Tensor,           # [B, n, n]
    x_inp: torch.Tensor,       # [B, n, C]
) -> torch.Tensor:
    """
    Stream distribute-mix-add - 纯 PyTorch 实现（支持 autograd）

    Returns:
        out: [B, n, C]
    """
    B, n, C = x_inp.shape

    # 1. Activate H_post: sigmoid * 2
    H_post_activated = 2.0 * torch.sigmoid(H_post_raw)  # [B, n]

    # 2. Distribute: y_dist[i] = H_post[i] * y_norm
    y_dist = H_post_activated.unsqueeze(-1) * y_norm.unsqueeze(1)  # [B, n, C]

    # 3. Mix: mix_out = M @ x_inp
    mix_out = torch.bmm(M, x_inp)  # [B, n, C]

    # 4. Add
    out = y_dist + mix_out

    return out


def compute_rms_pytorch(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute RMS - 纯 PyTorch 实现（支持 autograd）

    Args:
        x: [B, C]

    Returns:
        rms: [B]
    """
    return torch.sqrt((x ** 2).mean(dim=-1) + eps)


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    RMSNorm - 纯 PyTorch 实现（支持 autograd）

    Args:
        x: [B, C]
        weight: [C]

    Returns:
        y: [B, C]
    """
    rms = compute_rms_pytorch(x, eps)
    y = (x / rms.unsqueeze(-1)) * weight.unsqueeze(0)
    return y


def stream_aggregate_pytorch(x: torch.Tensor, H_pre_raw: torch.Tensor) -> torch.Tensor:
    """
    Stream aggregate - 纯 PyTorch 实现（支持 autograd）

    Args:
        x: [B, n, C]
        H_pre_raw: [B, n]

    Returns:
        y: [B, C]
    """
    # Activate H_pre
    H_pre_activated = torch.sigmoid(H_pre_raw)  # [B, n]

    # Aggregate: y = sum_i(H_pre[i] * x[i])
    y = (H_pre_activated.unsqueeze(-1) * x).sum(dim=1)  # [B, C]

    return y


def dynamic_h_projection_pytorch(
    x_flat: torch.Tensor,  # [B, n*C]
    phi: torch.Tensor,     # [n, n*C] or [n*n, n*C]
    alpha: torch.Tensor,   # scalar
    b: torch.Tensor,       # [n] or [n, n]
    rms: torch.Tensor,     # [B]
    is_res: bool = False   # phi_res uses different shape
) -> torch.Tensor:
    """
    Dynamic H projection - 纯 PyTorch 实现（支持 autograd）

    Args:
        x_flat: [B, n*C]
        phi: [n, n*C] for pre/post, [n*n, n*C] for res
        alpha: scalar
        b: [n] for pre/post, [n,n] for res
        rms: [B]
        is_res: whether this is for H_res (affects output shape)

    Returns:
        H_raw: [B, n] for pre/post, [B, n, n] for res
    """
    # Projection: p = x_flat @ phi.T
    p = torch.matmul(x_flat, phi.t())  # [B, n] or [B, n*n]

    # Compute tilde: alpha * p * (1/rms) + b
    rms_inv = 1.0 / rms  # [B]
    if is_res:
        # H_res: [B, n*n]
        tilde = alpha * p * rms_inv.unsqueeze(-1) + b.reshape(-1)
        H_raw = tilde.reshape(-1, b.size(0), b.size(1))  # [B, n, n]
    else:
        # H_pre/H_post: [B, n]
        tilde = alpha * p * rms_inv.unsqueeze(-1) + b
        H_raw = tilde

    return H_raw
