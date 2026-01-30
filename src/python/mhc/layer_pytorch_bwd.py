"""
MHC Layer with selectable backward implementation (Ascend C vs PyTorch)
用于调试：可以选择性地切换某些算子的backward到PyTorch实现
"""

import torch
import torch.nn as nn
from typing import Optional
import os

from .ops import (
    sinkhorn_knopp,
    rmsnorm,
    stream_aggregate,
    stream_distribute_mix_add,
    compute_rms,
)

from .ops_pytorch_backward import (
    sinkhorn_knopp_pytorch,
    rmsnorm_pytorch,
    stream_aggregate_pytorch,
    stream_distribute_mix_add_pytorch,
    compute_rms_pytorch,
    dynamic_h_projection_pytorch,
)


# 配置：哪些算子使用PyTorch backward
USE_PYTORCH_BWD = {
    'sinkhorn': os.getenv('MHC_SINKHORN_BWD', 'ascend') == 'pytorch',
    'stream_ops': os.getenv('MHC_STREAM_OPS_BWD', 'ascend') == 'pytorch',
    'rmsnorm': os.getenv('MHC_RMSNORM_BWD', 'ascend') == 'pytorch',
    'projection': os.getenv('MHC_PROJECTION_BWD', 'ascend') == 'pytorch',
}


class MHCLayerPyTorchBwd(nn.Module):
    """
    MHC Layer with selectable backward implementation.

    Environment variables control which operators use PyTorch backward:
    - MHC_SINKHORN_BWD=pytorch: Use PyTorch for Sinkhorn backward
    - MHC_STREAM_OPS_BWD=pytorch: Use PyTorch for stream ops backward
    - MHC_RMSNORM_BWD=pytorch: Use PyTorch for RMSNorm backward
    - MHC_PROJECTION_BWD=pytorch: Use PyTorch for projection backward
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        num_sinkhorn_iters: int = 20,
        sinkhorn_eps: float = 1e-8,
        rmsnorm_eps: float = 1e-5,
        use_dynamic_h: bool = True,
        alpha_init: float = 0.01,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.rmsnorm_eps = rmsnorm_eps
        self.use_dynamic_h = use_dynamic_h

        n = expansion_rate
        C = hidden_dim
        nC = n * C

        # RMSNorm weight
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))

        if use_dynamic_h:
            # Dynamic H parameters
            self.phi_pre = nn.Parameter(torch.randn(n, nC) * 0.02)
            self.phi_post = nn.Parameter(torch.randn(n, nC) * 0.02)
            self.phi_res = nn.Parameter(torch.randn(n * n, nC) * 0.02)

            self.b_pre = nn.Parameter(torch.zeros(n))
            self.b_post = nn.Parameter(torch.zeros(n))
            self.b_res = nn.Parameter(torch.zeros(n, n))

            self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
            self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
            self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        else:
            # Static H
            self.H_pre = nn.Parameter(torch.zeros(n, dtype=torch.float32))
            self.H_post = nn.Parameter(torch.zeros(n, dtype=torch.float32))
            H_res_init = alpha_init * torch.randn(n, n)
            self.H_res = nn.Parameter(H_res_init.float())

        # 记录使用的backward实现
        print(f"MHC Backward configuration:")
        print(f"  Sinkhorn: {'PyTorch' if USE_PYTORCH_BWD['sinkhorn'] else 'Ascend C'}")
        print(f"  Stream ops: {'PyTorch' if USE_PYTORCH_BWD['stream_ops'] else 'Ascend C'}")
        print(f"  RMSNorm: {'PyTorch' if USE_PYTORCH_BWD['rmsnorm'] else 'Ascend C'}")
        print(f"  Projection: {'PyTorch' if USE_PYTORCH_BWD['projection'] else 'Ascend C'}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selectable backward.

        Args:
            x: [B, n, C]

        Returns:
            out: [B, n, C]
        """
        B, n, C = x.shape
        x_f32 = x.float()

        if self.use_dynamic_h:
            # Dynamic H computation
            x_flat = x_f32.reshape(B, -1)  # [B, n*C]

            # Compute RMS
            if USE_PYTORCH_BWD['rmsnorm']:
                rms = compute_rms_pytorch(x_flat, self.sinkhorn_eps)
            else:
                rms = compute_rms(x_flat, self.sinkhorn_eps)

            # Projection - always use PyTorch (Ascend C has PyTorch backward anyway)
            H_pre_raw = dynamic_h_projection_pytorch(
                x_flat, self.phi_pre, self.alpha_pre, self.b_pre, rms, is_res=False
            )
            H_post_raw = dynamic_h_projection_pytorch(
                x_flat, self.phi_post, self.alpha_post, self.b_post, rms, is_res=False
            )
            H_res_raw = dynamic_h_projection_pytorch(
                x_flat, self.phi_res, self.alpha_res, self.b_res, rms, is_res=True
            )

            # Sinkhorn on H_res
            H_res_exp = torch.exp(H_res_raw)
            if USE_PYTORCH_BWD['sinkhorn']:
                M_normalized = sinkhorn_knopp_pytorch(H_res_exp, self.num_sinkhorn_iters, self.sinkhorn_eps)
            else:
                M_normalized = sinkhorn_knopp(H_res_exp, self.num_sinkhorn_iters, self.sinkhorn_eps)

        else:
            # Static H
            H_pre_raw = self.H_pre.unsqueeze(0).expand(B, -1)
            H_post_raw = self.H_post.unsqueeze(0).expand(B, -1)
            H_res_raw = self.H_res.unsqueeze(0).expand(B, -1, -1)

            H_res_exp = torch.exp(H_res_raw)
            if USE_PYTORCH_BWD['sinkhorn']:
                M_normalized = sinkhorn_knopp_pytorch(H_res_exp, self.num_sinkhorn_iters, self.sinkhorn_eps)
            else:
                M_normalized = sinkhorn_knopp(H_res_exp, self.num_sinkhorn_iters, self.sinkhorn_eps)

        # Stream aggregate
        if USE_PYTORCH_BWD['stream_ops']:
            x_aggregated = stream_aggregate_pytorch(x_f32, H_pre_raw)
        else:
            x_aggregated = stream_aggregate(x_f32, H_pre_raw)

        # RMSNorm
        if USE_PYTORCH_BWD['rmsnorm']:
            y_norm = rmsnorm_pytorch(x_aggregated, self.rmsnorm_weight.float(), self.rmsnorm_eps)
        else:
            y_norm = rmsnorm(x_aggregated, self.rmsnorm_weight, self.rmsnorm_eps)

        # Stream distribute-mix-add
        if USE_PYTORCH_BWD['stream_ops']:
            output = stream_distribute_mix_add_pytorch(y_norm, H_post_raw, M_normalized, x_f32)
        else:
            output = stream_distribute_mix_add(y_norm, H_post_raw, M_normalized, x_f32)

        return output
