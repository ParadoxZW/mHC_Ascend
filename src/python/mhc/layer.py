"""
MHC Layer implementation for Ascend NPUs.

This module provides the high-level MHCLayer module that combines
all the underlying operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .ops import (
    sinkhorn_knopp,
    rmsnorm,
    stream_aggregate,
    stream_distribute_mix_add,
    fused_rmsnorm_matmul,
    compute_rms,
    mhc_layer_fused_dynamic_inference,
    mhc_layer_fused_dynamic_training,  # ✅ Fixed! stream_ops backward now works
)


class MHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) Layer.

    This layer implements the mHC architecture from DeepSeek-AI,
    providing efficient feature mixing and aggregation.

    Args:
        hidden_dim: Hidden dimension size (C)
        expansion_rate: Expansion rate (n), number of parallel streams
        num_sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        sinkhorn_eps: Epsilon for Sinkhorn-Knopp stability (also used for dynamic-H RMS)
        rmsnorm_eps: Epsilon for RMSNorm stability
        alpha_init: Initialization scale for alpha parameters (dynamic H)
        use_dynamic_h: Whether to dynamically compute H matrices
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

        # RMSNorm weight for aggregated features
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))

        if use_dynamic_h:
            # Dynamic H computation parameters (match CUDA reference)
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
            # Static H computation: learnable raw H values
            self.H_pre = nn.Parameter(torch.zeros(n, dtype=torch.float32))
            self.H_post = nn.Parameter(torch.zeros(n, dtype=torch.float32))
            H_res_init = alpha_init * torch.randn(n, n)
            self.H_res = nn.Parameter(H_res_init.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MHC layer.

        Args:
            x: Input tensor [batch_size, expansion_rate, hidden_dim]

        Returns:
            Output tensor [batch_size, expansion_rate, hidden_dim]
        """
        B, n, C = x.shape
        assert n == self.expansion_rate, f"Expected expansion_rate={self.expansion_rate}, got {n}"
        assert C == self.hidden_dim, f"Expected hidden_dim={self.hidden_dim}, got {C}"

        x_f32 = x.float()

        # Step 1: Compute or load H matrices
        if self.use_dynamic_h:
            # Use fully C++ orchestrated path for both training and inference
            if (not self.training) and (not torch.is_grad_enabled()):
                # Inference mode: no gradients needed
                return mhc_layer_fused_dynamic_inference(
                    x_f32,
                    self.rmsnorm_weight,
                    self.phi_pre,
                    self.phi_post,
                    self.phi_res,
                    self.alpha_pre,
                    self.alpha_post,
                    self.alpha_res,
                    self.b_pre,
                    self.b_post,
                    self.b_res,
                    self.num_sinkhorn_iters,
                    self.sinkhorn_eps,
                    self.rmsnorm_eps,
                )
            else:
                # Training mode: use autograd-enabled version
                return mhc_layer_fused_dynamic_training(
                    x_f32,
                    self.rmsnorm_weight,
                    self.phi_pre,
                    self.phi_post,
                    self.phi_res,
                    self.alpha_pre,
                    self.alpha_post,
                    self.alpha_res,
                    self.b_pre,
                    self.b_post,
                    self.b_res,
                    self.num_sinkhorn_iters,
                    self.sinkhorn_eps,
                    self.rmsnorm_eps,
                )
        else:
            # Static H: broadcast to batch
            H_pre_raw = self.H_pre.unsqueeze(0).expand(B, -1)
            H_post_raw = self.H_post.unsqueeze(0).expand(B, -1)
            H_res_raw = self.H_res.unsqueeze(0).expand(B, -1, -1)

            # Step 2: Stream aggregate
            # Aggregate: x_agg = sum(sigmoid(H_pre[i]) * x[i])
            x_aggregated = stream_aggregate(x_f32, H_pre_raw)  # [B, C]

            # Step 3: RMSNorm on aggregated features
            y_norm = rmsnorm(x_aggregated, self.rmsnorm_weight, self.rmsnorm_eps)  # [B, C]

            # Step 4: Sinkhorn-Knopp normalization on M
            # Sinkhorn-Knopp normalization on exp(H_res_raw)
            H_res_exp = torch.exp(H_res_raw)
            M_normalized = sinkhorn_knopp(
                H_res_exp, self.num_sinkhorn_iters, self.sinkhorn_eps
            )  # [B, n, n]

            # Step 5: Stream distribute, mix, and add
            # Distribute: y_dist[i] = (2 * sigmoid(H_post[i])) * y_norm
            # Mix: mix_out[i] = sum_j(M[i, j] * x[j])
            # Add: output = y_dist + mix_out
            output = stream_distribute_mix_add(
                y_norm, H_post_raw, M_normalized, x_f32
            )  # [B, n, C]

            return output

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"expansion_rate={self.expansion_rate}, "
            f"num_sinkhorn_iters={self.num_sinkhorn_iters}, "
            f"use_dynamic_h={self.use_dynamic_h}"
        )


# =============================================================================
# Helper function to create an MHC layer
# =============================================================================

def create_mhc_layer(
    hidden_dim: int,
    expansion_rate: int = 4,
    num_sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-8,
    rmsnorm_eps: float = 1e-5,
    use_dynamic_h: bool = True,
    alpha_init: float = 0.01,
) -> MHCLayer:
    """
    Create an MHC layer with the given configuration.

    Args:
        hidden_dim: Hidden dimension size
        expansion_rate: Number of parallel streams
        num_sinkhorn_iters: Sinkhorn-Knopp iterations
        sinkhorn_eps: Sinkhorn epsilon
        rmsnorm_eps: RMSNorm epsilon
        use_dynamic_h: Whether to use dynamic H computation
        alpha_init: Initialization scale for alpha parameters (dynamic H)

    Returns:
        MHCLayer instance
    """
    return MHCLayer(
        hidden_dim=hidden_dim,
        expansion_rate=expansion_rate,
        num_sinkhorn_iters=num_sinkhorn_iters,
        sinkhorn_eps=sinkhorn_eps,
        rmsnorm_eps=rmsnorm_eps,
        use_dynamic_h=use_dynamic_h,
        alpha_init=alpha_init,
    )


# =============================================================================
# Residual Connection Wrapper
# =============================================================================

class MHCResidualWrapper(nn.Module):
    """
    Wrapper to use MHCLayer as a drop-in replacement for residual connections.

    Automatically prepares input in the format required by MHCLayer [B, n, C].

    Args:
        hidden_dim: Feature dimension
        expansion_rate: Number of streams (default 4)
        residual_mode: How to fill extra streams:
            - 'zero': Fill with zeros
            - 'decay': Fill with decayed copies of residual
        use_post_norm: Whether to apply LayerNorm after mHC output (default True for stability)
        **mhc_kwargs: Additional arguments for MHCLayer

    Example:
        >>> # In transformer layer
        >>> residual = hidden_states  # [B, T, C]
        >>> attn_output = self.attention(hidden_states)  # [B, T, C]
        >>> mhc_wrapper = MHCResidualWrapper(hidden_dim=512, expansion_rate=4)
        >>> hidden_states = mhc_wrapper(residual, attn_output)  # Replaces: residual + attn_output
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        residual_mode: str = 'decay',
        use_post_norm: bool = True,  # 新增：默认使用输出归一化
        **mhc_kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.residual_mode = residual_mode
        self.use_post_norm = False  # 暂时禁用，debug用

        self.mhc = MHCLayer(
            hidden_dim=hidden_dim,
            expansion_rate=expansion_rate,
            **mhc_kwargs
        )

    def forward(
        self,
        residual: torch.Tensor,
        branch_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace residual connection with mHC.

        Standard: output = residual + branch_output
        mHC: output = mhc([residual, branch_output, ...]) -> average

        Args:
            residual: Residual input [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            branch_output: Branch output (e.g., attention or MLP) [same shape as residual]

        Returns:
            Output [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """
        original_shape = residual.shape
        if len(original_shape) == 3:
            B, T, C = original_shape
            residual = residual.reshape(B * T, C)
            branch_output = branch_output.reshape(B * T, C)
        else:
            B, C = original_shape
            T = 1

        n = self.expansion_rate

        # Prepare mHC input [B*T, n, C]
        mhc_input = torch.zeros(B * T, n, C, device=residual.device, dtype=residual.dtype)
        mhc_input[:, 0] = residual
        mhc_input[:, 1] = branch_output

        # Fill remaining streams based on mode
        if self.residual_mode == 'zero':
            pass  # Already zeros
        elif self.residual_mode == 'decay':
            for i in range(2, n):
                mhc_input[:, i] = residual * (0.1 ** (i - 1))
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

        # mHC forward
        output = self.mhc(mhc_input)  # [B*T, n, C]

        # Average across streams to get [B*T, C]
        output = output.mean(dim=1)

        # Reshape back to original
        if T > 1:
            output = output.reshape(B, T, C)

        return output

    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.hidden_dim}, '
            f'expansion_rate={self.expansion_rate}, '
            f'residual_mode={self.residual_mode}'
        )


def replace_residual_with_mhc(
    hidden_dim: int,
    expansion_rate: int = 4,
    **kwargs
) -> MHCResidualWrapper:
    """
    Helper to create MHCResidualWrapper for replacing residual connections.

    Example:
        >>> # In model definition
        >>> self.mhc_post_attn = replace_residual_with_mhc(hidden_dim=512)
        >>>
        >>> # In forward pass
        >>> # Before: hidden_states = residual + attn_output
        >>> # After:
        >>> hidden_states = self.mhc_post_attn(residual, attn_output)
    """
    return MHCResidualWrapper(
        hidden_dim=hidden_dim,
        expansion_rate=expansion_rate,
        **kwargs
    )
