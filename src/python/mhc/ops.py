"""
PyTorch autograd functions for mHC Ascend kernels.

This module provides PyTorch-compatible autograd functions that wrap
the C++ kernels, enabling automatic differentiation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    import torch_npu
    _HAS_TORCH_NPU = True
except Exception:
    torch_npu = None
    _HAS_TORCH_NPU = False

try:
    import mhc_ascend
except ImportError:
    raise ImportError(
        "mhc_ascend C++ extension not found. Please build the project first:\n"
        "  cd mHC_ascend/build && cmake .. && make && make install"
    )


# =============================================================================
# Sinkhorn-Knopp Operations
# =============================================================================

class SinkhornKnoppFunction(torch.autograd.Function):
    """Sinkhorn-Knopp doubly stochastic matrix normalization."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
        """
        Forward pass for Sinkhorn-Knopp normalization.

        Args:
            inp: Input tensor [batch_size, M, N]
            num_iters: Number of normalization iterations
            eps: Epsilon for numerical stability

        Returns:
            out: Normalized tensor [batch_size, M, N]
        """
        out = mhc_ascend.sinkhorn_knopp_fwd(inp, num_iters, eps)
        ctx.save_for_backward(inp, out)
        ctx.num_iters = num_iters
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Backward pass for Sinkhorn-Knopp normalization."""
        inp, out = ctx.saved_tensors
        grad_inp = mhc_ascend.sinkhorn_knopp_bwd(
            grad_out, inp, out, ctx.num_iters, ctx.eps
        )
        return grad_inp, None, None


def sinkhorn_knopp(
    inp: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    """
    Apply Sinkhorn-Knopp normalization to a batch of matrices.

    Args:
        inp: Input tensor [batch_size, M, N]
        num_iters: Number of normalization iterations
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [batch_size, M, N]
    """
    if _HAS_TORCH_NPU:
        return SinkhornKnoppFunction.apply(inp.float(), num_iters, eps)
    return SinkhornKnoppFunction.apply(inp.float(), num_iters, eps)


# =============================================================================
# RMSNorm Operations
# =============================================================================

class RMSNormFunction(torch.autograd.Function):
    """Root Mean Square Normalization."""

    @staticmethod
    def forward(
        ctx, inp: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            inp: Input tensor [batch_size, hidden_dim]
            weight: Weight tensor [hidden_dim]
            eps: Epsilon for numerical stability

        Returns:
            out: Normalized tensor [batch_size, hidden_dim]
        """
        inp_bf16 = inp.bfloat16()
        weight_bf16 = weight.bfloat16()
        out, rms = mhc_ascend.rmsnorm_fwd(inp_bf16, weight_bf16, eps)
        ctx.save_for_backward(inp_bf16, weight_bf16, rms)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass for RMSNorm."""
        inp_bf16, weight_bf16, rms = ctx.saved_tensors
        grad_out_f32 = grad_out.float()
        grad_inp, grad_weight = mhc_ascend.rmsnorm_bwd(
            grad_out_f32, inp_bf16, weight_bf16, rms
        )
        return grad_inp, grad_weight, None


def rmsnorm(inp: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMS normalization.

    Args:
        inp: Input tensor [batch_size, hidden_dim]
        weight: Weight tensor [hidden_dim]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [batch_size, hidden_dim]
    """
    # Always use custom AscendC kernel on NPU to keep behavior aligned.
    return RMSNormFunction.apply(inp, weight, eps)


# =============================================================================
# Stream Aggregate Operations
# =============================================================================

class StreamAggregateFunction(torch.autograd.Function):
    """Stream aggregation with weighted sum."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor, H_pre_raw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for stream aggregation.

        Args:
            inp: Input tensor [batch_size, n, hidden_dim]
            H_pre_raw: Raw H_pre weights [batch_size, n] (before sigmoid)

        Returns:
            out: Aggregated tensor [batch_size, hidden_dim]
        """
        inp_f32 = inp.float()
        H_pre_raw_f32 = H_pre_raw.float()
        out, H_pre_activated = mhc_ascend.stream_aggregate_fwd(inp_f32, H_pre_raw_f32)
        ctx.save_for_backward(inp_f32, H_pre_activated)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass for stream aggregation."""
        inp_f32, H_pre_activated = ctx.saved_tensors
        grad_out_f32 = grad_out.float()
        grad_inp, grad_H_pre_activated = mhc_ascend.stream_aggregate_bwd(
            grad_out_f32, inp_f32, H_pre_activated
        )
        grad_H_pre = grad_H_pre_activated * H_pre_activated * (1.0 - H_pre_activated)
        return grad_inp, grad_H_pre


def stream_aggregate(inp: torch.Tensor, H_pre_raw: torch.Tensor) -> torch.Tensor:
    """
    Aggregate multiple streams with sigmoid-activated weights.

    Args:
        inp: Input tensor [batch_size, n, hidden_dim]
        H_pre_raw: Raw H_pre weights [batch_size, n]

    Returns:
        Aggregated tensor [batch_size, hidden_dim]
    """
    # Always use custom AscendC kernel on NPU to keep behavior aligned.
    return StreamAggregateFunction.apply(inp, H_pre_raw)


# =============================================================================
# Stream Distribute Mix Add Operations
# =============================================================================

class StreamDistributeMixAddFunction(torch.autograd.Function):
    """Stream distribution, mixing, and residual addition."""

    @staticmethod
    def forward(
        ctx,
        y_norm: torch.Tensor,
        H_post_raw: torch.Tensor,
        M: torch.Tensor,
        x_inp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for stream distribute-mix-add.

        Args:
            y_norm: Normalized aggregated features [batch_size, hidden_dim]
            H_post_raw: Raw H_post weights [batch_size, n]
            M: Mixing matrix [batch_size, n, n]
            x_inp: Original input [batch_size, n, hidden_dim]

        Returns:
            out: Output tensor [batch_size, n, hidden_dim]
        """
        y_norm_bf16 = y_norm.bfloat16()
        y_norm_f32 = y_norm_bf16.float()
        H_post_raw_f32 = H_post_raw.float()
        M_f32 = M.float()
        x_inp_f32 = x_inp.float()
        out, H_post_activated = mhc_ascend.stream_distribute_mix_add_fwd(
            y_norm_bf16, H_post_raw_f32, M_f32, x_inp_f32
        )
        ctx.save_for_backward(x_inp_f32, y_norm_f32, M_f32, H_post_activated)
        return out

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass for stream distribute-mix-add."""
        x_inp_f32, y_norm_f32, M_f32, H_post_activated = ctx.saved_tensors
        grad_x, grad_y_norm, grad_M, grad_H_post_activated = (
            mhc_ascend.stream_distribute_mix_add_bwd(
                grad_out, x_inp_f32, y_norm_f32, M_f32, H_post_activated
            )
        )
        grad_H_post = grad_H_post_activated * H_post_activated * (
            1.0 - H_post_activated / 2.0
        )
        return grad_y_norm.bfloat16(), grad_H_post, grad_M, grad_x


def stream_distribute_mix_add(
    y_norm: torch.Tensor,
    H_post_raw: torch.Tensor,
    M: torch.Tensor,
    x_inp: torch.Tensor,
) -> torch.Tensor:
    """
    Distribute, mix, and add streams.

    Args:
        y_norm: Normalized aggregated features [batch_size, hidden_dim]
        H_post_raw: Raw H_post weights [batch_size, n]
        M: Mixing matrix [batch_size, n, n]
        x_inp: Original input [batch_size, n, hidden_dim]

    Returns:
        Output tensor [batch_size, n, hidden_dim]
    """
    # Always use custom AscendC kernel on NPU to keep behavior aligned.
    return StreamDistributeMixAddFunction.apply(y_norm, H_post_raw, M, x_inp)


# =============================================================================
# Dynamic-H Projection (BF16 MatMul)
# =============================================================================

class FusedProjectionFunction(torch.autograd.Function):
    """BF16 matmul projection for dynamic-H (forward fused, backward in PyTorch)."""

    @staticmethod
    def forward(ctx, x_flat: torch.Tensor, phi_concat: torch.Tensor) -> torch.Tensor:
        x_flat_f32 = x_flat.float().contiguous()
        phi_concat_f32 = phi_concat.float().contiguous()
        out = mhc_ascend.fused_rmsnorm_matmul_fwd(
            x_flat_f32.bfloat16().contiguous(),
            phi_concat_f32.bfloat16().contiguous(),
        )
        ctx.save_for_backward(x_flat_f32, phi_concat_f32)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat_f32, phi_concat_f32 = ctx.saved_tensors
        grad_out_f32 = grad_out.float()
        grad_x, grad_phi = mhc_ascend.fused_rmsnorm_matmul_bwd(
            grad_out_f32.contiguous(),
            x_flat_f32.contiguous(),
            phi_concat_f32.contiguous(),
        )
        return grad_x, grad_phi


def fused_rmsnorm_matmul(x_flat: torch.Tensor, phi_concat: torch.Tensor) -> torch.Tensor:
    """
    Project x_flat with phi_concat using bf16 matmul kernel.

    Args:
        x_flat: [B, K] float tensor
        phi_concat: [out_dim, K] float tensor

    Returns:
        H_proj_concat: [B, out_dim] float tensor
    """
    if _HAS_TORCH_NPU:
        x = x_flat.float()
        w = phi_concat.float()
        # Match reference: bf16 matmul accumulation, then return float32
        out = torch.matmul(x.bfloat16(), w.t().bfloat16())
        return out.float()
    return FusedProjectionFunction.apply(x_flat, phi_concat)


def compute_rms(x_flat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute RMS per row using Ascend kernel.

    Args:
        x_flat: [B, K] float tensor
        eps: epsilon for numerical stability

    Returns:
        rms: [B] float tensor
    """
    # Always use custom AscendC kernel on NPU to keep behavior aligned.
    return mhc_ascend.compute_rms_fwd(x_flat.bfloat16().contiguous(), eps)


# =============================================================================
# MHC Layer (Dynamic-H) Fused Forward (Inference)
# =============================================================================

def mhc_layer_fused_dynamic_inference(
    x_expanded: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    phi_pre: torch.Tensor,
    phi_post: torch.Tensor,
    phi_res: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    b_pre: torch.Tensor,
    b_post: torch.Tensor,
    b_res: torch.Tensor,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-8,
    rmsnorm_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused dynamic-H forward for inference (no autograd).
    Uses host-side orchestration to call AscendC kernels.
    """
    n = phi_pre.size(0)
    phi_concat = torch.cat(
        [phi_pre, phi_post, phi_res.view(n * n, -1)], dim=0
    ).contiguous()

    alpha_pre_val = float(alpha_pre.item()) if torch.is_tensor(alpha_pre) else float(alpha_pre)
    alpha_post_val = float(alpha_post.item()) if torch.is_tensor(alpha_post) else float(alpha_post)
    alpha_res_val = float(alpha_res.item()) if torch.is_tensor(alpha_res) else float(alpha_res)

    output, *_ = mhc_ascend.mhc_layer_fwd_dynamic(
        x_expanded.contiguous(),
        rmsnorm_weight.contiguous(),
        phi_concat,
        alpha_pre_val,
        alpha_post_val,
        alpha_res_val,
        b_pre.contiguous(),
        b_post.contiguous(),
        b_res.contiguous(),
        sinkhorn_iters,
        sinkhorn_eps,
        rmsnorm_eps,
    )
    return output


# =============================================================================
# MHC Layer (Dynamic-H) Training with Autograd
# =============================================================================

class MHCLayerDynamicFunction(torch.autograd.Function):
    """
    MHC Layer with dynamic H computation for training.
    Forward is fully C++ orchestrated, backward uses PyTorch autograd with C++ kernels.
    """

    @staticmethod
    def forward(
        ctx,
        x_expanded: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        phi_pre: torch.Tensor,
        phi_post: torch.Tensor,
        phi_res: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        b_pre: torch.Tensor,
        b_post: torch.Tensor,
        b_res: torch.Tensor,
        sinkhorn_iters: int,
        sinkhorn_eps: float,
        rmsnorm_eps: float,
    ) -> torch.Tensor:
        """
        Fully C++ orchestrated forward pass for training.
        Returns only output, saves intermediate results for backward.
        """
        n = phi_pre.size(0)
        phi_concat = torch.cat(
            [phi_pre, phi_post, phi_res.view(n * n, -1)], dim=0
        ).contiguous()

        alpha_pre_val = float(alpha_pre.item()) if torch.is_tensor(alpha_pre) else float(alpha_pre)
        alpha_post_val = float(alpha_post.item()) if torch.is_tensor(alpha_post) else float(alpha_post)
        alpha_res_val = float(alpha_res.item()) if torch.is_tensor(alpha_res) else float(alpha_res)

        # Call C++ function that returns all intermediate results
        (
            output,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            x_flat_bf16,
            rms_h,
        ) = mhc_ascend.mhc_layer_fwd_dynamic(
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            phi_concat,
            alpha_pre_val,
            alpha_post_val,
            alpha_res_val,
            b_pre.contiguous(),
            b_post.contiguous(),
            b_res.contiguous(),
            sinkhorn_iters,
            sinkhorn_eps,
            rmsnorm_eps,
        )

        # Save for backward
        ctx.save_for_backward(
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            x_flat_bf16,
            rms_h,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            alpha_pre,
            alpha_post,
            alpha_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.sinkhorn_eps = sinkhorn_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using unified C++ orchestration.
        All kernel calls are now managed in C++ layer for better performance.
        """
        (
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            x_flat_bf16,
            rms_h,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            alpha_pre,
            alpha_post,
            alpha_res,
        ) = ctx.saved_tensors

        # Call unified C++ backward
        (
            grad_x,
            grad_rmsnorm_weight,
            d_phi_pre,
            d_phi_post,
            d_phi_res,
            d_alpha_pre,
            d_alpha_post,
            d_alpha_res,
            d_b_pre,
            d_b_post,
            d_b_res,
        ) = mhc_ascend.mhc_layer_bwd_dynamic(
            grad_output.contiguous(),
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            rms.contiguous(),
            x_agg_bf16.contiguous(),
            H_pre_activated.contiguous(),
            H_post_activated.contiguous(),
            M.contiguous(),
            y_norm_bf16.contiguous(),
            x_flat_bf16.contiguous(),
            rms_h.contiguous(),
            phi_pre.contiguous(),
            phi_post.contiguous(),
            phi_res.contiguous(),
            float(alpha_pre.item()),
            float(alpha_post.item()),
            float(alpha_res.item()),
            b_pre.contiguous(),
            b_post.contiguous(),
            b_res.contiguous(),
            ctx.sinkhorn_iters,
            ctx.sinkhorn_eps,
            ctx.rmsnorm_eps,
        )

        # Ensure gradients have correct dtypes
        d_alpha_pre = d_alpha_pre.to(alpha_pre.dtype).view_as(alpha_pre)
        d_alpha_post = d_alpha_post.to(alpha_post.dtype).view_as(alpha_post)
        d_alpha_res = d_alpha_res.to(alpha_res.dtype).view_as(alpha_res)

        return (
            grad_x,
            grad_rmsnorm_weight,
            d_phi_pre,
            d_phi_post,
            d_phi_res,
            d_alpha_pre,
            d_alpha_post,
            d_alpha_res,
            d_b_pre,
            d_b_post,
            d_b_res,
            None,  # sinkhorn_iters
            None,  # sinkhorn_eps
            None,  # rmsnorm_eps
        )


def mhc_layer_fused_dynamic_training(
    x_expanded: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    phi_pre: torch.Tensor,
    phi_post: torch.Tensor,
    phi_res: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    b_pre: torch.Tensor,
    b_post: torch.Tensor,
    b_res: torch.Tensor,
    sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-8,
    rmsnorm_eps: float = 1e-5,
) -> torch.Tensor:
    """
    MHC Layer forward with dynamic H computation for training.
    Uses fully C++ orchestrated forward pass with autograd support.

    Args:
        x_expanded: Input tensor [B, n, C]
        rmsnorm_weight: RMSNorm weight [C]
        phi_pre: Projection matrix for H_pre [n, n*C]
        phi_post: Projection matrix for H_post [n, n*C]
        phi_res: Projection matrix for H_res [n*n, n*C]
        alpha_pre: Scale parameter for H_pre (scalar tensor)
        alpha_post: Scale parameter for H_post (scalar tensor)
        alpha_res: Scale parameter for H_res (scalar tensor)
        b_pre: Bias for H_pre [n]
        b_post: Bias for H_post [n]
        b_res: Bias for H_res [n, n]
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        sinkhorn_eps: Epsilon for Sinkhorn-Knopp
        rmsnorm_eps: Epsilon for RMSNorm

    Returns:
        Output tensor [B, n, C]
    """
    if not torch.is_tensor(alpha_pre):
        alpha_pre = torch.tensor(alpha_pre, device=phi_pre.device, dtype=phi_pre.dtype)
    if not torch.is_tensor(alpha_post):
        alpha_post = torch.tensor(alpha_post, device=phi_pre.device, dtype=phi_pre.dtype)
    if not torch.is_tensor(alpha_res):
        alpha_res = torch.tensor(alpha_res, device=phi_pre.device, dtype=phi_res.dtype)

    return MHCLayerDynamicFunction.apply(
        x_expanded.float(),
        rmsnorm_weight.bfloat16(),
        phi_pre.float(),
        phi_post.float(),
        phi_res.float(),
        alpha_pre,
        alpha_post,
        alpha_res,
        b_pre.float(),
        b_post.float(),
        b_res.float(),
        sinkhorn_iters,
        sinkhorn_eps,
        rmsnorm_eps,
    )
