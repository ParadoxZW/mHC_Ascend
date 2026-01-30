"""
纯PyTorch版本的backward（用于对比和debug）
"""

import torch
import mhc_ascend


class MHCLayerDynamicFunction_PurePyTorchBackward(torch.autograd.Function):
    """
    MHC Layer with Dynamic-H (pure PyTorch backward for debugging)
    """

    @staticmethod
    def forward(ctx, x_expanded, rmsnorm_weight, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res, b_pre, b_post, b_res,
                sinkhorn_iters, sinkhorn_eps, rmsnorm_eps):
        """Forward using Ascend ops"""
        B, n, C = x_expanded.shape
        nC = n * C

        # Concatenate phi matrices
        phi_concat = torch.cat([phi_pre, phi_post, phi_res], dim=0)

        # Call Ascend forward
        (output, rms, x_agg_bf16, H_pre_activated, H_post_activated, M,
         y_norm_bf16, x_flat_bf16, rms_h) = mhc_ascend.mhc_layer_fwd_dynamic(
            x_expanded,
            rmsnorm_weight,
            phi_concat,
            float(alpha_pre.item() if alpha_pre.numel() == 1 else alpha_pre),
            float(alpha_post.item() if alpha_post.numel() == 1 else alpha_post),
            float(alpha_res.item() if alpha_res.numel() == 1 else alpha_res),
            b_pre,
            b_post,
            b_res,
            sinkhorn_iters,
            sinkhorn_eps,
            rmsnorm_eps,
        )

        # Save for backward (save BFloat16 tensors)
        ctx.save_for_backward(
            x_expanded, rmsnorm_weight, rms, x_agg_bf16,
            H_pre_activated, H_post_activated, M, y_norm_bf16,
            x_flat_bf16, rms_h,
            phi_pre, phi_post, phi_res,
            b_pre, b_post, b_res,
            alpha_pre, alpha_post, alpha_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.sinkhorn_eps = sinkhorn_eps
        ctx.rmsnorm_eps = rmsnorm_eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Pure PyTorch backward"""
        (x_expanded, rmsnorm_weight, rms, x_agg_bf16,
         H_pre_activated, H_post_activated, M, y_norm_bf16,
         x_flat_bf16, rms_h,
         phi_pre, phi_post, phi_res,
         b_pre, b_post, b_res,
         alpha_pre, alpha_post, alpha_res) = ctx.saved_tensors

        B, n, C = x_expanded.shape
        nC = n * C

        # Convert to Float32
        x = x_expanded.float()
        x_flat = x_flat_bf16.float()
        y_norm = y_norm_bf16.float()
        grad_out = grad_output.float()

        # ========== Step 1: Backward through stream_distribute_mix_add (Pure PyTorch) ==========
        # output = H_post.unsqueeze(-1) * y_norm.unsqueeze(1) + M @ x
        # y_dist = H_post.unsqueeze(-1) * y_norm.unsqueeze(1)  # [B, n, C]
        # x_mixed = M @ x  # [B, n, C]

        # grad_x_mix = M.t() @ grad_out  # [B, n, C]
        grad_x_mix = torch.matmul(M.transpose(-2, -1), grad_out)

        # grad_M = grad_out @ x.transpose(-2, -1)  # [B, n, n]
        grad_M = torch.matmul(grad_out, x.transpose(-2, -1))

        # grad_y_norm = (H_post.unsqueeze(-1) * grad_out).sum(dim=1)  # [B, C]
        grad_y_norm = (H_post_activated.unsqueeze(-1) * grad_out).sum(dim=1)

        # grad_H_post = (grad_out * y_norm.unsqueeze(1)).sum(dim=-1)  # [B, n]
        grad_H_post_activated = (grad_out * y_norm.unsqueeze(1)).sum(dim=-1)

        # Activation derivative: H_post = 2 * sigmoid(tilde_post)
        # d_H_post/d_tilde = 2 * sigmoid' = 2 * sigmoid * (1 - sigmoid) = H_post * (1 - H_post / 2)
        grad_H_post = grad_H_post_activated * H_post_activated * (1.0 - H_post_activated / 2.0)

        # ========== Step 2: Backward through RMSNorm (Pure PyTorch) ==========
        # y_norm = x_agg / rms * rmsnorm_weight
        x_agg = x_agg_bf16.float()

        # d_x_agg = grad_y_norm * rmsnorm_weight / rms  # [B, C]
        # rms is [B, 1], need to broadcast
        grad_x_agg = grad_y_norm * rmsnorm_weight / rms.view(B, 1)

        # d_rmsnorm_weight = (grad_y_norm * x_agg / rms).sum(dim=0)  # [C]
        grad_rmsnorm_weight = (grad_y_norm * x_agg / rms.view(B, 1)).sum(dim=0)

        # Backward through RMS normalization
        #rms_term = -(grad_y_norm * rmsnorm_weight * x_agg / (rms ** 3)).sum(dim=-1, keepdim=True)
        # grad_x_agg += rms_term * x_agg / C

        # ========== Step 3: Backward through stream_aggregate (Pure PyTorch) ==========
        # x_agg = sum_j(H_pre[j] * x[j])  # [B, C]

        # grad_x_from_agg = H_pre.unsqueeze(-1) * grad_x_agg.unsqueeze(1)  # [B, n, C]
        grad_x_from_agg = H_pre_activated.unsqueeze(-1) * grad_x_agg.unsqueeze(1)

        # grad_H_pre_activated = (grad_x_agg.unsqueeze(1) * x).sum(dim=-1)  # [B, n]
        grad_H_pre_activated = (grad_x_agg.unsqueeze(1) * x).sum(dim=-1)

        # Activation derivative: H_pre = sigmoid(tilde_pre)
        # d_H_pre/d_tilde = sigmoid * (1 - sigmoid)
        grad_H_pre = grad_H_pre_activated * H_pre_activated * (1.0 - H_pre_activated)

        # ========== Step 4: Sum gradients from both paths ==========
        grad_x = grad_x_mix + grad_x_from_agg

        # ========== Step 5: Backward through Dynamic-H computation (Pure PyTorch) ==========
        rms_h_f32 = rms_h.float()
        rms_inv = 1.0 / rms_h_f32
        rms_inv2 = rms_inv * rms_inv

        phi_pre_f32 = phi_pre.float()
        phi_post_f32 = phi_post.float()
        phi_res_f32 = phi_res.float()

        # Recompute projections (parameterization B: p = x_flat @ phi)
        p_pre = x_flat @ phi_pre_f32.t()
        p_post = x_flat @ phi_post_f32.t()
        p_res_flat = x_flat @ phi_res_f32.t()
        p_res = p_res_flat.view(B, n, n)

        # grad_H_post and grad_H_pre are d_tilde_post and d_tilde_pre
        d_tilde_pre = grad_H_pre
        d_tilde_post = grad_H_post

        # Backward through Sinkhorn (parameterization B: tilde_res = alpha * p / rms + b)
        alpha_res_f32 = alpha_res.float()
        b_res_f32 = b_res.float()
        tilde_res = alpha_res_f32 * p_res * rms_inv.view(B, 1, 1) + b_res_f32
        H_res_exp = torch.exp(tilde_res)

        d_H_res_exp = mhc_ascend.sinkhorn_knopp_bwd(
            grad_M.contiguous(),
            H_res_exp.contiguous(),
            M.contiguous(),
            ctx.sinkhorn_iters,
            ctx.sinkhorn_eps,
        )

        d_tilde_res = d_H_res_exp * H_res_exp

        # Bias gradients
        d_b_pre = d_tilde_pre.sum(dim=0)
        d_b_post = d_tilde_post.sum(dim=0)
        d_b_res = d_tilde_res.sum(dim=0)

        # Alpha gradients (parameterization B: d_tilde/d_alpha = p / rms)
        alpha_pre_f32 = alpha_pre.float()
        alpha_post_f32 = alpha_post.float()
        # Fix: rms_inv is [B], need [B, 1] for broadcasting with [B, n]
        rms_inv_2d = rms_inv.unsqueeze(-1)
        d_alpha_pre = (d_tilde_pre * (p_pre * rms_inv_2d)).sum()
        d_alpha_post = (d_tilde_post * (p_post * rms_inv_2d)).sum()
        d_alpha_res = (d_tilde_res * (p_res * rms_inv.view(B, 1, 1))).sum()

        # Projection gradients (parameterization B: d_tilde/d_p = alpha / rms)
        d_p_pre = d_tilde_pre * (alpha_pre_f32 * rms_inv_2d)
        d_p_post = d_tilde_post * (alpha_post_f32 * rms_inv_2d)
        d_p_res = d_tilde_res * (alpha_res_f32 * rms_inv.view(B, 1, 1))
        d_p_res_flat = d_p_res.reshape(B, n * n)

        # Phi gradients (parameterization B: p = x_flat @ phi)
        d_phi_pre = d_p_pre.t() @ x_flat
        d_phi_post = d_p_post.t() @ x_flat
        d_phi_res = d_p_res_flat.t() @ x_flat

        # x_flat gradients
        d_x_flat = d_p_pre @ phi_pre_f32
        d_x_flat += d_p_post @ phi_post_f32
        d_x_flat += d_p_res_flat @ phi_res_f32

        # rms_h gradients (parameterization B: d_tilde/d_rms = -alpha * p / rms^2)
        # Fix: rms_inv2 is [B], need [B, 1] for broadcasting with [B, n]
        rms_inv2_2d = rms_inv2.unsqueeze(-1)
        d_r = -(d_tilde_pre * (alpha_pre_f32 * p_pre) * rms_inv2_2d).sum(dim=1)
        d_r -= (d_tilde_post * (alpha_post_f32 * p_post) * rms_inv2_2d).sum(dim=1)
        d_r -= (d_tilde_res * (alpha_res_f32 * p_res) * rms_inv2.view(B, 1, 1)).sum(dim=(1, 2))

        # Backward through RMS computation
        # rms_inv is [B], need to broadcast to [B, 1]
        d_x_flat += d_r[:, None] * x_flat * (rms_inv.view(B, 1) / float(nC))

        # Add to total x gradient
        grad_x = grad_x + d_x_flat.view(B, n, C)

        # Convert to appropriate dtypes
        d_b_pre = d_b_pre.to(b_pre.dtype)
        d_b_post = d_b_post.to(b_post.dtype)
        d_b_res = d_b_res.to(b_res.dtype)
        d_alpha_pre = d_alpha_pre.to(alpha_pre.dtype)
        d_alpha_post = d_alpha_post.to(alpha_post.dtype)
        d_alpha_res = d_alpha_res.to(alpha_res.dtype)
        d_phi_pre = d_phi_pre.to(phi_pre.dtype)
        d_phi_post = d_phi_post.to(phi_post.dtype)
        d_phi_res = d_phi_res.to(phi_res.dtype)

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


def mhc_layer_fused_dynamic_pure_pytorch_backward(
    x_expanded,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    b_pre,
    b_post,
    b_res,
    sinkhorn_iters=20,
    sinkhorn_eps=1e-8,
    rmsnorm_eps=1e-5,
):
    """Wrapper using pure PyTorch backward (matching original signature)"""
    if not torch.is_tensor(alpha_pre):
        alpha_pre = torch.tensor(alpha_pre, device=phi_pre.device, dtype=phi_pre.dtype)
    if not torch.is_tensor(alpha_post):
        alpha_post = torch.tensor(alpha_post, device=phi_pre.device, dtype=phi_pre.dtype)
    if not torch.is_tensor(alpha_res):
        alpha_res = torch.tensor(alpha_res, device=phi_pre.device, dtype=phi_pre.dtype)

    return MHCLayerDynamicFunction_PurePyTorchBackward.apply(
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
