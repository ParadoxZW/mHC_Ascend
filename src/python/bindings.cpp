/**
 * @file bindings.cpp
 * @brief PyTorch bindings for mHC Ascend kernels
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * This file provides pybind11 bindings for all mHC Ascend C kernels,
 * allowing them to be called from PyTorch Python code.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <unordered_map>
#include <cstdlib>  // for std::getenv, std::atoi

#include "../csrc/include/mhc_types.h"
#include "mhc_kernels/aclrtlaunch_fused_rmsnorm_matmul_backward.h"
#include "mhc_kernels/aclrtlaunch_fused_rmsnorm_matmul_forward.h"
#include "mhc_kernels/aclrtlaunch_compute_rms_forward.h"
#include "mhc_kernels/aclrtlaunch_rmsnorm_backward.h"
#include "mhc_kernels/aclrtlaunch_rmsnorm_forward.h"
#include "mhc_kernels/aclrtlaunch_sinkhorn_knopp_backward.h"
#include "mhc_kernels/aclrtlaunch_sinkhorn_knopp_forward.h"
#include "mhc_kernels/aclrtlaunch_stream_aggregate_backward.h"
#include "mhc_kernels/aclrtlaunch_stream_aggregate_forward.h"
#include "mhc_kernels/aclrtlaunch_stream_distribute_mix_add_backward.h"
#include "mhc_kernels/aclrtlaunch_stream_distribute_mix_add_forward.h"
#include "mhc_kernels/aclrtlaunch_test_vectorize.h"

using namespace mhc_ascend;
namespace py = pybind11;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Launch an Ascend kernel
 *
 * Strategy for optimization:
 * - Keep stream.synchronize() before launch (needed for correctness)
 * - Remove event.synchronize() after launch (use delayed cleanup)
 * - Use a static cache for tiling memory to avoid repeated malloc/free
 */

// Thread-local cache for tiling memory (to avoid repeated malloc/free)
thread_local static std::unordered_map<size_t, void*> tiling_cache;

template<typename TilingT, typename LaunchFunc, typename... Args>
void launch_kernel(
    LaunchFunc launch_func,
    const TilingT& tiling,
    Args... args)
{
    auto cur_stream = c10_npu::getCurrentNPUStream();

    // Get or allocate tiling data from cache
    size_t tiling_size = sizeof(TilingT);
    void* tiling_device = nullptr;

    auto it = tiling_cache.find(tiling_size);
    if (it != tiling_cache.end()) {
        tiling_device = it->second;
    } else {
        aclrtMalloc(&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
        tiling_cache[tiling_size] = tiling_device;
    }

    // Copy tiling data
    aclrtMemcpy(tiling_device, tiling_size, &tiling, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Ensure preceding ops complete (needed for correctness with Sinkhorn etc.)
    cur_stream.synchronize();

    // Launch kernel
    launch_func(
        tiling.used_core_num,
        cur_stream.stream(false),
        args...,
        tiling_device
    );

    // OPTIMIZATION: No synchronization after launch
    // The tiling memory is cached and reused, so we don't need to wait
    // PyTorch's stream ordering guarantees the kernel will execute in order
}

// =============================================================================
// Sinkhorn-Knopp Bindings
// =============================================================================

// Helper to get multi-core setting from environment
static int get_multicore_setting() {
    static int multicore = -1;
    if (multicore < 0) {
        const char* env = std::getenv("MHC_MULTICORE");
        // Default to 8 cores for performance, set MHC_MULTICORE=1 for single-core debug
        multicore = env ? std::atoi(env) : 8;
        if (multicore <= 0) multicore = 1;
        if (multicore > 32) multicore = 32;  // Cap at reasonable max
    }
    return multicore;
}

// Helper to get Sinkhorn forward implementation mode from environment
// MHC_SINKHORN_FWD_IMPL: auto (default), scalar, vectorized
static int get_sinkhorn_fwd_impl_mode() {
    static int impl_mode = -1;
    if (impl_mode < 0) {
        const char* env = std::getenv("MHC_SINKHORN_FWD_IMPL");
        if (env == nullptr) {
            impl_mode = 0;  // AUTO
        } else {
            std::string mode(env);
            if (mode == "scalar") {
                impl_mode = 1;  // SCALAR (v2_Optimized)
            } else if (mode == "vectorized") {
                impl_mode = 2;  // VECTORIZED (v3_Vectorized)
            } else {
                impl_mode = 0;  // AUTO (default for unknown values)
            }
        }
    }
    return impl_mode;
}

// Helper to get Sinkhorn backward implementation mode from environment
// MHC_SINKHORN_BWD_IMPL: auto (default), scalar, vectorized, n4_optimized
static int get_sinkhorn_bwd_impl_mode() {
    static int impl_mode = -1;
    if (impl_mode < 0) {
        const char* env = std::getenv("MHC_SINKHORN_BWD_IMPL");
        if (env == nullptr) {
            impl_mode = 0;  // AUTO
        } else {
            std::string mode(env);
            if (mode == "scalar") {
                impl_mode = 1;  // SCALAR
            } else if (mode == "vectorized") {
                impl_mode = 2;  // VECTORIZED
            } else if (mode == "n4_optimized" || mode == "n4") {
                impl_mode = 3;  // N4_OPTIMIZED
            } else {
                impl_mode = 0;  // AUTO (default for unknown values)
            }
        }
    }
    return impl_mode;
}

at::Tensor sinkhorn_knopp_fwd(
    const at::Tensor& inp,
    int num_iters,
    float eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(inp.dim() == 3, "Input must be 3D tensor [B, M, N]");
    TORCH_CHECK(inp.size(1) == inp.size(2),
        "Sinkhorn-Knopp requires square matrices (M == N), got M=", inp.size(1), ", N=", inp.size(2));

    auto out = at::empty_like(inp);

    SinkhornTiling tiling;
    tiling.batch_size = inp.size(0);
    tiling.M = inp.size(1);
    tiling.N = inp.size(2);
    tiling.num_iters = num_iters;
    tiling.eps = eps;
    // Use multi-core for performance, limited by batch size
    int num_cores = get_multicore_setting();
    tiling.used_core_num = std::min(num_cores, static_cast<int>(inp.size(0)));
    // Set forward implementation mode from environment variable
    tiling.fwd_impl_mode = get_sinkhorn_fwd_impl_mode();
    tiling.bwd_impl_mode = 0;  // Not used in forward

    launch_kernel(
        aclrtlaunch_sinkhorn_knopp_forward,
        tiling,
        inp.data_ptr(),
        out.data_ptr());

    return out;
}

at::Tensor sinkhorn_knopp_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& inp,
    const at::Tensor& out,
    int num_iters,
    float eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_out), "Gradient must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(out), "Output must be on NPU device");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out must be float32");
    TORCH_CHECK(inp.scalar_type() == at::kFloat, "inp must be float32");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(grad_out.sizes() == inp.sizes(), "grad_out shape must match inp");
    TORCH_CHECK(out.sizes() == inp.sizes(), "out shape must match inp");
    TORCH_CHECK(inp.size(1) == inp.size(2),
        "Sinkhorn-Knopp requires square matrices (M == N), got M=", inp.size(1), ", N=", inp.size(2));

    auto grad_inp = at::empty_like(inp);

    SinkhornTiling tiling;
    tiling.batch_size = inp.size(0);
    tiling.M = inp.size(1);
    tiling.N = inp.size(2);
    tiling.num_iters = num_iters;
    tiling.eps = eps;
    // Use multi-core for performance, limited by batch size
    int num_cores = get_multicore_setting();
    tiling.used_core_num = std::min(num_cores, static_cast<int>(inp.size(0)));
    // Set implementation modes from environment variables
    tiling.fwd_impl_mode = 0;  // Not used in backward
    tiling.bwd_impl_mode = get_sinkhorn_bwd_impl_mode();

    launch_kernel(
        aclrtlaunch_sinkhorn_knopp_backward,
        tiling,
        grad_out.data_ptr(),
        inp.data_ptr(),
        out.data_ptr(),
        grad_inp.data_ptr());

    return grad_inp;
}

// =============================================================================
// RMSNorm Bindings
// =============================================================================

std::tuple<at::Tensor, at::Tensor> rmsnorm_fwd(
    const at::Tensor& inp,
    const at::Tensor& weight,
    float eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(inp.dim() == 2, "Input must be 2D tensor [B, C]");
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "Weight must be bfloat16");

    auto out = at::empty_like(inp);
    auto rms = at::empty({inp.size(0)}, inp.options().dtype(at::kFloat));

    RMSNormTiling tiling;
    tiling.batch_size = inp.size(0);
    tiling.hidden_dim = inp.size(1);
    tiling.eps = eps;
    tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(inp.size(0)));
    tiling.tile_size = inp.size(1);

    launch_kernel(
        aclrtlaunch_rmsnorm_forward,
        tiling,
        inp.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        rms.data_ptr());

    return std::make_tuple(out, rms);
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& inp,
    const at::Tensor& weight,
    const at::Tensor& rms)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_out), "Gradient must be on NPU device");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out must be float32");
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(rms.scalar_type() == at::kFloat, "rms must be float32");

    auto grad_inp = at::empty_like(inp);
    auto grad_weight = at::zeros({weight.size(0)}, weight.options().dtype(at::kFloat));

    RMSNormTiling tiling;
    tiling.batch_size = inp.size(0);
    tiling.hidden_dim = inp.size(1);
    tiling.eps = 1e-5f;  // Default, actual value from forward
    tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(inp.size(0)));
    tiling.tile_size = inp.size(1);

    launch_kernel(
        aclrtlaunch_rmsnorm_backward,
        tiling,
        grad_out.data_ptr(),
        inp.data_ptr(),
        weight.data_ptr(),
        rms.data_ptr(),
        grad_inp.data_ptr(),
        grad_weight.data_ptr());

    return std::make_tuple(grad_inp, grad_weight);
}

// =============================================================================
// Stream Aggregate Bindings
// =============================================================================

std::tuple<at::Tensor, at::Tensor> stream_aggregate_fwd(
    const at::Tensor& inp,
    const at::Tensor& H_pre_raw)
{
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(inp.dim() == 3, "Input must be 3D tensor [B, n, C]");
    TORCH_CHECK(inp.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(H_pre_raw.scalar_type() == at::kFloat, "H_pre_raw must be float32");

    int B = inp.size(0);
    int n = inp.size(1);
    int C = inp.size(2);

    auto out = at::empty({B, C}, inp.options().dtype(at::kBFloat16));
    auto H_pre_activated = at::empty_like(H_pre_raw);

    StreamAggregateTiling tiling;
    tiling.batch_size = B;
    tiling.n = n;
    tiling.C = C;
    tiling.used_core_num = std::min(get_multicore_setting(), B);

    launch_kernel(
        aclrtlaunch_stream_aggregate_forward,
        tiling,
        inp.data_ptr(),
        H_pre_raw.data_ptr(),
        out.data_ptr(),
        H_pre_activated.data_ptr());

    return std::make_tuple(out, H_pre_activated);
}

std::tuple<at::Tensor, at::Tensor> stream_aggregate_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& inp,
    const at::Tensor& H_pre_activated)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_out), "Gradient must be on NPU device");

    int B = inp.size(0);
    int n = inp.size(1);
    int C = inp.size(2);

    auto grad_inp = at::empty_like(inp);
    auto grad_H_pre = at::empty_like(H_pre_activated);

    StreamAggregateTiling tiling;
    tiling.batch_size = B;
    tiling.n = n;
    tiling.C = C;
    tiling.used_core_num = std::min(get_multicore_setting(), B);

    launch_kernel(
        aclrtlaunch_stream_aggregate_backward,
        tiling,
        grad_out.data_ptr(),
        inp.data_ptr(),
        H_pre_activated.data_ptr(),
        grad_inp.data_ptr(),
        grad_H_pre.data_ptr());

    // Explicit sync to ensure kernel completion before returning
    // This is needed because GlobalTensor::SetValue writes may not be immediately visible
    c10_npu::getCurrentNPUStream().synchronize();

    return std::make_tuple(grad_inp, grad_H_pre);
}

// =============================================================================
// Stream Distribute Mix Add Bindings
// =============================================================================

std::tuple<at::Tensor, at::Tensor> stream_distribute_mix_add_fwd(
    const at::Tensor& y_norm,
    const at::Tensor& H_post_raw,
    const at::Tensor& M,
    const at::Tensor& x_inp)
{
    TORCH_CHECK(torch_npu::utils::is_npu(y_norm), "Input must be on NPU device");
    TORCH_CHECK(y_norm.scalar_type() == at::kBFloat16, "y_norm must be bfloat16");
    TORCH_CHECK(H_post_raw.scalar_type() == at::kFloat, "H_post_raw must be float32");
    TORCH_CHECK(M.scalar_type() == at::kFloat, "M must be float32");
    TORCH_CHECK(x_inp.scalar_type() == at::kFloat, "x_inp must be float32");

    int B = x_inp.size(0);
    int n = x_inp.size(1);
    int C = x_inp.size(2);

    auto out = at::empty_like(x_inp);
    auto H_post_activated = at::empty_like(H_post_raw);

    StreamDistributeMixAddTiling tiling;
    tiling.batch_size = B;
    tiling.n = n;
    tiling.C = C;
    tiling.used_core_num = std::min(get_multicore_setting(), B);

    launch_kernel(
        aclrtlaunch_stream_distribute_mix_add_forward,
        tiling,
        y_norm.data_ptr(),
        H_post_raw.data_ptr(),
        M.data_ptr(),
        x_inp.data_ptr(),
        out.data_ptr(),
        H_post_activated.data_ptr());

    return std::make_tuple(out, H_post_activated);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> stream_distribute_mix_add_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x_inp,
    const at::Tensor& y_norm,
    const at::Tensor& M,
    const at::Tensor& H_post_activated)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_out), "Gradient must be on NPU device");

    int B = x_inp.size(0);
    int n = x_inp.size(1);
    int C = x_inp.size(2);

    auto grad_x = at::empty_like(x_inp);
    auto grad_y_norm = at::empty({B, C}, y_norm.options().dtype(at::kFloat));
    auto grad_M = at::empty_like(M);
    auto grad_H_post = at::empty_like(H_post_activated);

    StreamDistributeMixAddTiling tiling;
    tiling.batch_size = B;
    tiling.n = n;
    tiling.C = C;
    tiling.used_core_num = std::min(get_multicore_setting(), B);

    // IMPORTANT: y_norm is BF16 but kernel expects float32
    // Must convert before passing to kernel (same as fused layer backward at line 648)
    auto y_norm_f32 = y_norm.to(at::kFloat).contiguous();

    launch_kernel(
        aclrtlaunch_stream_distribute_mix_add_backward,
        tiling,
        grad_out.data_ptr(),
        x_inp.data_ptr(),
        y_norm_f32.data_ptr(),
        M.data_ptr(),
        H_post_activated.data_ptr(),
        grad_x.data_ptr(),
        grad_y_norm.data_ptr(),
        grad_M.data_ptr(),
        grad_H_post.data_ptr());

    return std::make_tuple(grad_x, grad_y_norm, grad_M, grad_H_post);
}

// =============================================================================
// Fused MatMul (Dynamic-H Projection)
// =============================================================================

/**
 * @brief BF16 MatMul for dynamic-H projection
 *
 * Performance comparison (benchmark_fused_matmul.py):
 * - PyTorch native matmul: ~0.038ms (uses optimized NPU kernels)
 * - Custom scalar kernel: 0.2-3.4ms (30-100x slower)
 *
 * Implementation options:
 * - Default: PyTorch matmul (fastest, uses optimized CANN ops)
 * - MHC_USE_SCALAR_MATMUL=1: Custom scalar kernel (for debugging/verification)
 *
 * Note: Ascend Matmul API integration requires significant build system changes
 * and is left as a future optimization. PyTorch already uses optimized NPU ops.
 */
at::Tensor fused_rmsnorm_matmul_fwd(
    const at::Tensor& inp,
    const at::Tensor& weight)
{
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(inp.dim() == 2, "Input must be 2D tensor [B, K]");
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "Weight must be bfloat16");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D tensor [out_dim, K]");
    TORCH_CHECK(weight.size(1) == inp.size(1), "Weight K dimension must match input");

    int64_t B = inp.size(0);
    int64_t hidden_dim = inp.size(1);
    int64_t out_dim = weight.size(0);

    // Check if we should use scalar kernel instead of PyTorch matmul
    // Environment variable MHC_USE_SCALAR_MATMUL=1 enables custom scalar kernel
    const char* use_scalar_env = std::getenv("MHC_USE_SCALAR_MATMUL");
    bool use_scalar_kernel = use_scalar_env && std::atoi(use_scalar_env) == 1;

    if (use_scalar_kernel) {
        // Custom Ascend C scalar kernel (for debugging/verification)
        auto out = at::empty({B, out_dim}, inp.options().dtype(at::kFloat));

        FusedRMSNormMatMulTiling tiling;
        tiling.batch_size = B;
        tiling.hidden_dim = hidden_dim;
        tiling.out_dim = out_dim;
        tiling.eps = 1e-8f;  // Not used in forward, but set for consistency
        tiling.used_core_num = get_multicore_setting();

        // Note: rms is not computed in this kernel (computed separately)
        auto dummy_rms = at::empty({B}, inp.options().dtype(at::kFloat));

        launch_kernel(
            aclrtlaunch_fused_rmsnorm_matmul_forward,
            tiling,
            inp.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            dummy_rms.data_ptr());

        return out;
    }

    // Default: Use PyTorch matmul (fastest, uses optimized CANN ops)
    auto out = torch::matmul(inp, weight.t());
    return out.to(at::kFloat);
}

std::tuple<at::Tensor, at::Tensor> fused_rmsnorm_matmul_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& x_flat,
    const at::Tensor& phi_concat)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_out), "Inputs must be on NPU device");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "Inputs must be float32");

    auto grad_x = at::matmul(grad_out, phi_concat);
    auto grad_phi = at::matmul(grad_out.transpose(0, 1), x_flat);
    return std::make_tuple(grad_x, grad_phi);
}

at::Tensor compute_rms_fwd(
    const at::Tensor& inp,
    float eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(inp), "Input must be on NPU device");
    TORCH_CHECK(inp.dim() == 2, "Input must be 2D tensor [B, K]");
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "Input must be bfloat16");

    auto rms = at::empty({inp.size(0)}, inp.options().dtype(at::kFloat));

    ComputeRMSTiling tiling;
    tiling.batch_size = inp.size(0);
    tiling.hidden_dim = inp.size(1);
    tiling.eps = eps;
    tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(inp.size(0)));

    launch_kernel(
        aclrtlaunch_compute_rms_forward,
        tiling,
        inp.data_ptr(),
        rms.data_ptr());

    return rms;
}

// =============================================================================
// MHC Layer (Dynamic-H) Fused Forward - Host Orchestration
// =============================================================================

// Forward declaration
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mhc_layer_fwd_dynamic(
    const at::Tensor& x_expanded,
    const at::Tensor& rmsnorm_weight,
    const at::Tensor& phi_concat,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    const at::Tensor& b_pre,
    const at::Tensor& b_post,
    const at::Tensor& b_res,
    int sinkhorn_iters,
    float sinkhorn_eps,
    float rmsnorm_eps);

// =============================================================================
// MHC Layer (Dynamic-H) Fused Backward - Host Orchestration
// =============================================================================

/**
 * @brief Unified backward for MHC Layer with dynamic-H computation
 *
 * This function orchestrates all backward kernel calls in C++ layer,
 * reducing Python-C++ boundary overhead and matching CUDA architecture.
 *
 * @return Tuple of gradients for all learnable parameters
 */
std::tuple<
    at::Tensor,  // grad_x_expanded [B, n, C]
    at::Tensor,  // grad_rmsnorm_weight [C]
    at::Tensor,  // grad_phi_pre [n, nC]
    at::Tensor,  // grad_phi_post [n, nC]
    at::Tensor,  // grad_phi_res [n*n, nC]
    at::Tensor,  // grad_alpha_pre [1]
    at::Tensor,  // grad_alpha_post [1]
    at::Tensor,  // grad_alpha_res [1]
    at::Tensor,  // grad_b_pre [n]
    at::Tensor,  // grad_b_post [n]
    at::Tensor   // grad_b_res [n*n]
>
mhc_layer_bwd_dynamic(
    // Input gradient
    const at::Tensor& grad_output,      // [B, n, C]
    // Forward-saved tensors
    const at::Tensor& x_expanded,       // [B, n, C]
    const at::Tensor& rmsnorm_weight,   // [C]
    const at::Tensor& rms,              // [B]
    const at::Tensor& x_agg_bf16,       // [B, C]
    const at::Tensor& H_pre_activated,  // [B, n]
    const at::Tensor& H_post_activated, // [B, n]
    const at::Tensor& M,                // [B, n, n]
    const at::Tensor& y_norm_bf16,      // [B, C]
    const at::Tensor& x_flat_bf16,      // [B, nC]
    const at::Tensor& rms_h,            // [B]
    // Model parameters
    const at::Tensor& phi_pre,          // [n, nC]
    const at::Tensor& phi_post,         // [n, nC]
    const at::Tensor& phi_res,          // [n*n, nC]
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    const at::Tensor& b_pre,            // [n]
    const at::Tensor& b_post,           // [n]
    const at::Tensor& b_res,            // [n*n]
    // Hyperparameters
    int sinkhorn_iters,
    float sinkhorn_eps,
    float rmsnorm_eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(grad_output), "grad_output must be on NPU device");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be 3D tensor [B, n, C]");

    int64_t B = x_expanded.size(0);
    int64_t n = x_expanded.size(1);
    int64_t C = x_expanded.size(2);
    int64_t nC = n * C;

    auto x_f32 = x_expanded.to(at::kFloat).contiguous();
    auto grad_out_f32 = grad_output.to(at::kFloat).contiguous();
    auto y_norm_f32 = y_norm_bf16.to(at::kFloat).contiguous();

    // =========================================================================
    // Step 1: Backward through stream_distribute_mix_add
    // =========================================================================
    auto grad_x_mix = at::empty({B, n, C}, grad_output.options().dtype(at::kFloat));
    auto grad_y_norm = at::empty({B, C}, y_norm_bf16.options().dtype(at::kFloat));
    auto grad_M = at::empty_like(M);
    auto grad_H_post_activated = at::empty_like(H_post_activated);

    {
        StreamDistributeMixAddTiling tiling;
        tiling.batch_size = B;
        tiling.n = n;
        tiling.C = C;
        tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(B));

        launch_kernel(
            aclrtlaunch_stream_distribute_mix_add_backward,
            tiling,
            grad_out_f32.data_ptr(),
            x_f32.data_ptr(),
            y_norm_f32.data_ptr(),
            M.data_ptr(),
            H_post_activated.data_ptr(),
            grad_x_mix.data_ptr(),
            grad_y_norm.data_ptr(),
            grad_M.data_ptr(),
            grad_H_post_activated.data_ptr());
    }

    // Step 1b: H_post sigmoid derivative (soft_sign activation)
    // d_tilde_post = d_H_post_activated * H_post * (1 - H_post / 2)
    auto grad_H_post = grad_H_post_activated * H_post_activated * (1.0f - H_post_activated / 2.0f);

    // =========================================================================
    // Step 2: Backward through RMSNorm
    // IMPORTANT: rmsnorm_backward kernel expects bf16 weight input
    // =========================================================================
    auto grad_x_agg = at::empty_like(x_agg_bf16);
    auto grad_rmsnorm_weight = at::zeros({C}, rmsnorm_weight.options().dtype(at::kFloat));
    auto rmsnorm_weight_bf16 = rmsnorm_weight.to(at::kBFloat16).contiguous();

    {
        RMSNormTiling tiling;
        tiling.batch_size = B;
        tiling.hidden_dim = C;
        tiling.eps = rmsnorm_eps;
        tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(B));
        tiling.tile_size = C;

        launch_kernel(
            aclrtlaunch_rmsnorm_backward,
            tiling,
            grad_y_norm.data_ptr(),
            x_agg_bf16.data_ptr(),
            rmsnorm_weight_bf16.data_ptr(),
            rms.data_ptr(),
            grad_x_agg.data_ptr(),
            grad_rmsnorm_weight.data_ptr());
    }

    // =========================================================================
    // Step 3: Backward through stream_aggregate
    // IMPORTANT: rmsnorm_bwd returns BF16 grad_x_agg, but stream_aggregate_bwd
    // kernel expects Float32 inputs. Convert to avoid incorrect gradient values.
    // =========================================================================
    auto grad_x_from_agg = at::empty({B, n, C}, x_expanded.options().dtype(at::kFloat));
    auto grad_H_pre_activated = at::empty_like(H_pre_activated);

    {
        StreamAggregateTiling tiling;
        tiling.batch_size = B;
        tiling.n = n;
        tiling.C = C;
        tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(B));

        launch_kernel(
            aclrtlaunch_stream_aggregate_backward,
            tiling,
            grad_x_agg.to(at::kFloat).contiguous().data_ptr(),  // Convert BF16 -> F32
            x_f32.data_ptr(),
            H_pre_activated.data_ptr(),
            grad_x_from_agg.data_ptr(),
            grad_H_pre_activated.data_ptr());

        // Explicit sync for SetValue writes
        c10_npu::getCurrentNPUStream().synchronize();
    }

    // Step 3b: H_pre sigmoid derivative
    // d_tilde_pre = d_H_pre_activated * H_pre * (1 - H_pre)
    auto grad_H_pre = grad_H_pre_activated * H_pre_activated * (1.0f - H_pre_activated);

    // =========================================================================
    // Step 4: Combine x gradients from both paths
    // =========================================================================
    auto grad_x = grad_x_mix + grad_x_from_agg;

    // =========================================================================
    // Step 5: Backward through Sinkhorn-Knopp
    // =========================================================================
    auto x_flat = x_flat_bf16.to(at::kFloat).contiguous();
    auto rms_h_f32 = rms_h.to(at::kFloat).contiguous();
    auto rms_inv = 1.0f / rms_h_f32;
    auto rms_inv2 = rms_inv * rms_inv;

    auto phi_pre_f32 = phi_pre.to(at::kFloat).contiguous();
    auto phi_post_f32 = phi_post.to(at::kFloat).contiguous();
    auto phi_res_f32 = phi_res.to(at::kFloat).view({n * n, nC}).contiguous();

    // Recompute projections (parameterization B: p = x_flat @ phi)
    auto p_pre = at::matmul(x_flat, phi_pre_f32.t());
    auto p_post = at::matmul(x_flat, phi_post_f32.t());
    auto p_res_flat = at::matmul(x_flat, phi_res_f32.t());
    auto p_res = p_res_flat.view({B, n, n});

    // Recompute tilde_res for Sinkhorn backward
    // tilde = alpha * p / rms + b
    auto b_res_f32 = b_res.to(at::kFloat).contiguous();
    auto tilde_res = alpha_res * p_res * rms_inv.view({B, 1, 1}) + b_res_f32;
    auto H_res_exp = at::exp(tilde_res);

    // Sinkhorn backward
    at::Tensor d_H_res_exp;
    {
        SinkhornTiling tiling;
        tiling.batch_size = B;
        tiling.M = n;
        tiling.N = n;
        tiling.num_iters = sinkhorn_iters;
        tiling.eps = sinkhorn_eps;
        int num_cores = get_multicore_setting();
        tiling.used_core_num = std::min(num_cores, static_cast<int>(B));

        d_H_res_exp = at::empty_like(H_res_exp);

        launch_kernel(
            aclrtlaunch_sinkhorn_knopp_backward,
            tiling,
            grad_M.data_ptr(),
            H_res_exp.data_ptr(),
            M.data_ptr(),
            d_H_res_exp.data_ptr());
    }

    auto d_tilde_res = d_H_res_exp * H_res_exp;

    // =========================================================================
    // Step 6: Compute parameter gradients (dynamic-H)
    // =========================================================================
    // grad_H_pre and grad_H_post are already d_tilde_pre and d_tilde_post
    auto d_tilde_pre = grad_H_pre;
    auto d_tilde_post = grad_H_post;

    // Bias gradients: d_b = sum(d_tilde, dim=0)
    auto d_b_pre = d_tilde_pre.sum(0);
    auto d_b_post = d_tilde_post.sum(0);
    auto d_b_res = d_tilde_res.sum(0);

    // Alpha gradients (parameterization B: tilde = alpha * p / rms + b)
    // d_tilde/d_alpha = p / rms
    auto rms_inv_pre = rms_inv.view({B, 1});
    auto rms_inv_res = rms_inv.view({B, 1, 1});
    auto d_alpha_pre = (d_tilde_pre * (p_pre * rms_inv_pre)).sum().view({1});
    auto d_alpha_post = (d_tilde_post * (p_post * rms_inv_pre)).sum().view({1});
    auto d_alpha_res = (d_tilde_res * (p_res * rms_inv_res)).sum().view({1});

    // Projection gradients (parameterization B)
    // d_tilde/d_p = alpha / rms
    auto d_p_pre = d_tilde_pre * (alpha_pre * rms_inv_pre);
    auto d_p_post = d_tilde_post * (alpha_post * rms_inv_pre);
    auto d_p_res = d_tilde_res * (alpha_res * rms_inv_res);
    auto d_p_res_flat = d_p_res.reshape({B, n * n});

    // Phi gradients (parameterization B: p = x_flat @ phi)
    // d_phi = d_p.T @ x_flat
    auto d_phi_pre = at::matmul(d_p_pre.t(), x_flat);
    auto d_phi_post = at::matmul(d_p_post.t(), x_flat);
    auto d_phi_res = at::matmul(d_p_res_flat.t(), x_flat);

    // x_flat gradients from projections
    auto d_x_flat = at::matmul(d_p_pre, phi_pre_f32);
    d_x_flat = d_x_flat + at::matmul(d_p_post, phi_post_f32);
    d_x_flat = d_x_flat + at::matmul(d_p_res_flat, phi_res_f32);

    // =========================================================================
    // Step 7: Backward through RMS computation in tilde
    // rms = sqrt(mean(x_flat^2) + eps), d_rms/d_x = x / (nC * rms)
    // d_tilde/d_rms = -alpha * p / rms^2
    // =========================================================================
    auto d_r = -(d_tilde_pre * (alpha_pre * p_pre) * rms_inv2.view({B, 1})).sum(1);
    d_r = d_r - (d_tilde_post * (alpha_post * p_post) * rms_inv2.view({B, 1})).sum(1);
    d_r = d_r - (d_tilde_res * (alpha_res * p_res) * rms_inv2.view({B, 1, 1})).sum({1, 2});

    // Add RMS gradient contribution to x_flat
    d_x_flat = d_x_flat + d_r.view({B, 1}) * x_flat * (rms_inv.view({B, 1}) / static_cast<float>(nC));

    // Add x_flat gradient to total x gradient
    grad_x = grad_x + d_x_flat.view({B, n, C});

    // Convert gradients to appropriate dtypes
    d_b_pre = d_b_pre.to(b_pre.dtype());
    d_b_post = d_b_post.to(b_post.dtype());
    d_b_res = d_b_res.to(b_res.dtype());
    d_phi_pre = d_phi_pre.to(phi_pre.dtype());
    d_phi_post = d_phi_post.to(phi_post.dtype());
    d_phi_res = d_phi_res.view_as(phi_res).to(phi_res.dtype());

    return std::make_tuple(
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
        d_b_res
    );
}

// =============================================================================
// MHC Layer (Dynamic-H) Fused Forward Implementation
// =============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mhc_layer_fwd_dynamic(
    const at::Tensor& x_expanded,
    const at::Tensor& rmsnorm_weight,
    const at::Tensor& phi_concat,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    const at::Tensor& b_pre,
    const at::Tensor& b_post,
    const at::Tensor& b_res,
    int sinkhorn_iters,
    float sinkhorn_eps,
    float rmsnorm_eps)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x_expanded), "Input must be on NPU device");
    TORCH_CHECK(x_expanded.dim() == 3, "x_expanded must be [B, n, C]");
    TORCH_CHECK(phi_concat.dim() == 2, "phi_concat must be [out_dim, n*C]");

    int64_t B = x_expanded.size(0);
    int64_t n = x_expanded.size(1);
    int64_t C = x_expanded.size(2);
    int64_t nC = n * C;

    auto x_f32 = x_expanded.to(at::kFloat).contiguous();
    auto x_flat = x_f32.view({B, nC});
    auto x_flat_bf16 = x_flat.to(at::kBFloat16).contiguous();

    // Compute RMS for x_flat (used in activation calculation)
    auto rms_h = compute_rms_fwd(x_flat_bf16, rmsnorm_eps);

    // Parameterization B (matching CUDA):
    // p = x_flat @ phi (NOT normalized), then tilde = alpha * p / rms + b
    auto phi_concat_bf16 = phi_concat.to(at::kBFloat16).contiguous();
    auto H_proj_concat = fused_rmsnorm_matmul_fwd(x_flat_bf16, phi_concat_bf16);

    auto p_pre = H_proj_concat.slice(1, 0, n);
    auto p_post = H_proj_concat.slice(1, n, 2 * n);
    auto p_res = H_proj_concat.slice(1, 2 * n, 2 * n + n * n).view({B, n, n});

    // Parameterization B: tilde = alpha * p * (1/rms) + b (matching CUDA)
    auto rms_h_inv = 1.0f / rms_h;
    auto rms_h_inv_pre = rms_h_inv.view({B, 1});
    auto rms_h_inv_res = rms_h_inv.view({B, 1, 1});
    auto H_pre_raw = (p_pre * alpha_pre * rms_h_inv_pre).add(b_pre);
    auto H_post_raw = (p_post * alpha_post * rms_h_inv_pre).add(b_post);
    auto H_res_raw = (p_res * alpha_res * rms_h_inv_res).add(b_res);

    auto H_res_exp = H_res_raw.exp();
    auto M = sinkhorn_knopp_fwd(H_res_exp, sinkhorn_iters, sinkhorn_eps);

    at::Tensor x_agg_bf16, H_pre_activated;
    std::tie(x_agg_bf16, H_pre_activated) = stream_aggregate_fwd(x_f32, H_pre_raw);

    at::Tensor y_norm_bf16, rms;
    std::tie(y_norm_bf16, rms) =
        rmsnorm_fwd(x_agg_bf16, rmsnorm_weight.to(at::kBFloat16), rmsnorm_eps);

    at::Tensor output, H_post_activated;
    std::tie(output, H_post_activated) =
        stream_distribute_mix_add_fwd(y_norm_bf16, H_post_raw, M, x_f32);

    return std::make_tuple(
        output,
        rms,
        x_agg_bf16,
        H_pre_activated,
        H_post_activated,
        M,
        y_norm_bf16,
        x_flat_bf16,
        rms_h);
}

// =============================================================================
// Vectorization Test Functions
// =============================================================================

// Tiling structure for test kernel (must match test_vectorize.cpp)
struct VectorizeTestTiling {
    int32_t batch_size;
    int32_t N;           // Vector length (e.g., 4, 8, 16)
    int32_t test_mode;   // Which test method to use (0=scalar, 1=mask, 2=pad, etc.)
    int32_t used_core_num;
};

/**
 * @brief Test vectorization with different modes
 * @param x Input tensor 1 [batch, N]
 * @param y Input tensor 2 [batch, N]
 * @param test_mode 0=scalar_add, 1=masked_add, 2=padded_add, 3=scalar_reduce,
 *                  4=masked_reduce, 5=scalar_norm, 6=masked_norm
 * @return Output tensor [batch, N]
 *
 * Test modes:
 * 0: Scalar vector add (baseline)
 * 1: Masked vector add (SetMaskCount + SetVectorMask)
 * 2: Padded vector add (DataCopyPad + aligned compute)
 * 3: Scalar reduce sum (baseline)
 * 4: Masked reduce sum (WholeReduceSum with mask)
 * 5: Scalar row normalization (sum + divide)
 * 6: Masked row normalization (masked reduce + masked multiply)
 */
at::Tensor test_vectorize_op(
    const at::Tensor& x,
    const at::Tensor& y,
    int test_mode)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "Input x must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(y), "Input y must be on NPU device");
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D tensor [batch, N]");
    TORCH_CHECK(y.dim() == 2, "Input y must be 2D tensor [batch, N]");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have same shape");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be float32");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    auto out = at::empty_like(x_contig);

    VectorizeTestTiling tiling;
    tiling.batch_size = x.size(0);
    tiling.N = x.size(1);
    tiling.test_mode = test_mode;
    tiling.used_core_num = std::min(get_multicore_setting(), static_cast<int>(x.size(0)));

    launch_kernel(
        aclrtlaunch_test_vectorize,
        tiling,
        x_contig.data_ptr(),
        y_contig.data_ptr(),
        out.data_ptr());

    return out;
}

// =============================================================================
// Pybind11 Module Definition
// =============================================================================

PYBIND11_MODULE(mhc_ascend, m) {
    m.doc() = "mHC (Manifold-Constrained Hyper-Connections) for Huawei Ascend NPUs";

    // Sinkhorn-Knopp operations
    m.def("sinkhorn_knopp_fwd", &sinkhorn_knopp_fwd,
          "Sinkhorn-Knopp forward pass",
          py::arg("inp"), py::arg("num_iters") = 20, py::arg("eps") = 1e-8f);
    m.def("sinkhorn_knopp_bwd", &sinkhorn_knopp_bwd,
          "Sinkhorn-Knopp backward pass",
          py::arg("grad_out"), py::arg("inp"), py::arg("out"),
          py::arg("num_iters") = 20, py::arg("eps") = 1e-8f);

    // RMSNorm operations
    m.def("rmsnorm_fwd", &rmsnorm_fwd,
          "RMSNorm forward pass",
          py::arg("inp"), py::arg("weight"), py::arg("eps") = 1e-5f);
    m.def("rmsnorm_bwd", &rmsnorm_bwd,
          "RMSNorm backward pass",
          py::arg("grad_out"), py::arg("inp"), py::arg("weight"), py::arg("rms"));

    // Stream aggregate operations
    m.def("stream_aggregate_fwd", &stream_aggregate_fwd,
          "Stream aggregate forward pass",
          py::arg("inp"), py::arg("H_pre_raw"));
    m.def("stream_aggregate_bwd", &stream_aggregate_bwd,
          "Stream aggregate backward pass",
          py::arg("grad_out"), py::arg("inp"), py::arg("H_pre_activated"));

    // Stream distribute mix add operations
    m.def("stream_distribute_mix_add_fwd", &stream_distribute_mix_add_fwd,
          "Stream distribute-mix-add forward pass",
          py::arg("y_norm"), py::arg("H_post_raw"), py::arg("M"), py::arg("x_inp"));
    m.def("stream_distribute_mix_add_bwd", &stream_distribute_mix_add_bwd,
          "Stream distribute-mix-add backward pass",
          py::arg("grad_out"), py::arg("x_inp"), py::arg("y_norm"),
          py::arg("M"), py::arg("H_post_activated"));

    // Fused projection (bf16 matmul -> f32)
    m.def("fused_rmsnorm_matmul_fwd", &fused_rmsnorm_matmul_fwd,
          "Fused bf16 matmul for dynamic-H projection",
          py::arg("inp"), py::arg("weight"));
    m.def("fused_rmsnorm_matmul_bwd", &fused_rmsnorm_matmul_bwd,
          "Fused bf16 matmul backward (NPU matmul)",
          py::arg("grad_out"), py::arg("x_flat"), py::arg("phi_concat"));
    m.def("compute_rms_fwd", &compute_rms_fwd,
          "Compute RMS per row (bf16 input)",
          py::arg("inp"), py::arg("eps") = 1e-8f);

    // MHC Layer (dynamic-H) fused forward
    m.def("mhc_layer_fwd_dynamic", &mhc_layer_fwd_dynamic,
          "MHC Layer forward (dynamic-H, fused host orchestration)",
          py::arg("x_expanded"),
          py::arg("rmsnorm_weight"),
          py::arg("phi_concat"),
          py::arg("alpha_pre"),
          py::arg("alpha_post"),
          py::arg("alpha_res"),
          py::arg("b_pre"),
          py::arg("b_post"),
          py::arg("b_res"),
          py::arg("sinkhorn_iters"),
          py::arg("sinkhorn_eps"),
          py::arg("rmsnorm_eps"));

    // MHC Layer (dynamic-H) unified backward
    m.def("mhc_layer_bwd_dynamic", &mhc_layer_bwd_dynamic,
          "MHC Layer backward (dynamic-H, unified C++ orchestration)",
          py::arg("grad_output"),
          py::arg("x_expanded"),
          py::arg("rmsnorm_weight"),
          py::arg("rms"),
          py::arg("x_agg_bf16"),
          py::arg("H_pre_activated"),
          py::arg("H_post_activated"),
          py::arg("M"),
          py::arg("y_norm_bf16"),
          py::arg("x_flat_bf16"),
          py::arg("rms_h"),
          py::arg("phi_pre"),
          py::arg("phi_post"),
          py::arg("phi_res"),
          py::arg("alpha_pre"),
          py::arg("alpha_post"),
          py::arg("alpha_res"),
          py::arg("b_pre"),
          py::arg("b_post"),
          py::arg("b_res"),
          py::arg("sinkhorn_iters"),
          py::arg("sinkhorn_eps"),
          py::arg("rmsnorm_eps"));

    // Vectorization test function
    m.def("test_vectorize", &test_vectorize_op,
          "Test vectorization with different modes.\n"
          "Modes: 0=scalar_add, 1=masked_add, 2=padded_add, 3=scalar_reduce,\n"
          "       4=masked_reduce, 5=scalar_norm, 6=masked_norm",
          py::arg("x"), py::arg("y"), py::arg("test_mode") = 0);
}
