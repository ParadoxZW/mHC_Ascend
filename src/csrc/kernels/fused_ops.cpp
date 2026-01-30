/**
 * @file fused_ops.cpp
 * @brief Fused operations (BF16 MatMul) for mHC on Ascend
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * Implements fused operations for efficiency:
 * - fused_rmsnorm_matmul: BF16 MatMul for dynamic-H projection (RMS handled separately)
 *
 * CUDA Reference: src/csrc/kernels/fused_rmsnorm_matmul.cuh
 *
 * Note: This is a simplified version. For production use with Ascend Matmul API,
 * refer to Ascend C Matmul documentation and samples.
 *
 * IMPORTANT: DataCopy requires 32-byte alignment. For dimensions that may not satisfy
 * this (like out_dim=28 for n=4), we use GetValue/SetValue or DataCopyPad.
 * See debug_notes/stream_aggregate_backward_alignment_fix.md for reference.
 */

#include "kernel_operator.h"
#include "../include/mhc_types.h"
#include "../include/utils.h"

using namespace mhc_ascend;
using namespace AscendC;
using FusedRMSNormMatMulTilingData = FusedRMSNormMatMulTiling;

// Helper: Check if size satisfies 32-byte alignment for DataCopy
template<typename T>
__aicore__ inline bool is_32byte_aligned(int32_t count) {
    return (count * static_cast<int32_t>(sizeof(T))) % 32 == 0;
}

// =============================================================================
// Fused MatMul Forward Kernel
// =============================================================================

/**
 * @brief Simplified fused matmul kernel
 *
 * This kernel performs:
 * 1. BF16 -> F32 cast for input/weight
 * 2. Matrix multiplication: out = x @ weight^T (F32 accumulation)
 *
 * Note: RMS is computed outside this kernel to preserve autograd behavior.
 * For full performance, the Ascend Matmul API should be used.
 */
class FusedRMSNormMatMulKernel {
public:
    __aicore__ inline FusedRMSNormMatMulKernel() {}

    /**
     * @brief Initialize kernel
     * @param inp_gm Input tensor [batch_size, hidden_dim]
     * @param weight_gm Weight matrix [out_dim, hidden_dim]
     * @param out_gm Output tensor [batch_size, out_dim]
     * @param rms_gm RMS values [batch_size] (optional output)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR inp_gm,
        GM_ADDR weight_gm,
        GM_ADDR out_gm,
        GM_ADDR rms_gm,
        const FusedRMSNormMatMulTiling& tiling)
    {
        using DataT = floatX;
        using AccT = floatN;

        this->tiling = tiling;
        (void)rms_gm;

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        // Distribute batch across cores
        this->batch_per_core = CeilingDiv(tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;
        }

        this->hidden_dim = tiling.hidden_dim;
        this->out_dim = tiling.out_dim;

        // Set global buffers
        int32_t inp_offset = this->batch_start * this->hidden_dim;
        int32_t out_offset = this->batch_start * this->out_dim;

        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(inp_gm) + inp_offset,
                              this->batch_count * this->hidden_dim);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(weight_gm),
                                 this->out_dim * this->hidden_dim);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(out_gm) + out_offset,
                              this->batch_count * this->out_dim);

        // Initialize buffers
        pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->hidden_dim * sizeof(DataT));
        pipe.InitBuffer(outBuf, this->out_dim * sizeof(AccT));
        pipe.InitBuffer(weightRowBuf, this->hidden_dim * sizeof(DataT));
        pipe.InitBuffer(inpF32Buf, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(weightF32Buf, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(tmpBuf1, this->hidden_dim * sizeof(AccT));
    }

    /**
     * @brief Main processing function
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleRow(b);
        }
    }

private:
    /**
     * @brief Process a single batch element
     *
     * Steps:
     * 1. Load input row (bf16) and cast to float
     * 2. For each output dimension:
     *    - Load weight row (bf16) and cast to float
     *    - Compute dot product with input
     * 3. Write all results to GM at once
     *
     * Note: Uses alignment-safe loading for BF16 data. DataCopy requires 32-byte
     * alignment which may not be satisfied for arbitrary hidden_dim.
     * Output is written using DataCopyPad to handle arbitrary out_dim.
     */
    __aicore__ inline void ProcessSingleRow(int32_t batch_idx) {
        using DataT = floatX;
        using AccT = floatN;

        LocalTensor<DataT> inp = inpQueue.AllocTensor<DataT>();
        LocalTensor<AccT> inpF32 = inpF32Buf.Get<AccT>();
        LocalTensor<DataT> weight_row = weightRowBuf.Get<DataT>();
        LocalTensor<AccT> weight_row_f32 = weightF32Buf.Get<AccT>();
        LocalTensor<AccT> tmp = tmpBuf1.Get<AccT>();
        LocalTensor<AccT> result_buf = outBuf.Get<AccT>();

        // Check if hidden_dim is aligned for BF16 DataCopy (32 bytes = 16 BF16 elements)
        bool hidden_aligned = is_32byte_aligned<DataT>(this->hidden_dim);

        // Load input row (bf16)
        if (hidden_aligned) {
            DataCopy(inp, inpGm[batch_idx * this->hidden_dim], this->hidden_dim);
        } else {
            // Use DataCopyPad for unaligned access
            AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(this->hidden_dim * sizeof(DataT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DataT> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad<DataT>(inp, inpGm[batch_idx * this->hidden_dim], copyParams, padParams);
        }
        inpQueue.EnQue(inp);
        inp = inpQueue.DeQue<DataT>();

        // Convert input to F32
        ConvertBF16ToF32(inpF32, inp, this->hidden_dim);

        // Compute matrix multiplication using fully scalar operations
        // This avoids all pipeline synchronization issues
        int32_t out_offset = batch_idx * this->out_dim;

        for (int32_t i = 0; i < this->out_dim; ++i) {
            // Compute dot product: out[i] = sum_j(input[j] * weight[i, j])
            // Use scalar reads to avoid DataCopy alignment issues
            AccT dot_product = static_cast<AccT>(0.0);
            for (int32_t j = 0; j < this->hidden_dim; ++j) {
                // Read input (already in F32 from inpF32)
                AccT inp_val = inpF32.GetValue(j);
                // Read weight directly from GM (BF16 -> F32 conversion via cast)
                DataT w_bf16 = weightGm.GetValue(i * this->hidden_dim + j);
                // Convert BF16 to F32 by bit manipulation
                // BF16 is stored as uint16, shift left 16 bits to get F32 representation
                uint16_t w_bits = *reinterpret_cast<uint16_t*>(&w_bf16);
                uint32_t f32_bits = static_cast<uint32_t>(w_bits) << 16;
                AccT w_f32 = *reinterpret_cast<AccT*>(&f32_bits);
                dot_product += inp_val * w_f32;
            }

            // Write result directly to GM
            result_buf.SetValue(0, dot_product);
            PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams outCopyParams = {
                1, static_cast<uint32_t>(sizeof(AccT)), 0, 0, 0};
            AscendC::DataCopyPad<AccT>(
                outGm[out_offset + i],
                result_buf,
                outCopyParams);
            PipeBarrier<PIPE_MTE3>();
        }

        // Free tensors
        inpQueue.FreeTensor(inp);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> inpQueue;
    TBuf<QuePosition::VECCALC> outBuf, weightRowBuf, inpF32Buf, weightF32Buf, tmpBuf1;

    GlobalTensor<floatX> inpGm, weightGm;
    GlobalTensor<floatN> outGm;

    FusedRMSNormMatMulTiling tiling;

    int32_t hidden_dim, out_dim;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
};

// =============================================================================
// Fused RMSNorm + MatMul Forward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void fused_rmsnorm_matmul_forward(
    GM_ADDR inp,
    GM_ADDR weight,
    GM_ADDR out,
    GM_ADDR rms,
    GM_ADDR tiling)
{
    FusedRMSNormMatMulTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    FusedRMSNormMatMulKernel kernel;
    kernel.Init(inp, weight, out, rms, tiling_data);
    kernel.Process();
}

// =============================================================================
// Fused RMSNorm + MatMul Backward Kernel
// =============================================================================

/**
 * @brief Backward pass for fused RMSNorm + MatMul
 *
 * Computes:
 * - grad_weight: gradient w.r.t. weight matrix
 * - grad_inp: gradient w.r.t. input
 */
template<typename T = float>
class FusedRMSNormMatMulBackwardKernel {
public:
    __aicore__ inline FusedRMSNormMatMulBackwardKernel() {}

    /**
     * @brief Initialize backward kernel
     */
    __aicore__ inline void Init(
        GM_ADDR grad_out_gm,
        GM_ADDR inp_gm,
        GM_ADDR weight_gm,
        GM_ADDR rms_gm,
        GM_ADDR grad_inp_gm,
        GM_ADDR grad_weight_gm,
        const FusedRMSNormMatMulTiling& tiling)
    {
        this->tiling = tiling;

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        this->batch_per_core = CeilingDiv(tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;
        }

        this->hidden_dim = tiling.hidden_dim;
        this->out_dim = tiling.out_dim;

        int32_t inp_offset = this->batch_start * this->hidden_dim;
        int32_t out_offset = this->batch_start * this->out_dim;

        gradOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_out_gm) + out_offset,
                                   this->batch_count * this->out_dim);
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inp_gm) + inp_offset,
                               this->batch_count * this->hidden_dim);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight_gm),
                                  this->out_dim * this->hidden_dim);
        rmsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rms_gm) + this->batch_start,
                               this->batch_count);
        gradInpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_inp_gm) + inp_offset,
                                   this->batch_count * this->hidden_dim);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_weight_gm),
                                       this->out_dim * this->hidden_dim);

        // Initialize buffers
        pipe.InitBuffer(gradOutQueue, NUM_BUFFERS, this->out_dim * sizeof(T));
        pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(weightQueue, NUM_BUFFERS, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(rmsQueue, NUM_BUFFERS, ALIGN_UP(sizeof(T), BLK_LEN));
        pipe.InitBuffer(gradInpQueue, NUM_BUFFERS, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(tmpBuf1, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(tmpBuf2, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(normalizedBuf, this->hidden_dim * sizeof(T));
        pipe.InitBuffer(gradWeightAccum, this->out_dim * this->hidden_dim * sizeof(T));
    }

    /**
     * @brief Main backward processing
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        // Initialize grad_weight accumulator to zero
        LocalTensor<T> grad_weight_acc = gradWeightAccum.Get<T>();
        ZeroBuffer(grad_weight_acc, this->out_dim * this->hidden_dim);

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBackward(b, grad_weight_acc);
        }

        // Write accumulated grad_weight to GM (atomic add for multi-core accumulation)
        AscendC::SetAtomicAdd<T>();
        DataCopy(gradWeightGm[0], grad_weight_acc, this->out_dim * this->hidden_dim);
        AscendC::SetAtomicNone();
    }

private:
    /**
     * @brief Process backward for a single batch element
     *
     * Gradients:
     * - grad_weight[i, j] += grad_out[i] * normalized[j]
     * - grad_normalized[j] = sum_i(grad_out[i] * weight[i, j])
     * - grad_inp = grad_normalized / RMS - correction_term
     */
    __aicore__ inline void ProcessSingleBackward(int32_t batch_idx, LocalTensor<T>& grad_weight_acc) {
        // Load inputs
        LocalTensor<T> grad_out = gradOutQueue.AllocTensor<T>();
        LocalTensor<T> inp = inpQueue.AllocTensor<T>();
        LocalTensor<T> rms_val = rmsQueue.AllocTensor<T>();

        DataCopy(grad_out, gradOutGm[batch_idx * this->out_dim], this->out_dim);
        DataCopy(inp, inpGm[batch_idx * this->hidden_dim], this->hidden_dim);
        ReadScalarFromGM(rms_val, rmsGm, batch_idx);

        gradOutQueue.EnQue(grad_out);
        inpQueue.EnQue(inp);
        rmsQueue.EnQue(rms_val);

        grad_out = gradOutQueue.DeQue<T>();
        inp = inpQueue.DeQue<T>();
        rms_val = rmsQueue.DeQue<T>();

        // Compute normalized input
        LocalTensor<T> normalized = normalizedBuf.Get<T>();
        T rms = rms_val.GetValue(0);
        T rms_inv = static_cast<T>(1.0) / rms;
        Muls(normalized, inp, rms_inv, this->hidden_dim);

        // Compute grad_weight and grad_normalized
        LocalTensor<T> grad_normalized = tmpBuf1.Get<T>();
        LocalTensor<T> weight_row = weightQueue.AllocTensor<T>();
        LocalTensor<T> tmp = tmpBuf2.Get<T>();

        ZeroBuffer(grad_normalized, this->hidden_dim);

        for (int32_t i = 0; i < this->out_dim; ++i) {
            T grad_out_i = grad_out.GetValue(i);

            // Load weight row
            DataCopy(weight_row, weightGm[i * this->hidden_dim], this->hidden_dim);

            // grad_weight[i, :] += grad_out[i] * normalized
            LocalTensor<T> grad_weight_row = grad_weight_acc[i * this->hidden_dim];
            Muls(tmp, normalized, grad_out_i, this->hidden_dim);
            Add(grad_weight_row, grad_weight_row, tmp, this->hidden_dim);

            // grad_normalized += grad_out[i] * weight[i, :]
            Muls(tmp, weight_row, grad_out_i, this->hidden_dim);
            Add(grad_normalized, grad_normalized, tmp, this->hidden_dim);
        }

        // Apply RMSNorm backward
        LocalTensor<T> grad_inp = gradInpQueue.AllocTensor<T>();
        ApplyRMSNormBackward(grad_inp, grad_normalized, inp, rms_val, this->hidden_dim, tmp);

        // Store grad_inp
        gradInpQueue.EnQue(grad_inp);
        grad_inp = gradInpQueue.DeQue<T>();
        DataCopy(gradInpGm[batch_idx * this->hidden_dim], grad_inp, this->hidden_dim);

        // Free tensors
        gradOutQueue.FreeTensor(grad_out);
        inpQueue.FreeTensor(inp);
        rmsQueue.FreeTensor(rms_val);
        weightQueue.FreeTensor(weight_row);
        gradInpQueue.FreeTensor(grad_inp);
    }

    /**
     * @brief Apply RMSNorm backward
     *
     * grad_inp = grad_normalized / RMS - (inp * dot(grad_normalized, inp)) / (RMS^3 * hidden_dim)
     */
    __aicore__ inline void ApplyRMSNormBackward(
        LocalTensor<T>& grad_inp,
        const LocalTensor<T>& grad_normalized,
        const LocalTensor<T>& inp,
        const LocalTensor<T>& rms_val,
        int32_t length,
        LocalTensor<T>& tmp)
    {
        T rms = rms_val.GetValue(0);
        T rms_inv = static_cast<T>(1.0) / rms;

        // grad_inp = grad_normalized / RMS
        Muls(grad_inp, grad_normalized, rms_inv, length);

        // Compute dot product
        Mul(tmp, grad_normalized, inp, length);
        LocalTensor<T> dot_val = tmp[0];
        ReduceSumOptimized(dot_val, tmp, tmp, length);
        T dot = dot_val.GetValue(0);

        // Correction term
        T correction_scale = -dot / (rms * rms * rms * static_cast<T>(length));
        Muls(tmp, inp, correction_scale, length);

        // grad_inp += tmp
        Add(grad_inp, grad_inp, tmp, length);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> gradOutQueue, inpQueue, weightQueue, rmsQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> gradInpQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2, normalizedBuf, gradWeightAccum;

    GlobalTensor<T> gradOutGm, inpGm, weightGm, rmsGm, gradInpGm, gradWeightGm;

    FusedRMSNormMatMulTiling tiling;

    int32_t hidden_dim, out_dim;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
};

// =============================================================================
// Fused RMSNorm + MatMul Backward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void fused_rmsnorm_matmul_backward(
    GM_ADDR grad_out,
    GM_ADDR inp,
    GM_ADDR weight,
    GM_ADDR rms,
    GM_ADDR grad_inp,
    GM_ADDR grad_weight,
    GM_ADDR tiling)
{
    FusedRMSNormMatMulTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    FusedRMSNormMatMulBackwardKernel<float> kernel;
    kernel.Init(grad_out, inp, weight, rms, grad_inp, grad_weight, tiling_data);
    kernel.Process();
}
