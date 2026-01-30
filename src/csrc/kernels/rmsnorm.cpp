/**
 * @file rmsnorm.cpp
 * @brief RMS Normalization kernel for Ascend
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * Implements Root Mean Square (RMS) normalization:
 *   out[i] = weight[i] * (inp[i] / RMS)
 *   where RMS = sqrt(mean(inp²) + eps)
 *
 * This is a key component in the mHC layer for normalizing aggregated features.
 *
 * CUDA Reference: src/csrc/kernels/rmsnorm.cuh
 */

#include "kernel_operator.h"
#include "../include/mhc_types.h"
#include "../include/utils.h"

using namespace mhc_ascend;
using namespace AscendC;
using RMSNormTilingData = RMSNormTiling;

// =============================================================================
// RMSNorm Forward Kernel Class
// =============================================================================

class RMSNormKernel {
public:
    __aicore__ inline RMSNormKernel() {}

    /**
     * @brief Initialize kernel
     * @param inp_gm Input tensor [batch_size, hidden_dim]
     * @param weight_gm Weight tensor [hidden_dim]
     * @param out_gm Output tensor [batch_size, hidden_dim]
     * @param rms_gm RMS values [batch_size] (optional, can be nullptr)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR inp_gm,
        GM_ADDR weight_gm,
        GM_ADDR out_gm,
        GM_ADDR rms_gm,
        const RMSNormTiling& tiling)
    {
        using DataT = floatX;
        using AccT = floatN;

        this->tiling = tiling;
        this->output_rms = (rms_gm != nullptr);

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
        this->hidden_dim_aligned = ALIGN_UP(this->hidden_dim, BLK_LEN / static_cast<int32_t>(sizeof(DataT)));
        this->total_size = this->batch_count * this->hidden_dim;

        // Set global buffers
        int32_t offset = this->batch_start * this->hidden_dim;
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(inp_gm) + offset, this->total_size);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(out_gm) + offset, this->total_size);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(weight_gm), this->hidden_dim);

        if (this->output_rms) {
            rmsGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(rms_gm) + this->batch_start,
                                  this->batch_count);
        }

        // Initialize pipe and queues
        pipe.InitBuffer(inQueue, NUM_BUFFERS, this->hidden_dim_aligned * sizeof(DataT));
        pipe.InitBuffer(weightQueue, 1, this->hidden_dim_aligned * sizeof(DataT));
        pipe.InitBuffer(outQueue, NUM_BUFFERS, this->hidden_dim * sizeof(DataT));

        // Temporary buffers for computation (float32)
        pipe.InitBuffer(inpF32Buf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(weightF32Buf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(squaredBuf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(tmpCalcBuf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(outF32Buf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(rmsBuf, ALIGN_UP(sizeof(AccT), BLK_LEN));
    }

    /**
     * @brief Main processing function
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        // Load weight once (shared across all batch elements)
        using DataT = floatX;
        using AccT = floatN;

        LocalTensor<DataT> weight = weightQueue.AllocTensor<DataT>();
        AscendC::DataCopyExtParams wCopyParams = {
            1, static_cast<uint32_t>(this->hidden_dim * sizeof(DataT)), 0, 0, 0};
        int32_t wpad = this->hidden_dim_aligned - this->hidden_dim;
        AscendC::DataCopyPadExtParams<DataT> wPadParams = {
            false, 0, static_cast<uint8_t>(wpad), 0};
        AscendC::DataCopyPad<DataT>(weight, weightGm, wCopyParams, wPadParams);
        weightQueue.EnQue(weight);
        weight = weightQueue.DeQue<DataT>();

        // Cast weight to float once per core
        LocalTensor<AccT> weightF32 = weightF32Buf.Get<AccT>();
        ConvertBF16ToF32(weightF32, weight, this->hidden_dim_aligned);

        // Process each batch element
        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleRow(b, weightF32);
        }

        // Free weight tensor
        weightQueue.FreeTensor(weight);
    }

private:
    /**
     * @brief Process a single row (batch element)
     * @param batch_idx Index within this core's batch
     * @param weight Weight tensor
     */
    __aicore__ inline void ProcessSingleRow(int32_t batch_idx, LocalTensor<floatN>& weightF32) {
        using DataT = floatX;
        using AccT = floatN;

        // Load input
        LocalTensor<DataT> inp = inQueue.AllocTensor<DataT>();
        AscendC::DataCopyExtParams xCopyParams = {
            1, static_cast<uint32_t>(this->hidden_dim * sizeof(DataT)), 0, 0, 0};
        int32_t xpad = this->hidden_dim_aligned - this->hidden_dim;
        AscendC::DataCopyPadExtParams<DataT> xPadParams = {
            false, 0, static_cast<uint8_t>(xpad), 0};
        AscendC::DataCopyPad<DataT>(inp, inpGm[batch_idx * this->hidden_dim], xCopyParams, xPadParams);
        inQueue.EnQue(inp);

        inp = inQueue.DeQue<DataT>();

        LocalTensor<AccT> inpF32 = inpF32Buf.Get<AccT>();
        LocalTensor<AccT> squared = squaredBuf.Get<AccT>();
        LocalTensor<AccT> rms_val = rmsBuf.Get<AccT>();
        LocalTensor<AccT> tmpCalc = tmpCalcBuf.Get<AccT>();
        LocalTensor<AccT> outF32 = outF32Buf.Get<AccT>();

        // Cast input to float
        ConvertBF16ToF32(inpF32, inp, this->hidden_dim_aligned);

        // Compute RMS: sqrt(mean(inp²) + eps) in float
        ComputeRMS(rms_val, inpF32, squared, tmpCalc, this->hidden_dim,
                   static_cast<AccT>(tiling.eps));

        // Store RMS if requested
        if (this->output_rms) {
            WriteScalarToGM(rmsGm, batch_idx, rms_val);
        }

        // Normalize: inp / RMS (combine reciprocal and multiplication)
        AccT rms_scalar = rms_val.GetValue(0);
        AccT rms_inv = static_cast<AccT>(1.0) / rms_scalar;  // Scalar reciprocal
        Muls(outF32, inpF32, rms_inv, this->hidden_dim);

        // Allocate output
        LocalTensor<DataT> out = outQueue.AllocTensor<DataT>();

        // Apply weight: out = (inp / RMS) * weight
        Mul(outF32, outF32, weightF32, this->hidden_dim);
        ConvertF32ToBF16(out, outF32, this->hidden_dim);

        // Enqueue and copy to GM
        outQueue.EnQue(out);
        out = outQueue.DeQue<DataT>();
        DataCopy(outGm[batch_idx * this->hidden_dim], out, this->hidden_dim);

        // Free tensors
        inQueue.FreeTensor(inp);
        outQueue.FreeTensor(out);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> inQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> outQueue;
    TBuf<QuePosition::VECCALC> inpF32Buf, weightF32Buf, squaredBuf, tmpCalcBuf, outF32Buf, rmsBuf;

    GlobalTensor<floatX> inpGm, weightGm, outGm;
    GlobalTensor<floatN> rmsGm;

    RMSNormTiling tiling;

    int32_t hidden_dim;
    int32_t hidden_dim_aligned;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t total_size;
    bool output_rms;
};

// =============================================================================
// RMSNorm Forward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void rmsnorm_forward(
    GM_ADDR inp,
    GM_ADDR weight,
    GM_ADDR out,
    GM_ADDR rms,
    GM_ADDR tiling)
{
    RMSNormTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    RMSNormKernel kernel;
    kernel.Init(inp, weight, out, rms, tiling_data);
    kernel.Process();
}

// =============================================================================
// RMSNorm Backward Kernel Class
// =============================================================================

class RMSNormBackwardKernel {
public:
    __aicore__ inline RMSNormBackwardKernel() {}

    /**
     * @brief Initialize backward kernel
     * @param grad_out_gm Gradient w.r.t. output [batch_size, hidden_dim]
     * @param inp_gm Original input [batch_size, hidden_dim]
     * @param weight_gm Weight [hidden_dim]
     * @param rms_gm RMS values from forward [batch_size]
     * @param grad_inp_gm Gradient w.r.t. input [batch_size, hidden_dim] (output)
     * @param grad_weight_gm Gradient w.r.t. weight [hidden_dim] (output)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR grad_out_gm,
        GM_ADDR inp_gm,
        GM_ADDR weight_gm,
        GM_ADDR rms_gm,
        GM_ADDR grad_inp_gm,
        GM_ADDR grad_weight_gm,
        const RMSNormTiling& tiling)
    {
        using DataT = floatX;
        using AccT = floatN;

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
        this->total_size = this->batch_count * this->hidden_dim;

        int32_t offset = this->batch_start * this->hidden_dim;
        gradOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(grad_out_gm) + offset,
                                  this->total_size);
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(inp_gm) + offset, this->total_size);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(weight_gm), this->hidden_dim);
        rmsGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(rms_gm) + this->batch_start,
                              this->batch_count);
        gradInpGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(grad_inp_gm) + offset,
                                  this->total_size);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(grad_weight_gm), this->hidden_dim);

        // Initialize buffers
        pipe.InitBuffer(gradOutQueue, NUM_BUFFERS, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->hidden_dim * sizeof(DataT));
        pipe.InitBuffer(weightQueue, 1, this->hidden_dim * sizeof(DataT));
        pipe.InitBuffer(rmsQueue, NUM_BUFFERS, ALIGN_UP(sizeof(AccT), BLK_LEN));
        pipe.InitBuffer(gradInpQueue, NUM_BUFFERS, this->hidden_dim * sizeof(DataT));

        // Temp buffers
        pipe.InitBuffer(inpF32Buf, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(weightF32Buf, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(tmpBuf1, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(tmpBuf2, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(gradInpF32Buf, this->hidden_dim * sizeof(AccT));
        pipe.InitBuffer(tmpScalar, sizeof(AccT));

        // Accumulator for grad_weight
        pipe.InitBuffer(gradWeightAccum, this->hidden_dim * sizeof(AccT));
    }

    /**
     * @brief Main backward processing
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        // Initialize grad_weight accumulator to zero
        using DataT = floatX;
        using AccT = floatN;

        LocalTensor<AccT> gradWeightAcc = gradWeightAccum.Get<AccT>();
        ZeroBuffer(gradWeightAcc, this->hidden_dim);

        // Load weight
        LocalTensor<DataT> weight = weightQueue.AllocTensor<DataT>();
        DataCopy(weight, weightGm, this->hidden_dim);
        weightQueue.EnQue(weight);
        weight = weightQueue.DeQue<DataT>();

        LocalTensor<AccT> weightF32 = weightF32Buf.Get<AccT>();
        ConvertBF16ToF32(weightF32, weight, this->hidden_dim);

        // Process each batch element
        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBackward(b, weightF32, gradWeightAcc);
        }

        // Write accumulated grad_weight to GM (atomic add for multi-core accumulation)
        AscendC::SetAtomicAdd<AccT>();
        DataCopy(gradWeightGm, gradWeightAcc, this->hidden_dim);
        AscendC::SetAtomicNone();

        weightQueue.FreeTensor(weight);
    }

private:
    /**
     * @brief Process backward for a single row
     *
     * Gradient computation:
     * grad_inp = grad_out * weight / RMS - (inp * weight * dot(grad_out * weight, inp)) / (RMS^3 * hidden_dim)
     * grad_weight += grad_out * (inp / RMS)
     */
    __aicore__ inline void ProcessSingleBackward(
        int32_t batch_idx,
        LocalTensor<floatN>& weightF32,
        LocalTensor<floatN>& gradWeightAcc)
    {
        using DataT = floatX;
        using AccT = floatN;

        // Load inputs
        LocalTensor<AccT> gradOut = gradOutQueue.AllocTensor<AccT>();
        LocalTensor<DataT> inp = inpQueue.AllocTensor<DataT>();
        LocalTensor<AccT> rms_val = rmsQueue.AllocTensor<AccT>();

        DataCopy(gradOut, gradOutGm[batch_idx * this->hidden_dim], this->hidden_dim);
        DataCopy(inp, inpGm[batch_idx * this->hidden_dim], this->hidden_dim);
        ReadScalarFromGM(rms_val, rmsGm, batch_idx);

        gradOutQueue.EnQue(gradOut);
        inpQueue.EnQue(inp);
        rmsQueue.EnQue(rms_val);

        gradOut = gradOutQueue.DeQue<AccT>();
        inp = inpQueue.DeQue<DataT>();
        rms_val = rmsQueue.DeQue<AccT>();

        // Get temp buffers
        LocalTensor<AccT> inpF32 = inpF32Buf.Get<AccT>();
        LocalTensor<AccT> tmp1 = tmpBuf1.Get<AccT>();
        LocalTensor<AccT> tmp2 = tmpBuf2.Get<AccT>();
        LocalTensor<AccT> scalar = tmpScalar.Get<AccT>();
        LocalTensor<AccT> gradInpF32 = gradInpF32Buf.Get<AccT>();

        ConvertBF16ToF32(inpF32, inp, this->hidden_dim);

        AccT rms = rms_val.GetValue(0);
        AccT rms_inv = static_cast<AccT>(1.0) / rms;

        // Compute grad_weight contribution: grad_out * (inp / RMS)
        Muls(tmp1, inpF32, rms_inv, this->hidden_dim);    // tmp1 = inp / RMS
        Mul(tmp1, tmp1, gradOut, this->hidden_dim);    // tmp1 = grad_out * (inp / RMS)
        Add(gradWeightAcc, gradWeightAcc, tmp1, this->hidden_dim);  // Accumulate

        // Compute grad_inp: grad_out * weight / RMS - inp * dot(grad_out * weight, inp) / (RMS^3 * hidden_dim)
        Mul(tmp1, gradOut, weightF32, this->hidden_dim);  // tmp1 = grad_out * weight

        // dot_product = sum((grad_out * weight) * inp)
        Mul(tmp2, tmp1, inpF32, this->hidden_dim);        // tmp2 = (grad_out * weight) * inp
        ReduceSumOptimized(scalar, tmp2, tmp1, this->hidden_dim);

        AccT dot_val = scalar.GetValue(0);
        AccT correction_scale =
            -dot_val / (rms * rms * rms * static_cast<AccT>(this->hidden_dim));

        // grad_inp = (grad_out * weight) / RMS + inp * correction_scale
        Mul(tmp1, gradOut, weightF32, this->hidden_dim);  // recompute tmp1 = grad_out * weight
        Muls(tmp1, tmp1, rms_inv, this->hidden_dim);   // tmp1 = (grad_out * weight) / RMS
        Muls(tmp2, inpF32, correction_scale, this->hidden_dim);

        // grad_inp = tmp1 + tmp2
        Add(gradInpF32, tmp1, tmp2, this->hidden_dim);
        LocalTensor<DataT> gradInp = gradInpQueue.AllocTensor<DataT>();
        ConvertF32ToBF16(gradInp, gradInpF32, this->hidden_dim);

        // Write to GM
        gradInpQueue.EnQue(gradInp);
        gradInp = gradInpQueue.DeQue<DataT>();
        DataCopy(gradInpGm[batch_idx * this->hidden_dim], gradInp, this->hidden_dim);

        // Free tensors
        gradOutQueue.FreeTensor(gradOut);
        inpQueue.FreeTensor(inp);
        rmsQueue.FreeTensor(rms_val);
        gradInpQueue.FreeTensor(gradInp);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> gradOutQueue, inpQueue, rmsQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> gradInpQueue;
    TBuf<QuePosition::VECCALC> inpF32Buf, weightF32Buf, tmpBuf1, tmpBuf2, gradInpF32Buf, tmpScalar,
        gradWeightAccum;

    GlobalTensor<floatN> gradOutGm, rmsGm, gradWeightGm;
    GlobalTensor<floatX> inpGm, weightGm, gradInpGm;

    RMSNormTiling tiling;

    int32_t hidden_dim;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t total_size;
};

// =============================================================================
// RMSNorm Backward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void rmsnorm_backward(
    GM_ADDR grad_out,
    GM_ADDR inp,
    GM_ADDR weight,
    GM_ADDR rms,
    GM_ADDR grad_inp,
    GM_ADDR grad_weight,
    GM_ADDR tiling)
{
    RMSNormTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    RMSNormBackwardKernel kernel;
    kernel.Init(grad_out, inp, weight, rms, grad_inp, grad_weight, tiling_data);
    kernel.Process();
}
