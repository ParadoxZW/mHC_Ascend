/**
 * @file stream_ops.cpp
 * @brief Stream operations for mHC (aggregate and distribute) on Ascend
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * Implements stream-level operations:
 * - stream_aggregate: Weighted sum aggregation of n streams
 * - stream_distribute_mix_add: Distribution, mixing, and residual addition
 *
 * CUDA Reference: src/csrc/kernels/stream_ops.cuh
 */

#include "kernel_operator.h"
#include "../include/mhc_types.h"
#include "../include/utils.h"

using namespace mhc_ascend;
using namespace AscendC;
using StreamOpsTilingData = StreamOpsTiling;

// =============================================================================
// Stream Aggregate Forward Kernel
// =============================================================================

class StreamAggregateKernel {
public:
    __aicore__ inline StreamAggregateKernel() {}
    using InT = float;
    using OutT = floatX;
    using AccT = float;

    /**
     * @brief Initialize aggregate kernel
     * @param inp_gm Input tensor [B, n, C]
     * @param H_pre_raw_gm H_pre weights before sigmoid [B, n]
     * @param out_gm Output tensor [B, C]
     * @param H_pre_activated_gm Activated H_pre values [B, n] (optional output)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR inp_gm,
        GM_ADDR H_pre_raw_gm,
        GM_ADDR out_gm,
        GM_ADDR H_pre_activated_gm,
        const StreamAggregateTiling& tiling)
    {
        this->tiling = tiling;
        this->output_activated = (H_pre_activated_gm != nullptr);

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        // Distribute batch across cores
        this->batch_per_core = CeilingDiv(tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;
        }

        this->n = tiling.n;
        this->C = tiling.C;
        const int32_t align_elems = BLK_LEN / static_cast<int32_t>(sizeof(AccT));  // 8 for float32
        this->n_aligned = ALIGN_UP(this->n, align_elems);
        // Align C to 8 elements for float32 vector operations (8 * 4 = 32B)
        this->C_padded = ALIGN_UP(this->C, align_elems);
        this->needs_C_padding = (this->C != this->C_padded);

        // Set global buffers
        int32_t inp_offset = this->batch_start * this->n * this->C;
        int32_t H_offset = this->batch_start * this->n;
        int32_t out_offset = this->batch_start * this->C;

        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(inp_gm) + inp_offset,
                              this->batch_count * this->n * this->C);
        H_pre_rawGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT*>(H_pre_raw_gm) + H_offset,
                                     this->batch_count * this->n);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT*>(out_gm) + out_offset,
                              this->batch_count * this->C);

        if (this->output_activated) {
            H_pre_activatedGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ AccT*>(H_pre_activated_gm) + H_offset,
                this->batch_count * this->n);
        }

        // Initialize buffers with proper alignment
        // Use padded layout [n, C_padded] for input to ensure vector op alignment
        if (this->needs_C_padding) {
            pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->n * this->C_padded * sizeof(InT));
        } else {
            pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->n * this->C * sizeof(InT));
        }
        pipe.InitBuffer(outQueue, NUM_BUFFERS, this->C_padded * sizeof(OutT));
        pipe.InitBuffer(HPreActBuf, this->n_aligned * sizeof(AccT));
        pipe.InitBuffer(tmpBuf, this->C_padded * sizeof(AccT));
        pipe.InitBuffer(onesBuf, this->n_aligned * sizeof(AccT));
        pipe.InitBuffer(outF32Buf, this->C_padded * sizeof(AccT));
    }

    /**
     * @brief Main processing function
     *
     * For each batch element:
     * 1. Load H_pre_raw and apply sigmoid activation
     * 2. Load input streams [n, C]
     * 3. Compute weighted sum: out[c] = sum(H_pre[i] * inp[i, c])
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBatch(b);
        }
    }

private:
    /**
     * @brief Load from GM to VECIN queue with alignment handling.
     * Pattern from official Ascend C sample (DataCopyPadCustom_GM2UB):
     * DataCopy the 32B-aligned portion, then scalar SetValue for remainder.
     */
    __aicore__ inline void LoadGmToQueue(
        LocalTensor<InT> dst, GlobalTensor<InT> src, int32_t count)
    {
        const int32_t align_count = BLK_LEN / static_cast<int32_t>(sizeof(InT));
        int32_t aligned_count = (count / align_count) * align_count;

        if (aligned_count > 0) {
            DataCopy(dst, src, aligned_count);
        }
        // Scalar copy for remainder
        for (int32_t i = aligned_count; i < count; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

    __aicore__ inline void ProcessSingleBatch(int32_t batch_idx) {
        // Load and activate H_pre (scalar-safe to avoid alignment issues)
        LocalTensor<AccT> H_pre_activated = HPreActBuf.Get<AccT>();
        for (int32_t i = 0; i < this->n; ++i) {
            H_pre_activated.SetValue(i, H_pre_rawGm.GetValue(batch_idx * this->n + i));
        }
        for (int32_t i = this->n; i < this->n_aligned; ++i) {
            H_pre_activated.SetValue(i, static_cast<AccT>(0.0));
        }
        // Manual sigmoid: 1 / (1 + exp(-x)) to avoid unsupported Sigmoid on float.
        LocalTensor<AccT> ones = onesBuf.Get<AccT>();
        OnesBuffer(ones, this->n_aligned);
        Muls(H_pre_activated, H_pre_activated, static_cast<AccT>(-1.0), this->n_aligned);
        Exp(H_pre_activated, H_pre_activated, this->n_aligned);
        Adds(H_pre_activated, H_pre_activated, static_cast<AccT>(1.0), this->n_aligned);
        Div(H_pre_activated, ones, H_pre_activated, this->n_aligned);
        PipeBarrier<PIPE_V>();

        if (this->output_activated) {
            AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(this->n * sizeof(AccT)), 0, 0, 0};
            AscendC::DataCopyPad<AccT>(
                H_pre_activatedGm[batch_idx * this->n],
                H_pre_activated,
                copyParams);
        }

        // Load input [n, C] with alignment handling
        LocalTensor<InT> inp = inpQueue.AllocTensor<InT>();
        if (this->needs_C_padding) {
            // Load row by row with padding
            for (int32_t i = 0; i < this->n; ++i) {
                int32_t gm_offset = batch_idx * this->n * this->C + i * this->C;
                int32_t ub_offset = i * this->C_padded;
                // Load actual data
                LoadGmToQueue(inp[ub_offset], inpGm[gm_offset], this->C);
                // Zero-fill padding
                for (int32_t c = this->C; c < this->C_padded; ++c) {
                    inp.SetValue(ub_offset + c, static_cast<InT>(0.0));
                }
            }
        } else {
            // Fast path: C is already aligned
            DataCopy(inp, inpGm[batch_idx * this->n * this->C], this->n * this->C);
        }
        inpQueue.EnQue(inp);
        inp = inpQueue.DeQue<InT>();

        // Allocate output
        LocalTensor<OutT> out = outQueue.AllocTensor<OutT>();
        LocalTensor<AccT> outF32 = outF32Buf.Get<AccT>();
        ZeroBuffer(outF32, this->C_padded);

        // Compute weighted sum: out[c] = sum_i(H_pre[i] * inp[i, c])
        // Use C_padded for vector operations (padding is zero, so safe)
        int32_t row_stride = this->needs_C_padding ? this->C_padded : this->C;
        for (int32_t i = 0; i < this->n; ++i) {
            AccT h_val = H_pre_activated.GetValue(i);
            LocalTensor<InT> inp_stream = inp[i * row_stride];
            LocalTensor<AccT> tmp = tmpBuf.Get<AccT>();

            // tmp = h_val * inp_stream
            Muls(tmp, inp_stream, h_val, this->C_padded);

            // out += tmp
            Add(outF32, outF32, tmp, this->C_padded);
        }

        // Store output with alignment handling
        ConvertF32ToBF16(out, outF32, this->C_padded);
        outQueue.EnQue(out);
        out = outQueue.DeQue<OutT>();

        // Use DataCopyPad for non-aligned output
        bool out_aligned = (this->C * sizeof(OutT)) % BLK_LEN == 0;
        if (out_aligned) {
            DataCopy(outGm[batch_idx * this->C], out, this->C);
        } else {
            AscendC::DataCopyExtParams outCopyParams = {
                1, static_cast<uint32_t>(this->C * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[batch_idx * this->C], out, outCopyParams);
        }

        // Free tensors
        inpQueue.FreeTensor(inp);
        outQueue.FreeTensor(out);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> inpQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> outQueue;
    TBuf<QuePosition::VECCALC> HPreActBuf, tmpBuf, onesBuf, outF32Buf;

    GlobalTensor<InT> inpGm, H_pre_rawGm;
    GlobalTensor<OutT> outGm;
    GlobalTensor<AccT> H_pre_activatedGm;

    StreamAggregateTiling tiling;

    int32_t n, C;
    int32_t n_aligned;
    int32_t C_padded;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    bool output_activated;
    bool needs_C_padding;
};

// =============================================================================
// Stream Aggregate Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void stream_aggregate_forward(
    GM_ADDR inp,
    GM_ADDR H_pre_raw,
    GM_ADDR out,
    GM_ADDR H_pre_activated,
    GM_ADDR tiling)
{
    StreamAggregateTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    StreamAggregateKernel kernel;
    kernel.Init(inp, H_pre_raw, out, H_pre_activated, tiling_data);
    kernel.Process();
}

// =============================================================================
// Stream Distribute Mix Add Forward Kernel
// =============================================================================

class StreamDistributeMixAddKernel {
public:
    __aicore__ inline StreamDistributeMixAddKernel() {}
    using NormT = floatX;
    using AccT = float;

    /**
     * @brief Initialize kernel
     * @param y_norm_gm Normalized aggregated features [B, C]
     * @param H_post_raw_gm H_post weights before activation [B, n]
     * @param M_gm Mixing matrix [B, n, n]
     * @param x_inp_gm Original input [B, n, C]
     * @param out_gm Output [B, n, C]
     * @param H_post_activated_gm Activated H_post [B, n] (optional)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR y_norm_gm,
        GM_ADDR H_post_raw_gm,
        GM_ADDR M_gm,
        GM_ADDR x_inp_gm,
        GM_ADDR out_gm,
        GM_ADDR H_post_activated_gm,
        const StreamDistributeMixAddTiling& tiling)
    {
        this->tiling = tiling;
        this->output_activated = (H_post_activated_gm != nullptr);

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        this->batch_per_core = CeilingDiv(tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;
        }

        this->n = tiling.n;
        this->C = tiling.C;
        const int32_t align_elems = BLK_LEN / static_cast<int32_t>(sizeof(AccT));
        this->n_aligned = ALIGN_UP(this->n, align_elems);
        // Align C to 16 elements: satisfies both BF16 (16-elem=32B) and F32 (8-elem=32B) vector alignment
        this->C_padded = ALIGN_UP(this->C, 16);
        this->nn_padded = ALIGN_UP(this->n * this->n, align_elems);
        this->needs_padding = (this->C != this->C_padded);

        // Set global buffers
        int32_t y_offset = this->batch_start * this->C;
        int32_t H_offset = this->batch_start * this->n;
        int32_t M_offset = this->batch_start * this->n * this->n;
        int32_t x_offset = this->batch_start * this->n * this->C;

        y_normGm.SetGlobalBuffer(reinterpret_cast<__gm__ NormT*>(y_norm_gm) + y_offset,
                                 this->batch_count * this->C);
        H_post_rawGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(H_post_raw_gm) + H_offset,
                                     this->batch_count * this->n);
        MGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(M_gm) + M_offset,
                            this->batch_count * this->n * this->n);
        x_inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(x_inp_gm) + x_offset,
                                this->batch_count * this->n * this->C);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(out_gm) + x_offset,
                              this->batch_count * this->n * this->C);

        if (this->output_activated) {
            H_post_activatedGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ AccT*>(H_post_activated_gm) + H_offset,
                this->batch_count * this->n);
        }

        // Initialize buffers with proper alignment
        pipe.InitBuffer(y_normQueue, NUM_BUFFERS, this->C_padded * sizeof(NormT));
        pipe.InitBuffer(MQueue, NUM_BUFFERS, this->nn_padded * sizeof(AccT));
        if (this->needs_padding) {
            // Padded row layout [n, C_padded] for vector op alignment
            pipe.InitBuffer(x_inpQueue, NUM_BUFFERS, this->n * this->C_padded * sizeof(AccT));
            pipe.InitBuffer(outQueue, NUM_BUFFERS, this->n * this->C_padded * sizeof(AccT));
        } else {
            pipe.InitBuffer(x_inpQueue, NUM_BUFFERS, this->n * this->C * sizeof(AccT));
            pipe.InitBuffer(outQueue, NUM_BUFFERS, this->n * this->C * sizeof(AccT));
        }
        pipe.InitBuffer(tmpBuf1, this->n_aligned * sizeof(AccT));
        pipe.InitBuffer(tmpBuf2, this->C_padded * sizeof(AccT));
        pipe.InitBuffer(onesBuf, this->n_aligned * sizeof(AccT));
        pipe.InitBuffer(yNormF32Buf, this->C_padded * sizeof(AccT));
    }

    /**
     * @brief Main processing function
     *
     * For each batch element:
     * 1. Distribute: y_dist[i] = (2 * sigmoid(H_post[i])) * y_norm
     * 2. Mix: mix_out[i, c] = sum_j(M[i, j] * x_inp[j, c])
     * 3. Add: output[i, c] = y_dist[i, c] + mix_out[i, c]
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBatch(b);
        }
    }

private:
    /**
     * @brief Load from GM to VECIN queue with alignment handling.
     * Pattern from official Ascend C sample (DataCopyPadCustom_GM2UB):
     * DataCopy the 32B-aligned portion, then scalar SetValue for remainder.
     */
    template<typename T>
    __aicore__ inline void LoadGmToQueue(
        LocalTensor<T> dst, GlobalTensor<T> src, int32_t count)
    {
        const int32_t align_count = BLK_LEN / static_cast<int32_t>(sizeof(T));
        int32_t aligned_count = (count / align_count) * align_count;

        if (aligned_count > 0) {
            DataCopy(dst, src, aligned_count);
        }
        // Scalar copy for remainder (SetValue on queue tensor is supported per official samples)
        for (int32_t i = aligned_count; i < count; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

    __aicore__ inline void ProcessSingleBatch(int32_t batch_idx) {
        // Load H_post and apply 2 * sigmoid activation (scalar-safe)
        LocalTensor<AccT> H_post_activated = tmpBuf1.Get<AccT>();
        for (int32_t i = 0; i < this->n; ++i) {
            H_post_activated.SetValue(i, H_post_rawGm.GetValue(batch_idx * this->n + i));
        }
        for (int32_t i = this->n; i < this->n_aligned; ++i) {
            H_post_activated.SetValue(i, static_cast<AccT>(0.0));
        }
        // Manual sigmoid: 1 / (1 + exp(-x)) to avoid unsupported Sigmoid on float.
        LocalTensor<AccT> ones = onesBuf.Get<AccT>();
        OnesBuffer(ones, this->n_aligned);
        Muls(H_post_activated, H_post_activated, static_cast<AccT>(-1.0), this->n_aligned);
        Exp(H_post_activated, H_post_activated, this->n_aligned);
        Adds(H_post_activated, H_post_activated, static_cast<AccT>(1.0), this->n_aligned);
        Div(H_post_activated, ones, H_post_activated, this->n_aligned);
        Muls(H_post_activated, H_post_activated, static_cast<AccT>(2.0), this->n_aligned);
        PipeBarrier<PIPE_V>();

        if (this->output_activated) {
            AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(this->n * sizeof(AccT)), 0, 0, 0};
            AscendC::DataCopyPad<AccT>(
                H_post_activatedGm[batch_idx * this->n],
                H_post_activated,
                copyParams);
        }

        // Load y_norm (BF16) with alignment handling
        LocalTensor<NormT> y_norm = y_normQueue.AllocTensor<NormT>();
        if (this->needs_padding) {
            // Scalar load + zero padding for non-aligned BF16 data
            for (int32_t c = 0; c < this->C; ++c) {
                y_norm.SetValue(c, y_normGm.GetValue(batch_idx * this->C + c));
            }
            for (int32_t c = this->C; c < this->C_padded; ++c) {
                y_norm.SetValue(c, static_cast<NormT>(0));
            }
        } else {
            DataCopy(y_norm, y_normGm[batch_idx * this->C], this->C);
        }
        y_normQueue.EnQue(y_norm);
        y_norm = y_normQueue.DeQue<NormT>();
        LocalTensor<AccT> y_norm_f32 = yNormF32Buf.Get<AccT>();
        ConvertBF16ToF32(y_norm_f32, y_norm, this->C_padded);

        // Load M matrix with alignment handling
        LocalTensor<AccT> M = MQueue.AllocTensor<AccT>();
        int32_t nn = this->n * this->n;
        LoadGmToQueue(M, MGm[batch_idx * nn], nn);
        MQueue.EnQue(M);
        M = MQueue.DeQue<AccT>();

        // Load x_inp with alignment handling
        LocalTensor<AccT> x_inp = x_inpQueue.AllocTensor<AccT>();
        int32_t row_stride = this->needs_padding ? this->C_padded : this->C;
        if (this->needs_padding) {
            // Row-by-row load into padded UB layout [n, C_padded]
            for (int32_t i = 0; i < this->n; ++i) {
                int32_t gm_offset = batch_idx * this->n * this->C + i * this->C;
                int32_t ub_offset = i * this->C_padded;
                for (int32_t c = 0; c < this->C; ++c) {
                    x_inp.SetValue(ub_offset + c, x_inpGm.GetValue(gm_offset + c));
                }
                // Zero padding for vector op alignment
                for (int32_t c = this->C; c < this->C_padded; ++c) {
                    x_inp.SetValue(ub_offset + c, static_cast<AccT>(0.0));
                }
            }
        } else {
            DataCopy(x_inp, x_inpGm[batch_idx * this->n * this->C], this->n * this->C);
        }
        x_inpQueue.EnQue(x_inp);
        x_inp = x_inpQueue.DeQue<AccT>();

        // Allocate output
        LocalTensor<AccT> out = outQueue.AllocTensor<AccT>();

        // Process each output stream using padded vector length for alignment
        int32_t vec_len = this->needs_padding ? this->C_padded : this->C;
        for (int32_t i = 0; i < this->n; ++i) {
            LocalTensor<AccT> out_stream = out[i * row_stride];
            LocalTensor<AccT> tmp = tmpBuf2.Get<AccT>();

            // 1. Distribute: out_stream = H_post[i] * y_norm
            AccT h_post_val = H_post_activated.GetValue(i);
            Muls(out_stream, y_norm_f32, h_post_val, vec_len);

            // 2. Mix: add weighted sum of input streams
            for (int32_t j = 0; j < this->n; ++j) {
                AccT m_ij = M.GetValue(i * this->n + j);
                LocalTensor<AccT> x_stream = x_inp[j * row_stride];

                // tmp = m_ij * x_stream[j]
                Muls(tmp, x_stream, m_ij, vec_len);

                // out_stream += tmp
                Add(out_stream, out_stream, tmp, vec_len);
            }
        }

        // Store output with alignment handling
        outQueue.EnQue(out);
        out = outQueue.DeQue<AccT>();
        if (this->needs_padding) {
            // Row-by-row store: only write actual C elements per row
            for (int32_t i = 0; i < this->n; ++i) {
                AscendC::DataCopyExtParams oParams = {
                    1, static_cast<uint32_t>(this->C * sizeof(AccT)), 0, 0, 0};
                AscendC::DataCopyPad(outGm[batch_idx * this->n * this->C + i * this->C],
                                     out[i * this->C_padded],
                                     oParams);
            }
        } else {
            DataCopy(outGm[batch_idx * this->n * this->C], out, this->n * this->C);
        }

        // Free tensors
        y_normQueue.FreeTensor(y_norm);
        MQueue.FreeTensor(M);
        x_inpQueue.FreeTensor(x_inp);
        outQueue.FreeTensor(out);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> y_normQueue, MQueue, x_inpQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> outQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2, onesBuf, yNormF32Buf;

    GlobalTensor<NormT> y_normGm;
    GlobalTensor<AccT> H_post_rawGm, MGm, x_inpGm, outGm, H_post_activatedGm;

    StreamDistributeMixAddTiling tiling;

    int32_t n, C;
    int32_t n_aligned;
    int32_t C_padded;
    int32_t nn_padded;
    bool needs_padding;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    bool output_activated;
};

// =============================================================================
// Stream Distribute Mix Add Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void stream_distribute_mix_add_forward(
    GM_ADDR y_norm,
    GM_ADDR H_post_raw,
    GM_ADDR M,
    GM_ADDR x_inp,
    GM_ADDR out,
    GM_ADDR H_post_activated,
    GM_ADDR tiling)
{
    StreamDistributeMixAddTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    StreamDistributeMixAddKernel kernel;
    kernel.Init(y_norm, H_post_raw, M, x_inp, out, H_post_activated, tiling_data);
    kernel.Process();
}

// =============================================================================
// Stream Aggregate Backward Kernel
// =============================================================================

template<typename T = float>
class StreamAggregateBackwardKernel {
public:
    __aicore__ inline StreamAggregateBackwardKernel() {}

    /**
     * @brief Initialize backward kernel
     * @param grad_out_gm Gradient w.r.t. output [B, C]
     * @param inp_gm Original input [B, n, C]
     * @param H_pre_activated_gm Activated H_pre from forward [B, n]
     * @param grad_inp_gm Gradient w.r.t. input [B, n, C] (output)
     * @param grad_H_pre_gm Gradient w.r.t. H_pre_raw [B, n] (output)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR grad_out_gm,
        GM_ADDR inp_gm,
        GM_ADDR H_pre_activated_gm,
        GM_ADDR grad_inp_gm,
        GM_ADDR grad_H_pre_gm,
        const StreamAggregateTiling& tiling)
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

        this->n = tiling.n;
        this->C = tiling.C;

        // Align C to 8 elements for float32 vector operations (8 * 4 = 32B)
        const int32_t align_elems = BLK_LEN / static_cast<int32_t>(sizeof(T));
        this->C_padded = ALIGN_UP(this->C, align_elems);
        this->needs_C_padding = (this->C != this->C_padded);

        int32_t grad_out_offset = this->batch_start * this->C;
        int32_t inp_offset = this->batch_start * this->n * this->C;
        int32_t H_offset = this->batch_start * this->n;

        gradOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_out_gm) + grad_out_offset,
                                   this->batch_count * this->C);
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inp_gm) + inp_offset,
                               this->batch_count * this->n * this->C);
        H_pre_activatedGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(H_pre_activated_gm) + H_offset,
                                           this->batch_count * this->n);
        gradInpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_inp_gm) + inp_offset,
                                   this->batch_count * this->n * this->C);
        gradH_preGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_H_pre_gm) + H_offset,
                                     this->batch_count * this->n);

        // Initialize buffers with proper alignment (use C_padded for vector ops)
        pipe.InitBuffer(gradOutQueue, NUM_BUFFERS, this->C_padded * sizeof(T));
        if (this->needs_C_padding) {
            pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->n * this->C_padded * sizeof(T));
            pipe.InitBuffer(gradInpQueue, NUM_BUFFERS, this->n * this->C_padded * sizeof(T));
        } else {
            pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->n * this->C * sizeof(T));
            pipe.InitBuffer(gradInpQueue, NUM_BUFFERS, this->n * this->C * sizeof(T));
        }
        pipe.InitBuffer(H_preQueue, NUM_BUFFERS, ALIGN_UP(this->n * sizeof(T), BLK_LEN));
        pipe.InitBuffer(tmpBuf, this->C_padded * sizeof(T));
        pipe.InitBuffer(gradH_preBuf, ALIGN_UP(this->n * sizeof(T), BLK_LEN));
    }

    /**
     * @brief Backward pass
     *
     * Gradients:
     * - grad_inp[i, c] = grad_out[c] * H_pre[i]
     * - grad_H_pre[i] = sum_c(grad_out[c] * inp[i, c]) * sigmoid'(H_pre_raw[i])
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBackward(b);
        }
    }

private:
    /**
     * @brief Load from GM to VECIN queue with alignment handling.
     */
    __aicore__ inline void LoadGmToQueue(
        LocalTensor<T> dst, GlobalTensor<T> src, int32_t count)
    {
        const int32_t align_count = BLK_LEN / static_cast<int32_t>(sizeof(T));
        int32_t aligned_count = (count / align_count) * align_count;

        if (aligned_count > 0) {
            DataCopy(dst, src, aligned_count);
        }
        for (int32_t i = aligned_count; i < count; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

    __aicore__ inline void ProcessSingleBackward(int32_t batch_idx) {
        // Load grad_out with alignment handling
        LocalTensor<T> grad_out = gradOutQueue.AllocTensor<T>();
        if (this->needs_C_padding) {
            LoadGmToQueue(grad_out, gradOutGm[batch_idx * this->C], this->C);
            // Zero-fill padding
            for (int32_t c = this->C; c < this->C_padded; ++c) {
                grad_out.SetValue(c, static_cast<T>(0.0));
            }
        } else {
            DataCopy(grad_out, gradOutGm[batch_idx * this->C], this->C);
        }
        gradOutQueue.EnQue(grad_out);
        grad_out = gradOutQueue.DeQue<T>();

        // Load inp with alignment handling
        LocalTensor<T> inp = inpQueue.AllocTensor<T>();
        if (this->needs_C_padding) {
            // Load row by row with padding
            for (int32_t i = 0; i < this->n; ++i) {
                int32_t gm_offset = batch_idx * this->n * this->C + i * this->C;
                int32_t ub_offset = i * this->C_padded;
                LoadGmToQueue(inp[ub_offset], inpGm[gm_offset], this->C);
                // Zero-fill padding
                for (int32_t c = this->C; c < this->C_padded; ++c) {
                    inp.SetValue(ub_offset + c, static_cast<T>(0.0));
                }
            }
        } else {
            DataCopy(inp, inpGm[batch_idx * this->n * this->C], this->n * this->C);
        }
        inpQueue.EnQue(inp);
        inp = inpQueue.DeQue<T>();

        // Load H_pre_activated: use efficient path for 32B-aligned sizes, scalar for unaligned
        LocalTensor<T> H_pre_activated = H_preQueue.AllocTensor<T>();
        // For float32: n must be multiple of 8 (n*4 = 32 bytes alignment)
        bool n_aligned = (this->n * sizeof(T)) % 32 == 0;
        if (n_aligned) {
            DataCopy(H_pre_activated, H_pre_activatedGm[batch_idx * this->n], this->n);
        } else {
            for (int32_t i = 0; i < this->n; ++i) {
                H_pre_activated.SetValue(i, H_pre_activatedGm.GetValue(batch_idx * this->n + i));
            }
        }
        H_preQueue.EnQue(H_pre_activated);
        H_pre_activated = H_preQueue.DeQue<T>();

        // Allocate grad_inp and get buffers
        LocalTensor<T> grad_inp = gradInpQueue.AllocTensor<T>();
        LocalTensor<T> grad_H_pre = gradH_preBuf.Get<T>();
        LocalTensor<T> tmp = tmpBuf.Get<T>();

        // Compute grad_inp and grad_H_pre
        // Use C_padded for vector operations (padding is zero, so safe)
        int32_t row_stride = this->needs_C_padding ? this->C_padded : this->C;
        for (int32_t i = 0; i < this->n; ++i) {
            T h_val = H_pre_activated.GetValue(i);
            LocalTensor<T> grad_inp_stream = grad_inp[i * row_stride];
            LocalTensor<T> inp_stream = inp[i * row_stride];

            // grad_inp[i, :] = grad_out * h_val (PIPE_V)
            Muls(grad_inp_stream, grad_out, h_val, this->C_padded);

            // grad_H_pre[i] = sum_c(grad_out[c] * inp[i, c])
            // Use vector Mul then scalar sum with PipeBarrier
            Mul(tmp, grad_out, inp_stream, this->C_padded);

            // CRITICAL: Wait for PIPE_V (Mul) to complete before reading with PIPE_S (GetValue)
            PipeBarrier<PIPE_V>();

            // Only sum actual C elements (not padding)
            T sum = static_cast<T>(0);
            for (int32_t c = 0; c < this->C; ++c) {
                sum += tmp.GetValue(c);
            }
            grad_H_pre.SetValue(i, sum);
        }

        // Store grad_inp with alignment handling
        gradInpQueue.EnQue(grad_inp);
        grad_inp = gradInpQueue.DeQue<T>();
        if (this->needs_C_padding) {
            // Store row by row using DataCopyPad
            for (int32_t i = 0; i < this->n; ++i) {
                int32_t ub_offset = i * this->C_padded;
                int32_t gm_offset = batch_idx * this->n * this->C + i * this->C;
                AscendC::DataCopyExtParams copyParams = {
                    1, static_cast<uint32_t>(this->C * sizeof(T)), 0, 0, 0
                };
                AscendC::DataCopyPad(gradInpGm[gm_offset], grad_inp[ub_offset], copyParams);
            }
        } else {
            DataCopy(gradInpGm[batch_idx * this->n * this->C], grad_inp, this->n * this->C);
        }

        // Store grad_H_pre: use efficient path for 32B-aligned sizes, DataCopyPad for unaligned
        if (n_aligned) {
            DataCopy(gradH_preGm[batch_idx * this->n], grad_H_pre, this->n);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(this->n * sizeof(T)), 0, 0, 0
            };
            AscendC::DataCopyPad(gradH_preGm[batch_idx * this->n], grad_H_pre, copyParams);
        }

        // Free tensors
        gradOutQueue.FreeTensor(grad_out);
        inpQueue.FreeTensor(inp);
        H_preQueue.FreeTensor(H_pre_activated);
        gradInpQueue.FreeTensor(grad_inp);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> gradOutQueue, inpQueue, H_preQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> gradInpQueue;
    TBuf<QuePosition::VECCALC> tmpBuf, gradH_preBuf;

    GlobalTensor<T> gradOutGm, inpGm, H_pre_activatedGm, gradInpGm, gradH_preGm;

    StreamAggregateTiling tiling;

    int32_t n, C;
    int32_t C_padded;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    bool needs_C_padding;
};

// =============================================================================
// Stream Aggregate Backward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void stream_aggregate_backward(
    GM_ADDR grad_out,
    GM_ADDR inp,
    GM_ADDR H_pre_activated,
    GM_ADDR grad_inp,
    GM_ADDR grad_H_pre,
    GM_ADDR tiling)
{
    StreamAggregateTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    StreamAggregateBackwardKernel<float> kernel;
    kernel.Init(grad_out, inp, H_pre_activated, grad_inp, grad_H_pre, tiling_data);
    kernel.Process();
}

// =============================================================================
// Stream Distribute Mix Add Backward Kernel
// =============================================================================

template<typename T = float>
class StreamDistributeMixAddBackwardKernel {
public:
    __aicore__ inline StreamDistributeMixAddBackwardKernel() {}

    /**
     * @brief Initialize backward kernel
     */
    __aicore__ inline void Init(
        GM_ADDR grad_out_gm,
        GM_ADDR x_inp_gm,
        GM_ADDR y_norm_gm,
        GM_ADDR M_gm,
        GM_ADDR H_post_activated_gm,
        GM_ADDR grad_x_gm,
        GM_ADDR grad_y_norm_gm,
        GM_ADDR grad_M_gm,
        GM_ADDR grad_H_post_gm,
        const StreamDistributeMixAddTiling& tiling)
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

        this->n = tiling.n;
        this->C = tiling.C;

        int32_t grad_out_offset = this->batch_start * this->n * this->C;
        int32_t y_offset = this->batch_start * this->C;
        int32_t M_offset = this->batch_start * this->n * this->n;
        int32_t H_offset = this->batch_start * this->n;

        gradOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_out_gm) + grad_out_offset,
                                   this->batch_count * this->n * this->C);
        x_inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_inp_gm) + grad_out_offset,
                                 this->batch_count * this->n * this->C);
        y_normGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_norm_gm) + y_offset,
                                  this->batch_count * this->C);
        MGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(M_gm) + M_offset,
                             this->batch_count * this->n * this->n);
        H_post_activatedGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(H_post_activated_gm) + H_offset,
                                            this->batch_count * this->n);
        grad_xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_x_gm) + grad_out_offset,
                                  this->batch_count * this->n * this->C);
        grad_y_normGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_y_norm_gm) + y_offset,
                                       this->batch_count * this->C);
        grad_MGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_M_gm) + M_offset,
                                  this->batch_count * this->n * this->n);
        grad_H_postGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_H_post_gm) + H_offset,
                                       this->batch_count * this->n);

        // Initialize buffers (with proper alignment)
        pipe.InitBuffer(gradOutQueue, NUM_BUFFERS, ALIGN_UP(this->n * this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(x_inpQueue, NUM_BUFFERS, ALIGN_UP(this->n * this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(y_normQueue, NUM_BUFFERS, ALIGN_UP(this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(MQueue, NUM_BUFFERS, ALIGN_UP(this->n * this->n * sizeof(T), BLK_LEN));
        pipe.InitBuffer(H_postQueue, NUM_BUFFERS, ALIGN_UP(this->n * sizeof(T), BLK_LEN));
        pipe.InitBuffer(grad_xQueue, NUM_BUFFERS, ALIGN_UP(this->n * this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(tmpBuf1, ALIGN_UP(this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(tmpBuf2, ALIGN_UP(this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(grad_y_normBuf, ALIGN_UP(this->C * sizeof(T), BLK_LEN));
        pipe.InitBuffer(grad_MBuf, ALIGN_UP(this->n * this->n * sizeof(T), BLK_LEN));
        pipe.InitBuffer(grad_H_postBuf, ALIGN_UP(this->n * sizeof(T), BLK_LEN));
    }

    /**
     * @brief Backward pass
     *
     * Gradients:
     * - grad_x[j, c] = sum_i(grad_out[i, c] * M[i, j])
     * - grad_y_norm[c] = sum_i(grad_out[i, c] * H_post[i])
     * - grad_M[i, j] = sum_c(grad_out[i, c] * x[j, c])
     * - grad_H_post[i] = sum_c(grad_out[i, c] * y_norm[c]) * sigmoid'
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBackward(b);
        }
    }

private:
    /**
     * @brief Load from GM to VECIN queue with alignment handling.
     * Pattern from official Ascend C sample (DataCopyPadCustom_GM2UB):
     * DataCopy the 32B-aligned portion, then scalar SetValue for remainder.
     */
    __aicore__ inline void LoadGmToQueue(
        LocalTensor<T> dst, GlobalTensor<T> src, int32_t count)
    {
        const int32_t align_count = BLK_LEN / static_cast<int32_t>(sizeof(T));
        int32_t aligned_count = (count / align_count) * align_count;

        if (aligned_count > 0) {
            DataCopy(dst, src, aligned_count);
        }
        // Scalar copy for remainder (SetValue on queue tensor is supported per official samples)
        for (int32_t i = aligned_count; i < count; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

    __aicore__ inline void ProcessSingleBackward(int32_t batch_idx) {
        // Load inputs - use LoadGmToQueue for alignment-safe loading

        // Tensors: grad_out, x_inp, M - may not be 32B-aligned
        LocalTensor<T> grad_out_t = gradOutQueue.AllocTensor<T>();
        LocalTensor<T> x_inp_t = x_inpQueue.AllocTensor<T>();
        LocalTensor<T> M_t = MQueue.AllocTensor<T>();

        LoadGmToQueue(grad_out_t, gradOutGm[batch_idx * this->n * this->C], this->n * this->C);
        LoadGmToQueue(x_inp_t, x_inpGm[batch_idx * this->n * this->C], this->n * this->C);
        LoadGmToQueue(M_t, MGm[batch_idx * this->n * this->n], this->n * this->n);

        gradOutQueue.EnQue(grad_out_t);
        x_inpQueue.EnQue(x_inp_t);
        MQueue.EnQue(M_t);

        LocalTensor<T> grad_out = gradOutQueue.DeQue<T>();
        LocalTensor<T> x_inp = x_inpQueue.DeQue<T>();
        LocalTensor<T> M = MQueue.DeQue<T>();

        // Small tensors: y_norm (C floats), H_post (n floats) - read directly from GM with GetValue
        // to avoid DataCopy rounding down to 0 bytes for sizes < 32B

        // Allocate outputs
        LocalTensor<T> grad_x = grad_xQueue.AllocTensor<T>();
        LocalTensor<T> grad_y_norm = grad_y_normBuf.Get<T>();
        LocalTensor<T> grad_M = grad_MBuf.Get<T>();
        LocalTensor<T> grad_H_post = grad_H_postBuf.Get<T>();
        LocalTensor<T> tmp1 = tmpBuf1.Get<T>();
        LocalTensor<T> tmp2 = tmpBuf2.Get<T>();

        // Initialize all output buffers to zero
        ZeroBuffer(grad_x, this->n * this->C);
        ZeroBuffer(grad_y_norm, this->C);
        ZeroBuffer(grad_M, this->n * this->n);
        ZeroBuffer(grad_H_post, this->n);

        // Compute gradients - read small tensors directly from GM
        for (int32_t i = 0; i < this->n; ++i) {
            // Read H_post from GM (small tensor < 32B)
            T h_post_val = H_post_activatedGm.GetValue(batch_idx * this->n + i);
            T dot_H_post = static_cast<T>(0.0);

            for (int32_t c = 0; c < this->C; ++c) {
                T grad_out_ic = grad_out.GetValue(i * this->C + c);
                // Read y_norm from GM (small tensor < 32B)
                T y_norm_c = y_normGm.GetValue(batch_idx * this->C + c);

                // grad_y_norm[c] += H_post[i] * grad_out[i, c]
                T grad_y_val = grad_y_norm.GetValue(c);
                grad_y_norm.SetValue(c, grad_y_val + h_post_val * grad_out_ic);

                // Accumulate dot product for grad_H_post[i]
                dot_H_post += grad_out_ic * y_norm_c;
            }

            // grad_H_post[i] = dot(grad_out[i], y_norm)
            grad_H_post.SetValue(i, dot_H_post);

            // Compute grad_x and grad_M
            for (int32_t j = 0; j < this->n; ++j) {
                T m_ij = M.GetValue(i * this->n + j);
                T dot_M_ij = static_cast<T>(0.0);

                for (int32_t c = 0; c < this->C; ++c) {
                    T grad_out_ic = grad_out.GetValue(i * this->C + c);
                    T x_jc = x_inp.GetValue(j * this->C + c);

                    // grad_x[j, c] += M[i, j] * grad_out[i, c]
                    T grad_x_val = grad_x.GetValue(j * this->C + c);
                    grad_x.SetValue(j * this->C + c, grad_x_val + m_ij * grad_out_ic);

                    // Accumulate dot product for grad_M[i, j]
                    dot_M_ij += grad_out_ic * x_jc;
                }

                // grad_M[i, j] = dot(grad_out[i], x[j])
                grad_M.SetValue(i * this->n + j, dot_M_ij);
            }
        }

        // Store outputs with alignment-safe writes
        grad_xQueue.EnQue(grad_x);
        grad_x = grad_xQueue.DeQue<T>();

        // grad_x store: use DataCopyPad if not 32B-aligned
        bool nC_aligned = (this->n * this->C * sizeof(T)) % BLK_LEN == 0;
        if (nC_aligned) {
            DataCopy(grad_xGm[batch_idx * this->n * this->C], grad_x, this->n * this->C);
        } else {
            AscendC::DataCopyExtParams x_copyParams = {
                1, static_cast<uint32_t>(this->n * this->C * sizeof(T)), 0, 0, 0
            };
            AscendC::DataCopyPad(grad_xGm[batch_idx * this->n * this->C], grad_x, x_copyParams);
        }

        // Write small tensors using DataCopyPad to avoid alignment issues
        // (SetValue doesn't work properly for small tensor writes to GM)
        bool y_norm_aligned = (this->C * sizeof(T)) % BLK_LEN == 0;
        if (y_norm_aligned) {
            DataCopy(grad_y_normGm[batch_idx * this->C], grad_y_norm, this->C);
        } else {
            AscendC::DataCopyExtParams y_copyParams = {
                1, static_cast<uint32_t>(this->C * sizeof(T)), 0, 0, 0
            };
            AscendC::DataCopyPad(grad_y_normGm[batch_idx * this->C], grad_y_norm, y_copyParams);
        }

        // grad_M store: use DataCopyPad if not 32B-aligned
        bool nn_aligned = (this->n * this->n * sizeof(T)) % BLK_LEN == 0;
        if (nn_aligned) {
            DataCopy(grad_MGm[batch_idx * this->n * this->n], grad_M, this->n * this->n);
        } else {
            AscendC::DataCopyExtParams M_copyParams = {
                1, static_cast<uint32_t>(this->n * this->n * sizeof(T)), 0, 0, 0
            };
            AscendC::DataCopyPad(grad_MGm[batch_idx * this->n * this->n], grad_M, M_copyParams);
        }

        // Write grad_H_post using DataCopyPad for small tensors
        bool h_post_aligned = (this->n * sizeof(T)) % 32 == 0;
        if (h_post_aligned) {
            DataCopy(grad_H_postGm[batch_idx * this->n], grad_H_post, this->n);
        } else {
            AscendC::DataCopyExtParams h_copyParams = {
                1, static_cast<uint32_t>(this->n * sizeof(T)), 0, 0, 0
            };
            AscendC::DataCopyPad(grad_H_postGm[batch_idx * this->n], grad_H_post, h_copyParams);
        }

        // Free tensors (y_norm and H_post read directly from GM, no FreeTensor needed)
        gradOutQueue.FreeTensor(grad_out);
        x_inpQueue.FreeTensor(x_inp);
        MQueue.FreeTensor(M);
        grad_xQueue.FreeTensor(grad_x);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> gradOutQueue, x_inpQueue, y_normQueue, MQueue, H_postQueue;
    TQue<QuePosition::VECOUT, NUM_BUFFERS> grad_xQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2, grad_y_normBuf, grad_MBuf, grad_H_postBuf;

    GlobalTensor<T> gradOutGm, x_inpGm, y_normGm, MGm, H_post_activatedGm;
    GlobalTensor<T> grad_xGm, grad_y_normGm, grad_MGm, grad_H_postGm;

    StreamDistributeMixAddTiling tiling;

    int32_t n, C;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
};

// =============================================================================
// Stream Distribute Mix Add Backward Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void stream_distribute_mix_add_backward(
    GM_ADDR grad_out,
    GM_ADDR x_inp,
    GM_ADDR y_norm,
    GM_ADDR M,
    GM_ADDR H_post_activated,
    GM_ADDR grad_x,
    GM_ADDR grad_y_norm,
    GM_ADDR grad_M,
    GM_ADDR grad_H_post,
    GM_ADDR tiling)
{
    StreamDistributeMixAddTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    StreamDistributeMixAddBackwardKernel<float> kernel;
    kernel.Init(grad_out, x_inp, y_norm, M, H_post_activated,
                grad_x, grad_y_norm, grad_M, grad_H_post, tiling_data);
    kernel.Process();
}
