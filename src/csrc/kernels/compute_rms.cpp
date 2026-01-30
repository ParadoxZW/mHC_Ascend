/**
 * @file compute_rms.cpp
 * @brief Compute RMS for dynamic-H on Ascend
 *
 * Computes RMS per row: rms = sqrt(mean(x^2) + eps)
 * Input: bf16 [B, K], Output: float [B]
 */

#include "kernel_operator.h"
#include "../include/mhc_types.h"
#include "../include/utils.h"

using namespace mhc_ascend;
using namespace AscendC;
using ComputeRMSTilingData = ComputeRMSTiling;

class ComputeRMSKernel {
public:
    __aicore__ inline ComputeRMSKernel() {}

    __aicore__ inline void Init(
        GM_ADDR inp_gm,
        GM_ADDR rms_gm,
        const ComputeRMSTiling& tiling)
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
        this->hidden_dim_aligned = ALIGN_UP(this->hidden_dim, BLK_LEN / static_cast<int32_t>(sizeof(DataT)));
        this->total_size = this->batch_count * this->hidden_dim;

        int32_t offset = this->batch_start * this->hidden_dim;
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataT*>(inp_gm) + offset, this->total_size);
        rmsGm.SetGlobalBuffer(reinterpret_cast<__gm__ AccT*>(rms_gm) + this->batch_start,
                              this->batch_count);

        pipe.InitBuffer(inpQueue, NUM_BUFFERS, this->hidden_dim_aligned * sizeof(DataT));
        pipe.InitBuffer(inpF32Buf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(squaredBuf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(tmpCalcBuf, this->hidden_dim_aligned * sizeof(AccT));
        pipe.InitBuffer(rmsBuf, ALIGN_UP(sizeof(AccT), BLK_LEN));
    }

    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleRow(b);
        }
    }

private:
    __aicore__ inline void ProcessSingleRow(int32_t batch_idx) {
        using DataT = floatX;
        using AccT = floatN;

        LocalTensor<DataT> inp = inpQueue.AllocTensor<DataT>();
        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(this->hidden_dim * sizeof(DataT)), 0, 0, 0};
        int32_t rpad = this->hidden_dim_aligned - this->hidden_dim;
        AscendC::DataCopyPadExtParams<DataT> padParams = {
            false, 0, static_cast<uint8_t>(rpad), 0};
        AscendC::DataCopyPad<DataT>(inp, inpGm[batch_idx * this->hidden_dim], copyParams, padParams);
        inpQueue.EnQue(inp);
        inp = inpQueue.DeQue<DataT>();

        LocalTensor<AccT> inpF32 = inpF32Buf.Get<AccT>();
        LocalTensor<AccT> squared = squaredBuf.Get<AccT>();
        LocalTensor<AccT> tmpCalc = tmpCalcBuf.Get<AccT>();
        LocalTensor<AccT> rms_val = rmsBuf.Get<AccT>();

        ConvertBF16ToF32(inpF32, inp, this->hidden_dim_aligned);

        // Debug path: if eps < 0, return the first converted element for BF16->F32 validation.
        if (tiling.eps < 0) {
            AccT first = inpF32.GetValue(0);
            rms_val.SetValue(0, first);
            WriteScalarToGM(rmsGm, batch_idx, rms_val);
            inpQueue.FreeTensor(inp);
            return;
        }
        ComputeRMS(rms_val, inpF32, squared, tmpCalc, this->hidden_dim,
                   static_cast<AccT>(tiling.eps));

        WriteScalarToGM(rmsGm, batch_idx, rms_val);

        inpQueue.FreeTensor(inp);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> inpQueue;
    TBuf<QuePosition::VECCALC> inpF32Buf, squaredBuf, tmpCalcBuf, rmsBuf;

    GlobalTensor<floatX> inpGm;
    GlobalTensor<floatN> rmsGm;

    ComputeRMSTiling tiling;

    int32_t hidden_dim;
    int32_t hidden_dim_aligned;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t total_size;
};

extern "C" __global__ __aicore__ void compute_rms_forward(
    GM_ADDR inp,
    GM_ADDR rms,
    GM_ADDR tiling)
{
    ComputeRMSTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    ComputeRMSKernel kernel;
    kernel.Init(inp, rms, tiling_data);
    kernel.Process();
}
