# mHC Ascend

Ascend C implementation of **mHC (Manifold-Constrained Hyper-Connections)** for Huawei Ascend NPUs.

## Introduction

mHC is a novel technique proposed by DeepSeek-AI that replaces traditional residual connections in Transformer architectures with learnable manifold-constrained hyper-connections. The core innovation uses **Sinkhorn-Knopp normalization** to produce doubly-stochastic mixing matrices, enabling more expressive information flow between layers while maintaining mathematical constraints that improve training stability.

Key components of mHC:
- **Stream Aggregate**: Weighted aggregation of multiple input streams using learned gating
- **RMSNorm**: Root Mean Square normalization for stable training
- **Sinkhorn-Knopp**: Iterative algorithm to produce doubly-stochastic matrices
- **Stream Distribute**: Distribution and mixing of normalized features back to streams

This implementation provides optimized Ascend C kernels with full forward and backward pass support, enabling end-to-end training on Huawei Ascend NPUs.

### Acknowledgments

This project is based on the CUDA implementation from [AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu). Special thanks to the original authors for their excellent work.

### Testing Status

This implementation has been **comprehensively tested on Ascend 910B1** including:
- Numerical accuracy validation against CUDA reference
- End-to-end training verification on real arithmetic tasks
- Multi-core parallelization correctness

**Note**: We cannot guarantee correct operation on other Huawei accelerator models. If you encounter issues on different hardware, please open an issue.

### Development Story

This project was developed through **100% vibe coding** with AI assistance over 5 intensive days. The journey involved navigating numerous Ascend C quirks including 32-byte alignment requirements, multi-core Mask API interference, and GlobalTensor write semantics. See the `docs/` directory for detailed documentation of challenges encountered and solutions developed.

---

## Installation

### Requirements

- **Hardware**: Huawei Ascend 910B1 NPU
- **CANN Toolkit**: Version 8.0+ (aarch64-linux)
- **Python**: 3.8+ with PyTorch 2.x and torch-npu

### Environment Setup

```bash
# Set CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Verify NPU availability
npu-smi info
```

### Build and Install

```bash
cd mHC_ascend

# Build kernels and install Python package
bash scripts/build.sh
pip install -e .
```

The build script handles:
1. CMake configuration for Ascend910B1
2. Ascend C kernel compilation
3. Python extension building
4. Package installation

### Verify Installation

```python
import torch
import torch_npu
from mhc import MHCLayer

# Quick test
layer = MHCLayer(hidden_dim=64, expansion_rate=4).npu()
x = torch.randn(2, 4, 64, device='npu')
y = layer(x)
print(f"Output shape: {y.shape}")  # [2, 4, 64]
```

---

## Usage

### Basic Usage

```python
import torch
import torch_npu  # Must import before mhc
from mhc import MHCLayer

# Create mHC layer with Dynamic H (paper default)
layer = MHCLayer(
    hidden_dim=4096,      # Feature dimension (C)
    expansion_rate=4,     # Number of streams (n)
    num_sinkhorn_iters=20,
    use_dynamic_h=True,   # Input-dependent H computation
).npu()

# Input: [batch_size, expansion_rate, hidden_dim]
x = torch.randn(32, 4, 4096, device='npu')
output = layer(x)  # [32, 4, 4096]
```

### Static H Mode

For inference or when H values should be shared across batch:

```python
# Static H path (faster, shared H across batch)
layer_static = MHCLayer(
    hidden_dim=4096,
    expansion_rate=4,
    use_dynamic_h=False,  # Shared H values
).npu()
```

### Using Individual Operators

```python
from mhc import sinkhorn_knopp, rmsnorm, stream_aggregate

# Sinkhorn-Knopp normalization (requires non-negative square matrix)
matrix = torch.rand(32, 4, 4, device='npu')  # Use rand, not randn!
doubly_stochastic = sinkhorn_knopp(matrix, num_iters=20, eps=1e-8)
# Verify: rows and columns sum to 1
print(doubly_stochastic.sum(dim=-1))  # ~1.0
print(doubly_stochastic.sum(dim=-2))  # ~1.0

# RMS Normalization
x = torch.randn(1024, 4096, device='npu')
weight = torch.ones(4096, device='npu')
normalized = rmsnorm(x, weight, eps=1e-5)

# Stream Aggregate (weighted sum with sigmoid gating)
streams = torch.randn(32, 4, 256, device='npu')
H_pre = torch.randn(32, 4, device='npu')
aggregated = stream_aggregate(streams, H_pre)  # [32, 256]
```

### Integrating with Transformer Models

The `MHCResidualWrapper` provides a drop-in replacement for residual connections:

```python
from mhc import MHCResidualWrapper

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, expansion_rate=4):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForward(hidden_dim)

        # Replace residual connections with mHC
        self.mhc_attn = MHCResidualWrapper(hidden_dim, expansion_rate).npu()
        self.mhc_ffn = MHCResidualWrapper(hidden_dim, expansion_rate).npu()

    def forward(self, x):
        # Traditional: x = x + self.attention(x)
        # With mHC:
        attn_out = self.attention(x)
        x = self.mhc_attn(residual=x, branch_output=attn_out)

        # Traditional: x = x + self.ffn(x)
        # With mHC:
        ffn_out = self.ffn(x)
        x = self.mhc_ffn(residual=x, branch_output=ffn_out)

        return x
```

### Environment Variables

Control runtime behavior through environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MHC_MULTICORE` | 1-32 | 8 | Number of AI cores to use |
| `MHC_SINKHORN_FWD_IMPL` | auto, scalar, vectorized | auto | Sinkhorn forward implementation |
| `MHC_SINKHORN_BWD_IMPL` | auto, scalar, vectorized, n4 | auto | Sinkhorn backward implementation |
| `MHC_USE_SCALAR_MATMUL` | 0, 1 | 0 | Use scalar MatMul (debug only) |

```bash
# Debug mode: single core, scalar implementations
MHC_MULTICORE=1 MHC_SINKHORN_FWD_IMPL=scalar python train.py

# Production mode (default settings are optimal)
python train.py
```

---

## Benchmark

The `benchmark.py` script compares Ascend C implementation performance against naive PyTorch implementation.

### Running Benchmarks

```bash
cd mHC_ascend

# Single configuration benchmark
python benchmark.py --batch 64 --hidden 1280 --expansion 4

# All standard configurations from paper
python benchmark.py --all-configs

# Include backward pass
python benchmark.py --all-configs --backward

# Only Dynamic H path (paper default)
python benchmark.py --all-configs --dynamic-only
```

<!-- ### Benchmark Results

Tested on Ascend 910B1. Speedup is relative to naive PyTorch implementation on the same hardware.

**Dynamic H Path** (per-batch H values computed via Equations 7-9 from paper):

| Batch | Hidden | n  | Forward Speedup | Backward Speedup |
|-------|--------|----|-----------------|------------------|
| 320   | 1280   | 4  | TBD             | TBD              |
| 512   | 1920   | 4  | TBD             | TBD              |
| 1280  | 2560   | 4  | TBD             | TBD              |
| 2560  | 1280   | 4  | TBD             | TBD              |
| 128   | 1280   | 8  | TBD             | TBD              |
| 256   | 1280   | 8  | TBD             | TBD              |
| 32    | 1280   | 32 | TBD             | TBD              |
| 64    | 1280   | 32 | TBD             | TBD              |
| 128   | 1280   | 32 | TBD             | TBD              |

**Static H Path** (shared H across batch):

| Batch | Hidden | n  | Forward Speedup | Backward Speedup |
|-------|--------|----|-----------------|------------------|
| 320   | 1280   | 4  | TBD             | TBD              |
| 512   | 1920   | 4  | TBD             | TBD              |
| 1280  | 2560   | 4  | TBD             | TBD              |
| 2560  | 1280   | 4  | TBD             | TBD              |
| 128   | 1280   | 8  | TBD             | TBD              |
| 256   | 1280   | 8  | TBD             | TBD              |
| 32    | 1280   | 32 | TBD             | TBD              |
| 64    | 1280   | 32 | TBD             | TBD              |
| 128   | 1280   | 32 | TBD             | TBD              |

*Results will be updated after comprehensive benchmarking.* -->

---

## Documentation

Detailed documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | Project structure and layered design |
| [ops_implementation_details.md](docs/ops_implementation_details.md) | Detailed operator implementations |
| [vectorization_and_multicore.md](docs/vectorization_and_multicore.md) | Vectorization and multi-core support |
| [api_reference.md](docs/api_reference.md) | Complete Python API reference |
| [limitations_and_future_work.md](docs/limitations_and_future_work.md) | Known limitations and roadmap |
| [troubleshooting.md](docs/troubleshooting.md) | Common issues and debugging guide |

Chinese versions (`*_zh.md`) are also available.

---

## Citation

If you use this implementation in your research, please cite the original mHC paper:

```bibtex
@article{xie2025mhc,
  title={Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2409.19606},
  year={2024}
}
```

---

## Acknowledgments

- **DeepSeek-AI** for the mHC architecture and paper
- **[AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu)** for the reference CUDA implementation
- **Huawei** for the Ascend platform and CANN toolkit
- **Claude** for vibe coding assistance through 5 days of intensive debugging

---

## License

This project is released under the [MIT License](LICENSE).