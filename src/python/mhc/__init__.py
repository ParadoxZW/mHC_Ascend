"""
mHC (Manifold-Constrained Hyper-Connections) for Huawei Ascend NPUs.

This package provides efficient implementations of the mHC architecture
on Ascend 910B NPUs using CANN and Ascend C.

Example:
    >>> import torch
    >>> import torch_npu
    >>> from mhc import MHCLayer
    >>>
    >>> # Create layer
    >>> layer = MHCLayer(hidden_dim=4096, expansion_rate=4).npu()
    >>>
    >>> # Forward pass
    >>> x = torch.randn(32, 4, 4096, device='npu')
    >>> output = layer(x)
"""

__version__ = "0.1.0"

from .layer import MHCLayer, create_mhc_layer, MHCResidualWrapper, replace_residual_with_mhc
from .ops import (
    sinkhorn_knopp,
    rmsnorm,
    stream_aggregate,
    stream_distribute_mix_add,
)

__all__ = [
    "MHCLayer",
    "MHCResidualWrapper",
    "create_mhc_layer",
    "replace_residual_with_mhc",
    "sinkhorn_knopp",
    "rmsnorm",
    "stream_aggregate",
    "stream_distribute_mix_add",
]
