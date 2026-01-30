#!/usr/bin/env python3
"""
mHC Ascend Layer Benchmark - Compare Ascend C implementation vs PyTorch autograd.

This script benchmarks the mHC layer performance on Ascend NPUs, comparing:
1. Ascend C kernel implementation (mhc module)
2. Naive PyTorch autograd implementation

Uses the same input configurations as the CUDA benchmark (H100 SXM5):
- Static H Path (shared H across batch)
- Dynamic H Path (per-batch H values computed via Equations 7-9)

Usage:
    python3 bench_layer_vs_pytorch.py                 # Single default config
    python3 bench_layer_vs_pytorch.py --all-configs   # All standard configs
    python3 bench_layer_vs_pytorch.py --backward      # Include backward pass
"""

import argparse
import time
import torch
import torch.nn as nn
import torch_npu

# Import Ascend mHC implementation
from mhc import MHCLayer


class NaiveMHCLayer(nn.Module):
    """
    Naive PyTorch implementation of Static H mHC layer.
    Uses PyTorch autograd for backward pass.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.H_pre = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
        self.H_post = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
        H_res_init = alpha_init * torch.randn(expansion_rate, expansion_rate)
        self.H_res = nn.Parameter(H_res_init.float())

    def sinkhorn_knopp(self, M: torch.Tensor) -> torch.Tensor:
        M = torch.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=1, keepdim=True) + self.eps)
            M = M / (M.sum(dim=0, keepdim=True) + self.eps)
        return M

    def rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.rmsnorm_weight

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape

        H_pre_activated = torch.sigmoid(self.H_pre)
        H_post_activated = 2.0 * torch.sigmoid(self.H_post)
        M = self.sinkhorn_knopp(self.H_res)

        x_agg = torch.einsum("i,bic->bc", H_pre_activated, x_expanded)
        x_normed = self.rmsnorm(x_agg)
        y_dist = H_post_activated.view(1, n, 1) * x_normed.view(B, 1, C)
        x_mixed = torch.einsum("ij,bjc->bic", M, x_expanded)

        return x_mixed + y_dist


class NaiveMHCLayerDynamic(nn.Module):
    """
    Naive PyTorch implementation of Dynamic H mHC layer.
    Dynamic H computation as per the paper (Equations 7-9):
    1. Flatten x -> RMSNorm -> Linear projections to get tilde_H values
    2. H_pre = sigmoid(tilde_H_pre), H_post = 2*sigmoid(tilde_H_post)
    3. M = Sinkhorn-Knopp(tilde_H_res)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        n = expansion_rate
        C = hidden_dim

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

        self.phi_pre = nn.Parameter(torch.randn(n * C, n) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(n * C, n) * 0.02)
        self.phi_res = nn.Parameter(torch.randn(n * C, n * n) * 0.02)

        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

    def sinkhorn_knopp(self, M: torch.Tensor) -> torch.Tensor:
        M = torch.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)
        return M

    def rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.rmsnorm_weight

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape

        x_flat = x_expanded.reshape(B, n * C)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + self.eps)
        x_norm = x_flat / rms

        tilde_H_pre = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre
        tilde_H_post = self.alpha_post * (x_norm @ self.phi_post) + self.b_post
        tilde_H_res = (
            self.alpha_res * (x_norm @ self.phi_res).reshape(B, n, n) + self.b_res
        )

        H_pre = torch.sigmoid(tilde_H_pre)
        H_post = 2.0 * torch.sigmoid(tilde_H_post)
        M = self.sinkhorn_knopp(tilde_H_res)

        x_agg = torch.einsum("bi,bic->bc", H_pre, x_expanded)
        x_normed = self.rmsnorm(x_agg)
        y_dist = H_post.unsqueeze(-1) * x_normed.unsqueeze(1)
        x_mixed = torch.einsum("bij,bjc->bic", M, x_expanded)

        return x_mixed + y_dist


class NPUContextCleaner:
    """Context cleaner for NPU to flush cache between benchmark runs."""

    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("npu:0")
        self.device = device
        # NPU doesn't expose L2 cache size, use a reasonable default
        self.flush_size = 64 * 1024 * 1024  # 64MB
        self.flush_tensor = torch.empty(
            self.flush_size // 4, dtype=torch.float32, device=device
        )

    def clear(self):
        torch_npu.npu.synchronize(self.device)
        self.flush_tensor.fill_(1.0)
        torch_npu.npu.synchronize(self.device)


def benchmark_forward(layer, B, n, C, device, cleaner, warmup=10, runs=100):
    """Benchmark forward pass."""
    layer.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            x = torch.randn(B, n, C, device=device, dtype=torch.float32)
            _ = layer(x)
            del x
        torch_npu.npu.synchronize()

        torch_npu.npu.empty_cache()

        # Benchmark runs
        times = []
        for _ in range(runs):
            x = torch.randn(B, n, C, device=device, dtype=torch.float32)

            cleaner.clear()

            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)

            start_event.record()
            _ = layer(x)
            end_event.record()

            torch_npu.npu.synchronize()
            times.append(start_event.elapsed_time(end_event))

            del x

    # Trim outliers (10% from each end)
    times = sorted(times)
    trim = runs // 10
    if trim > 0:
        times = times[trim:-trim]

    avg_ms = sum(times) / len(times)
    return avg_ms


def benchmark_backward(layer, B, n, C, device, cleaner, warmup=10, runs=100):
    """Benchmark forward + backward pass."""
    layer.train()

    # Warmup
    for _ in range(warmup):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        del x, out, loss
    torch_npu.npu.synchronize()

    for p in layer.parameters():
        if p.grad is not None:
            p.grad.zero_()

    torch_npu.npu.empty_cache()

    # Benchmark runs
    times = []
    for _ in range(runs):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)

        cleaner.clear()

        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)

        start_event.record()
        out = layer(x)
        loss = out.sum()
        loss.backward()
        end_event.record()

        torch_npu.npu.synchronize()
        times.append(start_event.elapsed_time(end_event))

        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

        del x, out, loss

    # Trim outliers (10% from each end)
    times = sorted(times)
    trim = runs // 10
    if trim > 0:
        times = times[trim:-trim]

    avg_ms = sum(times) / len(times)
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC Ascend Layer vs PyTorch")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, default=1280, help="Hidden dimension")
    parser.add_argument("--expansion", type=int, default=4, help="Expansion rate (n)")
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=20, help="Sinkhorn iterations"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument(
        "--backward", action="store_true", help="Benchmark backward pass too"
    )
    parser.add_argument(
        "--all-configs", action="store_true", help="Run all standard configs from paper"
    )
    parser.add_argument(
        "--dynamic-only", action="store_true", help="Only benchmark Dynamic H path"
    )
    parser.add_argument(
        "--static-only", action="store_true", help="Only benchmark Static H path"
    )
    args = parser.parse_args()

    device = torch.device("npu:0")

    cleaner = NPUContextCleaner(device)

    # Standard configs from paper Appendix A section A.1
    if args.all_configs:
        configs = [
            (320, 1280, 4),
            (512, 1920, 4),
            (1280, 2560, 4),
            (2560, 1280, 4),
            (128, 1280, 8),
            (256, 1280, 8),
            (32, 1280, 32),
            (64, 1280, 32),
            (128, 1280, 32),
        ]
    else:
        configs = [(args.batch, args.hidden, args.expansion)]

    print("=" * 90)
    print("mHC Ascend Layer Python Benchmark")
    print("=" * 90)
    print(f"Device: Ascend NPU ({torch_npu.npu.get_device_name(0)})")
    print(f"Sinkhorn iterations: {args.sinkhorn_iters}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    # ===========================================================================
    # Static H Path Benchmark
    # ===========================================================================
    if not args.dynamic_only:
        print("Static H Path (shared H across batch)")
        print("-" * 90)
        print(
            f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Ascend (ms)':>14} {'PyTorch (ms)':>14} "
            f"{'Speedup':>10}"
        )
        print("-" * 90)

        for B, C, n in configs:
            try:
                # Ascend C implementation
                ascend_layer = MHCLayer(
                    hidden_dim=C,
                    expansion_rate=n,
                    num_sinkhorn_iters=args.sinkhorn_iters,
                    use_dynamic_h=False,
                ).to(device)

                # Naive PyTorch implementation
                naive_layer = NaiveMHCLayer(
                    hidden_dim=C,
                    expansion_rate=n,
                    sinkhorn_iters=args.sinkhorn_iters,
                ).to(device)

                ascend_time = benchmark_forward(
                    ascend_layer, B, n, C, device, cleaner, args.warmup, args.runs
                )
                pytorch_time = benchmark_forward(
                    naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
                )

                speedup = pytorch_time / ascend_time

                print(
                    f"{B:>6} {C:>6} {n:>4} {ascend_time:>14.3f} {pytorch_time:>14.3f} "
                    f"{speedup:>9.2f}x"
                )

                del ascend_layer, naive_layer
                torch_npu.npu.empty_cache()

            except Exception as e:
                print(f"{B:>6} {C:>6} {n:>4} {'ERROR':>14} {str(e)[:40]}")

        print()

    # ===========================================================================
    # Dynamic H Path Benchmark
    # ===========================================================================
    if not args.static_only:
        print(
            "Dynamic H Path (per-batch H values - paper implementation)"
        )
        print("-" * 90)
        print(
            f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Ascend (ms)':>14} {'PyTorch (ms)':>14} "
            f"{'Speedup':>10}"
        )
        print("-" * 90)

        for B, C, n in configs:
            try:
                # Ascend C implementation with dynamic H
                ascend_layer = MHCLayer(
                    hidden_dim=C,
                    expansion_rate=n,
                    num_sinkhorn_iters=args.sinkhorn_iters,
                    use_dynamic_h=True,
                ).to(device)

                # Naive PyTorch implementation with dynamic H
                naive_layer = NaiveMHCLayerDynamic(
                    hidden_dim=C,
                    expansion_rate=n,
                    sinkhorn_iters=args.sinkhorn_iters,
                ).to(device)

                ascend_time = benchmark_forward(
                    ascend_layer, B, n, C, device, cleaner, args.warmup, args.runs
                )
                pytorch_time = benchmark_forward(
                    naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
                )

                speedup = pytorch_time / ascend_time

                print(
                    f"{B:>6} {C:>6} {n:>4} {ascend_time:>14.3f} {pytorch_time:>14.3f} "
                    f"{speedup:>9.2f}x"
                )

                del ascend_layer, naive_layer
                torch_npu.npu.empty_cache()

            except Exception as e:
                print(f"{B:>6} {C:>6} {n:>4} {'ERROR':>14} {str(e)[:40]}")

        print()

    # ===========================================================================
    # Backward Pass Benchmarks
    # ===========================================================================
    if args.backward:
        # Static H Backward
        if not args.dynamic_only:
            print("Backward Pass Static H (forward + backward)")
            print("-" * 90)
            print(
                f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Ascend (ms)':>14} {'PyTorch (ms)':>14} "
                f"{'Speedup':>10}"
            )
            print("-" * 90)

            for B, C, n in configs:
                try:
                    ascend_layer = MHCLayer(
                        hidden_dim=C,
                        expansion_rate=n,
                        num_sinkhorn_iters=args.sinkhorn_iters,
                        use_dynamic_h=False,
                    ).to(device)

                    naive_layer = NaiveMHCLayer(
                        hidden_dim=C,
                        expansion_rate=n,
                        sinkhorn_iters=args.sinkhorn_iters,
                    ).to(device)

                    # Copy weights for fair comparison
                    naive_layer.H_pre.data = ascend_layer.H_pre.data.clone()
                    naive_layer.H_post.data = ascend_layer.H_post.data.clone()
                    naive_layer.H_res.data = ascend_layer.H_res.data.clone()
                    naive_layer.rmsnorm_weight.data = (
                        ascend_layer.rmsnorm_weight.data.float().clone()
                    )

                    ascend_time = benchmark_backward(
                        ascend_layer, B, n, C, device, cleaner, args.warmup, args.runs
                    )
                    pytorch_time = benchmark_backward(
                        naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
                    )

                    speedup = pytorch_time / ascend_time

                    print(
                        f"{B:>6} {C:>6} {n:>4} {ascend_time:>14.3f} {pytorch_time:>14.3f} "
                        f"{speedup:>9.2f}x"
                    )

                    del ascend_layer, naive_layer
                    torch_npu.npu.empty_cache()

                except Exception as e:
                    print(f"{B:>6} {C:>6} {n:>4} {'ERROR':>14} {str(e)[:40]}")

            print()

        # Dynamic H Backward
        if not args.static_only:
            print(
                "Backward Pass Dynamic H (forward + backward - paper implementation)"
            )
            print("-" * 90)
            print(
                f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Ascend (ms)':>14} {'PyTorch (ms)':>14} "
                f"{'Speedup':>10}"
            )
            print("-" * 90)

            for B, C, n in configs:
                try:
                    ascend_layer = MHCLayer(
                        hidden_dim=C,
                        expansion_rate=n,
                        num_sinkhorn_iters=args.sinkhorn_iters,
                        use_dynamic_h=True,
                    ).to(device)

                    naive_layer = NaiveMHCLayerDynamic(
                        hidden_dim=C,
                        expansion_rate=n,
                        sinkhorn_iters=args.sinkhorn_iters,
                    ).to(device)

                    ascend_time = benchmark_backward(
                        ascend_layer, B, n, C, device, cleaner, args.warmup, args.runs
                    )
                    pytorch_time = benchmark_backward(
                        naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
                    )

                    speedup = pytorch_time / ascend_time

                    print(
                        f"{B:>6} {C:>6} {n:>4} {ascend_time:>14.3f} {pytorch_time:>14.3f} "
                        f"{speedup:>9.2f}x"
                    )

                    del ascend_layer, naive_layer
                    torch_npu.npu.empty_cache()

                except Exception as e:
                    print(f"{B:>6} {C:>6} {n:>4} {'ERROR':>14} {str(e)[:40]}")

            print()

    print("=" * 90)
    print("Benchmark complete.")
    print()


if __name__ == "__main__":
    main()
