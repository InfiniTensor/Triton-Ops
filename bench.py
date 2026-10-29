# -*- coding: utf-8 -*-
"""Performance comparison: native PyTorch vs handwritten Triton (vs NineToothed-generated, a later phase).

Usage: python bench.py
Output: timings and metrics for each implementation across sizes, auto-archived to results/.

Metrics (chosen per operator type):
  - elementwise ops (add/abs etc.) are "memory-bound": the bottleneck is data movement → watch bandwidth in GB/s
    (closer to the memory-bandwidth ceiling is better; ~336 GB/s on an RTX 3060 Laptop)
  - matmul-like ops (addmm etc.) are "compute-bound": the bottleneck is multiply-add → watch TFLOPS
    (= 2·M·N·K ÷ time; closer to the Tensor Core peak is better)

Current scope: PyTorch vs Triton only. The NineToothed column is a reserved hook — once an operator
gets a NineToothed version, add one "ninetoothed" entry to its fns and the table gains a column automatically.
"""
import os
import sys
import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import triton

from kernels import triton_impl

DEVICE = os.environ.get("TB_DEVICE", "cuda")  # defaults to local CUDA; export TB_DEVICE=mlu on the MLU server
if DEVICE == "mlu":
    import torch_mlu  # noqa: F401  registers the "mlu" device
DTYPE = torch.float32
from reference_precision import configure_reference
REFERENCE_PRECISION = configure_reference(DEVICE)
REPEATS = 3
DETAILS = {}


# ---------------------------------------------------------------- Input construction
def _ew_inputs(n):
    """Elementwise ops: 1-D, length n."""
    x = torch.randn(n, dtype=DTYPE, device=DEVICE)
    y = torch.randn(n, dtype=DTYPE, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


def _square_inputs(n):
    """Matmul ops: n×n square matrices + bias."""
    a = torch.randn(n, n, dtype=DTYPE, device=DEVICE)
    b = torch.randn(n, n, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(n, dtype=DTYPE, device=DEVICE)
    out = torch.empty(n, n, dtype=DTYPE, device=DEVICE)
    return bias, a, b, out


def _ew_pos_inputs(n):
    """rsqrt-like ops: 1-D positive inputs (√ is undefined for negatives)."""
    x = torch.rand(n, dtype=DTYPE, device=DEVICE) * 2 + 0.1
    y = torch.randn(n, dtype=DTYPE, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


def _isnan_inputs(n):
    """isnan: float input + boolean output (z is bool, 1 byte per element)."""
    x = torch.randn(n, dtype=DTYPE, device=DEVICE)
    y = torch.randn(n, dtype=DTYPE, device=DEVICE)  # placeholder; keeps the three-return-value signature consistent
    z = torch.empty(n, dtype=torch.bool, device=DEVICE)
    return x, y, z


def _int_inputs(n):
    """Integer ops: int32 inputs (bench counts 4 bytes/element). The upper bound is clamped within INT32_MAX (CNNL limitation)."""
    x = torch.randint(-2**31, 2**31 - 1, (n,), dtype=torch.int32, device=DEVICE)
    y = torch.randint(-2**31, 2**31 - 1, (n,), dtype=torch.int32, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


# ---------------------------------------------------------------- Metrics
def _bandwidth(bytes_per_elem, integer=False):
    def metric(n, ms):
        # Registration uses fp32 byte counts; bool outputs stay one byte.
        width = 4 if integer or DTYPE == torch.float32 else 2
        actual_bytes = bytes_per_elem // 4 * width + bytes_per_elem % 4
        return f"{n * actual_bytes / (ms * 1e-3) / 1e9:6.1f}GB/s"
    return metric


def _tflops(n, ms):
    """Total multiply-adds for square M=N=K=n = 2·n³ floating-point operations."""
    return f"{2 * n ** 3 / (ms * 1e-3) / 1e12:6.2f}TFLOPS"


# ---------------------------------------------------------------- Operator registry
# Each operator: sizes + make_inputs + fns (implementations) + metric (metric policy)
def _add_fns(x, y, z):
    fns = {
        "PyTorch": lambda: torch.add(x, y, out=z),
        "Triton": lambda: triton_impl.add(x, y, z),
    }
    return fns


def _abs_fns(x, y, z):
    fns = {
        "PyTorch": lambda: torch.abs(x, out=z),
        "Triton": lambda: triton_impl.abs(x, z),
    }
    return fns


def _addmm_fns(bias, a, b, out):
    return {
        "PyTorch": lambda: torch.addmm(bias, a, b, out=out),
        "Triton": lambda: triton_impl.addmm(bias, a, b, out),
    }


def _neg_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.neg(x, out=z),
        "Triton": lambda: triton_impl.neg(x, z),
    }


def _cos_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.cos(x, out=z),
        "Triton": lambda: triton_impl.cos(x, z),
    }


def _sin_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.sin(x, out=z),
        "Triton": lambda: triton_impl.sin(x, z),
    }


def _exp_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.exp(x, out=z),
        "Triton": lambda: triton_impl.exp(x, z),
    }


def _rsqrt_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.rsqrt(x, out=z),
        "Triton": lambda: triton_impl.rsqrt(x, z),
    }


def _sigmoid_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.sigmoid(x, out=z),
        "Triton": lambda: triton_impl.sigmoid(x, z),
    }


def _tanh_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.tanh(x, out=z),
        "Triton": lambda: triton_impl.tanh(x, z),
    }


def _relu_fns(x, y, z):
    return {
        # torch.relu doesn't accept out=; clamp_min(x, 0) is the same math and supports out
        "PyTorch": lambda: torch.clamp_min(x, 0, out=z),
        "Triton": lambda: triton_impl.relu(x, z),
    }


def _silu_fns(x, y, z):
    return {
        # F.silu is ATen's fused kernel (single pass) but doesn't accept out=; it allocates its own return value
        "PyTorch": lambda: torch.nn.functional.silu(x),
        "Triton": lambda: triton_impl.silu(x, z),
    }


def _gelu_fns(x, y, z):
    return {
        # keep the same formula as the Triton version (tanh approximation); F.gelu doesn't accept out=
        "PyTorch": lambda: torch.nn.functional.gelu(x, approximate="tanh"),
        "Triton": lambda: triton_impl.gelu(x, z),
    }


def _isnan_fns(x, y, z):
    return {
        # torch.isnan doesn't accept out=; it allocates the boolean output itself
        "PyTorch": lambda: torch.isnan(x),
        "Triton": lambda: triton_impl.isnan(x, z),
    }


def _isinf_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.isinf(x),
        "Triton": lambda: triton_impl.isinf(x, z),
    }


def _bitwise_not_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.bitwise_not(x, out=z),
        "Triton": lambda: triton_impl.bitwise_not(x, z),
    }


def _sub_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.sub(x, y, out=z),
        "Triton": lambda: triton_impl.sub(x, y, z),
    }


def _mul_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.mul(x, y, out=z),
        "Triton": lambda: triton_impl.mul(x, y, z),
    }


def _div_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.div(x, y, out=z),
        "Triton": lambda: triton_impl.div(x, y, z),
    }


def _pow_inputs(n):
    """pow: benchmark the "square special case" exactly as rms_norm uses it — the exponent is the Python scalar 2.0,
    the Triton side takes the x*x fast path (kernels/triton_impl.py::square), 8 bytes/element.
    The general-exponent path (tensor y) is covered by tests/test_correctness.py::test_pow_triton."""
    x = torch.rand(n, dtype=DTYPE, device=DEVICE) * 2 + 0.1
    z = torch.empty_like(x)
    return x, 2.0, z


def _pow_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.pow(x, y, out=z),
        "Triton": lambda: triton_impl.pow(x, y, z),
    }


def _bool_out_inputs(n):
    """Comparison ops (eq/ne/gt/...): two float inputs + boolean output, 9 bytes/element."""
    x = torch.randn(n, dtype=DTYPE, device=DEVICE)
    y = torch.randn(n, dtype=DTYPE, device=DEVICE)
    z = torch.empty(n, dtype=torch.bool, device=DEVICE)
    return x, y, z


def _eq_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.eq(x, y, out=z),
        "Triton": lambda: triton_impl.eq(x, y, z),
    }


def _ne_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.ne(x, y, out=z),
        "Triton": lambda: triton_impl.ne(x, y, z),
    }


def _gt_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.gt(x, y, out=z),
        "Triton": lambda: triton_impl.gt(x, y, z),
    }


def _ge_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.ge(x, y, out=z),
        "Triton": lambda: triton_impl.ge(x, y, z),
    }


def _lt_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.lt(x, y, out=z),
        "Triton": lambda: triton_impl.lt(x, y, z),
    }


def _le_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.le(x, y, out=z),
        "Triton": lambda: triton_impl.le(x, y, z),
    }


def _bitwise_and_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.bitwise_and(x, y, out=z),
        "Triton": lambda: triton_impl.bitwise_and(x, y, z),
    }


def _bitwise_or_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.bitwise_or(x, y, out=z),
        "Triton": lambda: triton_impl.bitwise_or(x, y, z),
    }


def _clamp_fns(x, y, z):
    return {
        "PyTorch": lambda: torch.clamp(x, -0.5, 0.5, out=z),
        "Triton": lambda: triton_impl.clamp(x, z, -0.5, 0.5),
    }


def _rms_norm_inputs(n):
    """rms_norm: (M, 4096) 2-D input; n = total elements (M = n ÷ 4096)."""
    N = 4096
    M = n // N
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, dtype=DTYPE, device=DEVICE)
    y = torch.empty_like(x)
    return x, w, y


def _rms_norm_fns(x, w, y):
    N = x.shape[1]
    return {
        # F.rms_norm is an ATen fused kernel and doesn't accept out=
        "PyTorch": lambda: torch.nn.functional.rms_norm(x, (N,), w, 1e-6),
        "Triton": lambda: triton_impl.rms_norm(x, w, y),
    }


def _softmax_inputs(n):
    """softmax: (M, 4096) 2-D input; n = total elements."""
    N = 4096
    M = n // N
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    y = torch.empty_like(x)
    return x, y


def _softmax_fns(x, y):
    return {
        "PyTorch": lambda: torch.softmax(x, dim=-1),
        "Triton": lambda: triton_impl.softmax(x, y),
    }


def _layer_norm_inputs(n):
    """layer_norm: (M, 4096) input + per-channel w, b; n = total elements."""
    N = 4096
    M = n // N
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, dtype=DTYPE, device=DEVICE)
    b = torch.randn(N, dtype=DTYPE, device=DEVICE)
    y = torch.empty_like(x)
    return x, w, b, y


def _layer_norm_fns(x, w, b, y):
    N = x.shape[1]
    return {
        "PyTorch": lambda: torch.nn.functional.layer_norm(x, (N,), w, b),
        "Triton": lambda: triton_impl.layer_norm(x, w, b, y),
    }


def _mm_fns(bias, a, b, out):
    return {
        "PyTorch": lambda: torch.mm(a, b, out=out),
        "Triton": lambda: triton_impl.mm(a, b, out),
    }


def _bmm_inputs(n):
    """bmm: n = batch count; each batch is a 512×512 square matrix."""
    a = torch.randn(n, 512, 512, dtype=DTYPE, device=DEVICE)
    b = torch.randn(n, 512, 512, dtype=DTYPE, device=DEVICE)
    out = torch.empty(n, 512, 512, dtype=DTYPE, device=DEVICE)
    return a, b, out


def _bmm_fns(a, b, out):
    return {
        "PyTorch": lambda: torch.bmm(a, b, out=out),
        "Triton": lambda: triton_impl.bmm(a, b, out),
    }


def _bmm_tflops(n, ms):
    """Total multiply-adds for n batches of 512³ matrices = n × 2·512³ FLOPs."""
    return f"{n * 2 * 512 ** 3 / (ms * 1e-3) / 1e12:6.2f}TFLOPS"


def _swiglu_fns(x, y, z):
    return {
        # PyTorch has no ready-made SwiGLU; it can only chain F.silu(x) * y as two kernels —
        # writing and re-reading the intermediate — exactly what fusion aims to beat
        "PyTorch": lambda: torch.nn.functional.silu(x) * y,
        "Triton": lambda: triton_impl.swiglu(x, y, z),
    }


def _rope_inputs(n):
    """RoPE: (M, 128) 2-D input (head_dim=128); cos/sin are (M, 64) each; n = total elements of x."""
    D = 128
    M = n // D
    x = torch.randn(M, D, dtype=DTYPE, device=DEVICE)
    angles = torch.randn(M, D // 2, dtype=DTYPE, device=DEVICE)
    cos, sin = angles.cos(), angles.sin()
    out = torch.empty_like(x)
    return x, cos, sin, out


def _rope_fns(x, cos, sin, out):
    d2 = x.shape[1] // 2

    def pytorch_rope():
        x1, x2 = x[:, :d2], x[:, d2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return {
        # PyTorch can only split into two halves + four elementwise kernels + cat, moving intermediates many times
        "PyTorch": pytorch_rope,
        "Triton": lambda: triton_impl.rotary_position_embedding(x, cos, sin, out),
    }


def _dropout_fns(x, y, z):
    return {
        # torch.dropout(x, p, training=True): same semantics as the Triton version (zeroing + 1/(1-p) scaling)
        "PyTorch": lambda: torch.dropout(x, 0.25, True),
        "Triton": lambda: triton_impl.dropout(x, z, p=0.25, seed=42),
    }


def _sdpa_inputs(n):
    """SDPA: n = sequence length (M=N); fixed B=2, H=4, D=64."""
    B, H, D = 2, 4, 64
    q = torch.randn(B, H, n, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, H, n, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, H, n, D, dtype=DTYPE, device=DEVICE)
    out = torch.empty_like(q)
    return q, k, v, out


def _sdpa_fns(q, k, v, out):
    return {
        # PyTorch's SDPA is a fused flash/mem-efficient kernel — the performance ceiling for this operator
        "PyTorch": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
        "Triton": lambda: triton_impl.scaled_dot_product_attention(q, k, v, out),
    }


def _sdpa_tflops(n, ms):
    """FLOPs = the two matmuls QK^T and P·V: 2 × (2·B·H·M·N·D), with B=2, H=4, D=64, M=N=n."""
    return f"{4 * 2 * 4 * n * n * 64 / (ms * 1e-3) / 1e12:6.2f}TFLOPS"


def _conv2d_inputs(n):
    """conv2d: n = input side length (H=W); fixed N=2, C_in=32, C_out=64, K=3, pad=1 (output keeps size)."""
    N, C_IN, C_OUT, K = 2, 32, 64, 3
    x = torch.randn(N, C_IN, n, n, dtype=DTYPE, device=DEVICE)
    w = torch.randn(C_OUT, C_IN, K, K, dtype=DTYPE, device=DEVICE)
    y = torch.empty(N, C_OUT, n, n, dtype=DTYPE, device=DEVICE)
    return x, w, y


def _conv2d_fns(x, w, y):
    return {
        # cuDNN convolution is the performance ceiling here (highly optimized paths like implicit GEMM/Winograd)
        "PyTorch": lambda: torch.nn.functional.conv2d(x, w, padding=1),
        "Triton": lambda: triton_impl.conv2d(x, w, y, stride=1, padding=1),
    }


def _conv2d_tflops(n, ms):
    """FLOPs = 2·N·C_out·H·W·C_in·K², with N=2, C_in=32, C_out=64, K=3, H=W=n."""
    return f"{2 * 2 * 64 * n * n * 32 * 9 / (ms * 1e-3) / 1e12:6.2f}TFLOPS"


def _max_pool2d_inputs(n):
    """max_pool2d: n = input side length (H=W); fixed N=2, C=32, K=2, stride=2."""
    x = torch.randn(2, 32, n, n, dtype=DTYPE, device=DEVICE)
    y = torch.empty(2, 32, n // 2, n // 2, dtype=DTYPE, device=DEVICE)
    return x, y


def _max_pool2d_fns(x, y):
    return {
        "PyTorch": lambda: torch.nn.functional.max_pool2d(x, 2),
        "Triton": lambda: triton_impl.max_pool2d(x, y, kernel_size=2),
    }


def _mp2d_bandwidth(n, ms):
    """Reads the whole input + writes a 1/4-size output; N=2, C=32, K=2, H=W=n."""
    bytes_total = (2 * 32 * n * n + 2 * 32 * (n // 2) ** 2) * (4 if DTYPE == torch.float32 else 2)
    return f"{bytes_total / (ms * 1e-3) / 1e9:6.1f}GB/s"


OPS = {
    "add": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _add_fns, "metric": _bandwidth(3 * 4)},
    "sub": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _sub_fns, "metric": _bandwidth(3 * 4)},
    "mul": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _mul_fns, "metric": _bandwidth(3 * 4)},
    "div": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _div_fns, "metric": _bandwidth(3 * 4)},
    "pow": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _pow_inputs, "fns": _pow_fns, "metric": _bandwidth(2 * 4)},  # square special case: read x + write z
    # comparison policy: read 4+4B + write bool 1B = 9 bytes/element
    "eq": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _eq_fns, "metric": _bandwidth(4 + 4 + 1)},
    "ne": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _ne_fns, "metric": _bandwidth(4 + 4 + 1)},
    "gt": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _gt_fns, "metric": _bandwidth(4 + 4 + 1)},
    "ge": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _ge_fns, "metric": _bandwidth(4 + 4 + 1)},
    "lt": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _lt_fns, "metric": _bandwidth(4 + 4 + 1)},
    "le": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
           "make_inputs": _bool_out_inputs, "fns": _le_fns, "metric": _bandwidth(4 + 4 + 1)},
    "abs": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _abs_fns, "metric": _bandwidth(2 * 4)},
    "neg": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _neg_fns, "metric": _bandwidth(2 * 4)},
    "cos": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _cos_fns, "metric": _bandwidth(2 * 4)},
    "sin": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _sin_fns, "metric": _bandwidth(2 * 4)},
    "exp": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
            "make_inputs": _ew_inputs, "fns": _exp_fns, "metric": _bandwidth(2 * 4)},
    "rsqrt": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
              "make_inputs": _ew_pos_inputs, "fns": _rsqrt_fns, "metric": _bandwidth(2 * 4)},
    "sigmoid": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
                "make_inputs": _ew_inputs, "fns": _sigmoid_fns, "metric": _bandwidth(2 * 4)},
    "tanh": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
             "make_inputs": _ew_inputs, "fns": _tanh_fns, "metric": _bandwidth(2 * 4)},
    "relu": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
             "make_inputs": _ew_inputs, "fns": _relu_fns, "metric": _bandwidth(2 * 4)},
    "silu": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
             "make_inputs": _ew_inputs, "fns": _silu_fns, "metric": _bandwidth(2 * 4)},
    "gelu": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
             "make_inputs": _ew_inputs, "fns": _gelu_fns, "metric": _bandwidth(2 * 4)},
    # bandwidth policy: read fp32(4B) + write bool(1B) = 5 bytes/element
    "isnan": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
              "make_inputs": _isnan_inputs, "fns": _isnan_fns, "metric": _bandwidth(4 + 1)},
    "isinf": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
              "make_inputs": _isnan_inputs, "fns": _isinf_fns, "metric": _bandwidth(4 + 1)},
    "bitwise_not": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
                    "make_inputs": _int_inputs, "fns": _bitwise_not_fns, "metric": _bandwidth(2 * 4, integer=True)},
    "bitwise_and": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
                    "make_inputs": _int_inputs, "fns": _bitwise_and_fns, "metric": _bandwidth(3 * 4, integer=True)},
    "bitwise_or": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
                   "make_inputs": _int_inputs, "fns": _bitwise_or_fns, "metric": _bandwidth(3 * 4, integer=True)},
    "clamp": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
              "make_inputs": _ew_inputs, "fns": _clamp_fns, "metric": _bandwidth(2 * 4)},
    # reduction policy: read x + write y (w has only N elements, negligible) = 8 bytes/element
    "rms_norm": {"sizes": [524_288, 4_194_304, 16_777_216, 50_331_648],
                 "make_inputs": _rms_norm_inputs, "fns": _rms_norm_fns, "metric": _bandwidth(2 * 4)},
    "softmax": {"sizes": [524_288, 4_194_304, 16_777_216, 50_331_648],
                "make_inputs": _softmax_inputs, "fns": _softmax_fns, "metric": _bandwidth(2 * 4)},
    "layer_norm": {"sizes": [524_288, 4_194_304, 16_777_216, 50_331_648],
                   "make_inputs": _layer_norm_inputs, "fns": _layer_norm_fns, "metric": _bandwidth(2 * 4)},
    "addmm": {"sizes": [512, 1024, 2048, 4096],
              "make_inputs": _square_inputs, "fns": _addmm_fns, "metric": _tflops},
    "mm": {"sizes": [512, 1024, 2048, 4096],
           "make_inputs": _square_inputs, "fns": _mm_fns, "metric": _tflops},
    "bmm": {"sizes": [4, 16, 64, 128],
            "make_inputs": _bmm_inputs, "fns": _bmm_fns, "metric": _bmm_tflops},
    # swiglu policy: read x + read gate + write z = 3 fp32 tensors = 12 bytes/element
    "swiglu": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
               "make_inputs": _ew_inputs, "fns": _swiglu_fns, "metric": _bandwidth(3 * 4)},
    # RoPE policy: read x(4B) + write out(4B) + half a row each of cos/sin (4B/element total) = 12 bytes/element
    "rotary_position_embedding": {"sizes": [524_288, 4_194_304, 16_777_216, 50_331_648],
                                  "make_inputs": _rope_inputs, "fns": _rope_fns, "metric": _bandwidth(3 * 4)},
    # dropout policy: read x + write y = 8 bytes/element (p=0.25; Philox numbers are generated in-kernel and cost no bandwidth)
    "dropout": {"sizes": [100_000, 1_000_000, 10_000_000, 50_000_000],
                "make_inputs": _ew_inputs, "fns": _dropout_fns, "metric": _bandwidth(2 * 4)},
    # SDPA is compute-bound: watch TFLOPS (see _sdpa_tflops); sizes sweep the sequence length
    "scaled_dot_product_attention": {"sizes": [128, 512, 1024, 2048],
                                     "make_inputs": _sdpa_inputs, "fns": _sdpa_fns,
                                     "metric": _sdpa_tflops},
    # conv2d is compute-bound: watch TFLOPS (see _conv2d_tflops); sizes sweep the input side length
    "conv2d": {"sizes": [32, 64, 128, 224],
               "make_inputs": _conv2d_inputs, "fns": _conv2d_fns, "metric": _conv2d_tflops},
    # max_pool2d is memory-bound: watch bandwidth (see _mp2d_bandwidth); sizes sweep the input side length
    "max_pool2d": {"sizes": [128, 256, 512, 1024],
                   "make_inputs": _max_pool2d_inputs, "fns": _max_pool2d_fns,
                   "metric": _mp2d_bandwidth},
}


def _fmt_time(ms):
    return f"{ms:7.3f}ms" if ms >= 1 else f"{ms*1000:7.1f}us"


def _input_meta(t):
    """For archiving: tensors record shape/stride/dtype; scalar parameters (e.g. pow's exponent 2.0) are recorded separately to avoid .shape crashing."""
    if torch.is_tensor(t):
        return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
    return {"scalar": t, "dtype": type(t).__name__}


def bench_one(op, n):
    spec = OPS[op]
    inputs = spec["make_inputs"](n)
    fns = spec["fns"](*inputs)
    samples = {name: [] for name in fns}
    for repeat in range(REPEATS):
        names = list(fns)
        if repeat % 2:
            names.reverse()
        for name in names:
            samples[name].append(triton.testing.do_bench(
                fns[name], warmup=25, rep=100, quantiles=[0.2, 0.5, 0.8]))
    DETAILS[n] = {"inputs": [_input_meta(t) for t in inputs],
                  "rounds_p20_p50_p80_ms": samples}
    return {name: statistics.median(s[1] for s in rounds) for name, rounds in samples.items()}


def _env_info():
    """Self-describing header for archives: anyone (or any AI) reading this file can fully reconstruct the experimental conditions."""
    import platform
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
        ).stdout.strip()
        commit_str = commit + ("(+uncommitted changes)" if dirty else "")
    except Exception:
        commit_str = "unknown"

    info = {
        "device": torch.mlu.get_device_name(0) if DEVICE == "mlu" else torch.cuda.get_device_name(0),
        "os": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "code version": commit_str,
        "reference precision": json.dumps(REFERENCE_PRECISION, ensure_ascii=False),
        "source SHA256": "; ".join(f"{p}: {hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()}"
                                for p in ("kernels/triton_impl.py", "kernels/mlu_matmul.py", "reference_precision.py", "bench.py")),
    }
    if DEVICE == "mlu":
        info["torch_mlu"] = torch_mlu.__version__
    backend = torch.backends.mlu if DEVICE == "mlu" else torch.backends.cuda
    info["matmul.allow_tf32"] = backend.matmul.allow_tf32
    if DEVICE == "mlu":
        info["cnnl.allow_tf32"] = torch.backends.cnnl.allow_tf32
    else:
        info["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    return info


def _bench_op(op):
    """Run one operator at all sizes; return (implementation names, {size: {impl: ms}})."""
    spec = OPS[op]
    table = {}
    for n in spec["sizes"]:
        table[n] = bench_one(op, n)
    names = list(table[spec["sizes"][0]].keys())  # columns adapt to the number of implementations
    return names, table


def _save_op_result(op, runs):
    """Each operator archives independently: results/bench_<op>.md, overwritten each run (history lives in git)."""
    spec = OPS[op]
    env = _env_info()
    lines = [
        f"# {op} performance test",
        "",
        f"- time: {datetime.now():%Y-%m-%d %H:%M}",
        *[f"- {k}: {v}" for k, v in env.items()],
        f"- sizes: {', '.join(f'{n:,}' for n in spec['sizes'])}",
        f"- method: do_bench, warmup=25ms/rep=100ms; {REPEATS} rounds with interleaved implementation order; p20/p50/p80 per round, table takes the median of per-round p50",
        "- inputs: implementations share the same inputs per size, random seed 20261022; first JIT/autotune compilation excluded; device operations triggered by the Python wrapper are included",
        "- output allocation: PyTorch interfaces that support out= reuse the output, others allocate their own; Triton reuses the output. Results compare these call paths and are not uniformly labeled pure kernel time",
        "- metrics: elementwise ops use bandwidth GB/s = bytes moved ÷ time; matmuls use TFLOPS = 2·N³ ÷ time",
        "- note: bandwidth is the effective bandwidth computed from the algorithm's minimal read+write volume; see per-round quantiles for variance — no fixed noise bound is assumed; performance results do not substitute for correctness acceptance",
        "",
    ]
    global DTYPE
    original_dtype = DTYPE
    for dtype, names, table, detail in runs:
        DTYPE = dtype
        actual_dtype = "torch.int32" if op.startswith("bitwise_") else str(dtype)
        lines += [f"**input dtype: {actual_dtype}**", "",
                  "| size | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for n, results in table.items():
            cells = [f"{_fmt_time(ms)} {spec['metric'](n, ms)}" for ms in results.values()]
            lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
        lines += ["", "<details><summary>Input shapes, strides, and per-round p20/p50/p80 (ms)</summary>",
                  "", "```json", json.dumps(detail, ensure_ascii=False, indent=2), "```", "", "</details>", ""]
    DTYPE = original_dtype

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"bench_{op}.md"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def main():
    # usage: python bench.py runs everything; python bench.py addmm abs runs only the given ops
    global DTYPE, REPEATS
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="*")
    parser.add_argument("--dtype", choices=["float32", "float16", "both"], default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    REPEATS = args.repeats
    ops = args.ops or list(OPS)
    unknown = [op for op in ops if op not in OPS]
    if unknown:
        print(f"unknown operators: {unknown}; registered: {list(OPS)}")
        sys.exit(1)

    env = _env_info()
    print(f"device: {env['device']}  dtype: {args.dtype}", flush=True)
    dtypes = [torch.float32, torch.float16] if args.dtype == "both" else [getattr(torch, args.dtype)]
    for op in ops:
        runs = []
        for dtype in (dtypes[:1] if op.startswith("bitwise_") else dtypes):
            DTYPE = dtype
            torch.manual_seed(20261022)
            DETAILS.clear()
            names, table = _bench_op(op)
            runs.append((dtype, names, table, dict(DETAILS)))
            print(f"\n=== operator: {op} {dtype} ===", flush=True)
            for n, results in table.items():
                print(n, results, flush=True)
        print(f"archived: {_save_op_result(op, runs)}", flush=True)


if __name__ == "__main__":
    main()
