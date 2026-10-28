# -*- coding: utf-8 -*-
"""Handwritten Triton operator implementations (baseline reference group).

Each operator has two parts:
  - one @triton.jit kernel: the core logic running in parallel on the device
  - one Python wrapper: computes the grid and launches the kernel

The standard Triton pattern (see vector add):
  1. program_id(0) —— "which parallel program am I"
  2. arange + offsets —— "which slice of data I own"
  3. mask —— "bounds guard when the tail is smaller than one block"
  4. load → compute → store
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # tl.math has no tanh; take it from libdevice directly; portable path (CUDA/MLU both work)

# Autotune configs for elementwise kernels: BLOCK_SIZE x num_warps chosen per n_elements.
# Motivation (measured 2026-10-22): the CUDA-tuned BLOCK=1024 underperforms ~5x on MLU
# (add @ n=50M: 318 GB/s vs 1576 GB/s at BLOCK=8192/warps=1~2; CNNL 2268 GB/s).
# The platforms prefer different shapes (CUDA: small blocks + more warps / MLU: large blocks + few warps); hardcoding either side hurts the other,
# so leave it to autotune: each platform measures every config on the first encounter of an n, then hits the cache.
# 2026-10-22 P2-6 A/B (MLU, n=50M): bigger BLOCK is better — add 8192→32768: 1572→2215 GB/s (+41%),
# exp 1035→1756 (+70%); num_warps=2 is unsupported by the MLU backend (warns, then falls back to 1). Hence the MLU-only 16384/32768.
# CUDA keeps the original 6 configs: BLOCK=32768+warps=1 makes ptxas extremely slow (same lesson as P2-5), so the split follows TB_DEVICE.
_IS_MLU = os.environ.get("TB_DEVICE", "cuda") == "mlu"
_EW_CONFIGS = [triton.Config({"BLOCK_SIZE": b}, num_warps=w) for b in (1024, 4096, 8192) for w in (1, 4)] + (
    [triton.Config({"BLOCK_SIZE": b}, num_warps=1) for b in (16384, 32768)] if _IS_MLU else [])


# ---------------------------------------------------------------- Fast transcendental implementations
# 2026-10-22 P1-3 A/B measurements (MLU590-M9, n=50M fp32, BLOCK=8192):
#   copy ceiling 1284 GB/s; tl.sigmoid 851 GB/s (already bandwidth-bound; exp2/fast_dividef was slower at 731);
#   libdevice.tanh 364 GB/s; exp2 + plain division reaches 627 GB/s.
# Conclusion: only tanh is worth replacing (gelu's tanh approximation benefits too); sigmoid/silu keep tl.sigmoid.
# Use plain `/` for division (measured slightly faster and more accurate than libdevice.fast_dividef).
@triton.jit
def _fast_tanh(x):
    """tanh(x) = 1 - 2/(e^{2x}+1), via exp2 + plain division; the clamp prevents exp overflow."""
    t = x * 2.885390081777927                    # 2x·log2(e)
    t = tl.minimum(tl.maximum(t, -30.0), 30.0)   # 2^±30 stays within fp32, and tanh has long saturated to ±1
    return 1.0 - 2.0 / (tl.exp2(t) + 1.0)


# ---------------------------------------------------------------- P2-5 search space
# The MLU backend's dot goes through SIMD and prefers large tiles + few warps; CUDA keeps the original 6 configs — bringing MLU-specific configs
# into CUDA makes ptxas compile combos like 128×128/warps=1 extremely slowly (>10 minutes once, measured).
# Decided at import time via TB_DEVICE (bench.py / pytest export it before launch).
# (_IS_MLU is defined above at _EW_CONFIGS.)

_BMM_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
] + ([
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=1, num_stages=2),
] if _IS_MLU else [])

_SDPA_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
] + ([
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=1, num_stages=2),
] if _IS_MLU else [])


# ---------------------------------------------------------------- add
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                      # which block am I
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # indices of the elements I own
    mask = offsets < n_elements                      # bounds guard
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x, y, z):
    """z = x + y; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _add_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- abs
# Unary op: y = |x|. One fewer input than add; otherwise the exact same pattern.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _abs_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.abs(x), mask=mask)


def abs(x, y):
    """y = |x|; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _abs_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- neg
# Unary op: y = -x. Isomorphic to abs, just with -x instead of tl.abs(x).
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _neg_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, -x, mask=mask)


def neg(x, y):
    """y = -x; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _neg_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- cos
# Unary op: y = cos(x). tl.cos bottoms out in libdevice, matching PyTorch's precision.
# Note: tl.cos only accepts fp32/fp64 (libdevice has no fp16 version), so fp16 inputs are
# first upcast via .to(tl.float32) for the math and then cast back — PyTorch's fp16 ops
# do the same internally, so both sides share the precision policy and the 1e-2 tolerance holds.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _cos_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.cos(x.to(tl.float32))                # upcast to fp32 for the math
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)  # cast back to the original dtype on store


def cos(x, y):
    """y = cos(x); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _cos_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- sin
# Unary op: y = sin(x). Like cos, tl.sin only takes fp32/fp64; fp16 is upcast then downcast.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _sin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sin(x.to(tl.float32))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def sin(x, y):
    """y = sin(x); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _sin_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- exp
# Unary op: y = e^x. tl.exp likewise only takes fp32/fp64; fp16 is upcast then downcast.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _exp_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x.to(tl.float32))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def exp(x, y):
    """y = e^x; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _exp_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- rsqrt
# Unary op: y = 1/√x. tl.rsqrt likewise only takes fp32/fp64; fp16 is upcast then downcast.
# Math constraint: x must be non-negative (√ of a negative is NaN); build test data with positives.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _rsqrt_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.rsqrt(x.to(tl.float32))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def rsqrt(x, y):
    """y = 1/√x; inputs are 1-D contiguous tensors with non-negative elements."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _rsqrt_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- sigmoid
# Unary op: y = 1 / (1 + e^-x). The core building block of silu (x·sigmoid(x)).
# 2026-10-22 P1-3 A/B: tl.sigmoid on MLU already reaches 851 GB/s (copy ceiling 1284),
# switching to exp2/fast_dividef was slower (731), so tl.sigmoid is kept unchanged.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sigmoid(x.to(tl.float32))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def sigmoid(x, y):
    """y = 1 / (1 + e^-x); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _sigmoid_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- tanh
# Unary op: y = (e^x - e^-x) / (e^x + e^-x), squashing reals into (-1, 1).
# Used inside gelu's tanh approximation.
# 2026-10-22 P1-3: libdevice.tanh reaches only 364 GB/s on MLU; switching to _fast_tanh
# (exp2 + plain division) gives 627 GB/s (1.72×); maxerr ~2e-7, within the fp32 1e-6 tolerance.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = _fast_tanh(x.to(tl.float32))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def tanh(x, y):
    """y = tanh(x); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _tanh_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- relu
# Unary op: y = max(x, 0); negatives clamp to 0 and positives pass through.
# Pure comparison, no libdevice involved; fp16 computes directly without upcasting.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0), mask=mask)


def relu(x, y):
    """y = max(x, 0); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _relu_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- silu
# Unary composite op: y = x * sigmoid(x) = x / (1 + e^-x). The activation in Llama-family FFNs.
# Two steps fused into one kernel (sigmoid, then multiply by x), moving data only once —
# this is "operator fusion": PyTorch would round-trip the intermediate through memory; fusion avoids that.
# The sigmoid part is upcast to fp32 as usual; the multiply also happens in fp32, then everything is cast back.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    y = x32 * tl.sigmoid(x32)                     # P1-3 A/B: sigmoid is bandwidth-bound; keep tl.sigmoid
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def silu(x, y):
    """y = x * sigmoid(x); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _silu_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- gelu
# Unary composite op (tanh approximation):
#   y = 0.5 * x * (1 + tanh( √(2/π) * (x + 0.044715 * x³) ))
# Llama doesn't use it (it uses silu), but BERT/GPT-family models do. The exact version needs erf; the tanh approximation
# is faster and accurate enough — the mainstream engineering choice; verify against approximate='tanh'.
SQRT_2_OVER_PI = 0.7978845608028654  # √(2/π), precomputed constant


@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                 SQRT_2_OVER_PI: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    inner = SQRT_2_OVER_PI * (x32 + 0.044715 * x32 * x32 * x32)
    y = 0.5 * x32 * (1.0 + _fast_tanh(inner))     # P1-3: tanh goes through the exp2 fast path
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def gelu(x, y):
    """y = gelu(x) (tanh approximation); inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _gelu_kernel[grid](x, y, n, SQRT_2_OVER_PI=SQRT_2_OVER_PI)
    return y


# ---------------------------------------------------------------- isnan
# Unary predicate: y = (is x NaN?); output is a boolean tensor.
# Trick: NaN is the only value that does not equal itself (IEEE 754), hence x != x.
# Pure comparison; fp16 computes directly without upcasting. Reads floats, writes booleans; the dtypes are independent.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _isnan_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x != x, mask=mask)


def isnan(x, y):
    """y = isnan(x); x is a 1-D contiguous float tensor, y a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _isnan_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- isinf
# Unary predicate: y = (is x ±∞?); output is a boolean tensor.
# Trick: |x| == +∞ covers both positive and negative infinity.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _isinf_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.abs(x) == float("inf"), mask=mask)


def isinf(x, y):
    """y = isinf(x); x is a 1-D contiguous float tensor, y a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _isinf_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- bitwise_not
# Integer op: y = ~x (bitwise NOT; flips every bit).
# Only meaningful for integers (floats have no "bitwise"); inputs are int32/int64.
# Pure bit manipulation — a single instruction, no math-library dependency.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _bitwise_not_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, ~x, mask=mask)


def bitwise_not(x, y):
    """y = ~x; inputs are 1-D contiguous integer tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _bitwise_not_kernel[grid](x, y, n)
    return y


# ---------------------------------------------------------------- sub
# Binary op: z = x - y. Isomorphic to add, with - instead of +.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _sub_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x - y, mask=mask)


def sub(x, y, z):
    """z = x - y; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _sub_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- mul
# Binary op: z = x * y. The gate multiply in SwiGLU is exactly this combined with silu.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _mul_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x * y, mask=mask)


def mul(x, y, z):
    """z = x * y; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _mul_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- div
# Binary op: z = x / y. Division is an order of magnitude slower than multiply-add, but elementwise ops
# remain bandwidth-bound. When y ≈ 0 the result can be huge or inf; matching PyTorch's behavior is enough.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _div_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x / y, mask=mask)


def div(x, y, z):
    """z = x / y; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _div_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- square (special case of pow)
# The squaring in rms_norm: z = x*x. Pure memory traffic (read 4B + write 4B = 8B/element),
# an order of magnitude faster than generic libdevice.pow (reads 12B + transcendental throughput) (2026-10-22 P0-1).
# Exported standalone for direct callers; pow's "scalar exponent 2.0" fast path also goes through here.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _square_kernel(x_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x * x, mask=mask)


def square(x, z):
    """z = x*x; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _square_kernel[grid](x, z, n)
    return z


# ---------------------------------------------------------------- pow
# Binary op: z = x ** y (general exponent; rms_norm's squaring is the special case y=2).
# libdevice.pow only takes fp32/fp64; fp16 is upcast then downcast.
# Math constraint: negative base with non-integer exponent = NaN; use positive bases in test data.
# fast path: when the exponent is scalar 2.0 (including a single-element tensor), use square, z=x*x; general exponents still go through libdevice.pow.
# No "scan the whole tensor for all-2s" check: that would need an extra pass over y plus a sync, costing more than it saves.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _pow_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = libdevice.pow(x.to(tl.float32), y.to(tl.float32))
    tl.store(z_ptr + offsets, z.to(z_ptr.dtype.element_ty), mask=mask)


def _is_scalar_two(y):
    """Whether the exponent is scalar 2.0: free for Python numbers; a single-element tensor needs one value read back (microsecond sync)."""
    if isinstance(y, (int, float)):
        return float(y) == 2.0
    if torch.is_tensor(y) and y.numel() == 1:
        return float(y.item()) == 2.0
    return False


def pow(x, y, z):
    """z = x ** y; inputs are 1-D contiguous tensors whose elements must be positive."""
    if _is_scalar_two(y):
        return square(x, z)  # rms_norm's squaring usage: take the x*x fast path
    if not torch.is_tensor(y):
        y = torch.full((1,), float(y), dtype=x.dtype, device=x.device)
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _pow_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- eq
# Comparison op: z = (x == y); output is a boolean tensor.
# Pure comparison; fp16 compares directly without upcasting; boolean results are verified strictly with torch.equal.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _eq_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x == y, mask=mask)


def eq(x, y, z):
    """z = (x == y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _eq_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- ne
# Comparison op: z = (x != y); output is a boolean tensor.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _ne_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x != y, mask=mask)


def ne(x, y, z):
    """z = (x != y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _ne_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- gt / ge / lt / le
# The four magnitude comparisons: z = (x > y) / (x >= y) / (x < y) / (x <= y); boolean output.
# Identical structure, only the operator differs; with randn inputs True/False split naturally ~50/50,
# no need for the manual sample-seeding used by eq/ne.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _gt_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x > y, mask=mask)


@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _ge_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x >= y, mask=mask)


@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _lt_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x < y, mask=mask)


@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _le_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x <= y, mask=mask)


def gt(x, y, z):
    """z = (x > y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _gt_kernel[grid](x, y, z, n)
    return z


def ge(x, y, z):
    """z = (x >= y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _ge_kernel[grid](x, y, z, n)
    return z


def lt(x, y, z):
    """z = (x < y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _lt_kernel[grid](x, y, z, n)
    return z


def le(x, y, z):
    """z = (x <= y); x/y are 1-D contiguous float tensors, z a boolean tensor of the same length."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _le_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- bitwise_and / bitwise_or
# Binary integer bitwise ops: z = x & y (1 only where both bits are 1), z = x | y (1 where either bit is 1).
# Same pattern as bitwise_not: integer inputs, one bit instruction, exact verification via torch.equal.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x & y, mask=mask)


@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _bitwise_or_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x | y, mask=mask)


def bitwise_and(x, y, z):
    """z = x & y; inputs are 1-D contiguous integer tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _bitwise_and_kernel[grid](x, y, z, n)
    return z


def bitwise_or(x, y, z):
    """z = x | y; inputs are 1-D contiguous integer tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _bitwise_or_kernel[grid](x, y, z, n)
    return z


# ---------------------------------------------------------------- clamp
# Ternary op: y = min(max(x, lo), hi), clamping x to [lo, hi].
# The bounds are scalars (not tensors), passed in as runtime kernel arguments;
# tl.maximum/minimum with scalars natively support fp16; no upcast needed.
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _clamp_kernel(x_ptr, y_ptr, lo, hi, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.minimum(tl.maximum(x, lo), hi)  # floor first, then cap
    tl.store(y_ptr + offsets, y, mask=mask)


def clamp(x, y, lo=-1.0, hi=1.0):
    """y = min(max(x, lo), hi); x is a 1-D contiguous tensor; lo/hi are scalars."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _clamp_kernel[grid](x, y, lo, hi, n)
    return y


# ---------------------------------------------------------------- rms_norm
# Reduction op: y = x * rsqrt(mean(x²) + eps) * w (the normalization used in every Llama layer).
# Essential difference from elementwise ops: every output depends on a whole row (mean(x²) is a cross-element
# sum), so the parallelization changes — one program per row, reducing within the row via tl.sum.
# x has shape (M, N) and is reduced along the last dim; w is the per-channel weight (N,).
@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)                     # which row I own
    cols = tl.arange(0, BLOCK_N)                    # indices of all columns in the row
    mask = cols < N                                 # bounds guard when N doesn't fill a block
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x) / N                       # reduction: mean of the squared row
    rs = tl.rsqrt(sumsq + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * N + cols, (x * rs * w).to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=[triton.Config({"ROWS": r}, num_warps=1) for r in (1, 4, 8)],
                 key=["M", "N", "KIND"])
@triton.jit
def _mlu_norm_rows(x_ptr, w_ptr, b_ptr, y_ptr, M, N: tl.constexpr, eps,
                   KIND: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask, 0).to(tl.float32)
    if KIND == 0:
        scale = tl.rsqrt(tl.sum(x * x, 1) / N + eps)
        w = tl.load(w_ptr + cols, cols < N, 0).to(tl.float32)
        value = x * scale[:, None] * w[None, :]
    elif KIND == 1:
        x = tl.where(cols[None, :] < N, x, float("-inf"))
        e = tl.exp(x - tl.max(x, 1)[:, None])
        value = e / tl.sum(e, 1)[:, None]
    else:
        mean = tl.sum(x, 1) / N
        centered = tl.where(cols[None, :] < N, x - mean[:, None], 0)
        scale = tl.rsqrt(tl.sum(centered * centered, 1) / N + eps)
        w = tl.load(w_ptr + cols, cols < N, 0).to(tl.float32)
        b = tl.load(b_ptr + cols, cols < N, 0).to(tl.float32)
        value = centered * scale[:, None] * w[None, :] + b[None, :]
    tl.store(y_ptr + rows[:, None] * N + cols[None, :], value, mask)


def rms_norm(x, w, y, eps=1e-6):
    """y = rms_norm(x) * w. x has shape (M, N), w has shape (N,), y matches x."""
    M, N = x.shape
    if x.device.type == "mlu":
        _mlu_norm_rows[lambda cfg: (triton.cdiv(M, cfg["ROWS"]),)](
            x, w, None, y, M, N, eps, KIND=0, COLS=triton.next_power_of_2(N))
        return y
    BLOCK_N = triton.next_power_of_2(N)  # one block holds a whole row (requires N ≤ limit; fine for the baseline)
    grid = (M,)
    _rms_norm_kernel[grid](x, w, y, N, eps, BLOCK_N=BLOCK_N)
    return y


# ---------------------------------------------------------------- softmax
# Reduction op: y = exp(x - max(x)) / Σexp(x - max(x)); rows are independent.
# Subtracting the max is the core of numerical stability: without it, exp(large) overflows to inf.
# Both reductions (max, then sum) stay within the row; same parallelization as rms_norm: one program per row.
# mask trick: pad out-of-bounds with -inf — max ignores -inf and exp(-inf)=0 keeps the denominator clean.
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(x)                              # first reduction: row max
    e = tl.exp(x - m)                          # out-of-bounds lanes get exp(-inf)=0, inherently safe
    s = tl.sum(e)                              # second reduction: denominator
    tl.store(y_ptr + row * N + cols, (e / s).to(y_ptr.dtype.element_ty), mask=mask)


def softmax(x, y):
    """y = softmax(x) along the last dim. x has shape (M, N), y matches x."""
    M, N = x.shape
    if x.device.type == "mlu":
        _mlu_norm_rows[lambda cfg: (triton.cdiv(M, cfg["ROWS"]),)](
            x, None, None, y, M, N, 0., KIND=1, COLS=triton.next_power_of_2(N))
        return y
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _softmax_kernel[grid](x, y, N, BLOCK_N=BLOCK_N)
    return y


# ---------------------------------------------------------------- layer_norm
# Reduction op: y = (x - mean) / sqrt(var + eps) * w + b (the normalization in BERT-family models).
# Two steps more than rms_norm: subtract the mean first (one extra reduction), add bias at the end.
# Variance uses the second-moment form: var = mean((x-mean)²); data already sits in registers, so two reductions cost zero extra memory traffic.
@triton.jit
def _layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / N                        # first reduction: mean
    xc = tl.where(mask, x - mean, 0.0)          # center; zero-padded lanes don't pollute the variance
    var = tl.sum(xc * xc) / N                   # second reduction: variance
    rs = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * N + cols, (xc * rs * w + b).to(y_ptr.dtype.element_ty), mask=mask)


def layer_norm(x, w, b, y, eps=1e-5):
    """y = layer_norm(x) * w + b. x has shape (M, N), w/b have shape (N,), y matches x."""
    M, N = x.shape
    if x.device.type == "mlu":
        _mlu_norm_rows[lambda cfg: (triton.cdiv(M, cfg["ROWS"]),)](
            x, w, b, y, M, N, eps, KIND=2, COLS=triton.next_power_of_2(N))
        return y
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _layer_norm_kernel[grid](x, w, b, y, N, eps, BLOCK_N=BLOCK_N)
    return y


# ---------------------------------------------------------------- mm
# Matrix multiply: out = a @ b. The pure form of addmm without bias and alpha/beta;
# The 2-D tiling + K-loop + tl.dot framework is reused unchanged.
@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn,
               stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def mm(a, b, out):
    """out = a @ b. a has shape (M, K), b has shape (K, N), out has shape (M, N)."""
    if a.device.type == "mlu":
        from .mlu_matmul import matmul
        return matmul(a, b, out)
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    _, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mm_kernel[grid](
        a, b, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ---------------------------------------------------------------- bmm
# Batched matmul: out[i] = a[i] @ b[i]; B independent matmuls packed together.
# One extra dim over mm: the grid gains a batch axis and addresses gain a "batch × batch stride" term.
# 2026-10-20 autotune added: a fixed 64×64 tile trailed cuBLAS by ~38% at large batch;
# let Triton pick the fastest of 6 tile/warps/stages configs per (M, N, K).
@triton.autotune(configs=_BMM_CONFIGS, key=["M", "N", "K"])  # re-tunes only when the shape changes; same shape hits the cache
@triton.jit
def _bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_ab, stride_am, stride_ak,
                stride_bb, stride_bk, stride_bn,
                stride_cb, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_b = tl.program_id(axis=0)   # which batch
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bmm(a, b, out):
    """out = a @ b (batched). a has shape (B, M, K), b has shape (B, K, N), out has shape (B, M, N)."""
    a, b = a.contiguous(), b.contiguous()
    B, M, K = a.shape
    _, _, N = b.shape
    # autotune version: the grid becomes a lambda; tile sizes come from the selected config
    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _bmm_kernel[grid](
        a, b, out, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out


# ---------------------------------------------------------------- addmm
# out = beta * bias + alpha * (mat1 @ mat2)
# the classic Triton matmul, which adds three concepts on top of elementwise ops:
#   1. 2-D tiling: each program owns one (BLOCK_M, BLOCK_N) tile of the output
#   2. K-dim loop: accumulate along K in small chunks (saves memory, exploits cache)
#   3. tl.dot: block-level matmul that maps onto Tensor Cores
@triton.jit
def _addmm_kernel(bias_ptr, a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn, alpha, beta,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)   # which row-tile I own
    pid_n = tl.program_id(axis=1)   # which column-tile I own
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # accumulators are always fp32 to prevent precision loss
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        # input_precision="ieee": disables the TF32 approximation and computes strictly in fp32,
        # which is required to match PyTorch's reference within 1e-4
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = alpha * acc + beta * bias[None, :]  # bias broadcasts per row, matching torch.addmm semantics
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def addmm(bias, mat1, mat2, out, alpha=1.0, beta=1.0):
    """out = beta * bias + alpha * (mat1 @ mat2). bias has shape (N,) and broadcasts per row."""
    if mat1.device.type == "mlu":
        from .mlu_matmul import matmul
        return matmul(mat1, mat2, out, bias=bias, alpha=alpha, beta=beta)
    mat1, mat2 = mat1.contiguous(), mat2.contiguous()
    M, K = mat1.shape
    _, N = mat2.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _addmm_kernel[grid](
        bias, mat1, mat2, out, M, N, K,
        mat1.stride(0), mat1.stride(1), mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1), alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ---------------------------------------------------------------- swiglu
# Binary composite op: z = silu(x) * gate = x * sigmoid(x) * gate. The body of Llama-family FFNs:
# after two linear projections (w1 → x, w3 → gate), one branch goes through silu and the other acts as the "gate"; multiplied elementwise.
# Structure: the silu kernel with one extra gate read; the silu part is upcast to fp32 as usual, and the multiply happens in fp32 too.
# The payoff of fusion: PyTorch has no ready-made SwiGLU, so it chains F.silu(x)*gate as two kernels,
# round-tripping the intermediate; fused, only 3 data movements remain (read x, read gate, write z).
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _swiglu_kernel(x_ptr, gate_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    z = x32 * tl.sigmoid(x32) * gate.to(tl.float32)
    tl.store(z_ptr + offsets, z.to(z_ptr.dtype.element_ty), mask=mask)


def swiglu(x, gate, z):
    """z = silu(x) * gate; inputs are 1-D contiguous tensors."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _swiglu_kernel[grid](x, gate, z, n)
    return z


# ---------------------------------------------------------------- rotary_position_embedding
# RoPE (rotary position embedding): treat the feature dim of each row as D/2 pairs of "2-D vectors"; pair i rotates by angle θ_i:
#   out[i]      = x1[i] * cos[i] - x2[i] * sin[i]     (x1 = first half)
#   out[i+D/2]  = x2[i] * cos[i] + x1[i] * sin[i]     (x2 = second half)
# Llama's rotate_half convention: pairs are (i-th, i+D/2-th), not adjacent.
# cos/sin are precomputed per-position tables (one set per row, shape (M, D/2)); the kernel only does multiply-add,
# no trig calls — the sin/cos ops were already used when the tables were built.
# Parallelization: 2-D grid (row, first-half block); one program writes both the first-half and second-half positions.
# MLU adaptation: the first grid dimension on MLU caps at 65535 (2^31-1 on CUDA); when rows exceed it,
# the kernel strided-loops by the grid row count (on CUDA with M≤65535 the loop runs exactly once — unchanged behavior).
@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, M, D2,
                 BLOCK: tl.constexpr):
    row0 = tl.program_id(axis=0)                    # starting row
    nrows = tl.num_programs(axis=0)                 # row stride = grid row count
    pid = tl.program_id(axis=1)                     # which block within the row
    offs = pid * BLOCK + tl.arange(0, BLOCK)        # indices within the first half
    mask = offs < D2                                # bounds guard when D2 doesn't fill a block
    for row in range(row0, M, nrows):
        row_x = x_ptr + row * (2 * D2)              # start of this row (2*D2 elements per row)
        x1 = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(row_x + D2 + offs, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + row * D2 + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + row * D2 + offs, mask=mask, other=0.0).to(tl.float32)
        row_o = out_ptr + row * (2 * D2)
        tl.store(row_o + offs, (x1 * c - x2 * s).to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(row_o + D2 + offs, (x2 * c + x1 * s).to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=[triton.Config({"ROWS": r}, num_warps=1) for r in (8, 32, 128)],
                 key=["M", "D2"])
@triton.jit
def _rope_rows_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, M, D2: tl.constexpr,
                       ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    valid = (rows[:, None] < M) & (cols[None, :] < D2)
    offset = rows[:, None] * (2 * D2) + cols[None, :]
    pair = rows[:, None] * D2 + cols[None, :]
    x1 = tl.load(x_ptr + offset, valid, 0).to(tl.float32)
    x2 = tl.load(x_ptr + offset + D2, valid, 0).to(tl.float32)
    c = tl.load(cos_ptr + pair, valid, 0).to(tl.float32)
    s = tl.load(sin_ptr + pair, valid, 0).to(tl.float32)
    tl.store(out_ptr + offset, x1 * c - x2 * s, valid)
    tl.store(out_ptr + offset + D2, x2 * c + x1 * s, valid)


def rotary_position_embedding(x, cos, sin, out):
    """RoPE. x has shape (M, D) (D must be even), cos/sin have shape (M, D/2), out matches x."""
    x, cos, sin = x.contiguous(), cos.contiguous(), sin.contiguous()
    M, D = x.shape
    D2 = D // 2
    if x.device.type == "mlu":
        _rope_rows_kernel[lambda cfg: (triton.cdiv(M, cfg["ROWS"]),)](
            x, cos, sin, out, M, D2, COLS=triton.next_power_of_2(D2))
        return out
    BLOCK = min(triton.next_power_of_2(D2), 1024)
    grid = (min(M, 65535), triton.cdiv(D2, BLOCK))  # 65535 = MLU grid cap; extra rows are covered by the in-kernel loop
    _rope_kernel[grid](x, cos, sin, out, M, D2, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------- dropout
# dropout: y = x * keep / (1 - p); each element is zeroed with probability p, survivors scaled by 1/(1-p)
# to keep the expectation unchanged (training only; disabled at inference). Not used in DeepSeek inference, but it's on the list.
# Randomness uses Triton's Philox counter: tl.rand(seed, offsets) —
# the same (seed, element index) always yields the same number, making results reproducible and testable.
# Semantics match PyTorch's torch.dropout(x, p, training=True).
@triton.autotune(configs=_EW_CONFIGS, key=["n_elements"])
@triton.jit
def _dropout_kernel(x_ptr, y_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    r = tl.rand(seed, offsets)              # one uniform [0,1) random number per element
    keep_scale = 1.0 / (1.0 - p)            # scale factor for survivors
    y = tl.where(r > p, x.to(tl.float32) * keep_scale, 0.0)
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def dropout(x, y, p=0.5, seed=0):
    """y = dropout(x, p); inputs are 1-D contiguous tensors. Results are reproducible for the same seed."""
    n = x.numel()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)  # grid follows the block size chosen by autotune
    _dropout_kernel[grid](x, y, n, p, seed)
    return y


# ---------------------------------------------------------------- scaled_dot_product_attention
# SDPA: o = softmax(q @ k^T * scale) @ v — the body of attention.
# The naive version materializes the whole M×N score matrix in memory (write + read), which blows up as seq grows.
# FlashAttention approach: stream K/V in blocks, correcting on the fly with "online softmax" —
# for each new K block, update the per-row max m first, rescale the old accumulator by e^(m_old - m_new), then accumulate;
# mathematically identical to the one-shot version, but the score matrix lives in registers the whole time.
# Parallelization: one program owns (one batch·head, one row-block of Q) and loops over all K/V of that head.
# This baseline covers only the non-causal, fixed head_dim, dropout-free core path.
# 2026-10-20 autotune added: fixed 64×64 was ~7x slower than PyTorch; tile/warps/stages are now chosen per shape.
@triton.autotune(configs=_SDPA_CONFIGS, key=["M", "N", "D"])  # re-tunes only when the shape changes; same shape hits the cache
@triton.jit
def _sdpa_kernel(q_ptr, k_ptr, v_ptr, o_ptr, M, N, scale,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(axis=0)                   # which row-block of Q
    pid_bh = tl.program_id(axis=1)                  # which batch·head
    # With contiguous memory, each (b, h) base address = index × elements per head
    q_base = q_ptr + pid_bh * M * D
    k_base = k_ptr + pid_bh * N * D
    v_base = v_ptr + pid_bh * N * D
    o_base = o_ptr + pid_bh * M * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(q_base + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)  # running per-row max
    l_i = tl.zeros((BLOCK_M,), tl.float32)                # per-row denominator (unnormalized)
    acc = tl.zeros((BLOCK_M, D), tl.float32)              # per-row output accumulator (fp32)

    for start_n in range(0, N, BLOCK_N):                # sweep K/V block by block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k = tl.load(k_base + offs_n[:, None] * D + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        qk = tl.where(n_mask[None, :], qk, float("-inf"))  # pad out-of-bounds columns with -inf so softmax zeroes them
        m_new = tl.maximum(m_i, tl.max(qk, 1))            # new per-row max
        alpha = tl.exp(m_i - m_new)                       # rescale factor for the old accumulator
        p = tl.exp(qk - m_new[:, None])                   # this block's unnormalized attention
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * D + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]                              # divide by the denominator once at the end
    tl.store(o_base + offs_m[:, None] * D + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < M)


def scaled_dot_product_attention(q, k, v, out, scale=None):
    """Non-causal SDPA. q has shape (B, H, M, D); k/v have shape (B, H, N, D); out matches q.
    scale defaults to 1/√D, matching F.scaled_dot_product_attention."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, M, D = q.shape
    N = k.shape[2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    # autotune version: the grid becomes a lambda; BLOCK_M comes from the selected config
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), B * H)
    _sdpa_kernel[grid](q, k, v, out, M, N, scale, D=D)
    return out


# ---------------------------------------------------------------- conv2d
# 2-D convolution: y[n, co, oh, ow] = Σ_{ci,kh,kw} x[n, ci, oh*stride+kh-pad, ow*stride+kw-pad]
#                                          * w[co, ci, kh, kw]
# Key insight: convolution is a matrix multiply —
#   M dim = output pixels (batch folded in, N·H_out·W_out total)
#   N dim = output channels C_out
#   K dim = C_in·KH·KW (terms accumulated per output pixel)
# This is "implicit GEMM": the im2col matrix never materializes; each address is computed on the fly in the K loop,
# then multiplied block-wise via tl.dot (the earlier naive scalar version was 15x slower than cuDNN; rewritten 2026-10-20).
# K index k decomposes into (ci, kh, kw): ci = k // (KH·KW), kh = (k % (KH·KW)) // KW, kw = k % KW.
# Out-of-bounds (padding or a partial tail block) is masked to 0 — equivalent to zero-padded convolution.
# baseline scope: dilation=1, no bias, contiguous NCHW; stride/padding passed at runtime.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["C_in", "H", "W", "C_out", "KH", "KW"],  # re-tunes when the shape changes
)
@triton.jit
def _conv2d_kernel(x_ptr, w_ptr, y_ptr,
                   C_in, H, W, C_out, KH, KW, H_out, W_out, M_total,
                   stride, pad,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)                   # which output-pixel tile
    pid_n = tl.program_id(axis=1)                   # which output-channel tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M_total
    n_mask = offs_n < C_out

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out                        # flattened pixel index → (sample, row, col)
    rest = offs_m % HW_out
    oh = rest // W_out
    ow = rest % W_out

    K_total = C_in * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total
        ci = offs_k // (KH * KW)                    # K index → (input channel, kernel row, kernel col)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        ih = oh[:, None] * stride + kh[None, :] - pad   # (BLOCK_M, BLOCK_K) input coordinates
        iw = ow[:, None] * stride + kw[None, :] - pad
        a_mask = m_mask[:, None] & k_mask[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        a_ptrs = x_ptr + n_idx[:, None] * (C_in * H * W) + ci[None, :] * (H * W) + ih * W + iw
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)     # a small slice of im2col, computed and loaded on the fly
        b_ptrs = w_ptr + offs_n[None, :] * K_total + offs_k[:, None]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")

    # y layout (N, C_out, H_out, W_out): address = within-channel offset rest + channel offset co·HW_out
    # + batch offset n·C_out·HW_out (do NOT use the flattened pixel index offs_m directly —
    # it only carries n·HW_out and misses the C_out factor — that bug once broke every N>1 case)
    c_ptrs = y_ptr + rest[:, None] + offs_n[None, :] * HW_out \
        + n_idx[:, None] * (C_out * HW_out)
    tl.store(c_ptrs, acc.to(y_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def conv2d(x, w, y, stride=1, padding=0):
    """y = conv2d(x, w). x has shape (N, C_in, H, W), w has shape (C_out, C_in, KH, KW),
    y has shape (N, C_out, H_out, W_out). Baseline with dilation=1 and no bias."""
    x, w = x.contiguous(), w.contiguous()
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    M_total = N * H_out * W_out  # total output pixels (including batch), i.e. the M dim of the implicit GEMM
    grid = lambda META: (triton.cdiv(M_total, META["BLOCK_M"]),
                         triton.cdiv(C_out, META["BLOCK_N"]))
    _conv2d_kernel[grid](x, w, y, C_in, H, W, C_out, KH, KW, H_out, W_out,
                         M_total, stride, padding)
    return y


# ---------------------------------------------------------------- max_pool2d
# Max pooling: y[n, c, oh, ow] = max_{kh,kw} x[n, c, oh*stride+kh, ow*stride+kw]
# Same skeleton as conv2d (one program owns a tile of output pixels for a (sample, channel)),
# But instead of a weighted sum it takes the max: initialize the accumulator to -inf and tl.maximum each window element.
# Read-only with no intermediates; bandwidth-bound (reads k² elements per output).
# baseline scope: padding=0 (PyTorch default), contiguous NCHW; stride defaults to kernel_size.
# 2026-10-22 P0-2 rework (previously the repo's only memory-bound kernel without autotune, T/P 0.048):
#   1) autotune added to the generic path (BLOCK x num_warps); BLOCK=256 was previously hardcoded;
#   2) the accumulator is no longer upcast to fp32: max is pure selection, so compare in the original dtype and save registers;
#   3) PyTorch's default k=2/stride=2 takes a contiguous-read fast path: each row reads 2*BLOCK_W
#      contiguous elements, reshaped to (BLOCK_W, 2) for the reduction, replacing per-element stride-2 gathers.
@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w)
                          for b in (256, 1024) for w in (1, 4)],
                 key=["H_out", "W_out"])
@triton.jit
def _max_pool2d_kernel(x_ptr, y_ptr, C, H, W, KH, KW, H_out, W_out, stride,
                       BLOCK: tl.constexpr):
    pid_nc = tl.program_id(axis=0)                  # flattened (sample, channel) index
    pid_blk = tl.program_id(axis=1)
    n = pid_nc // C
    c = pid_nc % C

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    o_mask = offs < H_out * W_out
    oh = offs // W_out
    ow = offs % W_out

    x_base = x_ptr + n * (C * H * W) + c * (H * W)
    acc = tl.full((BLOCK,), float("-inf"), dtype=y_ptr.dtype.element_ty)  # original dtype, no upcast
    for kh in range(KH):
        ih = oh * stride + kh                       # padding=0, the window never crosses the left/top edge
        for kw in range(KW):
            iw = ow * stride + kw
            valid = o_mask & (ih < H) & (iw < W)    # only guard right/bottom overflow
            xv = tl.load(x_base + ih * W + iw, mask=valid, other=float("-inf"))
            acc = tl.maximum(acc, xv.to(acc.dtype))

    y_base = y_ptr + n * (C * H_out * W_out) + c * (H_out * W_out)
    tl.store(y_base + offs, acc, mask=o_mask)


# Contiguous-read fast path for k=2, stride=2: one program owns "one output row of one (n,c) plus a column block".
# Read 2*BLOCK_W contiguous elements per row → reshape to (BLOCK_W, 2) and take the max of the two window columns,
# then max across the two rows; write BLOCK_W contiguous outputs. Avoids one stride-2 gather per element.
@triton.autotune(configs=[triton.Config({"BLOCK_W": b}, num_warps=w)
                          for b in (128, 256, 512) for w in (1, 4)],
                 key=["W_out"])
@triton.jit
def _max_pool2d_2x2_kernel(x_ptr, y_ptr, H, W, H_out, W_out, BLOCK_W: tl.constexpr):
    pid_row = tl.program_id(axis=0)                 # (n*C + c) * H_out + oh
    pid_w = tl.program_id(axis=1)
    oh = pid_row % H_out
    nc = pid_row // H_out

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    o_mask = ow < W_out
    iw = pid_w * BLOCK_W * 2 + tl.arange(0, BLOCK_W * 2)   # 2*BLOCK_W contiguous inputs per row
    i_mask = iw < W

    base = nc * H * W + oh * 2 * W                  # two input rows: oh*2 and oh*2+1
    r0 = tl.load(x_ptr + base + iw, mask=i_mask, other=float("-inf"))
    r1 = tl.load(x_ptr + base + W + iw, mask=i_mask, other=float("-inf"))
    v0 = tl.max(tl.reshape(r0, (BLOCK_W, 2)), axis=1)      # max of the two columns in the window
    v1 = tl.max(tl.reshape(r1, (BLOCK_W, 2)), axis=1)
    acc = tl.maximum(v0, v1)                               # max across the two rows
    tl.store(y_ptr + nc * H_out * W_out + oh * W_out + ow, acc, mask=o_mask)


def max_pool2d(x, y, kernel_size=2, stride=None):
    """y = max_pool2d(x). x has shape (N, C, H, W), y has shape (N, C, H_out, W_out).
    stride defaults to kernel_size (same as PyTorch); padding=0."""
    x = x.contiguous()
    if stride is None:
        stride = kernel_size
    N, C, H, W = x.shape
    KH = KW = kernel_size
    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1
    if KH == 2 and KW == 2 and stride == 2:          # PyTorch default configuration (bench takes this path)
        grid = lambda META: (N * C * H_out, triton.cdiv(W_out, META["BLOCK_W"]))
        _max_pool2d_2x2_kernel[grid](x, y, H, W, H_out, W_out)
        return y
    grid = lambda META: (N * C, triton.cdiv(H_out * W_out, META["BLOCK"]))
    _max_pool2d_kernel[grid](x, y, C, H, W, KH, KW, H_out, W_out, stride)
    return y
