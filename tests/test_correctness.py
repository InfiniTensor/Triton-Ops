# -*- coding: utf-8 -*-
"""Correctness tests: every implementation is checked against the PyTorch reference.

Usage: python -m pytest tests/ -v
Tolerance policy and rationale:
  - fp16 uses the mixed tolerance atol=rtol=1e-2 as an engineering regression only; it is not equivalent to a pure ≤1% relative error;
  - the fp32 tolerance 1e-6 is an internal tightening that catches precision bugs the loose fp16 tolerance misses;
  - boolean/integer ops have no notion of error; compare exactly with torch.equal.
"""
import os

import torch

from kernels import triton_impl

try:
    from kernels import ninetoothed_impl
    HAS_NINETOOHED = True
except ImportError:  # when the Cambricon docker image lacks the ninetoothed library, the legacy ninetoothed tests are skipped automatically
    HAS_NINETOOHED = False

DEVICE = os.environ.get("TB_DEVICE", "cuda")  # defaults to local CUDA; export TB_DEVICE=mlu on the MLU server
if DEVICE == "mlu":
    import torch_mlu  # noqa: F401  registers the "mlu" device; not installed locally — only reached inside the server container
SHAPES = [1, 100, 1024, 100_000, 1_000_001]  # deliberately includes non-block-aligned lengths to exercise the mask


def _make_inputs(n, dtype):
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.randn(n, dtype=dtype, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


def _check(z, ref, dtype, atol32=1e-6, rtol32=1e-6):
    tol = {"atol": 1e-2, "rtol": 1e-2} if dtype == torch.float16 else {"atol": atol32, "rtol": rtol32}
    assert torch.allclose(z, ref, **tol), f"max error: {(z - ref).abs().max().item()}"


import pytest


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_triton(n, dtype):
    x, y, z = _make_inputs(n, dtype)
    triton_impl.add(x, y, z)
    _check(z, x + y, dtype)


@pytest.mark.skipif(not HAS_NINETOOHED, reason="ninetoothed library not installed in this environment")
@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_ninetoothed(n, dtype):
    x, y, z = _make_inputs(n, dtype)
    ninetoothed_impl.add(x, y, z)
    _check(z, x + y, dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sub_triton(n, dtype):
    x, y, z = _make_inputs(n, dtype)
    triton_impl.sub(x, y, z)
    _check(z, x - y, dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mul_triton(n, dtype):
    x, y, z = _make_inputs(n, dtype)
    triton_impl.mul(x, y, z)
    _check(z, x * y, dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_div_triton(n, dtype):
    x, y, z = _make_inputs(n, dtype)
    triton_impl.div(x, y, z)
    _check(z, x / y, dtype)


def _make_pow_inputs(n, dtype):
    """Inputs for pow: base x must be positive (negative base + non-integer exponent = NaN); exponent y is unrestricted."""
    x = torch.rand(n, dtype=dtype, device=DEVICE) * 2 + 0.1  # [0.1, 2.1)
    y = torch.randn(n, dtype=dtype, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_pow_triton(n, dtype):
    x, y, z = _make_pow_inputs(n, dtype)
    triton_impl.pow(x, y, z)
    _check(z, torch.pow(x, y), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_pow_square_scalar_triton(n, dtype):
    """pow square fast path: exponent as Python scalar 2.0 takes x*x (the rms_norm usage)."""
    x = torch.rand(n, dtype=dtype, device=DEVICE) * 2 + 0.1  # [0.1, 2.1)
    z = torch.empty_like(x)
    triton_impl.pow(x, 2.0, z)
    _check(z, torch.pow(x, 2.0), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_pow_square_one_elem_tensor(dtype):
    """The fast path also applies when the exponent is a single-element tensor 2.0."""
    x = torch.rand(1024, dtype=dtype, device=DEVICE) * 2 + 0.1
    z = torch.empty_like(x)
    y = torch.full((1,), 2.0, dtype=dtype, device=DEVICE)
    triton_impl.pow(x, y, z)
    _check(z, x * x, dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("exponent", [3.0, 0.5])
def test_pow_scalar_non2_triton(n, dtype, exponent):
    """Non-2 Python-scalar exponents take the broadcast path (no out-of-bounds read of a 1-element tensor)."""
    x = torch.rand(n, dtype=dtype, device=DEVICE) * 2 + 0.1  # [0.1, 2.1)
    z = torch.empty_like(x)
    triton_impl.pow(x, exponent, z)
    _check(z, torch.pow(x, exponent), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("exponent", [3.0, 0.5])
def test_pow_one_elem_tensor_non2(dtype, exponent):
    """A single-element exponent tensor with a value other than 2.0 also broadcasts."""
    x = torch.rand(1024, dtype=dtype, device=DEVICE) * 2 + 0.1
    z = torch.empty_like(x)
    y = torch.full((1,), exponent, dtype=dtype, device=DEVICE)
    triton_impl.pow(x, y, z)
    _check(z, torch.pow(x, y), dtype)


@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_square_triton(n, dtype):
    """Direct square() calls checked against PyTorch's x*x."""
    x = torch.rand(n, dtype=dtype, device=DEVICE) * 2 + 0.1
    z = torch.empty_like(x)
    triton_impl.square(x, z)
    _check(z, x * x, dtype)


def _make_eq_inputs(n, dtype):
    """Inputs for eq: two random tensors are almost never equal, so True cases must be manufactured —
    copy part of y into x so both True and False are exercised."""
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.randn(n, dtype=dtype, device=DEVICE)
    x[::5] = y[::5]  # force one equal pair every 5 elements
    z = torch.empty(n, dtype=torch.bool, device=DEVICE)
    return x, y, z


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_eq_triton(n, dtype):
    x, y, z = _make_eq_inputs(n, dtype)
    triton_impl.eq(x, y, z)
    assert torch.equal(z, x == y)  # boolean results compared bit-exactly


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ne_triton(n, dtype):
    # Symmetric trap: randn makes ne almost all True; reuse eq's input builder to seed 20% equal (False) samples
    x, y, z = _make_eq_inputs(n, dtype)
    triton_impl.ne(x, y, z)
    assert torch.equal(z, x != y)


def _make_cmp_inputs(n, dtype):
    """Magnitude comparisons (gt/ge/lt/le): randn pairs split True/False naturally ~50/50; no seeding needed."""
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.randn(n, dtype=dtype, device=DEVICE)
    z = torch.empty(n, dtype=torch.bool, device=DEVICE)
    return x, y, z


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gt_triton(n, dtype):
    x, y, z = _make_cmp_inputs(n, dtype)
    triton_impl.gt(x, y, z)
    assert torch.equal(z, x > y)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ge_triton(n, dtype):
    x, y, z = _make_cmp_inputs(n, dtype)
    triton_impl.ge(x, y, z)
    assert torch.equal(z, x >= y)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lt_triton(n, dtype):
    x, y, z = _make_cmp_inputs(n, dtype)
    triton_impl.lt(x, y, z)
    assert torch.equal(z, x < y)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_le_triton(n, dtype):
    x, y, z = _make_cmp_inputs(n, dtype)
    triton_impl.le(x, y, z)
    assert torch.equal(z, x <= y)


def _make_unary_inputs(n, dtype):
    x = torch.randn(n, dtype=dtype, device=DEVICE)  # randn includes negatives naturally — good for exercising abs
    y = torch.empty_like(x)
    return x, y


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_abs_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.abs(x, y)
    _check(y, x.abs(), dtype)


@pytest.mark.skipif(not HAS_NINETOOHED, reason="ninetoothed library not installed in this environment")
@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_abs_ninetoothed(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    ninetoothed_impl.abs(x, y)
    _check(y, x.abs(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_neg_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.neg(x, y)
    _check(y, -x, dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cos_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.cos(x, y)
    _check(y, x.cos(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sin_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.sin(x, y)
    _check(y, x.sin(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_exp_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.exp(x, y)
    _check(y, x.exp(), dtype)


def _make_pos_inputs(n, dtype):
    """Positive inputs: rsqrt/sqrt-like ops require x ≥ 0. rand∈[0,1), ×2+0.1 gives [0.1, 2.1)."""
    x = torch.rand(n, dtype=dtype, device=DEVICE) * 2 + 0.1
    y = torch.empty_like(x)
    return x, y


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rsqrt_triton(n, dtype):
    x, y = _make_pos_inputs(n, dtype)
    triton_impl.rsqrt(x, y)
    _check(y, x.rsqrt(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sigmoid_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.sigmoid(x, y)
    _check(y, x.sigmoid(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tanh_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.tanh(x, y)
    _check(y, x.tanh(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_relu_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.relu(x, y)
    _check(y, x.relu(), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_silu_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.silu(x, y)
    _check(y, torch.nn.functional.silu(x), dtype)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gelu_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.gelu(x, y)
    _check(y, torch.nn.functional.gelu(x, approximate="tanh"), dtype)


def _make_nan_inputs(n, dtype):
    """Inputs seeded with NaN: randn never produces NaN, so seed manually — one in every 7 elements."""
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    x[::7] = float("nan")
    y = torch.empty(n, dtype=torch.bool, device=DEVICE)  # output is boolean
    return x, y


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_isnan_triton(n, dtype):
    x, y = _make_nan_inputs(n, dtype)
    triton_impl.isnan(x, y)
    # Boolean predicates have no "error" — they must match bit-exactly: use equal, not allclose
    assert torch.equal(y, x.isnan())


def _make_inf_inputs(n, dtype):
    """Inputs seeded with ±∞: one in every 7 elements, alternating signs — both directions must be checked."""
    x = torch.randn(n, dtype=dtype, device=DEVICE)
    x[::7] = float("inf")
    x[3::14] = float("-inf")
    y = torch.empty(n, dtype=torch.bool, device=DEVICE)
    return x, y


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_isinf_triton(n, dtype):
    x, y = _make_inf_inputs(n, dtype)
    triton_impl.isinf(x, y)
    assert torch.equal(y, x.isinf())


def _int32_safe_range(dtype):
    """randint bounds: for int32, the upper bound is clamped into [INT32_MIN, INT32_MAX] (Cambricon CNNL's
    cnnlGenerateRandDiscreteUniform rejects out-of-range bounds; CUDA has no such restriction). randint's upper bound is exclusive."""
    info = torch.iinfo(dtype)
    return max(-2**31, info.min), min(2**31, info.max)


def _make_int_inputs(n, dtype):
    """Integer inputs: randint covers the full positive/negative range (the "domain" of bitwise ops is the entire integer domain)."""
    lo, hi = _int32_safe_range(dtype)
    x = torch.randint(lo, hi, (n,), dtype=dtype, device=DEVICE)
    y = torch.empty_like(x)
    return x, y


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])  # integer ops; dtype becomes an integer type
def test_bitwise_not_triton(n, dtype):
    x, y = _make_int_inputs(n, dtype)
    triton_impl.bitwise_not(x, y)
    assert torch.equal(y, ~x)  # bitwise results are exact; compare bit-exactly


def _make_int_bin_inputs(n, dtype):
    """Binary integer bitwise ops: two randint inputs + an empty output."""
    lo, hi = _int32_safe_range(dtype)
    x = torch.randint(lo, hi, (n,), dtype=dtype, device=DEVICE)
    y = torch.randint(lo, hi, (n,), dtype=dtype, device=DEVICE)
    z = torch.empty_like(x)
    return x, y, z


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_and_triton(n, dtype):
    x, y, z = _make_int_bin_inputs(n, dtype)
    triton_impl.bitwise_and(x, y, z)
    assert torch.equal(z, x & y)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise_or_triton(n, dtype):
    x, y, z = _make_int_bin_inputs(n, dtype)
    triton_impl.bitwise_or(x, y, z)
    assert torch.equal(z, x | y)


@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_triton(n, dtype):
    x, y = _make_unary_inputs(n, dtype)
    # ~38% of randn falls outside ±0.5, so all three clamp branches (upper/lower/pass-through) are well covered
    triton_impl.clamp(x, y, lo=-0.5, hi=0.5)
    _check(y, torch.clamp(x, -0.5, 0.5), dtype)


# ---------------------------------------------------------------- rms_norm
# Reduction ops: 2-D input (M, N), reduced along the last dim.
# fp32 tolerance relaxed to 1e-5: the sum of squares accumulates over N terms, and different reduction orders accumulate error.
NORM_SHAPES = [  # (M, N), deliberately mixing in non-power-of-2 N to exercise the in-row mask
    (1, 64), (32, 768), (128, 4096), (7, 1000), (513, 257),
]


def _make_rms_inputs(m, n, dtype):
    x = torch.randn(m, n, dtype=dtype, device=DEVICE)
    w = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.empty_like(x)
    return x, w, y


@pytest.mark.parametrize("m,n", NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_triton(m, n, dtype):
    x, w, y = _make_rms_inputs(m, n, dtype)
    triton_impl.rms_norm(x, w, y)
    ref = torch.nn.functional.rms_norm(x.float(), (n,), w.float(), eps=1e-6).to(dtype)
    _check(y, ref, dtype, atol32=1e-5, rtol32=1e-5)


@pytest.mark.parametrize("m,n", NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softmax_triton(m, n, dtype):
    x, _, y = _make_rms_inputs(m, n, dtype)  # w not needed; reuse the structure
    triton_impl.softmax(x, y)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    _check(y, ref, dtype, atol32=1e-5, rtol32=1e-5)


def _make_ln_inputs(m, n, dtype):
    x = torch.randn(m, n, dtype=dtype, device=DEVICE)
    w = torch.randn(n, dtype=dtype, device=DEVICE)
    b = torch.randn(n, dtype=dtype, device=DEVICE)
    y = torch.empty_like(x)
    return x, w, b, y


@pytest.mark.parametrize("m,n", NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_layer_norm_triton(m, n, dtype):
    x, w, b, y = _make_ln_inputs(m, n, dtype)
    triton_impl.layer_norm(x, w, b, y)
    ref = torch.nn.functional.layer_norm(x.float(), (n,), w.float(), b.float()).to(dtype)
    _check(y, ref, dtype, atol32=1e-5, rtol32=1e-5)


# ---------------------------------------------------------------- addmm
# Matmul tolerances are wider than elementwise: different K-dim accumulation orders accumulate error.
# fp32 runs at ieee precision (not TF32), so 1e-4 suffices; fp16 inputs are inherently less precise, so 1e-2.
MM_SHAPES = [  # (M, K, N), deliberately including odd shapes that don't divide evenly into tiles
    (64, 64, 64), (128, 256, 512), (100, 200, 300), (513, 257, 129),
]


def _make_addmm_inputs(m, k, n, dtype):
    a = torch.randn(m, k, dtype=dtype, device=DEVICE)
    b = torch.randn(k, n, dtype=dtype, device=DEVICE)
    bias = torch.randn(n, dtype=dtype, device=DEVICE)
    out = torch.empty(m, n, dtype=dtype, device=DEVICE)
    return bias, a, b, out


def _check_mm(out, ref, dtype):
    tol = {"atol": 1e-2, "rtol": 1e-2} if dtype == torch.float16 else {"atol": 1e-4, "rtol": 1e-4}
    assert torch.allclose(out, ref, **tol), f"max error: {(out.float() - ref.float()).abs().max().item()}"


@pytest.mark.parametrize("m,k,n", MM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_triton(m, k, n, dtype):
    bias, a, b, out = _make_addmm_inputs(m, k, n, dtype)
    triton_impl.addmm(bias, a, b, out)
    _check_mm(out, torch.addmm(bias, a, b), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addmm_triton_scales(dtype):
    """The case with alpha/beta coefficients (full torch.addmm semantics)."""
    bias, a, b, out = _make_addmm_inputs(128, 256, 512, dtype)
    triton_impl.addmm(bias, a, b, out, alpha=0.5, beta=0.25)
    _check_mm(out, torch.addmm(bias, a, b, beta=0.25, alpha=0.5), dtype)


@pytest.mark.parametrize("m,k,n", MM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mm_triton(m, k, n, dtype):
    _, a, b, out = _make_addmm_inputs(m, k, n, dtype)  # bias ignored; reuse the structure
    triton_impl.mm(a, b, out)
    _check_mm(out, torch.mm(a, b), dtype)


@pytest.mark.skipif(DEVICE != "mlu", reason="MLU schedule regression")
@pytest.mark.parametrize("m,k,n", [(1, 256, 128), (65, 129, 257), (513, 257, 769)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mlu_matmul_schedule(m, k, n, dtype):
    """Decode, masked persistent tiles, uneven cluster work, and output strides."""
    torch.manual_seed(20261022)
    a = torch.randn(k, m, dtype=dtype, device=DEVICE).T
    b = torch.randn(n, k, dtype=dtype, device=DEVICE).T
    bias = torch.randn(n, dtype=dtype, device=DEVICE)
    out = torch.empty(m, n * 2, dtype=dtype, device=DEVICE)[:, ::2]
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    triton_impl.mm(a, b, out)
    assert torch.allclose(out, a @ b, atol=tol, rtol=tol)
    triton_impl.addmm(bias, a, b, out, alpha=.5, beta=.25)
    assert torch.allclose(out, torch.addmm(bias, a, b, alpha=.5, beta=.25), atol=tol, rtol=tol)


BMM_SHAPES = [  # (B, M, K, N), again with odd shapes mixed in
    (2, 64, 64, 64), (4, 128, 256, 512), (3, 100, 200, 300), (2, 513, 257, 129),
]


@pytest.mark.parametrize("b,m,k,n", BMM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_bmm_triton(b, m, k, n, dtype):
    a = torch.randn(b, m, k, dtype=dtype, device=DEVICE)
    bb = torch.randn(b, k, n, dtype=dtype, device=DEVICE)
    out = torch.empty(b, m, n, dtype=dtype, device=DEVICE)
    triton_impl.bmm(a, bb, out)
    _check_mm(out, torch.bmm(a, bb), dtype)


# ---------------------------------------------------------------- swiglu
# Composite op: z = silu(x) * gate. The reference is exactly PyTorch's chaining (F.silu, then multiply).
# The Triton version computes in fp32 throughout, same policy as F.silu's fp32 path, so the default 1e-6 tolerance suffices.
@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_triton(n, dtype):
    x, y, z = _make_inputs(n, dtype)  # x as the main input, y as the gate
    triton_impl.swiglu(x, y, z)
    _check(z, torch.nn.functional.silu(x) * y, dtype)


# ---------------------------------------------------------------- rotary_position_embedding
# RoPE: x has shape (M, D); cos/sin have shape (M, D/2). The reference follows Llama's rotate_half convention.
# Pure multiply-add with no reduction; the default fp32 1e-6 tolerance suffices.
ROPE_SHAPES = [  # (M, D), deliberately mixing in D=100 (D/2=50 is not a power of 2) and odd row counts
    (1, 64), (32, 128), (128, 256), (7, 100), (513, 258),
]


def _make_rope_inputs(m, d, dtype):
    d2 = d // 2
    x = torch.randn(m, d, dtype=dtype, device=DEVICE)
    angles = torch.randn(m, d2, dtype=dtype, device=DEVICE)
    cos, sin = angles.cos(), angles.sin()  # precomputed frequency tables, mimicking real usage
    out = torch.empty_like(x)
    return x, cos, sin, out


def _rope_ref(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2].float(), x[..., d2:].float()
    c, s = cos.float(), sin.float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


@pytest.mark.parametrize("m,d", ROPE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rotary_position_embedding_triton(m, d, dtype):
    x, cos, sin, out = _make_rope_inputs(m, d, dtype)
    triton_impl.rotary_position_embedding(x, cos, sin, out)
    _check(out, _rope_ref(x, cos, sin), dtype)


# ---------------------------------------------------------------- dropout
# dropout is a random op; the two sides use different RNGs, so elementwise comparison with PyTorch is impossible. Verify on three levels:
#   1. with p=0 it must be the identity map (bit-exact);
#   2. with p>0: surviving positions must equal exactly x/(1-p) (checked exactly), and the zeroed ratio ≈ p;
#   3. two calls with the same seed must produce identical results (Philox reproducibility).
@pytest.mark.parametrize("n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_triton_p0(n, dtype):
    """p=0: nothing is dropped; y must equal x bit-exactly."""
    x, y = _make_unary_inputs(n, dtype)
    triton_impl.dropout(x, y, p=0.0)
    assert torch.equal(y, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_triton_mask(dtype):
    """p=0.25: exact check of surviving values + statistical check of the zeroed ratio."""
    p = 0.25
    x, y = _make_unary_inputs(1_000_001, dtype)  # large sample; ratio fluctuates ~4e-4
    triton_impl.dropout(x, y, p=p, seed=42)
    keep = y != 0  # randn almost never yields a true 0, so a zeroed slot is a dropped slot
    # surviving positions must equal exactly x/(1-p); this part can be checked exactly
    _check(y[keep], (x / (1 - p))[keep], dtype)
    # zeroed ratio ≈ p: ±0.02 is ~50 standard deviations, so no flakiness
    drop_ratio = 1.0 - keep.float().mean().item()
    assert abs(drop_ratio - p) < 0.02, f"zeroed ratio {drop_ratio:.4f}, expected {p}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_triton_deterministic(dtype):
    """Two calls with the same seed must match bit-exactly in both mask and values."""
    x, y1 = _make_unary_inputs(100_000, dtype)
    y2 = torch.empty_like(x)
    triton_impl.dropout(x, y1, p=0.5, seed=123)
    triton_impl.dropout(x, y2, p=0.5, seed=123)
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------- scaled_dot_product_attention
# Non-causal SDPA. Same tolerances as matmul: both tl.dot calls accumulate over K, so 1e-4 for fp32 and 1e-2 for fp16.
SDPA_SHAPES = [  # (B, H, M, N, D), deliberately mixing in M≠N, tile-unfriendly odd shapes, and small head_dim
    (1, 2, 64, 64, 64), (2, 4, 128, 128, 64),
    (1, 2, 300, 300, 64), (2, 3, 100, 150, 32),
]


@pytest.mark.parametrize("b,h,m,n,d", SDPA_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_scaled_dot_product_attention_triton(b, h, m, n, d, dtype):
    q = torch.randn(b, h, m, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, n, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, n, d, dtype=dtype, device=DEVICE)
    out = torch.empty_like(q)
    triton_impl.scaled_dot_product_attention(q, k, v, out)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    _check_mm(out, ref, dtype)


# ---------------------------------------------------------------- conv2d
# Direct convolution: accumulation length = C_in·KH·KW (tens to hundreds); same tolerance as matmul (fp32 1e-4).
CONV2D_SHAPES = [  # (N, C_in, C_out, H, W, K, stride, pad), mixing odd shapes / even kernels / stride=2
    (1, 3, 4, 8, 8, 3, 1, 0),
    (2, 8, 16, 17, 15, 3, 1, 1),
    (1, 4, 4, 10, 10, 5, 2, 2),
    (2, 3, 3, 9, 11, 4, 1, 0),
]


@pytest.mark.parametrize("n,cin,cout,h,w,k,s,p", CONV2D_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv2d_triton(n, cin, cout, h, w, k, s, p, dtype):
    x = torch.randn(n, cin, h, w, dtype=dtype, device=DEVICE)
    wt = torch.randn(cout, cin, k, k, dtype=dtype, device=DEVICE)
    h_out = (h + 2 * p - k) // s + 1
    w_out = (w + 2 * p - k) // s + 1
    y = torch.empty(n, cout, h_out, w_out, dtype=dtype, device=DEVICE)
    triton_impl.conv2d(x, wt, y, stride=s, padding=p)
    ref = torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
    _check_mm(y, ref, dtype)


# ---------------------------------------------------------------- max_pool2d
# max is pure selection: the output is always an original value from the input window, with zero arithmetic error,
# so like boolean/integer ops, verify bit-exactly with torch.equal.
MP2D_SHAPES = [  # (N, C, H, W, K, stride); None = defaults to K; includes non-divisible cases (tail window truncated)
    (1, 3, 8, 8, 2, None),
    (2, 8, 17, 15, 3, None),
    (2, 4, 10, 10, 2, 2),
    (1, 3, 9, 11, 3, 2),
    (2, 3, 9, 11, 2, 2),  # non-divisible tail block for the 2x2 fast path
]


@pytest.mark.parametrize("n,c,h,w,k,s", MP2D_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_max_pool2d_triton(n, c, h, w, k, s, dtype):
    x = torch.randn(n, c, h, w, dtype=dtype, device=DEVICE)
    stride = s if s is not None else k
    h_out, w_out = (h - k) // stride + 1, (w - k) // stride + 1
    y = torch.empty(n, c, h_out, w_out, dtype=dtype, device=DEVICE)
    triton_impl.max_pool2d(x, y, kernel_size=k, stride=s)
    ref = torch.nn.functional.max_pool2d(x, k, stride)
    assert torch.equal(y, ref)
