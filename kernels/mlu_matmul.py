"""MLU GEMM schedules; IEEE multiplication and fp32 accumulation are preserved.

Persistent work distribution follows the installed Cambricon Triton matmul
tutorial. CUDA continues to use the original kernels in triton_impl.py.
"""
import functools

import torch
import triton
import triton.language as tl


def _prune(configs, args, **kwargs):
    m, n, k = args["M"], args["N"], args["K"]
    # Retain a small fallback, but avoid running hundreds of padded rows for decode.
    return [c for c in configs if c.kwargs["BM"] <= max(16, triton.next_power_of_2(m))
            and c.kwargs["BN"] <= max(64, triton.next_power_of_2(n))
            and c.kwargs["BK"] <= max(64, triton.next_power_of_2(k))] or configs[:1]


@triton.autotune(configs=[
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=1, num_stages=0),
    triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=1, num_stages=0),
    triton.Config({"BM": 128, "BN": 128, "BK": 128}, num_warps=1, num_stages=0),
    triton.Config({"BM": 128, "BN": 128, "BK": 256}, num_warps=1, num_stages=0),
    triton.Config({"BM": 256, "BN": 128, "BK": 128}, num_warps=1, num_stages=0),
    triton.Config({"BM": 128, "BN": 256, "BK": 128}, num_warps=1, num_stages=0),
], key=["M", "N", "K", "BIAS"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _direct(a, b, bias, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, alpha, beta, BIAS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        av = tl.load(a + m[:, None] * K + k[None, :],
                     (m[:, None] < M) & (k[None, :] < K), 0)
        bv = tl.load(b + k[:, None] * N + n[None, :],
                     (k[:, None] < K) & (n[None, :] < N), 0)
        acc += tl.dot(av, bv, input_precision="ieee")
    if BIAS:
        value = tl.load(bias + n, n < N, 0)
        acc = alpha * acc + beta * value[None, :]
    tl.store(out + m[:, None] * S0 + n[None, :] * S1, acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.autotune(configs=[
    triton.Config({"BM": 16, "BN": 64, "BK": 64}, num_warps=4, num_stages=9, bottleneck="simd"),
    triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=9, bottleneck="simd"),
    triton.Config({"BM": 128, "BN": 128, "BK": 128}, num_warps=4, num_stages=9, bottleneck="simd"),
    triton.Config({"BM": 128, "BN": 128, "BK": 256}, num_warps=4, num_stages=9, bottleneck="simd"),
    triton.Config({"BM": 256, "BN": 128, "BK": 128}, num_warps=4, num_stages=9, bottleneck="simd"),
    triton.Config({"BM": 256, "BN": 256, "BK": 256}, num_warps=4, num_stages=9, bottleneck="simd"),
], key=["M", "N", "K", "BIAS"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _persistent(a, b, bias, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, alpha, beta, BIAS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    tiles_n = tl.cdiv(N, BN)
    tiles = tl.cdiv(M, BM) * tiles_n
    my_tiles = tiles // programs + (pid < tiles % programs)
    tiles_k = tl.cdiv(K, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    # Flatten output-tile/K loops so the MLU compiler can pipeline across tiles.
    for step in range(my_tiles * tiles_k):
        ki = step % tiles_k
        tile = pid + step // tiles_k * programs
        m = tile // tiles_n * BM + tl.arange(0, BM)
        n = tile % tiles_n * BN + tl.arange(0, BN)
        k = ki * BK + tl.arange(0, BK)
        av = tl.load(a + m[:, None] * K + k[None, :],
                     (m[:, None] < M) & (k[None, :] < K), 0)
        bv = tl.load(b + k[:, None] * N + n[None, :],
                     (k[:, None] < K) & (n[None, :] < N), 0)
        product = tl.dot(av, bv, input_precision="ieee")
        if ki == 0:
            acc = product
        else:
            acc += product
        if ki == tiles_k - 1:
            value = acc
            if BIAS:
                bias_value = tl.load(bias + n, n < N, 0)
                value = alpha * acc + beta * bias_value[None, :]
            tl.store(out + m[:, None] * S0 + n[None, :] * S1, value,
                     (m[:, None] < M) & (n[None, :] < N))


@functools.lru_cache(None)
def _cluster_count(device_index):
    from triton.backends.mlu.driver import BangUtils
    return BangUtils().get_device_properties(device_index)["cluster_num"]


def matmul(a, b, out, bias=None, alpha=1.0, beta=1.0):
    a, b = a.contiguous(), b.contiguous()
    m, k = a.shape
    n = b.shape[1]
    has_bias = bias is not None
    # Repeating out as an unused placeholder disables MLU restrict_ptr_hint.
    # None is eliminated by the constexpr BIAS=False branch without aliasing.
    bias_ptr = bias if has_bias else None
    args = (a, b, bias_ptr, out, m, n, k, *out.stride(), alpha, beta)
    if a.dtype == torch.float32 or m < 64:
        grid = lambda cfg: (triton.cdiv(m, cfg["BM"]), triton.cdiv(n, cfg["BN"]))
        _direct[grid](*args, BIAS=has_bias)
    else:
        clusters = _cluster_count(a.device.index)
        grid = lambda cfg: (min(clusters, triton.cdiv(m, cfg["BM"]) * triton.cdiv(n, cfg["BN"])),)
        _persistent[grid](*args, BIAS=has_bias)
    return out
