"""Isolated matmul diagnosis. Does not overwrite acceptance results."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels import triton_impl


@triton.jit
def persistent_mm(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tiles = tl.cdiv(M, BLOCK_M) * tiles_n
    my_tiles = tiles // programs + (pid < tiles % programs)
    tiles_k = tl.cdiv(K, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for step in range(my_tiles * tiles_k):
        ki = step % tiles_k
        tile = pid + step // tiles_k * programs
        mi = tile // tiles_n * BLOCK_M + tl.arange(0, BLOCK_M)
        ni = tile % tiles_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        av = tl.load(a + mi[:, None] * K + kk[None, :],
                     (mi[:, None] < M) & (kk[None, :] < K), 0)
        bv = tl.load(b + kk[:, None] * N + ni[None, :],
                     (kk[:, None] < K) & (ni[None, :] < N), 0)
        product = tl.dot(av, bv, input_precision="ieee")
        if ki == 0:
            acc = product
        else:
            acc += product
        if ki == tiles_k - 1:
            tl.store(c + mi[:, None] * N + ni[None, :], acc,
                     (mi[:, None] < M) & (ni[None, :] < N))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default=os.environ.get("TB_DEVICE", "mlu"))
    parser.add_argument("--dump", default="/tmp/codex_mm_asm")
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()
    if args.device == "mlu":
        import torch_mlu
    torch.manual_seed(20261022)
    dtype = getattr(torch, args.dtype)
    n = args.size
    a = torch.randn(n, n, device=args.device, dtype=dtype)
    b = torch.randn(n, n, device=args.device, dtype=dtype)
    c = torch.empty_like(a)
    ref = a @ b
    print(json.dumps({"event": "environment", "torch": torch.__version__,
                      "triton": triton.__version__, "dtype": args.dtype, "size": n,
                      "device": args.device}), flush=True)
    timings = triton.testing.do_bench(lambda: torch.mm(a, b, out=c),
                                     warmup=50, rep=200, quantiles=[0.2, 0.5, 0.8])
    print(json.dumps({"event": "pytorch", "ms_p20_p50_p80": timings}), flush=True)
    configs = [(64, 64, 32, 4), (64, 64, 128, 1), (64, 64, 256, 1),
               (64, 128, 128, 1), (128, 128, 128, 1), (128, 128, 256, 1),
               (128, 128, 512, 1), (128, 256, 128, 1), (256, 128, 128, 1),
               (128, 128, 128, 4), (128, 128, 256, 4)]
    if args.persistent:
        import triton.backends.mlu.driver as driver
        clusters = driver.BangUtils().get_device_properties(torch.mlu.current_device())["cluster_num"]
        configs = [(128, 128, 128, 4), (128, 128, 256, 4), (256, 128, 128, 4),
                   (256, 256, 128, 4), (256, 256, 256, 4), (384, 256, 128, 4)]
    for bm, bn, bk, warps in configs:
        config = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
        if args.persistent:
            config.update(num_stages=9, bottleneck="simd")
        start = time.monotonic()
        try:
            def run():
                if args.persistent:
                    return persistent_mm[(min(clusters, triton.cdiv(n, bm)*triton.cdiv(n, bn)),)](
                        a, b, c, n, n, n, **config)
                return triton_impl._mm_kernel[(triton.cdiv(n, bm), triton.cdiv(n, bn))](
                    a, b, c, n, n, n, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1), **config)
            compiled = run()
            torch.get_device_module(args.device).synchronize()
            diff = (c.float() - ref.float()).abs()
            error = dict(max_abs=diff.max().item(),
                         relative_l2=(diff.norm() / ref.float().norm()).item(),
                         legacy_allclose=torch.allclose(c, ref, atol=.01 if dtype == torch.float16 else .0001,
                                                        rtol=.01 if dtype == torch.float16 else .0001))
            timings = triton.testing.do_bench(run, warmup=25, rep=100,
                                             quantiles=[0.2, 0.5, 0.8])
            print(json.dumps({"event": "candidate", **config, **error,
                              "ms_p20_p50_p80": timings,
                              "compile_and_measure_s": time.monotonic()-start,
                              "asm_keys": list(compiled.asm)}), flush=True)
            if bm == 64 and bk == 32:
                dump = Path(args.dump)
                dump.mkdir(exist_ok=True, parents=True)
                for name, value in compiled.asm.items():
                    if isinstance(value, str):
                        (dump / f"{args.dtype}.{name}").write_text(value)
        except Exception as exc:
            print(json.dumps({"event": "error", **config,
                              "error": str(exc)[:1200]}), flush=True)


if __name__ == "__main__":
    main()
