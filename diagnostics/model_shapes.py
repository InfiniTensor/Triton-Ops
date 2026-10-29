"""Operator microbenchmarks from the public model config; no model is loaded.

Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
from kernels import triton_impl as kernels
from diagnostics.accuracy_metrics import accuracy_metrics


def cases(op, dtype):
    options = dict(dtype=dtype, device=bench.DEVICE)
    if op == "mm":
        for name, m, k, n in [("decode_q", 1, 4096, 4096), ("decode8_q", 8, 4096, 4096),
                               ("prefill128_q", 128, 4096, 4096),
                               ("prefill128_kv", 128, 4096, 1024),
                               ("prefill128_mlp_up", 128, 4096, 14336),
                               ("prefill128_mlp_down", 128, 14336, 4096)]:
            a, b = torch.randn(m, k, **options), torch.randn(k, n, **options)
            out = torch.empty(m, n, **options)
            yield name, [a, b, out], lambda: torch.mm(a, b, out=out), lambda: kernels.mm(a, b, out)
    elif op == "rms_norm":
        for tokens in (1, 128, 1024):
            x, w = torch.randn(tokens, 4096, **options), torch.randn(4096, **options)
            out = torch.empty_like(x)
            yield str(tokens), [x, w, out], lambda: torch.nn.functional.rms_norm(x, (4096,), w, 1e-5), lambda: kernels.rms_norm(x, w, out, eps=1e-5)
    elif op == "rotary_position_embedding":
        for tokens in (1, 128, 1024):
            x = torch.randn(tokens * 32, 128, **options)
            angles = torch.randn(tokens * 32, 64, **options)
            c, s = angles.cos(), angles.sin()
            out = torch.empty_like(x)
            def reference():
                a, b = x[:, :64].float(), x[:, 64:].float()
                return torch.cat((a*c.float()-b*s.float(), b*c.float()+a*s.float()), dim=-1).to(dtype)
            yield str(tokens), [x, c, s, out], reference, lambda: kernels.rotary_position_embedding(x, c, s, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="+", choices=["mm", "rms_norm", "rotary_position_embedding"])
    args = parser.parse_args()
    for op in args.ops:
        records = []
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(20261022)
            for name, inputs, reference, implementation in cases(op, dtype):
                expected = reference().clone()
                actual = implementation()
                row = {"name": name, "dtype": str(dtype), "shapes": [list(t.shape) for t in inputs],
                       "accuracy": accuracy_metrics(actual, expected)}
                rounds = {"PyTorch": [], "Triton": []}
                for repeat in range(3):
                    fns = [("PyTorch", reference), ("Triton", implementation)]
                    if repeat % 2:
                        fns.reverse()
                    for label, fn in fns:
                        rounds[label].append(triton.testing.do_bench(fn, quantiles=[.2, .5, .8]))
                row["rounds_p20_p50_p80_ms"] = rounds
                row["median_ms"] = {key: statistics.median(q[1] for q in values) for key, values in rounds.items()}
                records.append(row)
                print(op, name, dtype, row["median_ms"], flush=True)
        report = {"environment": bench._env_info(), "seed": 20261022,
                  "scope": "Operator microbenchmarks only; not an integrated model or end-to-end inference test. RoPE reference promotes arithmetic to fp32 as the kernel does.",
                  "model_config": {"hidden_size": 4096, "intermediate_size": 14336,
                                   "attention_heads": 32, "kv_heads": 8, "rms_norm_eps": 1e-5,
                                   "native_dtype": "bfloat16"}, "samples": records}
        path = Path(__file__).resolve().parents[1] / "results" / f"model_{op}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
