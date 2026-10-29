"""Compare batched-row RoPE against flattened gather addressing."""
import json
import sys
from pathlib import Path

import torch
import torch_mlu
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@triton.jit
def flat_kernel(x, c, s, out, M, D2: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pair < M * D2
    offset = pair // D2 * (2 * D2) + pair % D2
    x1 = tl.load(x + offset, valid, 0).to(tl.float32)
    x2 = tl.load(x + offset + D2, valid, 0).to(tl.float32)
    cv = tl.load(c + pair, valid, 0).to(tl.float32)
    sv = tl.load(s + pair, valid, 0).to(tl.float32)
    tl.store(out + offset, x1 * cv - x2 * sv, valid)
    tl.store(out + offset + D2, x2 * cv + x1 * sv, valid)


@triton.jit
def rows_kernel(x, c, s, out, M, D2: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    mask = (row[:, None] < M) & (col[None, :] < D2)
    offset = row[:, None] * (2 * D2) + col[None, :]
    pair = row[:, None] * D2 + col[None, :]
    x1 = tl.load(x + offset, mask, 0).to(tl.float32)
    x2 = tl.load(x + offset + D2, mask, 0).to(tl.float32)
    cv = tl.load(c + pair, mask, 0).to(tl.float32)
    sv = tl.load(s + pair, mask, 0).to(tl.float32)
    tl.store(out + offset, x1 * cv - x2 * sv, mask)
    tl.store(out + offset + D2, x2 * cv + x1 * sv, mask)


torch.manual_seed(20261022)
records = []
for dtype in (torch.float32, torch.float16):
    m, d2 = 393216, 64
    x = torch.randn(m, 2*d2, dtype=dtype, device="mlu")
    angle = torch.randn(m, d2, dtype=dtype, device="mlu")
    c, s = angle.cos(), angle.sin()
    out = torch.empty_like(x)
    flat_kernel[(triton.cdiv(m*d2,8192),)](x,c,s,out,m,d2,BLOCK=8192,num_warps=1)
    reference = out.clone()
    for rows in (1, 8, 32, 64, 128):
        fn = lambda: rows_kernel[(triton.cdiv(m, rows),)](x, c, s, out, m, d2, ROWS=rows, COLS=64, num_warps=1)
        try:
            fn()
            correct = torch.equal(out, reference)
            timings = triton.testing.do_bench(fn, quantiles=[.2,.5,.8])
            record = {"dtype": str(dtype), "rows": rows, "exact_to_flat": correct, "p20_p50_p80_ms": timings}
        except Exception as exc:
            record = {"dtype": str(dtype), "rows": rows, "error": str(exc)}
        records.append(record)
        print(json.dumps(record), flush=True)
_rope_out = Path(__file__).resolve().parents[1] / "results/diagnostic_rotary_position_embedding.json"
_rope_out.parent.mkdir(parents=True, exist_ok=True)
_rope_out.write_text(json.dumps(records, indent=2))
