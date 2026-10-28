"""Single-variable experiment: unused output alias versus None for bias."""
import json
import re
from pathlib import Path
import sys

import torch
import torch_mlu
import triton
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.mlu_matmul import _persistent

torch.manual_seed(20261022)
n = 4096
a = torch.randn(n, n, dtype=torch.float16, device="mlu")
b = torch.randn_like(a)
out = torch.empty_like(a)
reference = a @ b
from triton.backends.mlu.driver import BangUtils
clusters = BangUtils().get_device_properties(torch.mlu.current_device())["cluster_num"]
records = []
for label, bias in [("unused_alias_of_output", out), ("unused_none", None)]:
    def fn():
        return _persistent.fn[(clusters,)](
            a, b, bias, out, n, n, n, n, 1, 1., 1., BIAS=False,
            BM=256, BN=256, BK=256, num_warps=4, num_stages=9, bottleneck="simd")
    compiled = fn()
    correct = torch.allclose(out, reference, atol=.01, rtol=.01)
    timings = triton.testing.do_bench(fn, quantiles=[.2,.5,.8])
    hints = {key: re.findall(r'\w*restrict\w*\s*=\s*\w+', compiled.asm[key])
             for key in ("ttir", "mluir")}
    row = {"case": label, "dtype": "float16", "shape": [n,n,n],
           "engineering_allclose": correct, "p20_p50_p80_ms": timings, "restrict_attributes": hints}
    records.append(row)
    print(json.dumps(row), flush=True)
(Path(__file__).resolve().parents[1] / "results/diagnostic_mm_alias.json").write_text(json.dumps(records, indent=2))
