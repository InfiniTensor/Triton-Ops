"""Check whether the known fp32 convolution failures are reference precision mismatches."""
import json
import os
from pathlib import Path
import sys

os.environ["TB_DEVICE"] = "mlu"
import torch
import torch_mlu
import triton
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels import triton_impl
from diagnostics.accuracy_metrics import accuracy_metrics

torch.manual_seed(20261022)
original = torch.backends.cnnl.allow_tf32
records = []
try:
    for shape in [(1, 3, 4, 8, 8, 3, 1, 0), (2, 8, 16, 17, 15, 3, 1, 1),
                  (1, 4, 4, 10, 10, 5, 2, 2), (2, 3, 3, 9, 11, 4, 1, 0)]:
        batch, ci, co, h, w, k, stride, pad = shape
        x = torch.randn(batch, ci, h, w, device="mlu")
        weight = torch.randn(co, ci, k, k, device="mlu")
        out = torch.empty(batch, co, (h+2*pad-k)//stride+1, (w+2*pad-k)//stride+1, device="mlu")
        triton_impl.conv2d(x, weight, out, stride=stride, padding=pad)
        for enabled in (True, False):
            torch.backends.cnnl.allow_tf32 = enabled
            reference = torch.nn.functional.conv2d(x, weight, stride=stride, padding=pad)
            row = {"shape": shape, "cnnl_allow_tf32": enabled,
                   "fp32_engineering_allclose_1e4": torch.allclose(out, reference, atol=1e-4, rtol=1e-4),
                   **accuracy_metrics(out, reference)}
            records.append(row)
            print(json.dumps(row), flush=True)
finally:
    torch.backends.cnnl.allow_tf32 = original
report = {"original_cnnl_allow_tf32": original, "torch": torch.__version__,
          "torch_mlu": torch_mlu.__version__, "triton": triton.__version__, "samples": records}
_conv_out = Path(__file__).resolve().parents[1] / "results/diagnostic_conv2d.json"
_conv_out.parent.mkdir(parents=True, exist_ok=True)
_conv_out.write_text(
    json.dumps(report, indent=2), encoding="utf-8")
