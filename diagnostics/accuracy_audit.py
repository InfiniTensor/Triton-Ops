"""Save numerical evidence per operator; this script does not certify acceptance."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
from diagnostics.accuracy_metrics import accuracy_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="+")
    parser.add_argument("--size", type=int)
    args = parser.parse_args()
    for op in args.ops:
        if op == "dropout":
            raise ValueError("Dropout requires a separate statistical and RNG-state test")
        spec = bench.OPS[op]
        rows = []
        dtypes = [torch.float32] if op.startswith("bitwise_") else [torch.float32, torch.float16]
        for dtype in dtypes:
            torch.manual_seed(20261022)
            bench.DTYPE = dtype
            n = args.size or spec["sizes"][1]
            inputs = spec["make_inputs"](n)
            fns = spec["fns"](*inputs)
            reference = fns["PyTorch"]().clone()
            actual = fns["Triton"]()
            row = {"dtype": str(inputs[0].dtype), "size": n,
                   "inputs": [list(t.shape) if torch.is_tensor(t) else t for t in inputs],
                   **accuracy_metrics(actual, reference)}
            rows.append(row)
            print(op, json.dumps(row), flush=True)
        report = {"environment": bench._env_info(), "seed": 20261022,
                  "acceptance_status": "not_certified_error_definition_pending", "samples": rows}
        path = Path(__file__).resolve().parents[1] / "results" / f"accuracy_{op}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
