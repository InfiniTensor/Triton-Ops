# Cambricon MLU Platform Notes and Performance Baseline

> This document consolidates the platform knowledge, precision conventions and
> performance baseline gathered while porting and tuning the Triton baseline on
> Cambricon MLU, for future maintenance and optimization reference.
> Day-to-day process records (daily progress, acceptance checklists, etc.) are kept
> in the internal repository and are not part of this deliverable.

## Measurement environment (baseline spec)

- Device: MLU590-M9 (single card)
- Software stack: torch 2.11.0 + torch_mlu 1.33.1 + triton 3.4.0; runtime CNNL 2.2.14, driver 6.5.52
- Reference environment: RTX 3060 Laptop (CUDA), used for daily development and dual-platform regression
- Method: `triton.testing.do_bench`, warmup 25 ms / rep 100 ms; 3 rounds with alternating
  implementation order; p20/p50/p80 recorded per round, tables use the median of the three p50 values;
  inputs shared per size, random seed 20261022; first JIT/autotune compilation excluded
- Output allocation: PyTorch interfaces that support `out=` reuse the output buffer, others
  allocate their own; Triton reuses the output. The numbers compare these call paths and
  should not be read as pure kernel time.
- Metrics: elementwise ops report effective bandwidth in GB/s (minimal algorithmic bytes
  moved ÷ elapsed time); matmul-family ops report TFLOPS = 2·N³ ÷ elapsed time
- Source measurements: archived per-operator bench runs dated 2026-10-22, each carrying the
  source-tree SHA256 of `kernels/triton_impl.py`, `kernels/mlu_matmul.py`,
  `reference_precision.py` and `bench.py` (kept in the internal repository)
- Note: numbers are not strictly comparable across image/driver upgrades; always quote the
  environment version when citing them

## Known MLU platform characteristics (lessons learned)

1. **Hardware limit of 65535 on the first grid dimension** (CUDA allows 2³¹-1). Large
   tensors need a grid-stride loop fallback; see the RoPE implementation in
   `kernels/triton_impl.py` for an example.
2. **The best `num_warps` is usually 1–2**; warps ≥ 4 with a medium BLOCK easily triggers
   resource overflow. `num_warps=2` is not supported by the MLU backend on some kernels
   (it falls back to 1 after a warning).
3. **Never point an unused pointer argument at the output tensor** — it flips `restrictPtr`
   to false in the compiled IR and visibly slows memory access; pass `None` for unused
   arguments instead.
4. **Elementwise ops prefer the opposite block shape vs CUDA**: CUDA favors small blocks
   with more warps, MLU favors large blocks with few warps (BLOCK 16384/32768 still pays
   off on MLU, while the same config makes CUDA-side ptxas compilation extremely slow).
   Keep per-platform configs (this repo switches at import time via `TB_DEVICE`).
5. **libdevice goes through a compatibility import layer**; in NineToothed-related files
   `import torch` must come first (workaround for a Triton device-query encoding bug on
   Chinese-localized Windows).
6. **CNNL's `cnnlGenerateRandDiscreteUniform` rejects int32-out-of-range bounds**, so
   `randint` upper bounds must be clamped into INT32_MAX (CUDA has no such restriction).
7. **Triton 3.4.0's `tl.dot` lowers to a SIMD path on the MLU backend — no tensor-core
   path** (mlu_matmul config bottleneck=simd). Evidence: conv2d fp16 and fp32 measure
   identically (0.38 vs 0.38 TFLOPS). The gap of such ops vs CUDA Tensor Cores is a
   hardware-path difference, not a dtype or implementation issue.

## Precision conventions

- Default reference policy `TB_REFERENCE_PRECISION=ieee`: TF32 is explicitly disabled for
  both matmul and conv on the PyTorch reference side, aligned with Triton's
  `input_precision="ieee"`. `vendor-default` reproduces the vendor default path; results
  measured that way must keep the policy label and must not be mixed into same-precision
  comparisons.
- Acceptance criterion: fp16 relative error ≤ 1% (the contract requires relative error
  ≤ 1% with 100% correctness under fp16; no official error formula or acceptance script
  has been designated yet, so full error statistics are preserved).
- Engineering regression: mixed-tolerance atol=rtol=1e-2 is kept as a regression gate;
  it is not equivalent to the formal ≤1% relative-error criterion. Boolean/integer ops
  are verified bit-exact with `torch.equal`.

## Accuracy audit (MLU590-M9, seed 20261022, IEEE reference policy)

Per-element statistics against the PyTorch reference. `>1% count` / `>1% share` count
elements whose relative error against a nonzero reference exceeds 1%; `max rel` is the
largest such elementwise relative error; `rel L2` is the relative L2 norm of the whole
output (worst sample per op/dtype). Boolean/integer ops are bit-exact by construction.

Worst relative L2 across all ops and dtypes: **0.0353%** — well within the ≤1% acceptance line.

| Op | dtype | elements | >1% count | >1% share | max rel (%) | rel L2 (%) |
|---|---|---:|---:|---:|---:|---:|
| abs | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| abs | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| add | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| add | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| addmm | float32 | 1,048,576 | 5 | 0.0005% | 0.1359 | 0.0000 |
| addmm | float16 | 1,048,576 | 1 | 0.0001% | 0.0508 | 0.0007 |
| bitwise_and | int32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| bitwise_not | int32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| bitwise_or | int32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| bmm | float32 | 4,194,304 | 22 | 0.0005% | 4.0000 | 0.0000 |
| bmm | float16 | 4,194,304 | 10 | 0.0002% | 0.2500 | 0.0007 |
| clamp | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| clamp | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| conv2d | float32 | 524,288 | 4 | 0.0008% | 0.1111 | 0.0000 |
| conv2d | float16 | 524,288 | 1 | 0.0002% | 0.0257 | 0.0007 |
| cos | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| cos | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| div | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| div | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| eq | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| eq | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| exp | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| exp | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| ge | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| ge | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| gelu | float32 | 1,000,000 | 1 | 0.0001% | 0.1633 | 0.0000 |
| gelu | float16 | 1,000,000 | 1 | 0.0001% | 0.5000 | 0.0003 |
| gt | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| gt | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| isinf | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| isinf | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| isnan | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| isnan | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| layer_norm | float32 | 4,194,304 | 4 | 0.0001% | 0.0833 | 0.0000 |
| layer_norm | float16 | 4,194,304 | 1 | 0.0000% | 0.0122 | 0.0003 |
| le | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| le | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| lt | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| lt | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| max_pool2d | float32 | 1,048,576 | 0 | 0.0000% | 0.0000 | 0.0000 |
| max_pool2d | float16 | 1,048,576 | 0 | 0.0000% | 0.0000 | 0.0000 |
| mm | float32 | 1,048,576 | 0 | 0.0000% | 0.0000 | 0.0000 |
| mm | float16 | 1,048,576 | 0 | 0.0000% | 0.0000 | 0.0000 |
| mul | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| mul | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| ne | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| ne | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| neg | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| neg | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| pow | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| pow | float16 | 1,000,000 | 0 | 0.0000% | 0.0010 | 0.0000 |
| relu | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| relu | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| rms_norm | float32 | 4,194,304 | 0 | 0.0000% | 0.0000 | 0.0000 |
| rms_norm | float16 | 4,194,304 | 0 | 0.0000% | 0.0010 | 0.0005 |
| rotary_position_embedding | float32 | 4,194,304 | 0 | 0.0000% | 0.0000 | 0.0000 |
| rotary_position_embedding | float16 | 4,194,304 | 19,901 | 0.4745% | 0.9180 | 0.0353 |
| rsqrt | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| rsqrt | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| scaled_dot_product_attention | float32 | 262,144 | 1 | 0.0004% | 0.0312 | 0.0000 |
| scaled_dot_product_attention | float16 | 262,144 | 2,123 | 0.8099% | 7.3636 | 0.0246 |
| sigmoid | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| sigmoid | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| silu | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| silu | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| sin | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| sin | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| softmax | float32 | 4,194,304 | 0 | 0.0000% | 0.0000 | 0.0000 |
| softmax | float16 | 4,194,304 | 0 | 0.0000% | 0.0029 | 0.0008 |
| sub | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| sub | float16 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| swiglu | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| swiglu | float16 | 1,000,000 | 2 | 0.0002% | 0.0769 | 0.0348 |
| tanh | float32 | 1,000,000 | 0 | 0.0000% | 0.0000 | 0.0000 |
| tanh | float16 | 1,000,000 | 0 | 0.0000% | 0.0009 | 0.0004 |

Notes on the two visible outliers (both fp16, both still within acceptance):

- `rotary_position_embedding` fp16: 19,901 elements (0.4745%) exceed 1% elementwise,
  max 0.9180, all at near-zero reference magnitudes; the whole-tensor relative L2 is far
  below 1%.
- `scaled_dot_product_attention` fp16: 2,123 elements (0.8099%) exceed 1% elementwise,
  max 7.3636; relative L2 remains far below 1%.

## Performance baseline (MLU590-M9, fp32, Triton vs native PyTorch)

T/P = Triton ÷ PyTorch (ratio of reported bandwidth or TFLOPS); < 1 means Triton is
slower. The summary table quotes the largest size tier; the per-operator tables below
list every measured size. `us` = microseconds, `ms` = milliseconds.

### Summary (largest size tier per op)

| Op | T/P | Op | T/P | Op | T/P |
|---|---:|---|---:|---|---:|
| abs | 1.00 | neg | 1.00 | cos | 0.87 |
| sin | 0.87 | exp | 1.99 | rsqrt | 1.15 |
| sigmoid | 0.62 | tanh | 0.74 | relu | 1.19 |
| silu | 0.57 | gelu | 0.45 | isnan | 1.74 |
| isinf | 1.97 | bitwise_not | 1.16 | add | 0.98 |
| sub | 0.98 | mul | 0.98 | div | 0.83 |
| pow | 0.99 | eq | 1.27 | ne | 1.27 |
| gt | 1.27 | ge | 1.27 | lt | 1.27 |
| le | 1.27 | bitwise_and | 1.37 | bitwise_or | 1.37 |
| clamp | 1.13 | softmax | 0.47 | layer_norm | 0.56 |
| rms_norm | 0.70 | mm | 0.60 | bmm | 0.63 |
| addmm | 0.66 | swiglu | 1.20 | rotary_position_embedding | 4.42 |
| scaled_dot_product_attention | 1.14 | dropout | 0.17 | conv2d | 0.009 |
| max_pool2d | 0.35 |   |   |   |   |

### Per-operator detail

#### Elementwise unary

**abs**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.9us  206.2GB/s | 3.1us  259.7GB/s | 1.26 |
| 1,000,000 | 6.6us 1219.5GB/s | 7.1us 1129.9GB/s | 0.93 |
| 10,000,000 | 40.0us 1998.0GB/s | 41.6us 1924.9GB/s | 0.96 |
| 50,000,000 | 188.0us 2127.7GB/s | 188.6us 2120.4GB/s | 1.00 |

**neg**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.9us  204.1GB/s | 3.1us  259.7GB/s | 1.27 |
| 1,000,000 | 6.6us 1219.5GB/s | 7.1us 1123.6GB/s | 0.92 |
| 10,000,000 | 40.0us 2000.0GB/s | 41.6us 1923.1GB/s | 0.96 |
| 50,000,000 | 188.2us 2124.9GB/s | 188.6us 2120.7GB/s | 1.00 |

**cos**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.5us  229.9GB/s | 3.2us  253.2GB/s | 1.10 |
| 1,000,000 | 7.2us 1105.0GB/s | 8.1us  985.2GB/s | 0.89 |
| 10,000,000 | 37.3us 2143.6GB/s | 47.8us 1673.6GB/s | 0.78 |
| 50,000,000 | 180.8us 2212.1GB/s | 207.1us 1931.2GB/s | 0.87 |

**sin**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.4us  232.6GB/s | 3.2us  253.2GB/s | 1.09 |
| 1,000,000 | 7.4us 1087.0GB/s | 8.1us  990.1GB/s | 0.91 |
| 10,000,000 | 37.3us 2143.6GB/s | 47.8us 1673.6GB/s | 0.78 |
| 50,000,000 | 180.8us 2212.9GB/s | 207.2us 1930.9GB/s | 0.87 |

**exp**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 5.2us  153.8GB/s | 3.3us  241.0GB/s | 1.57 |
| 1,000,000 | 14.6us  546.4GB/s | 8.8us  909.1GB/s | 1.66 |
| 10,000,000 | 101.4us  789.0GB/s | 52.8us 1516.3GB/s | 1.92 |
| 50,000,000 | 452.7us  883.6GB/s | 227.7us 1756.4GB/s | 1.99 |

**rsqrt**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.5us  229.9GB/s | 3.2us  253.2GB/s | 1.10 |
| 1,000,000 | 7.0us 1142.9GB/s | 8.1us  985.2GB/s | 0.86 |
| 10,000,000 | 40.0us 2000.0GB/s | 47.9us 1670.8GB/s | 0.84 |
| 50,000,000 | 237.4us 1684.6GB/s | 207.1us 1931.2GB/s | 1.15 |

**sigmoid**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.5us  177.0GB/s | 3.7us  217.4GB/s | 1.23 |
| 1,000,000 | 9.4us  851.1GB/s | 11.1us  719.4GB/s | 0.85 |
| 10,000,000 | 40.7us 1966.6GB/s | 68.9us 1160.8GB/s | 0.59 |
| 50,000,000 | 186.5us 2145.0GB/s | 298.4us 1340.3GB/s | 0.62 |

**tanh**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 5.2us  153.8GB/s | 4.4us  181.8GB/s | 1.18 |
| 1,000,000 | 12.5us  641.0GB/s | 16.4us  486.6GB/s | 0.76 |
| 10,000,000 | 72.8us 1098.9GB/s | 105.6us  757.3GB/s | 0.69 |
| 50,000,000 | 342.3us 1168.5GB/s | 462.7us  864.5GB/s | 0.74 |

**relu**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.8us  210.5GB/s | 3.0us  266.7GB/s | 1.27 |
| 1,000,000 | 7.2us 1105.0GB/s | 7.1us 1129.9GB/s | 1.02 |
| 10,000,000 | 40.1us 1994.0GB/s | 41.5us 1926.8GB/s | 0.97 |
| 50,000,000 | 225.1us 1776.8GB/s | 188.6us 2121.3GB/s | 1.19 |

**silu**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.7us  169.5GB/s | 3.8us  210.5GB/s | 1.24 |
| 1,000,000 | 9.6us  829.9GB/s | 11.8us  680.3GB/s | 0.82 |
| 10,000,000 | 42.9us 1865.7GB/s | 73.8us 1084.0GB/s | 0.58 |
| 50,000,000 | 182.0us 2198.3GB/s | 320.2us 1249.2GB/s | 0.57 |

**gelu**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 5.3us  151.5GB/s | 4.8us  166.7GB/s | 1.10 |
| 1,000,000 | 11.2us  711.7GB/s | 18.5us  432.0GB/s | 0.61 |
| 10,000,000 | 54.9us 1457.7GB/s | 124.2us  644.1GB/s | 0.44 |
| 50,000,000 | 248.7us 1608.2GB/s | 547.9us  730.1GB/s | 0.45 |

**isnan**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.3us  115.7GB/s | 3.0us  166.7GB/s | 1.44 |
| 1,000,000 | 8.2us  606.8GB/s | 6.1us  817.0GB/s | 1.35 |
| 10,000,000 | 45.0us 1110.1GB/s | 34.5us 1450.1GB/s | 1.31 |
| 50,000,000 | 267.1us  936.0GB/s | 153.3us 1631.0GB/s | 1.74 |

**isinf**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 8.8us   57.1GB/s | 3.2us  158.2GB/s | 2.77 |
| 1,000,000 | 13.5us  369.8GB/s | 6.8us  731.0GB/s | 1.98 |
| 10,000,000 | 68.5us  729.7GB/s | 39.2us 1276.8GB/s | 1.75 |
| 50,000,000 | 342.2us  730.6GB/s | 173.9us 1437.6GB/s | 1.97 |

**bitwise_not**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.4us  232.6GB/s | 3.0us  266.7GB/s | 1.15 |
| 1,000,000 | 6.4us 1250.0GB/s | 7.1us 1123.6GB/s | 0.90 |
| 10,000,000 | 39.5us 2024.3GB/s | 41.6us 1924.9GB/s | 0.95 |
| 50,000,000 | 218.5us 1830.7GB/s | 188.8us 2118.9GB/s | 1.16 |

#### Elementwise binary / compare

**add**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.4us  352.9GB/s | 3.3us  365.9GB/s | 1.04 |
| 1,000,000 | 8.0us 1500.0GB/s | 8.8us 1369.9GB/s | 0.91 |
| 10,000,000 | 54.3us 2209.9GB/s | 58.2us 2061.9GB/s | 0.93 |
| 50,000,000 | 264.3us 2270.3GB/s | 270.7us 2216.3GB/s | 0.98 |

**sub**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.6us  333.3GB/s | 3.3us  365.9GB/s | 1.10 |
| 1,000,000 | 8.0us 1507.5GB/s | 8.8us 1369.9GB/s | 0.91 |
| 10,000,000 | 54.3us 2210.8GB/s | 58.2us 2063.3GB/s | 0.93 |
| 50,000,000 | 264.3us 2270.3GB/s | 270.8us 2216.0GB/s | 0.98 |

**mul**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.7us  322.6GB/s | 3.2us  370.4GB/s | 1.15 |
| 1,000,000 | 8.0us 1500.0GB/s | 8.8us 1369.9GB/s | 0.91 |
| 10,000,000 | 54.3us 2209.1GB/s | 58.2us 2060.4GB/s | 0.93 |
| 50,000,000 | 264.4us 2269.6GB/s | 270.8us 2215.3GB/s | 0.98 |

**div**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  300.0GB/s | 3.7us  326.1GB/s | 1.09 |
| 1,000,000 | 8.7us 1376.1GB/s | 12.3us  977.2GB/s | 0.71 |
| 10,000,000 | 56.5us 2124.6GB/s | 75.2us 1595.7GB/s | 0.75 |
| 50,000,000 | 265.4us 2261.1GB/s | 319.3us 1879.0GB/s | 0.83 |

**pow**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.7us  215.1GB/s | 3.1us  259.7GB/s | 1.21 |
| 1,000,000 | 6.9us 1162.8GB/s | 7.1us 1123.6GB/s | 0.97 |
| 10,000,000 | 40.6us 1970.4GB/s | 41.6us 1923.1GB/s | 0.98 |
| 50,000,000 | 186.6us 2143.6GB/s | 188.9us 2117.7GB/s | 0.99 |

**eq**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  222.8GB/s | 3.2us  281.3GB/s | 1.26 |
| 1,000,000 | 7.8us 1148.0GB/s | 8.0us 1130.7GB/s | 0.98 |
| 10,000,000 | 44.7us 2012.5GB/s | 47.8us 1884.4GB/s | 0.94 |
| 50,000,000 | 272.0us 1654.2GB/s | 214.3us 2100.1GB/s | 1.27 |

**ne**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.2us  212.3GB/s | 3.2us  281.3GB/s | 1.33 |
| 1,000,000 | 8.3us 1087.0GB/s | 7.9us 1136.4GB/s | 1.05 |
| 10,000,000 | 45.0us 2001.8GB/s | 48.0us 1875.0GB/s | 0.94 |
| 50,000,000 | 271.2us 1659.4GB/s | 214.2us 2101.0GB/s | 1.27 |

**gt**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  222.8GB/s | 3.2us  277.8GB/s | 1.25 |
| 1,000,000 | 7.8us 1148.0GB/s | 8.0us 1125.0GB/s | 0.98 |
| 10,000,000 | 44.7us 2013.4GB/s | 48.0us 1873.4GB/s | 0.93 |
| 50,000,000 | 271.8us 1655.9GB/s | 214.7us 2096.1GB/s | 1.27 |

**ge**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  222.8GB/s | 3.2us  277.8GB/s | 1.25 |
| 1,000,000 | 7.8us 1148.0GB/s | 8.0us 1125.0GB/s | 0.98 |
| 10,000,000 | 44.8us 2010.7GB/s | 47.8us 1882.8GB/s | 0.94 |
| 50,000,000 | 271.4us 1658.3GB/s | 214.2us 2101.0GB/s | 1.27 |

**lt**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  222.8GB/s | 3.2us  277.8GB/s | 1.25 |
| 1,000,000 | 7.8us 1148.0GB/s | 8.0us 1125.0GB/s | 0.98 |
| 10,000,000 | 44.7us 2014.3GB/s | 48.0us 1873.4GB/s | 0.93 |
| 50,000,000 | 272.0us 1654.7GB/s | 214.8us 2094.6GB/s | 1.27 |

**le**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  222.8GB/s | 3.2us  277.8GB/s | 1.25 |
| 1,000,000 | 7.8us 1148.0GB/s | 8.0us 1125.0GB/s | 0.98 |
| 10,000,000 | 44.7us 2014.3GB/s | 48.0us 1875.0GB/s | 0.93 |
| 50,000,000 | 272.0us 1654.7GB/s | 214.6us 2096.9GB/s | 1.27 |

**bitwise_and**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.8us  319.1GB/s | 3.2us  375.0GB/s | 1.18 |
| 1,000,000 | 8.4us 1435.4GB/s | 8.8us 1369.9GB/s | 0.95 |
| 10,000,000 | 59.0us 2033.2GB/s | 58.1us 2064.7GB/s | 1.02 |
| 50,000,000 | 370.5us 1619.3GB/s | 270.9us 2215.0GB/s | 1.37 |

**bitwise_or**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 3.8us  319.1GB/s | 3.3us  365.9GB/s | 1.15 |
| 1,000,000 | 8.4us 1435.4GB/s | 8.8us 1369.9GB/s | 0.95 |
| 10,000,000 | 59.1us 2031.1GB/s | 58.2us 2060.4GB/s | 1.01 |
| 50,000,000 | 371.2us 1616.6GB/s | 271.1us 2213.4GB/s | 1.37 |

**clamp**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 4.0us  200.0GB/s | 3.2us  253.2GB/s | 1.27 |
| 1,000,000 | 8.1us  985.2GB/s | 7.8us 1025.6GB/s | 1.04 |
| 10,000,000 | 40.8us 1962.7GB/s | 45.7us 1751.3GB/s | 0.89 |
| 50,000,000 | 225.7us 1772.1GB/s | 199.3us 2007.2GB/s | 1.13 |

#### Normalization

**softmax**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 524,288 | 11.2us  375.8GB/s | 9.4us  448.1GB/s | 1.19 |
| 4,194,304 | 25.8us 1298.5GB/s | 45.8us  732.6GB/s | 0.56 |
| 16,777,216 | 78.7us 1705.0GB/s | 159.5us  841.6GB/s | 0.49 |
| 50,331,648 | 212.7us 1893.2GB/s | 448.6us  897.7GB/s | 0.47 |

**layer_norm**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 524,288 | 11.4us  369.2GB/s | 8.0us  521.7GB/s | 1.41 |
| 4,194,304 | 20.6us 1625.7GB/s | 35.4us  948.9GB/s | 0.58 |
| 16,777,216 | 63.0us 2129.8GB/s | 118.4us 1133.6GB/s | 0.53 |
| 50,331,648 | 184.7us 2179.8GB/s | 327.1us 1231.1GB/s | 0.56 |

**rms_norm**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 524,288 | 9.0us  464.0GB/s | 6.7us  627.9GB/s | 1.35 |
| 4,194,304 | 17.8us 1880.9GB/s | 28.7us 1170.0GB/s | 0.62 |
| 16,777,216 | 62.2us 2156.5GB/s | 95.5us 1405.1GB/s | 0.65 |
| 50,331,648 | 183.8us 2190.2GB/s | 264.3us 1523.4GB/s | 0.70 |

#### Matmul family

**mm**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 512 | 9.5us  28.20TFLOPS | 23.8us  11.28TFLOPS | 0.40 |
| 1,024 | 35.5us  60.53TFLOPS | 78.4us  27.39TFLOPS | 0.45 |
| 2,048 | 232.1us  74.01TFLOPS | 444.3us  38.67TFLOPS | 0.52 |
| 4,096 | 1.940ms  70.83TFLOPS | 3.227ms  42.59TFLOPS | 0.60 |

**bmm**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 4 | 39.2us  27.42TFLOPS | 39.1us  27.48TFLOPS | 1.00 |
| 16 | 86.6us  49.62TFLOPS | 116.0us  37.01TFLOPS | 0.75 |
| 64 | 272.7us  62.99TFLOPS | 421.3us  40.78TFLOPS | 0.65 |
| 128 | 520.2us  66.05TFLOPS | 826.2us  41.59TFLOPS | 0.63 |

**addmm**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 512 | 15.6us  17.16TFLOPS | 23.6us  11.39TFLOPS | 0.66 |
| 1,024 | 48.2us  44.55TFLOPS | 70.9us  30.28TFLOPS | 0.68 |
| 2,048 | 256.0us  67.11TFLOPS | 395.0us  43.49TFLOPS | 0.65 |
| 4,096 | 1.867ms  73.62TFLOPS | 2.829ms  48.59TFLOPS | 0.66 |

#### NN composite

**swiglu**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 8.3us  144.2GB/s | 4.1us  294.1GB/s | 2.04 |
| 1,000,000 | 17.4us  689.7GB/s | 14.1us  849.9GB/s | 1.23 |
| 10,000,000 | 97.6us 1230.0GB/s | 87.4us 1373.6GB/s | 1.12 |
| 50,000,000 | 446.4us 1344.0GB/s | 372.4us 1611.3GB/s | 1.20 |

**rotary_position_embedding**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 524,288 | 48.0us  131.1GB/s | 6.6us  953.3GB/s | 7.27 |
| 4,194,304 | 139.5us  360.7GB/s | 32.6us 1542.0GB/s | 4.28 |
| 16,777,216 | 464.4us  433.6GB/s | 112.2us 1793.7GB/s | 4.14 |
| 50,331,648 | 1.413ms  427.3GB/s | 319.9us 1888.1GB/s | 4.42 |

**scaled_dot_product_attention**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 128 | 44.9us   0.75TFLOPS | 9.1us   3.68TFLOPS | 4.91 |
| 512 | 94.0us   5.71TFLOPS | 54.2us   9.91TFLOPS | 1.74 |
| 1,024 | 219.4us   9.79TFLOPS | 165.9us  12.94TFLOPS | 1.32 |
| 2,048 | 673.2us  12.76TFLOPS | 588.8us  14.59TFLOPS | 1.14 |

**dropout**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 100,000 | 7.8us  102.6GB/s | 11.5us   69.4GB/s | 0.68 |
| 1,000,000 | 18.2us  438.6GB/s | 55.6us  143.9GB/s | 0.33 |
| 10,000,000 | 87.0us  920.0GB/s | 472.6us  169.3GB/s | 0.18 |
| 50,000,000 | 388.8us 1028.7GB/s | 2.290ms  174.7GB/s | 0.17 |

**conv2d**

dtype: float32

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 32 | 19.7us   3.84TFLOPS | 223.0us   0.34TFLOPS | 0.089 |
| 64 | 29.0us  10.41TFLOPS | 932.4us   0.32TFLOPS | 0.031 |
| 128 | 47.2us  25.59TFLOPS | 3.423ms   0.35TFLOPS | 0.014 |
| 224 | 86.8us  42.64TFLOPS | 9.775ms   0.38TFLOPS | 0.009 |

dtype: float16

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 32 | 15.2us   4.97TFLOPS | 289.0us   0.26TFLOPS | 0.052 |
| 64 | 18.6us  16.24TFLOPS | 892.8us   0.34TFLOPS | 0.021 |
| 128 | 23.9us  50.58TFLOPS | 3.289ms   0.37TFLOPS | 0.007 |
| 224 | 40.4us  91.48TFLOPS | 9.802ms   0.38TFLOPS | 0.004 |

**max_pool2d**

| Size | PyTorch | Triton | T/P |
|---|---:|---:|---:|
| 128 | 18.0us  290.6GB/s | 157.1us   33.4GB/s | 0.11 |
| 256 | 44.0us  476.6GB/s | 312.3us   67.1GB/s | 0.14 |
| 512 | 149.5us  561.2GB/s | 693.7us  120.9GB/s | 0.22 |
| 1,024 | 586.7us  571.9GB/s | 1.697ms  197.8GB/s | 0.35 |

## Known gaps (⚠) and diagnosis

- **conv2d (T/P ≈ 0.009 at the largest tier)**: the largest gap. On MLU, Triton's `tl.dot`
  lowers to SIMD with no tensor-core path (platform note 7) — a hardware-path difference.
  Future directions: a schedule that fits MLU better, or substituting the vendor library.
  At size 224, fp32: PyTorch 86.8 us · 42.64 TFLOPS vs Triton 9.775 ms · 0.38 TFLOPS;
  fp16: PyTorch 40.4 us · 91.48 TFLOPS vs Triton 9.802 ms · 0.38 TFLOPS. Triton measures
  identically in fp16 and fp32 (0.38 TFLOPS), confirming the missing tensor-core path.
- **bmm (0.63) / mm (0.60) / addmm (0.66)**: same dot-SIMD cause. SDPA has already
  overtaken PyTorch after search-space expansion (1.14), showing there is still headroom
  in large-tile + few-warp schedules.
- **dropout (0.17)**: Philox random number generation has low throughput on MLU; to be
  optimized.
- **max_pool2d (0.34)**: improved from 0.05 to 0.34 after the memory-access rework;
  window reuse still has room.
- Elementwise ops overall: median T/P ≈ 1.1; most are at parity or ahead of PyTorch.
