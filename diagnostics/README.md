**Diagnostic scripts and re-measurement notes**

These scripts record measurement evidence; they do not replace the mentor's formal
acceptance scripts. The final formula for relative error has not been designated yet; the
user has asked to preserve full statistics first.

- `matmul_scan.py`: compares matmul scheduling configurations with a fixed random seed.
  By default it scans the original 2D tiling; `--persistent` scans persistent kernels and
  software pipelining. Output includes failing and rejected configurations — not every
  configuration is a delivery candidate.
- `accuracy_audit.py`: generates `results/accuracy_<op>.json` per operator, recording
  elementwise relative error against nonzero references, zero-reference deviation,
  non-finite mismatches, and relative L2 error. Relative L2 is not automatically the
  contractual acceptance metric.
- `model_shapes.py`: derives operator shapes from the public DeepSeek-R1-Distill-Llama-8B
  config, covering fp16 and the model's original bf16. Results go to
  `results/model_<op>.json`. No model weights were downloaded; this does not imply that
  model replacement or end-to-end inference is done. RMSNorm explicitly uses the model's
  eps=1e-5.

Cambricon examples (run from the repository root; the existing container already sets
TB_DEVICE=mlu):

```bash
python diagnostics/matmul_scan.py --size 4096 --dtype float16 --persistent
python diagnostics/accuracy_audit.py mm addmm --size 4096
python diagnostics/accuracy_audit.py rms_norm layer_norm softmax rotary_position_embedding scaled_dot_product_attention
python diagnostics/model_shapes.py mm rms_norm rotary_position_embedding
python bench.py mm addmm rms_norm softmax layer_norm rotary_position_embedding scaled_dot_product_attention --dtype both --repeats 3
```

Regular performance archives remain `results/bench_<op>.md`, with fp32/fp16 as sections in
one file. The current method takes the median of the three rounds' p50 values and keeps
each round's p20/p50/p80, input shapes, strides, actual dtype and code SHA256; the old
method averaged by default, and historical numbers must not be retroactively claimed as
medians. When comparing numbers across the change, note this convention change.

Git tag `baseline-before-codex-20261022` points to the original 8f82604, allowing the first
version to be restored. Current scheduling changes preserve IEEE multiplication and fp32
accumulation; speedups are never obtained by lowering precision, calling PyTorch instead of
Triton, or turning failing tests into passing ones.

`conv_precision.py` toggles only CNNL's TF32 switch on identical inputs, saving error data
for the four historically failing shapes. Regular tests and benchmarks explicitly disable
reference TF32 by default, aligned with Triton's IEEE path; this fixes the reference
precision configuration and does not relax numerical tolerances.

The single-variable experiment `matmul_alias.py` records the MLU JIT's conservative
handling of repeated storage pointers: if an unused bias argument points at the output,
restrictPtr becomes false; passing None turns it back to true. The same fp16 4096-order
GEMM on a 256×256×256 tile went from ~1.64 ms to ~0.67 ms with unchanged numerical
regression. The implementation keeps backend checks for genuine input/output aliasing and
does not force restrict on.

`rope_scan.py` compares 1D indexing with multi-row 2D access; the failure of ROWS=1 at
large sizes beyond the grid limit is kept in the diagnostic file and is not a delivery
configuration.
