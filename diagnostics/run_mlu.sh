#!/usr/bin/env bash
# Run serially on one MLU to avoid benchmarks competing with correctness tests.
set -uo pipefail
export TB_DEVICE=mlu
python bench.py mm addmm rms_norm softmax layer_norm rotary_position_embedding scaled_dot_product_attention conv2d --dtype both --repeats 3 || exit $?
python diagnostics/accuracy_audit.py mm addmm --size 4096 || exit $?
python diagnostics/accuracy_audit.py rms_norm layer_norm softmax rotary_position_embedding scaled_dot_product_attention conv2d || exit $?
python diagnostics/model_shapes.py mm rms_norm rotary_position_embedding || exit $?
python -m pytest tests/ -v > results/test_mlu.txt 2>&1
test_status=$?
tail -n 22 results/test_mlu.txt
echo "PYTEST_EXIT=$test_status" | tee -a results/test_mlu.txt
exit "$test_status"
