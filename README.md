# Triton-Ops

手写 Triton 算子库（baseline 对照组）：40 个公开算子的 Triton 实现，在 NVIDIA（CUDA）和寒武纪（MLU）双平台做正确性验证，并与 PyTorch 原生实现对比性能。用途是为九齿（NineToothed）生成算子提供正确性与性能对照基线。

## 平台注意事项与性能基线

见 [MLU_NOTES.md](MLU_NOTES.md)：寒武纪 MLU 平台特性（grid 上限、num_warps 偏好、
dot-SIMD 路径等）、精度口径（IEEE 参考 / fp16 相对误差 ≤1%）、以及 40 个算子在
MLU590-M9 上的 T/P 性能基线摘要。

## 目录结构

```
Triton-Ops/
├── README.md                ← 本文件
├── MLU_NOTES.md             ← MLU 平台注意事项 + 精度口径 + 性能基线摘要
├── kernels/
│   ├── triton_impl.py       ← 手写 Triton 版算子（baseline 主体）
│   ├── mlu_matmul.py        ← MLU 专用 matmul 调度
│   └── ninetoothed_impl.py  ← 九齿版算子（历史保留，本阶段不测）
├── tests/
│   ├── test_correctness.py  ← 正确性测试：和 PyTorch 参考实现对答案
│   └── test_accuracy_metrics.py
├── diagnostics/             ← 精度/性能排查脚本、模型形状微基准、MLU 运行脚本
├── bench.py                 ← 性能对比：do_bench 测速，结果按算子自动存档
├── reference_precision.py   ← 参考精度策略开关
└── conftest.py
```

## 环境

| 环境 | 配置 | 用途 |
|---|---|---|
| 本机 | RTX 3060 Laptop（CUDA） | 日常开发与回归 |
| MLU 服务器 | MLU590-M9，torch 2.11 + torch_mlu 1.33.1 + triton 3.4.0（CNNL 2.2.14，驱动 6.5.52） | 寒武纪口径的验收基准，性能与正确性数字以此为准 |

## 怎么跑

依赖：PyTorch + Triton（MLU 环境为 torch_mlu）。

```bash
# 正确性测试（全部；只跑某个算子加 -k <算子名>）
python -m pytest tests/ -v

# 性能对比（全部算子；只跑指定算子：python bench.py addmm abs）
# 结果自动按算子存档到 results/bench_<算子>.md，头部自带设备/版本/commit 信息
# （results/ 为本地产物，不入库；可随时复现）
python bench.py
```

MLU 服务器（寒武纪）：

```bash
export TB_DEVICE=mlu            # 设备切换由环境变量控制
python -m pytest tests/ -v      # 全量正确性
python bench.py <算子...> --dtype both --repeats 3   # 性能对比
# 关键算子的串行全量流程（正确性 + 性能 + 精度审计）见 diagnostics/run_mlu.sh
```

## 怎么加一个新算子（标准流程）

1. 在 `kernels/triton_impl.py` 里写 Triton 版 kernel（参考已有的 `add`/`abs`）；
2. 在 `tests/test_correctness.py` 里加测试：和 PyTorch 原生实现对答案（`torch.allclose`，fp16 容差放宽），跑通；
3. 在 `bench.py` 的 `OPS` 表里登记（PyTorch + Triton 两项），跑测速；
4. 提交 commit（算子级提交）。

> 九齿版是后续阶段的事。`kernels/ninetoothed_impl.py` 保留历史实现；本阶段 bench 仅测 PyTorch/Triton，避免导入九齿触发设备初始化。

## 约定

- **先对后快**：正确性测试全绿才谈性能；
- 性能数字一律用 `triton.testing.do_bench`（自动 warmup + 多次取统计值）；
- 每组数字要带上：shape、dtype、设备（RTX 3060 Laptop / MLU590-M9）、时间单位 ms、换算带宽 GB/s。

## 测量与精度口径

- `python bench.py mm addmm --dtype both --repeats 3`：同一算子的 fp32/fp16 在同一个结果文件分段保存，记录三轮 p20/p50/p80、形状、步长及源码 SHA256。表格使用三轮 p50 的中位数。
- 混合容差正确性测试（fp16 atol=rtol=1e-2）保留为工程回归，不能代替相对误差 ≤1% 的正式验收口径；正式误差统计由 diagnostics/accuracy_audit.py 产出。
- MLU 的 mm/addmm 使用独立调度（`kernels/mlu_matmul.py`），CUDA 保留原实现。模型形状微基准不代表模型集成已完成。
- 精度路径显式对齐：默认 `TB_REFERENCE_PRECISION=ieee`，关闭 PyTorch 参考的 matmul/conv TF32，与 Triton 的 `input_precision="ieee"` 比较。`TB_REFERENCE_PRECISION=vendor-default` 可重现厂商默认路径，结果必须保留策略标签。
