# 寒武纪 MLU 平台注意事项与基线摘要

> 本文汇总 triton baseline 在寒武纪 MLU 上适配与调优过程中沉淀的平台知识、
> 精度口径和性能基线，供后续维护与优化参考。
> 过程性记录（每日进度、验收清单等）不入库，存档于内部仓库。

## 测量环境（基线口径）

- 设备：MLU590-M9（单卡）
- 软件栈：torch 2.11.0 + torch_mlu 1.33.1 + triton 3.4.0；运行时 CNNL 2.2.14，驱动 6.5.52
- 对照环境：RTX 3060 Laptop（CUDA），用于日常开发与双平台回归
- 测量方法：`triton.testing.do_bench`，3 轮、每轮 p20/p50/p80，表取各轮 p50 的中位数；
  输入共享、随机种子 20261022；不计首次 JIT/autotune 编译
- 注意：更换镜像/驱动版本后数字不完全可比，引用时请带上环境版本

## MLU 平台已知特性（踩坑记录）

1. **grid 第一维硬件上限 65535**（CUDA 为 2³¹-1）。大规格张量需 grid-stride
   循环兜底，RoPE 的实现在 `kernels/triton_impl.py` 中有示范。
2. **最优 num_warps 通常是 1~2**；warps≥4 配中等 BLOCK 易触发资源超限。
   warps=2 在部分 kernel 上不被 MLU 后端支持（warning 后回退 1）。
3. **不要让闲置的指针参数指向输出 tensor**——会把编译 IR 的 restrictPtr
   打成 false，明显拖慢访存；不用的参数传 `None`。
4. **逐元素算子偏好的 block 形状与 CUDA 相反**：CUDA 小块多 warp、MLU 大块
   少 warp（BLOCK 16384/32768 在 MLU 上仍有显著收益，但同样的配置会让 CUDA
   侧 ptxas 编译极慢）。务必按平台分配置（本仓库用 `TB_DEVICE` 在 import 时区分）。
5. **libdevice 走兼容导入层**；九齿相关文件 `import torch` 必须在最前
   （中文 Windows 上 Triton 设备查询有编码 bug 的 workaround）。
6. **CNNL 的 `cnnlGenerateRandDiscreteUniform` 不接受 int32 越界边界**，
   randint 输入上限需收拢进 INT32_MAX（CUDA 无此限制）。
7. **Triton 3.4.0 的 `tl.dot` 在 MLU 后端走 SIMD、无张量核路径**
   （mlu_matmul 配置 bottleneck=simd）。conv2d fp16 与 fp32 实测同速
   （0.38 vs 0.38 TFLOPS）即是证据——这类算子与 CUDA 上 Tensor Core 的差距
   属硬件路径差异，不是 dtype 或实现问题。

## 精度口径

- 参考精度策略默认 `TB_REFERENCE_PRECISION=ieee`：matmul/conv 两侧 TF32 全关，
  与 Triton 的 `input_precision="ieee"` 对齐；`vendor-default` 可重现厂商默认路径，
  结果必须保留策略标签。
- 验收口径：fp16 相对误差 ≤1%（现行 39 个算子整体 relative L2 全部 ≤0.035%）。
- 工程回归：fp32 内部加严 atol=rtol=1e-2（混合容差，不等同于纯相对误差，
  仅用于回归拦截）；布尔/整数算子用 `torch.equal` 逐位精确比对。

## 性能基线摘要（MLU590-M9，fp32，各算子最大规模档，Triton vs PyTorch 原生）

T/P = Triton ÷ PyTorch（带宽或算力之比），<1 表示 Triton 更慢。
完整四档规模数据与逐算子明细存档于内部仓库（results/ 不入库，可用 bench.py 复现）。

| 算子 | T/P | 算子 | T/P | 算子 | T/P |
|---|---:|---|---:|---|---:|
| abs | 1.00 | neg | 1.00 | cos | 0.87 |
| sin | 0.87 | exp | 1.99 | rsqrt | 1.15 |
| sigmoid | 0.62 | tanh | 0.74 | relu | 1.19 |
| silu | 0.57 | gelu | 0.45 | isnan | 1.74 |
| isinf | 1.97 | bitwise_not | 1.16 | add | 0.98 |
| sub | 0.98 | mul | 0.98 | div | 0.83 |
| pow（平方特例） | 0.99 | eq/ne/gt/ge/lt/le | 1.27 | bitwise_and/or | 1.37 |
| clamp | 1.13 | softmax | 0.47 | layer_norm | 0.56 |
| rms_norm | 0.70 | mm | 0.60 | bmm | 0.63 ⚠ |
| addmm | 0.66 | swiglu | 1.20 | rotary_position_embedding | 4.42 |
| scaled_dot_product_attention | 1.14 | dropout | 0.17 ⚠ | conv2d | 0.009 ⚠ |
| max_pool2d | 0.34 |  |  |  |  |

已知差距项（⚠）与定位结论：

- **conv2d（0.009）**：差距最大。MLU 上 Triton `tl.dot` 走 SIMD、无张量核路径
  （见平台特性 7），属硬件路径差异；后续方向是更贴合 MLU 的调度或厂商库替代。
- **bmm（0.63）/ mm（0.60）/ addmm（0.66）**：同受 dot-SIMD 路径影响；
  SDPA 经搜索空间扩充已反超（1.14），说明大 tile + 少 warp 的调度仍有挖掘空间。
- **dropout（0.17）**：Philox 随机数生成在 MLU 上吞吐低，待优化。
- **max_pool2d（0.34）**：访存改造后已从 0.05 提升到 0.34，仍有窗口复用空间。
- 逐元素算子整体：T/P 中位数约 1.1，多数已打平或反超 PyTorch。
