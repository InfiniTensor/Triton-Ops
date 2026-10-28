**诊断脚本与复测说明**

这些脚本记录测量证据，不替代带教的正式验收脚本。相对误差的最终公式尚未指定；用户已要求先保存完整统计。

- `matmul_scan.py`：固定随机种子比较矩阵乘调度配置。默认扫描原二维分块；`--persistent` 扫描常驻任务与软件流水线。输出含错误和被拒绝的配置，不代表每个配置都用于交付。
- `accuracy_audit.py`：每个算子生成 `results/accuracy_<op>.json`，分别记录非零参考值的逐元素相对误差、零参考值偏差、非有限值不一致和相对 L2 误差。相对 L2 不是自动采用的合同验收标准。
- `model_shapes.py`：根据公开的 DeepSeek-R1-Distill-Llama-8B 配置测算子形状，包含 fp16 和模型原始配置的 bf16。结果写入 `results/model_<op>.json`。没有下载模型权重，不代表模型替换或端到端推理已完成。RMSNorm 显式使用模型的 eps=1e-5。

寒武纪示例（在仓库根运行，现有容器已设 TB_DEVICE=mlu）：

```bash
python diagnostics/matmul_scan.py --size 4096 --dtype float16 --persistent
python diagnostics/accuracy_audit.py mm addmm --size 4096
python diagnostics/accuracy_audit.py rms_norm layer_norm softmax rotary_position_embedding scaled_dot_product_attention
python diagnostics/model_shapes.py mm rms_norm rotary_position_embedding
python bench.py mm addmm rms_norm softmax layer_norm rotary_position_embedding scaled_dot_product_attention --dtype both --repeats 3
```

常规性能存档仍为 `results/bench_<op>.md`，fp32/fp16 在同一文件中分段。新方法取三轮各自 p50 的中位数，并保留每轮 p20/p50/p80、输入形状、步长、实际 dtype 和代码 SHA256；旧方法默认取均值，历史数字不能伪称中位数。比较前后的数值时，应注明这个口径变化。

Git 标签 `baseline-before-codex-20261022` 指向原始 8f82604，可恢复查看初版。当前调度改动保留 IEEE 乘法与 fp32 累加；不通过降低精度、调用 PyTorch 代替 Triton 或把失败测试改成通过获得加速。

`conv_precision.py` 在同一输入上仅切换 CNNL 的 TF32 开关，保存四个历史失败形状的误差数据。常规测试和测速默认显式关闭参考 TF32，与 Triton 的 IEEE 路径对齐；这是修正参考精度配置，没有放宽数值容差。

单变量实验 `matmul_alias.py` 记录了 MLU JIT 对重复 storage 指针的保守处理：不用的 bias 参数如果指向输出，会使 restrictPtr=false；传 None 后变为 true。同一 256×256×256 tile 的 fp16 4096 阶 GEMM 从约 1.64ms 到约 0.67ms，数值回归不变。实现保留对真实输入/输出别名的后端检查，没有强制开启 restrict。

`rope_scan.py` 对比一维索引与多行二维访问；大规格 ROWS=1 超过 grid 上限的失败保留在诊断文件，不作为交付配置。
