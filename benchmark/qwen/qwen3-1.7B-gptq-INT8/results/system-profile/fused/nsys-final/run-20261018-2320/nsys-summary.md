# Nsight Systems fused profile

运行方式：`run_nsys_fused.sh --max-tokens 2`，关闭进程内 PyTorch profiler，避免 CUPTI 多订阅冲突。

## CUDA kernel 汇总

来自 `summary_cuda_gpu_kern_sum.txt`：

| Kernel | 调用次数 | GPU 总时间 | 占 kernel 总时间 |
|---|---:|---:|---:|
| `_gptq_prefill_kernel` | 588 | 349.495 ms | 83.7% |
| `_gptq_decode_kernel` | 392 | 17.002 ms | 4.1% |
| `flash_attention_varlen_kernel` | 84 | 5.210 ms | 1.2% |
| `triton_poi_fused_mul_silu_split_0` | 140 | 4.517 ms | 1.1% |

这里的百分比只针对 Nsight 捕获到的 CUDA kernel activity，不是端到端 wall time；该次报告包含初始化/warmup，且使用 `max_tokens=2` 的短请求，因此首次 kernel/编译和初始化搬运会影响结果。当前脚本后续运行会为请求范围加 NVTX 标记，便于在时间线中定位。

## 解释

- fused GPTQ kernel 已在 GPU timeline 中出现，说明 fused 路径实际执行了 `_gptq_prefill_kernel` 和 `_gptq_decode_kernel`。
- 当前短 trace 中 prefill kernel 总时间明显高于 decode kernel 总时间；decode 仍需用更长 trace 或正式 benchmark 分析稳定的每 token 成本。
- `cuda_gpu_sum.txt` 还显示大量 Host-to-Device memcpy，短 trace 中约占 GPU activity 的 79.8%，这部分包含初始化/数据搬运，不能直接归因于 GPTQ 算子。

原始报告：`fused.nsys-rep`；可在 Nsight Systems 打开。机器可读统计：`summary_cuda_gpu_kern_sum.txt`、`summary_cuda_gpu_sum.txt`、`summary_cuda_api_sum.txt`。
