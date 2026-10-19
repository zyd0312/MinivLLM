# Fused 路径整体 GPU Profile

- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- Backend：`fused`
- Prompt tokens：20
- Profile max completion tokens：2

本报告包含 profiler instrumentation，只用于定位瓶颈；正式吞吐和延迟应以无 profiler 的 performance 报告为准。

⚠️ **当前 trace 未采集到 CUDA kernel activity。设备时间 unavailable；请勿把它解释为 0 ms。**


## 阶段级耗时

| 阶段 | 次数 | Wall P50 | Wall P90 | CUDA stream P50 | CUDA stream P90 | 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 90.180 ms | 90.180 ms | 90.066 ms | 90.066 ms | 20 |
| decode | 1 | 65.995 ms | 65.995 ms | 65.963 ms | 65.963 ms | 1 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `mlp.down_proj` | `[1, 6144]` | 28 | 0.194 | 0.176 | 0.236 |
| decode | `mlp.gate_proj` | `[1, 2048]` | 28 | 0.151 | 0.144 | 0.176 |
| decode | `self_attn.q_proj` | `[1, 2048]` | 28 | 0.173 | 0.134 | 0.214 |
| decode | `self_attn.o_proj` | `[1, 2048]` | 28 | 0.179 | 0.127 | 0.304 |
| decode | `mlp.up_proj` | `[1, 2048]` | 28 | 0.135 | 0.126 | 0.153 |
| decode | `self_attn.k_proj` | `[1, 2048]` | 28 | 0.177 | 0.123 | 0.169 |
| decode | `self_attn.v_proj` | `[1, 2048]` | 28 | 0.162 | 0.117 | 0.184 |
| prefill | `mlp.down_proj` | `[20, 6144]` | 28 | 0.314 | 0.302 | 0.371 |
| prefill | `mlp.gate_proj` | `[20, 2048]` | 28 | 0.212 | 0.188 | 0.243 |
| prefill | `self_attn.q_proj` | `[20, 2048]` | 28 | 0.244 | 0.183 | 0.507 |
| prefill | `self_attn.o_proj` | `[20, 2048]` | 28 | 0.209 | 0.178 | 0.301 |
| prefill | `self_attn.k_proj` | `[20, 2048]` | 28 | 0.171 | 0.145 | 0.227 |
| prefill | `self_attn.v_proj` | `[20, 2048]` | 28 | 0.180 | 0.141 | 0.218 |
| prefill | `mlp.up_proj` | `[20, 2048]` | 28 | 0.172 | 0.138 | 0.258 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/nsys-request/run-20261018-2310/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/nsys-request/run-20261018-2310/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
