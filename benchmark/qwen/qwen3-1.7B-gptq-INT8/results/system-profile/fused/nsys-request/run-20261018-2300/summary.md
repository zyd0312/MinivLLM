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
| prefill | 1 | 131.442 ms | 131.442 ms | 131.288 ms | 131.288 ms | 20 |
| decode | 1 | 112.436 ms | 112.436 ms | 112.405 ms | 112.405 ms | 1 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `mlp.down_proj` | `[1, 6144]` | 28 | 0.293 | 0.231 | 0.427 |
| decode | `mlp.gate_proj` | `[1, 2048]` | 28 | 0.331 | 0.191 | 0.544 |
| decode | `self_attn.q_proj` | `[1, 2048]` | 28 | 0.254 | 0.184 | 0.313 |
| decode | `self_attn.o_proj` | `[1, 2048]` | 28 | 0.239 | 0.177 | 0.426 |
| decode | `mlp.up_proj` | `[1, 2048]` | 28 | 0.268 | 0.163 | 0.559 |
| decode | `self_attn.k_proj` | `[1, 2048]` | 28 | 0.185 | 0.162 | 0.270 |
| decode | `self_attn.v_proj` | `[1, 2048]` | 28 | 0.190 | 0.146 | 0.224 |
| prefill | `mlp.down_proj` | `[20, 6144]` | 28 | 0.408 | 0.332 | 0.536 |
| prefill | `mlp.gate_proj` | `[20, 2048]` | 28 | 0.265 | 0.221 | 0.383 |
| prefill | `self_attn.o_proj` | `[20, 2048]` | 28 | 0.245 | 0.218 | 0.355 |
| prefill | `self_attn.q_proj` | `[20, 2048]` | 28 | 0.321 | 0.214 | 0.629 |
| prefill | `self_attn.k_proj` | `[20, 2048]` | 28 | 0.274 | 0.186 | 0.334 |
| prefill | `mlp.up_proj` | `[20, 2048]` | 28 | 0.234 | 0.174 | 0.425 |
| prefill | `self_attn.v_proj` | `[20, 2048]` | 28 | 0.234 | 0.174 | 0.307 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.prefill.engine_step` | 1 | 35.652 | 131.133 | unavailable | unavailable |
| `minivllm.decode.engine_step` | 1 | 32.251 | 112.350 | unavailable | unavailable |
| `TorchDynamo Cache Lookup` | 340 | 9.790 | 9.790 | unavailable | unavailable |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 57 | 4.683 | 7.165 | unavailable | unavailable |
| `aten::copy_` | 8 | 4.536 | 4.536 | unavailable | unavailable |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 4.434 | 6.750 | unavailable | unavailable |
| `aten::copy_` | 58 | 3.975 | 3.975 | unavailable | unavailable |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 3.929 | 6.127 | unavailable | unavailable |
| `aten::cat` | 112 | 3.917 | 3.917 | unavailable | unavailable |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 56 | 3.396 | 5.060 | unavailable | unavailable |
| `aten::empty` | 516 | 2.913 | 2.913 | unavailable | unavailable |
| `Torch-Compiled Region: 2/1` | 28 | 2.839 | 7.741 | unavailable | unavailable |
| `Pregraph bytecode` | 340 | 2.822 | 2.822 | unavailable | unavailable |
| `## Call CompiledFxGraph fk2lu652vjfvedbrrlhvwax2muhsq454v6rd7kwcvc7upvjxbggk ##` | 28 | 2.713 | 3.955 | unavailable | unavailable |
| `Torch-Compiled Region: 0/3` | 56 | 2.675 | 9.889 | unavailable | unavailable |
| `Torch-Compiled Region: 0/4` | 57 | 2.657 | 10.180 | unavailable | unavailable |
| `Torch-Compiled Region: 0/2` | 57 | 2.623 | 10.052 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 2.483 | 2.483 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 2.317 | 2.317 | unavailable | unavailable |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 28 | 2.268 | 3.438 | unavailable | unavailable |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 28 | 2.051 | 3.537 | unavailable | unavailable |
| `Torch-Compiled Region: 0/5` | 56 | 2.041 | 7.674 | unavailable | unavailable |
| `## Call CompiledFxGraph fl74kv6dompuxay73yluqhmsqkoyzttp4gcay7lax26nlrtik62p ##` | 28 | 2.015 | 3.367 | unavailable | unavailable |
| `aten::add` | 56 | 1.905 | 1.905 | unavailable | unavailable |
| `AOTDispatcher Runtime Wrapper Prologue` | 340 | 1.871 | 1.871 | unavailable | unavailable |
| `aten::add` | 56 | 1.702 | 1.702 | unavailable | unavailable |
| `Torch-Compiled Region: 2/2` | 28 | 1.434 | 5.254 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.8.self_attn.k_proj` | 2 | 1.428 | 1.575 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.19.mlp.gate_proj` | 2 | 1.428 | 1.496 | unavailable | unavailable |
| `aten::sub` | 30 | 1.377 | 1.377 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 1.283 | 1.283 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.8.self_attn.v_proj` | 2 | 1.261 | 1.503 | unavailable | unavailable |
| `triton_poi_fused_mul_silu_split_0` | 28 | 1.242 | 1.242 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.10.mlp.down_proj` | 2 | 1.239 | 1.345 | unavailable | unavailable |
| `aten::copy_` | 56 | 1.217 | 1.217 | unavailable | unavailable |
| `aten::copy_` | 30 | 1.179 | 1.179 | unavailable | unavailable |
| `triton_poi_fused_mul_silu_split_0` | 28 | 1.170 | 1.170 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.4.self_attn.q_proj` | 2 | 1.121 | 1.231 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.8.mlp.down_proj` | 2 | 1.036 | 1.076 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.22.mlp.gate_proj` | 2 | 1.034 | 1.076 | unavailable | unavailable |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/nsys-request/run-20261018-2300/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/nsys-request/run-20261018-2300/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
