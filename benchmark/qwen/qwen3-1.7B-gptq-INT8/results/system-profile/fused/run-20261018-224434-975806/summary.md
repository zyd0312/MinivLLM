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
| prefill | 1 | 116.337 ms | 116.337 ms | 116.170 ms | 116.170 ms | 20 |
| decode | 1 | 91.514 ms | 91.514 ms | 91.471 ms | 91.471 ms | 1 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `mlp.down_proj` | `[1, 6144]` | 28 | 0.244 | 0.202 | 0.345 |
| decode | `mlp.gate_proj` | `[1, 2048]` | 28 | 0.201 | 0.168 | 0.228 |
| decode | `self_attn.o_proj` | `[1, 2048]` | 28 | 0.193 | 0.165 | 0.272 |
| decode | `self_attn.q_proj` | `[1, 2048]` | 28 | 0.180 | 0.155 | 0.245 |
| decode | `mlp.up_proj` | `[1, 2048]` | 28 | 0.288 | 0.146 | 0.790 |
| decode | `self_attn.k_proj` | `[1, 2048]` | 28 | 0.187 | 0.131 | 0.212 |
| decode | `self_attn.v_proj` | `[1, 2048]` | 28 | 0.153 | 0.124 | 0.225 |
| prefill | `mlp.down_proj` | `[20, 6144]` | 28 | 0.345 | 0.327 | 0.397 |
| prefill | `mlp.gate_proj` | `[20, 2048]` | 28 | 0.240 | 0.210 | 0.307 |
| prefill | `self_attn.o_proj` | `[20, 2048]` | 28 | 0.306 | 0.207 | 0.321 |
| prefill | `self_attn.q_proj` | `[20, 2048]` | 28 | 0.262 | 0.191 | 0.339 |
| prefill | `mlp.up_proj` | `[20, 2048]` | 28 | 0.218 | 0.177 | 0.260 |
| prefill | `self_attn.k_proj` | `[20, 2048]` | 28 | 0.182 | 0.176 | 0.225 |
| prefill | `self_attn.v_proj` | `[20, 2048]` | 28 | 0.182 | 0.154 | 0.252 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.prefill.engine_step` | 1 | 33.794 | 116.055 | unavailable | unavailable |
| `minivllm.decode.engine_step` | 1 | 24.397 | 91.368 | unavailable | unavailable |
| `TorchDynamo Cache Lookup` | 340 | 7.907 | 7.907 | unavailable | unavailable |
| `aten::copy_` | 8 | 5.011 | 5.011 | unavailable | unavailable |
| `Torch-Compiled Region: 0/2` | 57 | 4.064 | 10.375 | unavailable | unavailable |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 4.033 | 5.721 | unavailable | unavailable |
| `aten::copy_` | 58 | 4.018 | 4.018 | unavailable | unavailable |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 3.496 | 5.246 | unavailable | unavailable |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 57 | 3.337 | 4.945 | unavailable | unavailable |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 56 | 3.307 | 4.995 | unavailable | unavailable |
| `aten::empty` | 516 | 3.142 | 3.142 | unavailable | unavailable |
| `aten::cat` | 112 | 2.690 | 2.690 | unavailable | unavailable |
| `## Call CompiledFxGraph fl74kv6dompuxay73yluqhmsqkoyzttp4gcay7lax26nlrtik62p ##` | 28 | 2.470 | 4.498 | unavailable | unavailable |
| `Torch-Compiled Region: 0/5` | 56 | 2.092 | 7.591 | unavailable | unavailable |
| `Torch-Compiled Region: 0/3` | 56 | 2.044 | 8.022 | unavailable | unavailable |
| `Pregraph bytecode` | 340 | 1.957 | 1.957 | unavailable | unavailable |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 28 | 1.902 | 2.805 | unavailable | unavailable |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 28 | 1.803 | 3.050 | unavailable | unavailable |
| `Torch-Compiled Region: 0/4` | 57 | 1.770 | 6.997 | unavailable | unavailable |
| `## Call CompiledFxGraph fk2lu652vjfvedbrrlhvwax2muhsq454v6rd7kwcvc7upvjxbggk ##` | 28 | 1.720 | 2.491 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 1.687 | 1.687 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 1.607 | 1.607 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.5.self_attn.q_proj` | 2 | 1.518 | 1.549 | unavailable | unavailable |
| `aten::add` | 56 | 1.504 | 1.504 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.21.mlp.up_proj` | 2 | 1.484 | 1.669 | unavailable | unavailable |
| `aten::copy_` | 56 | 1.313 | 1.313 | unavailable | unavailable |
| `aten::add` | 56 | 1.298 | 1.298 | unavailable | unavailable |
| `AOTDispatcher Runtime Wrapper Prologue` | 340 | 1.217 | 1.217 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.1.self_attn.k_proj` | 2 | 1.156 | 1.240 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_1` | 28 | 1.117 | 1.117 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.8.self_attn.o_proj` | 2 | 1.090 | 1.118 | unavailable | unavailable |
| `Torch-Compiled Region: 2/1` | 28 | 1.055 | 3.810 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 1.048 | 1.048 | unavailable | unavailable |
| `Torch-Compiled Region: 2/2` | 28 | 1.016 | 3.974 | unavailable | unavailable |
| `aten::copy_` | 30 | 0.924 | 0.924 | unavailable | unavailable |
| `Torch-Compiled Region: 1/2` | 28 | 0.912 | 4.282 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_0` | 28 | 0.911 | 0.911 | unavailable | unavailable |
| `triton_poi_fused_mul_silu_split_0` | 28 | 0.903 | 0.903 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 0.898 | 0.898 | unavailable | unavailable |
| `Torch-Compiled Region: 1/1` | 28 | 0.892 | 5.670 | unavailable | unavailable |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-224434-975806/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-224434-975806/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
