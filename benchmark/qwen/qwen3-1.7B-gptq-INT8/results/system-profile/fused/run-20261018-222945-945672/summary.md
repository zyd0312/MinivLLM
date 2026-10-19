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
| prefill | 1 | 101.026 ms | 101.026 ms | 100.938 ms | 100.938 ms | 20 |
| decode | 1 | 103.424 ms | 103.424 ms | 103.383 ms | 103.383 ms | 1 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `mlp.down_proj` | `[1, 6144]` | 28 | 0.260 | 0.205 | 0.412 |
| decode | `self_attn.q_proj` | `[1, 2048]` | 28 | 0.265 | 0.180 | 0.333 |
| decode | `self_attn.o_proj` | `[1, 2048]` | 28 | 0.201 | 0.171 | 0.252 |
| decode | `mlp.gate_proj` | `[1, 2048]` | 28 | 0.267 | 0.169 | 0.299 |
| decode | `self_attn.k_proj` | `[1, 2048]` | 28 | 0.179 | 0.159 | 0.204 |
| decode | `mlp.up_proj` | `[1, 2048]` | 28 | 0.194 | 0.155 | 0.252 |
| decode | `self_attn.v_proj` | `[1, 2048]` | 28 | 0.206 | 0.146 | 0.228 |
| prefill | `mlp.down_proj` | `[20, 6144]` | 28 | 0.348 | 0.316 | 0.350 |
| prefill | `self_attn.q_proj` | `[20, 2048]` | 28 | 0.240 | 0.212 | 0.350 |
| prefill | `mlp.gate_proj` | `[20, 2048]` | 28 | 0.228 | 0.206 | 0.273 |
| prefill | `self_attn.o_proj` | `[20, 2048]` | 28 | 0.232 | 0.205 | 0.248 |
| prefill | `self_attn.k_proj` | `[20, 2048]` | 28 | 0.209 | 0.175 | 0.246 |
| prefill | `self_attn.v_proj` | `[20, 2048]` | 28 | 0.206 | 0.173 | 0.280 |
| prefill | `mlp.up_proj` | `[20, 2048]` | 28 | 0.201 | 0.172 | 0.215 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.decode.engine_step` | 1 | 28.855 | 103.323 | unavailable | unavailable |
| `minivllm.prefill.engine_step` | 1 | 28.653 | 100.794 | unavailable | unavailable |
| `TorchDynamo Cache Lookup` | 340 | 7.339 | 7.339 | unavailable | unavailable |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 57 | 4.998 | 6.973 | unavailable | unavailable |
| `aten::cat` | 112 | 3.870 | 3.870 | unavailable | unavailable |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 56 | 3.817 | 6.061 | unavailable | unavailable |
| `aten::copy_` | 8 | 3.655 | 3.655 | unavailable | unavailable |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 3.404 | 5.273 | unavailable | unavailable |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 3.076 | 4.683 | unavailable | unavailable |
| `aten::copy_` | 58 | 2.932 | 2.932 | unavailable | unavailable |
| `aten::empty` | 516 | 2.838 | 2.838 | unavailable | unavailable |
| `Torch-Compiled Region: 0/2` | 57 | 2.400 | 8.291 | unavailable | unavailable |
| `Torch-Compiled Region: 0/5` | 56 | 2.161 | 8.914 | unavailable | unavailable |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 28 | 2.107 | 3.708 | unavailable | unavailable |
| `Torch-Compiled Region: 0/4` | 57 | 2.049 | 9.332 | unavailable | unavailable |
| `Pregraph bytecode` | 340 | 2.002 | 2.002 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 1.975 | 1.975 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 1.869 | 1.869 | unavailable | unavailable |
| `Torch-Compiled Region: 0/3` | 56 | 1.841 | 7.151 | unavailable | unavailable |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 28 | 1.777 | 2.601 | unavailable | unavailable |
| `## Call CompiledFxGraph fl74kv6dompuxay73yluqhmsqkoyzttp4gcay7lax26nlrtik62p ##` | 28 | 1.757 | 3.079 | unavailable | unavailable |
| `aten::add` | 56 | 1.746 | 1.746 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.1.self_attn.q_proj` | 2 | 1.648 | 1.812 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.0.mlp.gate_proj` | 2 | 1.531 | 1.670 | unavailable | unavailable |
| `## Call CompiledFxGraph fk2lu652vjfvedbrrlhvwax2muhsq454v6rd7kwcvc7upvjxbggk ##` | 28 | 1.462 | 2.287 | unavailable | unavailable |
| `aten::add` | 56 | 1.443 | 1.443 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 1.333 | 1.333 | unavailable | unavailable |
| `aten::sub` | 30 | 1.325 | 1.325 | unavailable | unavailable |
| `AOTDispatcher Runtime Wrapper Prologue` | 340 | 1.292 | 1.292 | unavailable | unavailable |
| `aten::copy_` | 56 | 1.108 | 1.108 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.3.mlp.gate_proj` | 2 | 1.027 | 1.087 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.3.self_attn.v_proj` | 2 | 1.010 | 1.092 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_0` | 28 | 0.999 | 0.999 | unavailable | unavailable |
| `Torch-Compiled Region: 2/2` | 28 | 0.986 | 3.749 | unavailable | unavailable |
| `aten::copy_` | 30 | 0.971 | 0.971 | unavailable | unavailable |
| `Torch-Compiled Region: 2/1` | 28 | 0.957 | 3.507 | unavailable | unavailable |
| `Torch-Compiled Region: 1/2` | 28 | 0.950 | 4.977 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 0.938 | 0.938 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 28 | 0.911 | 0.911 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.27.mlp.down_proj` | 2 | 0.870 | 0.920 | unavailable | unavailable |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-222945-945672/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-222945-945672/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
