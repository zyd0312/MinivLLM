# Fused 路径整体 GPU Profile

- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- Backend：`fused`
- Prompt tokens：20
- Profile max completion tokens：8

本报告包含 profiler instrumentation，只用于定位瓶颈；正式吞吐和延迟应以无 profiler 的 performance 报告为准。

⚠️ **CUDA kernel activity 未采集到（CUPTI_ERROR_INVALID_DEVICE）。下方 device time 的 0 表示 unavailable，不代表 kernel 耗时为 0。此 trace 仅能用于 CPU 范围检查。**

## 阶段级耗时

| 阶段 | 次数 | Wall P50 | Wall P90 | CUDA P50 | CUDA P90 | 处理 token |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 59.663 ms | 59.663 ms | 59.579 ms | 59.579 ms | 21 |
| decode | 7 | 54.235 ms | 64.891 ms | 54.202 ms | 64.827 ms | 7 |

## Profiler 热点

按 self device time 排序；若设备时间不可用则该列为 0。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.decode.engine_step` | 7 | 198.491 | 390.487 | 0.000 | 0.000 |
| `minivllm.prefill.engine_step` | 1 | 27.706 | 59.397 | 0.000 | 0.000 |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 399 | 19.601 | 27.656 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 392 | 18.406 | 27.216 | 0.000 | 0.000 |
| `aten::copy_` | 38 | 16.059 | 16.059 | 0.000 | 0.000 |
| `TorchDynamo Cache Lookup` | 1360 | 15.399 | 15.399 | 0.000 | 0.000 |
| `Torch-Compiled Region: 0/4` | 399 | 11.424 | 40.991 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 196 | 11.371 | 18.814 | 0.000 | 0.000 |
| `Torch-Compiled Region: 0/5` | 392 | 11.272 | 41.431 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 196 | 9.132 | 12.788 | 0.000 | 0.000 |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 399 | 8.055 | 8.055 | 0.000 | 0.000 |
| `aten::cat` | 448 | 7.101 | 7.101 | 0.000 | 0.000 |
| `aten::add` | 392 | 6.494 | 6.494 | 0.000 | 0.000 |
| `Torch-Compiled Region: 2/2` | 196 | 5.475 | 19.165 | 0.000 | 0.000 |
| `Pregraph bytecode` | 1360 | 4.999 | 4.999 | 0.000 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 4.918 | 4.918 | 0.000 | 0.000 |
| `aten::empty` | 1728 | 4.811 | 4.811 | 0.000 | 0.000 |
| `Torch-Compiled Region: 1/2` | 196 | 4.751 | 25.132 | 0.000 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_0` | 196 | 4.240 | 4.240 | 0.000 | 0.000 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 3.892 | 3.892 | 0.000 | 0.000 |
| `AOTDispatcher Runtime Wrapper Prologue` | 1360 | 3.815 | 3.815 | 0.000 | 0.000 |
| `aten::reshape` | 2940 | 3.780 | 5.328 | 0.000 | 0.000 |
| `aten::copy_` | 58 | 3.709 | 3.709 | 0.000 | 0.000 |
| `triton_poi_fused_mul_silu_split_0` | 196 | 3.657 | 3.657 | 0.000 | 0.000 |
| `aten::sub` | 204 | 3.477 | 3.477 | 0.000 | 0.000 |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_1` | 196 | 3.203 | 3.203 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 2.503 | 3.609 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 2.400 | 3.436 | 0.000 | 0.000 |
| `Torch-Compiled Region: 0/2` | 57 | 1.675 | 5.675 | 0.000 | 0.000 |
| `aten::view` | 3136 | 1.669 | 1.669 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fr3x7kfdj6slmdbpf4kvwjufwh4fb44h3z6b5ankvmtbivqff527 ##` | 8 | 1.636 | 3.143 | 0.000 | 0.000 |
| `Torch-Compiled Region: 0/3` | 56 | 1.414 | 5.358 | 0.000 | 0.000 |
| `aten::empty_strided` | 357 | 1.352 | 1.352 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fl74kv6dompuxay73yluqhmsqkoyzttp4gcay7lax26nlrtik62p ##` | 28 | 1.350 | 2.218 | 0.000 | 0.000 |
| `## Call CompiledFxGraph fk2lu652vjfvedbrrlhvwax2muhsq454v6rd7kwcvc7upvjxbggk ##` | 28 | 1.217 | 1.793 | 0.000 | 0.000 |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 57 | 1.106 | 1.106 | 0.000 | 0.000 |
| `aten::copy_` | 30 | 0.942 | 0.942 | 0.000 | 0.000 |
| `aten::reshape` | 784 | 0.845 | 1.318 | 0.000 | 0.000 |
| `Torch-Compiled Region: 2/1` | 28 | 0.750 | 2.752 | 0.000 | 0.000 |
| `aten::empty_like` | 196 | 0.737 | 1.625 | 0.000 | 0.000 |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-200904-731561/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-200904-731561/summary.json`
