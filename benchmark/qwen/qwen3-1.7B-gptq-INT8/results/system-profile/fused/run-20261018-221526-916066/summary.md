# Fused 路径整体 GPU Profile

- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- Backend：`fused`
- Prompt tokens：20
- Profile max completion tokens：8

本报告包含 profiler instrumentation，只用于定位瓶颈；正式吞吐和延迟应以无 profiler 的 performance 报告为准。

⚠️ **当前 trace 未采集到 CUDA kernel activity。设备时间 unavailable；请勿把它解释为 0 ms。**


## 阶段级耗时

| 阶段 | 次数 | Wall P50 | Wall P90 | CUDA stream P50 | CUDA stream P90 | 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 88.542 ms | 88.542 ms | 88.387 ms | 88.387 ms | 20 |
| decode | 7 | 86.813 ms | 107.371 ms | 86.782 ms | 107.304 ms | 7 |

CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。

## GPTQ 投影（CUDA event）

按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。

| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | `mlp.down_proj` | `[1, 6144]` | 196 | 0.250 | 0.212 | 0.294 |
| decode | `mlp.gate_proj` | `[1, 2048]` | 196 | 0.224 | 0.178 | 0.335 |
| decode | `self_attn.q_proj` | `[1, 2048]` | 196 | 0.193 | 0.175 | 0.282 |
| decode | `self_attn.o_proj` | `[1, 2048]` | 196 | 0.195 | 0.170 | 0.302 |
| decode | `mlp.up_proj` | `[1, 2048]` | 196 | 0.192 | 0.160 | 0.267 |
| decode | `self_attn.k_proj` | `[1, 2048]` | 196 | 0.172 | 0.153 | 0.260 |
| decode | `self_attn.v_proj` | `[1, 2048]` | 196 | 0.170 | 0.148 | 0.260 |
| prefill | `mlp.down_proj` | `[20, 6144]` | 28 | 0.312 | 0.300 | 0.352 |
| prefill | `mlp.gate_proj` | `[20, 2048]` | 28 | 0.208 | 0.192 | 0.242 |
| prefill | `self_attn.o_proj` | `[20, 2048]` | 28 | 0.206 | 0.187 | 0.244 |
| prefill | `self_attn.q_proj` | `[20, 2048]` | 28 | 0.208 | 0.184 | 0.293 |
| prefill | `self_attn.k_proj` | `[20, 2048]` | 28 | 0.187 | 0.161 | 0.237 |
| prefill | `mlp.up_proj` | `[20, 2048]` | 28 | 0.194 | 0.151 | 0.189 |
| prefill | `self_attn.v_proj` | `[20, 2048]` | 28 | 0.159 | 0.150 | 0.195 |

## Profiler 热点

按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。

| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |
|---|---:|---:|---:|---:|---:|
| `minivllm.decode.engine_step` | 7 | 192.047 | 628.743 | unavailable | unavailable |
| `minivllm.prefill.engine_step` | 1 | 24.501 | 88.176 | unavailable | unavailable |
| `## Call CompiledFxGraph fmfuv5i3nqtdmdg6aeyozimpba2zcaixbbbfel7jxzzpbnrw3tgn ##` | 392 | 24.399 | 35.507 | unavailable | unavailable |
| `## Call CompiledFxGraph ffvf7qxxpt7qrrdceopzzwi7ee6wf5ryetiqw3g7mfykkm6b3afd ##` | 399 | 23.462 | 33.576 | unavailable | unavailable |
| `TorchDynamo Cache Lookup` | 1360 | 22.637 | 22.637 | unavailable | unavailable |
| `aten::copy_` | 38 | 15.987 | 15.987 | unavailable | unavailable |
| `## Call CompiledFxGraph fhwel7axcf4i23zceukczppz5a2iulevtgccmcfip6ks72gahuze ##` | 196 | 14.090 | 23.392 | unavailable | unavailable |
| `Torch-Compiled Region: 0/5` | 392 | 13.932 | 53.253 | unavailable | unavailable |
| `Torch-Compiled Region: 0/4` | 399 | 13.604 | 49.409 | unavailable | unavailable |
| `## Call CompiledFxGraph fbpcpdnscprhs5imt4potezua4yibn25rtuduszr7b7qditcbkwr ##` | 196 | 11.841 | 16.688 | unavailable | unavailable |
| `aten::cat` | 448 | 10.924 | 10.924 | unavailable | unavailable |
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 399 | 10.114 | 10.114 | unavailable | unavailable |
| `aten::add` | 392 | 8.912 | 8.912 | unavailable | unavailable |
| `aten::empty` | 1728 | 6.983 | 6.983 | unavailable | unavailable |
| `Torch-Compiled Region: 2/2` | 196 | 6.810 | 24.620 | unavailable | unavailable |
| `Pregraph bytecode` | 1360 | 6.220 | 6.220 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 6.060 | 6.060 | unavailable | unavailable |
| `Torch-Compiled Region: 1/2` | 196 | 5.418 | 30.542 | unavailable | unavailable |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 196 | 5.047 | 5.047 | unavailable | unavailable |
| `triton_poi_fused_mul_silu_split_0` | 196 | 4.847 | 4.847 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_0` | 196 | 4.833 | 4.833 | unavailable | unavailable |
| `aten::sub` | 204 | 4.512 | 4.512 | unavailable | unavailable |
| `triton_poi_fused__to_copy_add_cat_index_mul_split_sub_unsqueeze_1` | 196 | 4.469 | 4.469 | unavailable | unavailable |
| `AOTDispatcher Runtime Wrapper Prologue` | 1360 | 4.393 | 4.393 | unavailable | unavailable |
| `aten::reshape` | 2940 | 3.381 | 5.600 | unavailable | unavailable |
| `## Call CompiledFxGraph fintnwvbeurtnqc423scnhreq5lm7h27cuym6zp2q52ydf57gzfi ##` | 57 | 3.286 | 4.659 | unavailable | unavailable |
| `## Call CompiledFxGraph fp6vfnrengrg37jjoqnfkaszxxlshqogby4uuy5eutjhvcnqq3ua ##` | 56 | 2.875 | 4.061 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.19.mlp.gate_proj` | 8 | 2.734 | 2.831 | unavailable | unavailable |
| `aten::copy_` | 58 | 2.694 | 2.694 | unavailable | unavailable |
| `aten::view` | 3136 | 2.396 | 2.396 | unavailable | unavailable |
| `Torch-Compiled Region: 0/2` | 57 | 2.055 | 7.212 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.27.mlp.up_proj` | 8 | 2.021 | 2.193 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.21.mlp.up_proj` | 8 | 1.990 | 2.141 | unavailable | unavailable |
| `## Call CompiledFxGraph fr3x7kfdj6slmdbpf4kvwjufwh4fb44h3z6b5ankvmtbivqff527 ##` | 8 | 1.910 | 3.814 | unavailable | unavailable |
| `## Call CompiledFxGraph fl74kv6dompuxay73yluqhmsqkoyzttp4gcay7lax26nlrtik62p ##` | 28 | 1.745 | 2.760 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.0.mlp.gate_proj` | 8 | 1.661 | 1.788 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.16.mlp.down_proj` | 8 | 1.646 | 1.778 | unavailable | unavailable |
| `minivllm.gptq_projection.model.layers.0.mlp.down_proj` | 8 | 1.641 | 1.734 | unavailable | unavailable |
| `Torch-Compiled Region: 0/3` | 56 | 1.587 | 6.240 | unavailable | unavailable |
| `aten::empty_strided` | 357 | 1.578 | 1.578 | unavailable | unavailable |

## 输出文件

- Chrome trace：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-221526-916066/trace.json`
- 原始 JSON：`/home/zzzz/project/vllm-projects/MinivLLM/benchmark/qwen/qwen3-1.7B-gptq-INT8/results/system-profile/fused/run-20261018-221526-916066/summary.json`
- CUDA device activity：`缺失（0 个 device events）`
