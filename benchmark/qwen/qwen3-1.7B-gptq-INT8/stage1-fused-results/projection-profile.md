# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T07:42:03Z`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- PyTorch/CUDA：`2.9.1+cu128` / `12.8`
- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- GPTQ backend：`reference`

## 测试配置

| 项目 | 值 |
|---|---:|
| Batch size | 1 |
| Prompt tokens（本轮） | 20 |
| Max completion tokens | 32 |
| Max model length | 512 |
| 正式迭代次数 | 5 |
| Profiler hooks | 开启 |

## 核心性能指标

统计值为正式迭代结果；P50 是主要比较指标，P90 用于观察抖动。

| 指标 | P50 | P90 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| TTFT | 384.50 ms | 437.68 ms | 388.88 ms | 323.03 ms | 441.76 ms |
| Prefill 吞吐 | 52.02 tok/s | 59.15 tok/s | 52.11 tok/s | 45.27 tok/s | 61.91 tok/s |
| Decode 单 token 延迟 | 304.65 ms | 342.94 ms | 304.47 ms | 246.15 ms | 360.39 ms |
| Decode 吞吐 | 3.28 tok/s | 3.80 tok/s | 3.33 tok/s | 2.77 tok/s | 4.06 tok/s |
| 端到端延迟 | 9767.57 ms | 11050.45 ms | 9827.89 ms | 7994.54 ms | 11614.28 ms |
| 端到端生成吞吐 | 3.28 tok/s | 3.74 tok/s | 3.30 tok/s | 2.76 tok/s | 4.00 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 17498.53 ms |
| 首次显式 warmup 端到端延迟 | 10083.89 ms |
| 峰值 allocated | 6.34 GiB |
| 峰值 reserved | 6.38 GiB |
| KV cache blocks | 138 |
| Block size | 256 tokens |
| 理论 KV token 容量 | 35328 |
| 正式测试后空闲 blocks | 138 |

## GPTQ 投影 profile

以下为所有 28 层同名投影的中位数；profile 会增加少量 event 记录开销，不能直接当作无 instrumentation 的端到端基线。

| 阶段 | 投影 | 反量化 | MatMul | Forward | 临时显存 |
|---|---|---:|---:|---:|---:|
| prefill | `q_proj` | 0.48 ms | 0.13 ms | 0.63 ms | 8.08 MiB |
| prefill | `k_proj` | 0.34 ms | 0.12 ms | 0.49 ms | 4.04 MiB |
| prefill | `v_proj` | 0.30 ms | 0.09 ms | 0.40 ms | 4.04 MiB |
| prefill | `o_proj` | 0.44 ms | 0.10 ms | 0.55 ms | 8.08 MiB |
| prefill | `gate_proj` | 1.58 ms | 0.07 ms | 1.65 ms | 24.23 MiB |
| prefill | `up_proj` | 1.51 ms | 0.07 ms | 1.59 ms | 24.23 MiB |
| prefill | `down_proj` | 1.55 ms | 0.06 ms | 1.61 ms | 24.08 MiB |
| decode | `q_proj` | 0.32 ms | 0.02 ms | 0.34 ms | 8.00 MiB |
| decode | `k_proj` | 0.16 ms | 0.01 ms | 0.18 ms | 4.00 MiB |
| decode | `v_proj` | 0.22 ms | 0.05 ms | 0.27 ms | 4.00 MiB |
| decode | `o_proj` | 0.44 ms | 0.07 ms | 0.51 ms | 8.00 MiB |
| decode | `gate_proj` | 1.45 ms | 0.06 ms | 1.51 ms | 24.01 MiB |
| decode | `up_proj` | 1.34 ms | 0.06 ms | 1.40 ms | 24.01 MiB |
| decode | `down_proj` | 1.37 ms | 0.06 ms | 1.43 ms | 24.00 MiB |

## 正确性对齐

本次未执行 Transformers/GPTQModel logits 对齐。

## 结果文件

- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。
- 本 Markdown 文件用于快速汇报核心指标。
