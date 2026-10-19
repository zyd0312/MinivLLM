# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T07:17:54Z`
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
| TTFT | 332.01 ms | 413.21 ms | 343.59 ms | 247.62 ms | 415.10 ms |
| Prefill 吞吐 | 60.24 tok/s | 74.03 tok/s | 60.37 tok/s | 48.18 tok/s | 80.77 tok/s |
| Decode 单 token 延迟 | 271.12 ms | 303.40 ms | 267.67 ms | 216.59 ms | 311.90 ms |
| Decode 吞吐 | 3.69 tok/s | 4.38 tok/s | 3.80 tok/s | 3.21 tok/s | 4.62 tok/s |
| 端到端延迟 | 8737.15 ms | 9817.99 ms | 8641.66 ms | 6962.43 ms | 10079.51 ms |
| 端到端生成吞吐 | 3.66 tok/s | 4.36 tok/s | 3.77 tok/s | 3.17 tok/s | 4.60 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 20817.28 ms |
| 首次显式 warmup 端到端延迟 | 11716.33 ms |
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
| prefill | `q_proj` | 0.44 ms | 0.10 ms | 0.56 ms | 8.08 MiB |
| prefill | `k_proj` | 0.29 ms | 0.08 ms | 0.38 ms | 4.04 MiB |
| prefill | `v_proj` | 0.26 ms | 0.07 ms | 0.33 ms | 4.04 MiB |
| prefill | `o_proj` | 0.41 ms | 0.04 ms | 0.46 ms | 8.08 MiB |
| prefill | `gate_proj` | 1.42 ms | 0.07 ms | 1.49 ms | 24.23 MiB |
| prefill | `up_proj` | 1.34 ms | 0.07 ms | 1.40 ms | 24.23 MiB |
| prefill | `down_proj` | 1.37 ms | 0.06 ms | 1.43 ms | 24.08 MiB |
| decode | `q_proj` | 0.32 ms | 0.02 ms | 0.34 ms | 8.00 MiB |
| decode | `k_proj` | 0.16 ms | 0.01 ms | 0.17 ms | 4.00 MiB |
| decode | `v_proj` | 0.16 ms | 0.01 ms | 0.18 ms | 4.00 MiB |
| decode | `o_proj` | 0.41 ms | 0.02 ms | 0.45 ms | 8.00 MiB |
| decode | `gate_proj` | 1.41 ms | 0.06 ms | 1.48 ms | 24.01 MiB |
| decode | `up_proj` | 1.34 ms | 0.06 ms | 1.40 ms | 24.01 MiB |
| decode | `down_proj` | 1.37 ms | 0.06 ms | 1.43 ms | 24.00 MiB |

## 正确性对齐

本次未执行 Transformers/GPTQModel logits 对齐。

## 结果文件

- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。
- 本 Markdown 文件用于快速汇报核心指标。
