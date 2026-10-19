# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T05:58:19Z`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- PyTorch/CUDA：`2.9.1+cu128` / `12.8`
- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`

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
| TTFT | 314.54 ms | 377.89 ms | 320.79 ms | 263.00 ms | 409.28 ms |
| Prefill 吞吐 | 63.58 tok/s | 73.57 tok/s | 63.76 tok/s | 48.87 tok/s | 76.05 tok/s |
| Decode 单 token 延迟 | 221.46 ms | 273.69 ms | 231.57 ms | 194.02 ms | 298.80 ms |
| Decode 吞吐 | 4.52 tok/s | 5.02 tok/s | 4.41 tok/s | 3.35 tok/s | 5.15 tok/s |
| 端到端延迟 | 7180.04 ms | 8806.15 ms | 7499.91 ms | 6301.41 ms | 9526.03 ms |
| 端到端生成吞吐 | 4.46 tok/s | 4.94 tok/s | 4.35 tok/s | 3.36 tok/s | 5.08 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 13462.31 ms |
| 首次显式 warmup 端到端延迟 | 7133.05 ms |
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
| prefill | `q_proj` | 0.45 ms | 0.09 ms | 0.55 ms | 8.08 MiB |
| prefill | `k_proj` | 0.29 ms | 0.09 ms | 0.39 ms | 4.04 MiB |
| prefill | `v_proj` | 0.25 ms | 0.07 ms | 0.32 ms | 4.04 MiB |
| prefill | `o_proj` | 0.41 ms | 0.05 ms | 0.46 ms | 8.08 MiB |
| prefill | `gate_proj` | 1.42 ms | 0.07 ms | 1.49 ms | 24.23 MiB |
| prefill | `up_proj` | 1.34 ms | 0.06 ms | 1.40 ms | 24.23 MiB |
| prefill | `down_proj` | 1.37 ms | 0.06 ms | 1.43 ms | 24.08 MiB |
| decode | `q_proj` | 0.32 ms | 0.02 ms | 0.34 ms | 8.00 MiB |
| decode | `k_proj` | 0.16 ms | 0.01 ms | 0.17 ms | 4.00 MiB |
| decode | `v_proj` | 0.16 ms | 0.01 ms | 0.17 ms | 4.00 MiB |
| decode | `o_proj` | 0.37 ms | 0.02 ms | 0.39 ms | 8.00 MiB |
| decode | `gate_proj` | 1.40 ms | 0.06 ms | 1.46 ms | 24.01 MiB |
| decode | `up_proj` | 1.34 ms | 0.06 ms | 1.40 ms | 24.01 MiB |
| decode | `down_proj` | 1.37 ms | 0.06 ms | 1.43 ms | 24.00 MiB |

## 正确性对齐

本次未执行 Transformers/GPTQModel logits 对齐。

## 结果文件

- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。
- 本 Markdown 文件用于快速汇报核心指标。
