# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T05:57:05Z`
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
| Profiler hooks | 关闭 |

## 核心性能指标

统计值为正式迭代结果；P50 是主要比较指标，P90 用于观察抖动。

| 指标 | P50 | P90 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| TTFT | 220.44 ms | 235.40 ms | 221.02 ms | 196.43 ms | 235.86 ms |
| Prefill 吞吐 | 90.73 tok/s | 97.84 tok/s | 90.89 tok/s | 84.80 tok/s | 101.82 tok/s |
| Decode 单 token 延迟 | 157.13 ms | 159.46 ms | 157.56 ms | 154.94 ms | 160.02 ms |
| Decode 吞吐 | 6.36 tok/s | 6.42 tok/s | 6.35 tok/s | 6.25 tok/s | 6.45 tok/s |
| 端到端延迟 | 5105.84 ms | 5173.32 ms | 5105.59 ms | 4999.88 ms | 5196.68 ms |
| 端到端生成吞吐 | 6.27 tok/s | 6.36 tok/s | 6.27 tok/s | 6.16 tok/s | 6.40 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 22313.25 ms |
| 首次显式 warmup 端到端延迟 | 11170.93 ms |
| 峰值 allocated | 6.34 GiB |
| 峰值 reserved | 6.42 GiB |
| KV cache blocks | 138 |
| Block size | 256 tokens |
| 理论 KV token 容量 | 35328 |
| 正式测试后空闲 blocks | 138 |

## GPTQ 投影 profile

本次未开启 projection profiler。使用 `PROFILE_PROJECTIONS=1` 单独运行 profile，避免污染端到端性能基线。

## 正确性对齐

本次未执行 Transformers/GPTQModel logits 对齐。

## 结果文件

- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。
- 本 Markdown 文件用于快速汇报核心指标。
