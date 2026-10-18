# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T07:16:22Z`
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
| Profiler hooks | 关闭 |

## 核心性能指标

统计值为正式迭代结果；P50 是主要比较指标，P90 用于观察抖动。

| 指标 | P50 | P90 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| TTFT | 274.08 ms | 290.94 ms | 270.29 ms | 245.13 ms | 291.47 ms |
| Prefill 吞吐 | 72.97 tok/s | 80.87 tok/s | 74.38 tok/s | 68.62 tok/s | 81.59 tok/s |
| Decode 单 token 延迟 | 161.04 ms | 166.11 ms | 162.06 ms | 158.54 ms | 166.71 ms |
| Decode 吞吐 | 6.21 tok/s | 6.30 tok/s | 6.17 tok/s | 6.00 tok/s | 6.31 tok/s |
| 端到端延迟 | 5282.80 ms | 5424.93 ms | 5294.39 ms | 5167.83 ms | 5459.85 ms |
| 端到端生成吞吐 | 6.06 tok/s | 6.18 tok/s | 6.05 tok/s | 5.86 tok/s | 6.19 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 24395.89 ms |
| 首次显式 warmup 端到端延迟 | 6672.18 ms |
| 峰值 allocated | 6.34 GiB |
| 峰值 reserved | 6.38 GiB |
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
