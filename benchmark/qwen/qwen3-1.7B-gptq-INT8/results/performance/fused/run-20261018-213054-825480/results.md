# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T13:31:31Z`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- PyTorch/CUDA：`2.9.1+cu128` / `12.8`
- 模型：`/home/zzzz/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8`
- GPTQ backend：`fused`

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
| TTFT | 72.32 ms | 110.73 ms | 83.24 ms | 62.03 ms | 131.76 ms |
| Prefill 吞吐 | 276.54 tok/s | 306.30 tok/s | 257.09 tok/s | 151.79 tok/s | 322.41 tok/s |
| Decode 单 token 延迟 | 54.67 ms | 63.96 ms | 57.14 ms | 53.00 ms | 69.96 ms |
| Decode 吞吐 | 18.29 tok/s | 18.85 tok/s | 17.70 tok/s | 14.29 tok/s | 18.87 tok/s |
| 端到端延迟 | 1757.01 ms | 2090.45 ms | 1854.87 ms | 1719.33 ms | 2301.00 ms |
| 端到端生成吞吐 | 18.21 tok/s | 18.60 tok/s | 17.47 tok/s | 13.91 tok/s | 18.61 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 15289.03 ms |
| 首次显式 warmup 端到端延迟 | 3734.43 ms |
| 峰值 allocated | 6.32 GiB |
| 峰值 reserved | 6.35 GiB |
| KV cache blocks | 141 |
| Block size | 256 tokens |
| 理论 KV token 容量 | 36096 |
| 正式测试后空闲 blocks | 141 |

## GPTQ 投影 profile

本次未开启 projection profiler。使用 `PROFILE_PROJECTIONS=1` 单独运行 profile，避免污染端到端性能基线。

## 正确性对齐

本次未执行 Transformers/GPTQModel logits 对齐。

## 结果文件

- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。
- 本 Markdown 文件用于快速汇报核心指标。
