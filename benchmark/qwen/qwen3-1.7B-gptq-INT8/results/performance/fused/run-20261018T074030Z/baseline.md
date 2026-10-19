# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-18T07:40:30Z`
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
| TTFT | 105.52 ms | 116.64 ms | 107.53 ms | 97.76 ms | 117.58 ms |
| Prefill 吞吐 | 189.54 tok/s | 201.53 tok/s | 186.94 tok/s | 170.10 tok/s | 204.59 tok/s |
| Decode 单 token 延迟 | 85.87 ms | 88.09 ms | 85.47 ms | 79.89 ms | 88.54 ms |
| Decode 吞吐 | 11.65 tok/s | 12.18 tok/s | 11.72 tok/s | 11.29 tok/s | 12.52 tok/s |
| 端到端延迟 | 2770.04 ms | 2843.74 ms | 2757.35 ms | 2578.64 ms | 2862.55 ms |
| 端到端生成吞吐 | 11.55 tok/s | 12.08 tok/s | 11.62 tok/s | 11.18 tok/s | 12.41 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 18080.43 ms |
| 首次显式 warmup 端到端延迟 | 4452.06 ms |
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
