# Qwen3-1.7B GPTQ-INT8 推理基线报告

- 运行时间（UTC）：`2026-10-17T12:01:32Z`
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
| TTFT | 187.45 ms | 208.64 ms | 193.10 ms | 177.37 ms | 210.46 ms |
| Prefill 吞吐 | 106.70 tok/s | 111.06 tok/s | 104.03 tok/s | 95.03 tok/s | 112.76 tok/s |
| Decode 单 token 延迟 | 153.57 ms | 153.85 ms | 153.39 ms | 152.44 ms | 154.04 ms |
| Decode 吞吐 | 6.51 tok/s | 6.54 tok/s | 6.52 tok/s | 6.49 tok/s | 6.56 tok/s |
| 端到端延迟 | 4948.39 ms | 4975.08 ms | 4948.43 ms | 4903.43 ms | 4985.81 ms |
| 端到端生成吞吐 | 6.47 tok/s | 6.50 tok/s | 6.47 tok/s | 6.42 tok/s | 6.53 tok/s |

## 初始化、显存与 KV cache

| 指标 | 值 |
|---|---:|
| 模型加载时间 | 8390.23 ms |
| 首次显式 warmup 端到端延迟 | 6293.08 ms |
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
