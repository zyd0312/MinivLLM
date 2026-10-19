# Benchmark layout

Benchmark assets are grouped by model and test purpose:

```text
benchmark/
  qwen/qwen3-1.7B-gptq-INT8/
    performance/       formal throughput and latency runs
    operator-profile/  individual GPTQ projection measurements
    system-profile/    end-to-end traces and stage attribution
    correctness/       logits and backend comparison
    results/           timestamped reports grouped by test type
```

Use the model-specific [benchmark README](qwen/qwen3-1.7B-gptq-INT8/README.md)
for commands, options, and interpretation of each report. Formal performance
results and profiler diagnostics are written to separate directories.
