# Qwen3-1.7B GPTQ-INT8 benchmark

Run the baseline from the repository root with:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_benchmark.sh
```

The script writes `results.json` and the corresponding `results.md` summary
report. It reports model loading time, warmup and
formal runs, prefill prompt tokens/s and TTFT, per-token decode latency and
tokens/s, end-to-end latency, peak allocated/reserved memory, runtime KV-cache
block capacity, and (when enabled) per-projection GPTQ dequantization and
matrix-multiplication timings for q/k/v/o/gate/up/down. The JSON also stores
the exact prompt token ids and every raw iteration.

Use `--report /path/to/report.md` to choose a different Markdown report path.

The default run measures end-to-end performance without profiler hooks. Run a
separate projection profile when investigating GPTQLinear:

```bash
PROFILE_PROJECTIONS=1 OUTPUT=profile.json \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_benchmark.sh
```

To run the complete Stage 1 pair with one fixed workload, use
`run_stage1.sh`. It starts a fresh process for the formal baseline and another
fresh process for the projection profile, so profiler hooks cannot affect the
baseline. The two reports are written to `stage1-results/` by default:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_stage1.sh
```

Set `BATCH_SIZE`, `ITERATIONS`, `WARMUP`, `MAX_TOKENS`, or `PROMPT_FILE` to
change the shared workload. Both runs record model loading, first warmup,
prefill TTFT/tokens per second, decode latency/tokens per second, end-to-end
latency, peak allocated/reserved memory, KV-cache blocks, and exact token IDs.
The profile report additionally records CUDA-event timings for dequantization,
matrix multiplication, total projection time, temporary allocation, and input
shapes for every q/k/v/o/gate/up/down projection in both phases.

Here, a profiler is measurement instrumentation. The benchmark temporarily
wraps `GPTQLinear.forward`, records CUDA events around dequantization and
`F.linear`, and resolves those events after the GPU work completes. It answers
"where is the time going?"; it does not change the GPTQ algorithm or optimize
the model. The instrumentation adds event and hook overhead, which is why its
numbers must be read from `projection-profile.md` and never used as the formal
throughput baseline.

For example, to profile four concurrent sequences with a fixed prompt file:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
BATCH_SIZE=4 PROMPT_FILE=/path/to/prompts.txt \
OUTPUT_DIR=benchmark/qwen/qwen3-1.7B-gptq-INT8/stage1-batch4 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_stage1.sh
```

The Markdown profile table is the quick view. The JSON `projection_profile.by_type`
field contains machine-readable p50/p90/mean/min/max values grouped by phase
and projection type; `samples` retains each individual layer call and input
shape.

To compare logits with a Transformers/GPTQModel installation, set both paths:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
REFERENCE_MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_benchmark.sh
```

The reference comparison is optional because Transformers GPTQ backends differ
between environments. A failed or unavailable reference is recorded in the
JSON instead of invalidating the performance baseline.
