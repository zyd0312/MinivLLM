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
baseline. The two reports are written to `stage1-fused-results/` by default:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_stage1.sh
```

Set `BATCH_SIZE`, `ITERATIONS`, `WARMUP`, `MAX_TOKENS`, or `PROMPT_FILE` to
change the shared workload. Set `GPTQ_BACKEND=fused` to benchmark the fused
backend in the baseline process; the projection profile always uses the
reference backend because its hooks measure dequantization and `F.linear`
separately. Both runs record model loading, first warmup,
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

To verify that a fused backend preserves the reference model's outputs, run
`run_verify.sh` after the performance benchmark. It saves timestamped reports
under `stage1-fused-results/` by default, so a new check does not overwrite an
earlier result:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_verify.sh
```

The defaults compare `reference` and `fused` with four decode steps and the
chat template enabled. Set `CANDIDATE_BACKEND=fused_decode` or
`fused_prefill` to check one fused path separately. `PROMPT` or `PROMPT_FILE`
selects the input; `DECODE_STEPS`, `ATOL`, `RTOL`, `MIN_ALLCLOSE_RATE`,
`MIN_TOP1_RATE`, `MAX_MEAN_KL`, and `MIN_PROJECTION_ALLCLOSE_RATE` adjust the
run and its acceptance thresholds. Set `CHAT_TEMPLATE=0` to use the prompt as
plain text, or `KEEP_LOGITS=1` to save the compared tensors.

Use `OUTPUT_DIR` to change the report directory, or specify exact paths with
`OUTPUT_JSON` and `REPORT_MD`. Additional verifier options can be passed after
the script name, such as `run_verify.sh --decode-steps 8`; `PYTHON` selects the
Python interpreter.

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
CANDIDATE_BACKEND=fused_decode DECODE_STEPS=8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_verify.sh
```

For direct control of every verifier argument, call
`verify_gptq_backends.py` with Python:

```bash
.venv/bin/python benchmark/qwen/qwen3-1.7B-gptq-INT8/verify_gptq_backends.py \
  --model-path /path/to/Qwen3-1.7B-GPTQ-Int8 \
  --candidate-backend fused \
  --chat-template \
  --output benchmark/qwen/qwen3-1.7B-gptq-INT8/stage1-fused-results/backend-correctness.json \
  --report benchmark/qwen/qwen3-1.7B-gptq-INT8/stage1-fused-results/backend-correctness.md
```

The verifier runs reference and candidate sequentially in separate processes,
so it does not keep two model copies on the GPU. It compares prefill logits,
then several decode steps using the exact same greedy token prefix. During the
reference pass, it immediately compares each candidate projection on the same
activation. The JSON/Markdown report records logit errors, Top-1 agreement,
total-variation distance, KL divergence, per-projection errors and pass
thresholds. The process exits with status 1 if any configured criterion fails.

Use `--prompt` or `--prompt-file` to select another fixed input,
`--decode-steps` to extend the teacher-forced decode comparison, and
`--keep-logits` to preserve the full tensors for further analysis. Default
thresholds are `atol=0.1`, `rtol=0.02`, logit allclose rate >= 99.9%, Top-1
agreement >= 99%, mean KL <= 1e-3, and per-projection allclose rate >= 99.9%;
all can be adjusted from the command line.

To run correctness automatically after the baseline and profile, set
`RUN_CORRECTNESS=1`. This adds two model-load runs, so it is off by default.
`CORRECTNESS_BACKEND` selects the candidate (default: `GPTQ_BACKEND`) and
`CORRECTNESS_DECODE_STEPS` controls the number of teacher-forced decode steps:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
RUN_CORRECTNESS=1 GPTQ_BACKEND=fused \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_stage1.sh
```

The standalone verifier reports the prefill next-token distribution and each
decode-step distribution under the same reference-selected token prefix. It
also checks every GPTQ projection on the exact same activation. Logit error,
Top-1 agreement, KL divergence and projection error serve different purposes:
small tensor differences alone do not guarantee that the model assigns similar
probability to likely tokens. Total variation and KL measure distributional
differences directly, while the fixed-prefix comparison avoids random sampling
noise.

If the baseline completed but profiling failed, rerun only the profile phase
with `RUN_BASELINE=0`; this preserves the existing baseline files:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
OUTPUT_DIR=benchmark/qwen/qwen3-1.7B-gptq-INT8/stage1-fused-results \
RUN_BASELINE=0 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/run_stage1.sh
```

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
