# Qwen3-1.7B GPTQ-INT8 benchmark

The benchmark is split by test purpose so that a diagnostic profile never
looks like a formal performance result:

```text
performance/       formal throughput/latency benchmark and Stage 1 runner
operator-profile/  GPTQ projection instrumentation
system-profile/    whole fused-path PyTorch/CUDA timeline
correctness/       reference/fused logits and projection verification
results/           timestamped JSON/Markdown/trace artifacts
```

Existing reports were moved into `results/` without changing their contents.
New runs use timestamped directories and do not overwrite earlier reports.

Run the formal benchmark from the repository root with:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/performance/run_benchmark.sh
```

It records model loading, warmup and formal iterations, prefill TTFT and
tokens/s, per-token decode latency and tokens/s, end-to-end latency, peak
allocated/reserved memory, KV-cache capacity, and exact prompt token IDs. The
default fused report is written to `results/performance/fused/run-<timestamp>/`;
set `GPTQ_BACKEND=reference` to write a matching reference directory. Use
`OUTPUT_DIR` or `OUTPUT` to choose another location. The default run has no
profiler hooks and is the performance result to use for A/B comparisons.

To run the complete Stage 1 pair with one fixed workload:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/performance/run_stage1.sh
```

The script starts a fresh process for the formal fused baseline and another
fresh process for the reference-only GPTQ projection profile. By default they
are stored separately in `results/performance/fused/` and
`results/operator-profile/reference/`. The profile process measures
dequantization and `F.linear` with CUDA events; it adds instrumentation and
must not be used as the formal throughput result. Set `BATCH_SIZE`,
`ITERATIONS`, `WARMUP`, `MAX_TOKENS`, or `PROMPT_FILE` to keep the workload
fixed while changing its size. Use `BASELINE_OUTPUT_DIR`,
`PROFILE_OUTPUT_DIR`, and `CORRECTNESS_OUTPUT_DIR` for exact destinations.
`RESULTS_ROOT` changes the root for all three output groups.

If only the reference projection profile must be repeated, keep the existing
baseline by supplying its output directory and disabling the baseline:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
RUN_BASELINE=0 \
BASELINE_OUTPUT_DIR=benchmark/qwen/qwen3-1.7B-gptq-INT8/results/performance/fused/run-existing \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/performance/run_stage1.sh
```

To verify that the fused backend preserves the reference output, run:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/correctness/run_verify.sh
```

The verifier compares prefill and teacher-forced decode logits, Top-1
agreement, total variation, KL divergence, and every GPTQ projection on the
same activation. It defaults to `reference` versus `fused`, four decode steps,
and the chat template. Set `CANDIDATE_BACKEND=fused_decode` or
`fused_prefill` to isolate one explicit fused backend. `PROMPT`,
`PROMPT_FILE`, `DECODE_STEPS`, `ATOL`, `RTOL`, `MIN_ALLCLOSE_RATE`,
`MIN_TOP1_RATE`, `MAX_MEAN_KL`, and `MIN_PROJECTION_ALLCLOSE_RATE` adjust the
run. Reports go to `results/correctness/<backend>/` by default. Additional
verifier options can be passed after the script name.

For direct control of every verifier argument:

```bash
.venv/bin/python benchmark/qwen/qwen3-1.7B-gptq-INT8/correctness/verify_gptq_backends.py \
  --model-path /path/to/Qwen3-1.7B-GPTQ-Int8 \
  --candidate-backend fused \
  --chat-template \
  --output benchmark/qwen/qwen3-1.7B-gptq-INT8/results/correctness/fused/manual/backend-correctness.json \
  --report benchmark/qwen/qwen3-1.7B-gptq-INT8/results/correctness/fused/manual/backend-correctness.md
```

## Profile the current fused path

The whole-path profiler is separate from both the formal benchmark and the
reference-only projection profile. It performs one short fused request after
warmup, records host wall time and CUDA stream elapsed time for each engine
step, and records CUDA event spans around every GPTQ projection. Projection
spans include the fused kernel launch and can be grouped by layer, phase, and
shape. They are coarse attribution data, not a sum of device busy time. The
output contains a Markdown summary, raw JSON, and `trace.json` for Chrome trace
inspection. Profiler instrumentation changes timing, so use the run only to
locate candidates for optimization.

Run it with:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/system-profile/run_profile_fused.sh
```

The default trace uses the `fused` backend, one warmup request, and eight
completion tokens. `--max-tokens 1` captures prefill only; use `--max-tokens 2`
to include one decode step, or 32 to inspect more repeated decode steps. The
`fused_decode` and `fused_prefill` choices force that kernel for both engine
phases, so use `fused` when you want to observe normal runtime dispatch. Set
`OUTPUT_DIR` to choose the report directory; otherwise each backend gets its
own `results/system-profile/<backend>/run-<timestamp>/` directory.

The generated `summary.md` reports prefill/decode wall and CUDA stream times,
per-projection CUDA event timings, and PyTorch profiler events. The profiler
also checks whether CUDA device activities actually appear in `trace.json`. If
CUPTI initialization fails, device time is marked unavailable; zero-valued
device columns must not be read as zero-cost kernels. In that case, the
per-projection CUDA-event section still gives a coarse comparison, while the
trace can only be used for CPU-side ranges. CUDA event elapsed time may include
gaps between work on the stream, and wall time minus event time is not a valid
estimate of CPU time.

Open `trace.json` in Perfetto or `chrome://tracing`. When CUDA kernel events are
present, use them to inspect device activity and correlate it with the
`minivllm.gptq_projection.*` and engine-step ranges. Use the per-projection
section to find which layers and matrix shapes dominate, then confirm any
change with the profiler-free benchmark.

If PyTorch reports a CUPTI error, a short Nsight Systems capture can collect a
separate system-level report. The profile script marks the request with an
NVTX range so model loading and warmup stay outside collection:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B-GPTQ-Int8 \
  benchmark/qwen/qwen3-1.7B-gptq-INT8/system-profile/run_nsys_fused.sh --max-tokens 2
```

This disables the in-process PyTorch profiler to avoid two CUPTI subscribers.
The capture includes process initialization and warmup; use it for kernel
identity and rough attribution rather than a clean request-only latency
measurement. The request is marked with an NVTX range for timeline correlation.
It produces `fused.nsys-rep` under the run directory. Open it in Nsight
Systems, or summarize it with `nsys stats`; CUDA API/kernel reports are the
source for per-kernel durations, while the PyTorch CUDA-event section provides
projection-level grouping.

The recommended loop is: run the formal fused benchmark, run this short whole
path profile, identify the largest device-time region, then change one
operator and repeat the same benchmark/profile pair. Keep the formal benchmark
free of profiler instrumentation.
