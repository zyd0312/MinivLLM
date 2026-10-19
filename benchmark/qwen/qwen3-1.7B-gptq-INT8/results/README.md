# Benchmark results

Each run is kept in a purpose-specific directory. Do not place profiled runs
beside formal benchmark results:

- `performance/<backend>/run-<timestamp>/`: profiler-free throughput/latency results.
- `operator-profile/<backend>/run-<timestamp>/`: GPTQ per-projection diagnostics.
- `correctness/<backend>/run-<timestamp>/`: logits and projection comparisons.
- `system-profile/<backend>/run-<timestamp>/`: whole-path PyTorch profiler summaries, CUDA event attribution, and Chrome traces.

Historical reports were moved without editing their contents. Runs whose
backend or exact invocation was not recorded are kept under a
`legacy-*` directory name to avoid implying more certainty than the metadata
provides.

The initial whole-path profile attempt at
`system-profile/fused/run-20261018-200904-731561/` completed inference but
CUPTI returned `CUPTI_ERROR_INVALID_DEVICE`. Its trace contains CPU ranges only;
device-time values from that attempt are unavailable and must not be treated as
zero-cost kernels. Later profile runs explicitly report CUDA activity status
and retain per-projection CUDA-event spans as a coarse fallback.

The validated Nsight Systems capture is under
`system-profile/fused/nsys-final/run-20261018-2320/`. It contains the
`.nsys-rep` timeline and text summaries for CUDA API and kernel activity.
