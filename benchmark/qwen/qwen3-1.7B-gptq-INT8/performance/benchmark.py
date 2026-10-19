#!/usr/bin/env python3
"""Reproducible Qwen3 GPTQ-INT8 baseline and GPTQLinear profiler.

The benchmark deliberately calls ``LLMEngine.step`` instead of timing the
high-level ``generate`` method.  The first step is prefill and every later
step is decode, so the report contains separate measurements for both paths.
The script also records the exact token ids used by each run in the JSON
result, which makes a later comparison reproducible.

Example::

    python benchmark.py --model-path /path/to/Qwen3-1.7B-GPTQ-Int8 \
        --profile-projections --correctness --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate the repository root from benchmark.py")


REPO_ROOT = _find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from myvllm.engine.llm_engine import LLMEngine
from myvllm.layers.gptq import GPTQLinear, dequantize_gptq_int8
from myvllm.sampling_parameters import SamplingParams


DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in two concise paragraphs.",
    "Write a short Python function that computes the Fibonacci sequence.",
]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None, "stdev": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


class ProjectionProfiler:
    """Instrument GPTQLinear without changing the production implementation.

    CUDA events isolate dequantization from F.linear while preserving the
    exact reference execution.  Events are resolved only after a step has
    finished, so the timed region is not accidentally reduced to launch time.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled and torch.cuda.is_available()
        self.phase = "unknown"
        self.records: list[dict[str, Any]] = []
        self._original = GPTQLinear.forward
        self._installed = False

    def install(self) -> None:
        if not self.enabled or self._installed:
            return
        profiler = self

        def forward(layer: GPTQLinear, x: torch.Tensor) -> torch.Tensor:
            name = getattr(layer, "_benchmark_name", "GPTQLinear")
            baseline = torch.cuda.memory_allocated()
            deq_start = torch.cuda.Event(enable_timing=True)
            deq_end = torch.cuda.Event(enable_timing=True)
            mm_start = torch.cuda.Event(enable_timing=True)
            mm_end = torch.cuda.Event(enable_timing=True)
            deq_start.record()
            weight = dequantize_gptq_int8(layer.qweight, layer.scales)
            deq_end.record()
            after_dequant = torch.cuda.memory_allocated()
            mm_start.record()
            output = F.linear(x, weight)
            mm_end.record()
            after_linear = torch.cuda.memory_allocated()
            profiler.records.append({
                "phase": profiler.phase,
                "projection": name,
                "input_shape": list(x.shape),
                "dequant_start": deq_start,
                "dequant_end": deq_end,
                "matmul_start": mm_start,
                "matmul_end": mm_end,
                "baseline_allocated": baseline,
                "after_dequant_allocated": after_dequant,
                "after_linear_allocated": after_linear,
                "weight_bytes": int(weight.numel() * weight.element_size()),
            })
            return output

        GPTQLinear.forward = forward
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            GPTQLinear.forward = self._original
            self._installed = False

    def resolve(self) -> list[dict[str, Any]]:
        if not self.records:
            return []
        sync()
        resolved = []
        for record in self.records:
            dequant_ms = record["dequant_start"].elapsed_time(record["dequant_end"])
            matmul_ms = record["matmul_start"].elapsed_time(record["matmul_end"])
            baseline = record.pop("baseline_allocated")
            after_dequant = record.pop("after_dequant_allocated")
            after_linear = record.pop("after_linear_allocated")
            for key in ("dequant_start", "dequant_end", "matmul_start", "matmul_end"):
                record.pop(key)
            record.update({
                "dequant_ms": float(dequant_ms),
                "matmul_ms": float(matmul_ms),
                "forward_ms": float(dequant_ms + matmul_ms),
                "temporary_allocated_bytes": int(max(0, max(after_dequant, after_linear) - baseline)),
            })
            resolved.append(record)
        self.records = []
        return resolved


def attach_projection_names(engine: LLMEngine) -> None:
    for name, module in engine.model_runner.model.named_modules():
        if isinstance(module, GPTQLinear):
            module._benchmark_name = name


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    # Config reads config.json and derives all Qwen3 dimensions and GPTQ
    # metadata.  Keeping those values out of this file avoids stale benchmark
    # settings when a checkpoint is replaced.
    return {
        "model_name_or_path": str(Path(args.model_path).expanduser().resolve()),
        "quantization": "gptq",
        "gptq_backend": args.gptq_backend,
        "world_size": 1,
        "enforce_eager": True,
        "max_num_sequences": args.batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_cached_blocks": args.max_cached_blocks,
        "block_size": args.block_size,
        "max_model_length": args.max_model_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def load_prompts(args: argparse.Namespace, tokenizer) -> list[str]:
    if args.prompt_file:
        prompts = [line.rstrip("\n") for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
    else:
        prompts = list(args.prompt) if args.prompt else list(DEFAULT_PROMPTS)
    if not prompts:
        raise ValueError("At least one prompt is required")
    prompts = prompts[: args.batch_size]
    if args.chat_template:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts]
    return prompts


def run_one(
    engine: LLMEngine,
    prompts: list[str],
    sampling: SamplingParams,
    profiler: ProjectionProfiler,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token_ids = [engine.tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
    if any(not ids for ids in token_ids):
        raise ValueError("A prompt encoded to zero tokens")
    profiler.phase = "prefill"
    profiler.install()
    engine_start = time.perf_counter()
    for prompt in prompts:
        engine.add_prompt(prompt, sampling)

    phase_records: list[dict[str, Any]] = []
    first_prefill_ms = None
    prefill_total_ms = 0.0
    decode_ms: list[float] = []
    # ``LLMEngine.step`` samples the first completion token during prefill and
    # then reports ``len(seq)`` after postprocess.  Keep its raw counter for
    # diagnosis, but use prompt ids and decode steps for benchmark metrics so
    # prompt throughput and generated-token throughput have the usual meaning.
    raw_processed_prefill_tokens = 0
    decode_tokens = 0
    while not engine.scheduler.is_finished():
        # Scheduler gives waiting requests prefill priority.  Looking at the
        # queue before the call also labels a prefill split over several steps
        # correctly in the projection profile.
        profiler.phase = "prefill" if engine.scheduler.waiting else "decode"
        start = time.perf_counter()
        _, processed, is_prefill = engine.step()
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        phase_records.append({
            "phase": "prefill" if is_prefill else "decode",
            "processed_tokens": int(processed),
            "elapsed_ms": elapsed_ms,
        })
        if is_prefill:
            if first_prefill_ms is None:
                first_prefill_ms = elapsed_ms
            prefill_total_ms += elapsed_ms
            raw_processed_prefill_tokens += int(processed)
        else:
            decode_ms.append(elapsed_ms)
            decode_tokens += int(processed)

    total_ms = (time.perf_counter() - engine_start) * 1000.0
    profile_rows = profiler.resolve()
    prompt_tokens = sum(len(ids) for ids in token_ids)
    # Every request receives one sampled token from its prefill logits.  The
    # remaining completion tokens are emitted by decode steps.
    prefill_requests = len(prompts)
    generated_tokens = prefill_requests + decode_tokens
    decode_total = sum(decode_ms)
    result = {
        "prompt_token_ids": token_ids,
        "prompt_tokens": prompt_tokens,
        "prefill_processed_tokens": prompt_tokens,
        "raw_engine_prefill_processed_tokens": raw_processed_prefill_tokens,
        "prefill_first_completion_tokens": prefill_requests,
        "generated_tokens": generated_tokens,
        "ttft_ms": first_prefill_ms,
        "prefill_ms": prefill_total_ms,
        "prefill_tokens_per_second": (prompt_tokens / prefill_total_ms * 1000.0) if prefill_total_ms else None,
        "decode_ms": decode_total,
        "decode_steps": len(decode_ms),
        "decode_tokens": decode_tokens,
        "decode_token_latency_ms": (decode_total / decode_tokens) if decode_tokens else None,
        "decode_tokens_per_second": (decode_tokens / decode_total * 1000.0) if decode_total else None,
        "end_to_end_ms": total_ms,
        "end_to_end_generated_tokens_per_second": (generated_tokens / total_ms * 1000.0) if total_ms else None,
        "steps": phase_records,
    }
    return result, profile_rows


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "ttft_ms", "prefill_ms", "prefill_tokens_per_second", "decode_ms",
        "decode_token_latency_ms", "decode_tokens_per_second", "end_to_end_ms",
        "end_to_end_generated_tokens_per_second",
    ]
    return {field: summarize([float(run[field]) for run in runs if run.get(field) is not None]) for field in fields}


def projection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["projection"])].append(row)
    output = {}
    for (phase, projection), values in sorted(grouped.items()):
        key = f"{phase}/{projection}"
        output[key] = {
            "calls": len(values),
            "input_shapes": [list(shape) for shape in sorted({tuple(v["input_shape"]) for v in values})],
            "dequant_ms": summarize([v["dequant_ms"] for v in values]),
            "matmul_ms": summarize([v["matmul_ms"] for v in values]),
            "forward_ms": summarize([v["forward_ms"] for v in values]),
            "temporary_allocated_bytes": summarize([float(v["temporary_allocated_bytes"]) for v in values]),
            "weight_bytes": int(values[0]["weight_bytes"]),
        }
    return output


def projection_type_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate projection timings by q/k/v/o/gate/up/down across layers."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        projection = str(row["projection"]).rsplit(".", 1)[-1]
        grouped[(row["phase"], projection)].append(row)
    output = {}
    for (phase, projection), values in sorted(grouped.items()):
        output[f"{phase}/{projection}"] = {
            "calls": len(values),
            "input_shapes": [list(shape) for shape in sorted({tuple(v["input_shape"]) for v in values})],
            "dequant_ms": summarize([v["dequant_ms"] for v in values]),
            "matmul_ms": summarize([v["matmul_ms"] for v in values]),
            "forward_ms": summarize([v["forward_ms"] for v in values]),
            "temporary_allocated_bytes": summarize([float(v["temporary_allocated_bytes"]) for v in values]),
            "weight_bytes": sorted({int(v["weight_bytes"]) for v in values}),
        }
    return output


def _report_value(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def _report_stat(summary: dict[str, Any], key: str, suffix: str = "") -> str:
    return _report_value(summary.get(key), suffix=suffix)


def _bytes_to_gib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1024**3:.2f} GiB"


def _bytes_to_mib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1024**2:.2f} MiB"


def markdown_report(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary beside the raw JSON report."""
    metadata = report["metadata"]
    measured = report["measured"]
    summary = measured["summary"]
    loading = report["loading"]
    kv = report["kv_cache"]
    memory = measured["memory"]
    input_info = report["input"]
    first_compile_run = ((report.get("first_compile") or {}).get("run") or {})
    lines = [
        "# Qwen3-1.7B GPTQ-INT8 推理基线报告",
        "",
        f"- 运行时间（UTC）：`{metadata.get('timestamp_utc', '-')}`",
        f"- GPU：`{metadata.get('device', '-')}`",
        f"- PyTorch/CUDA：`{metadata.get('torch', '-')}` / `{metadata.get('cuda', '-')}`",
        f"- 模型：`{metadata.get('model_path', '-')}`",
        f"- GPTQ backend：`{metadata.get('gptq_backend', '-')}`",
        "",
        "## 测试配置",
        "",
        "| 项目 | 值 |",
        "|---|---:|",
        f"| Batch size | {input_info.get('batch_size', '-')} |",
        f"| Prompt tokens（本轮） | {sum(len(x) for x in [r for run in measured.get('runs', []) for r in run.get('prompt_token_ids', [])]) // max(1, len(measured.get('runs', [])))} |",
        f"| Max completion tokens | {input_info.get('max_tokens', '-')} |",
        f"| Max model length | {input_info.get('max_model_length', '-')} |",
        f"| 正式迭代次数 | {measured.get('iterations', '-')} |",
        f"| Profiler hooks | {'开启' if report['projection_profile'].get('enabled') else '关闭'} |",
        "",
        "## 核心性能指标",
        "",
        "统计值为正式迭代结果；P50 是主要比较指标，P90 用于观察抖动。",
        "",
        "| 指标 | P50 | P90 | 平均 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| TTFT | {_report_stat(summary['ttft_ms'], 'median', ' ms')} | {_report_stat(summary['ttft_ms'], 'p90', ' ms')} | {_report_stat(summary['ttft_ms'], 'mean', ' ms')} | {_report_stat(summary['ttft_ms'], 'min', ' ms')} | {_report_stat(summary['ttft_ms'], 'max', ' ms')} |",
        f"| Prefill 吞吐 | {_report_stat(summary['prefill_tokens_per_second'], 'median', ' tok/s')} | {_report_stat(summary['prefill_tokens_per_second'], 'p90', ' tok/s')} | {_report_stat(summary['prefill_tokens_per_second'], 'mean', ' tok/s')} | {_report_stat(summary['prefill_tokens_per_second'], 'min', ' tok/s')} | {_report_stat(summary['prefill_tokens_per_second'], 'max', ' tok/s')} |",
        f"| Decode 单 token 延迟 | {_report_stat(summary['decode_token_latency_ms'], 'median', ' ms')} | {_report_stat(summary['decode_token_latency_ms'], 'p90', ' ms')} | {_report_stat(summary['decode_token_latency_ms'], 'mean', ' ms')} | {_report_stat(summary['decode_token_latency_ms'], 'min', ' ms')} | {_report_stat(summary['decode_token_latency_ms'], 'max', ' ms')} |",
        f"| Decode 吞吐 | {_report_stat(summary['decode_tokens_per_second'], 'median', ' tok/s')} | {_report_stat(summary['decode_tokens_per_second'], 'p90', ' tok/s')} | {_report_stat(summary['decode_tokens_per_second'], 'mean', ' tok/s')} | {_report_stat(summary['decode_tokens_per_second'], 'min', ' tok/s')} | {_report_stat(summary['decode_tokens_per_second'], 'max', ' tok/s')} |",
        f"| 端到端延迟 | {_report_stat(summary['end_to_end_ms'], 'median', ' ms')} | {_report_stat(summary['end_to_end_ms'], 'p90', ' ms')} | {_report_stat(summary['end_to_end_ms'], 'mean', ' ms')} | {_report_stat(summary['end_to_end_ms'], 'min', ' ms')} | {_report_stat(summary['end_to_end_ms'], 'max', ' ms')} |",
        f"| 端到端生成吞吐 | {_report_stat(summary['end_to_end_generated_tokens_per_second'], 'median', ' tok/s')} | {_report_stat(summary['end_to_end_generated_tokens_per_second'], 'p90', ' tok/s')} | {_report_stat(summary['end_to_end_generated_tokens_per_second'], 'mean', ' tok/s')} | {_report_stat(summary['end_to_end_generated_tokens_per_second'], 'min', ' tok/s')} | {_report_stat(summary['end_to_end_generated_tokens_per_second'], 'max', ' tok/s')} |",
        "",
        "## 初始化、显存与 KV cache",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 模型加载时间 | {_report_value(loading.get('elapsed_ms'), suffix=' ms')} |",
        f"| 首次显式 warmup 端到端延迟 | {_report_value(first_compile_run.get('end_to_end_ms'), suffix=' ms')} |",
        f"| 峰值 allocated | {_bytes_to_gib(memory.get('peak_allocated_bytes'))} |",
        f"| 峰值 reserved | {_bytes_to_gib(memory.get('peak_reserved_bytes'))} |",
        f"| KV cache blocks | {kv.get('total_blocks', '-')} |",
        f"| Block size | {kv.get('block_size', '-')} tokens |",
        f"| 理论 KV token 容量 | {_report_value((kv.get('total_blocks', 0) or 0) * (kv.get('block_size', 0) or 0), digits=0)} |",
        f"| 正式测试后空闲 blocks | {kv.get('free_blocks_after_runs', '-')} |",
        "",
    ]

    profile = report.get("projection_profile", {})
    if profile.get("enabled"):
        rows = profile.get("samples", [])
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("phase", "unknown"), row.get("projection", "").split(".")[-1])].append(row)
        lines += [
            "## GPTQ 投影 profile",
            "",
            "以下为所有 28 层同名投影的中位数；profile 会增加少量 event 记录开销，不能直接当作无 instrumentation 的端到端基线。",
            "",
            "| 阶段 | 投影 | 反量化 | MatMul | Forward | 临时显存 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for phase in ("prefill", "decode"):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                values = grouped.get((phase, projection), [])
                if not values:
                    continue
                lines.append(
                    f"| {phase} | `{projection}` | "
                    f"{_report_value(statistics.median(x['dequant_ms'] for x in values), suffix=' ms')} | "
                    f"{_report_value(statistics.median(x['matmul_ms'] for x in values), suffix=' ms')} | "
                    f"{_report_value(statistics.median(x['forward_ms'] for x in values), suffix=' ms')} | "
                    f"{_bytes_to_mib(statistics.median(x['temporary_allocated_bytes'] for x in values))} |"
                )
        lines.append("")
    else:
        lines += [
            "## GPTQ 投影 profile",
            "",
            "本次未开启 projection profiler。使用 `PROFILE_PROJECTIONS=1` 单独运行 profile，避免污染端到端性能基线。",
            "",
        ]

    correctness = report.get("correctness")
    lines += ["## 正确性对齐", ""]
    if not correctness:
        lines += ["本次未执行 Transformers/GPTQModel logits 对齐。", ""]
    elif correctness.get("status") == "ok":
        lines += [
            "| 指标 | 值 |",
            "|---|---:|",
            f"| Max absolute error | {_report_value(correctness.get('max_absolute_error'))} |",
            f"| Relative error | {_report_value(correctness.get('relative_error'))} |",
            f"| Top-1 一致率 | {_report_value(correctness.get('top1_match_rate'), suffix='')} |",
            f"| KL divergence | {_report_value(correctness.get('kl_divergence_reference_to_mini'))} |",
            "",
        ]
    else:
        lines += [f"状态：`{correctness.get('status', 'unknown')}`；{correctness.get('reason', '-')}", ""]

    lines += [
        "## 结果文件",
        "",
        "- JSON 原始结果包含每轮明细、token IDs 和投影 profile 样本。",
        "- 本 Markdown 文件用于快速汇报核心指标。",
        "",
    ]
    return "\n".join(lines)


def correctness_check(engine: LLMEngine, prompt: str, reference_path: str | None) -> dict[str, Any]:
    """Compare final-token logits (and hidden state when available) to HF."""
    result: dict[str, Any] = {"status": "skipped", "reason": None}
    if not reference_path:
        result["reason"] = "no --reference-model-path supplied"
        return result
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - environment dependent
        result["reason"] = f"transformers unavailable: {exc}"
        return result

    captured_logits: list[torch.Tensor] = []
    captured_hidden: list[torch.Tensor] = []
    runner = engine.model_runner
    original_run_model = runner.run_model
    original_forward = runner.model.forward

    def capture_run_model(input_ids, is_prefill):
        logits = original_run_model(input_ids, is_prefill)
        captured_logits.append(logits.detach().float().cpu())
        return logits

    def capture_forward(input_ids):
        hidden = original_forward(input_ids)
        captured_hidden.append(hidden.detach().float().cpu())
        return hidden

    runner.run_model = capture_run_model
    runner.model.forward = capture_forward
    try:
        sampling = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True, max_model_length=engine.config["max_model_length"])
        engine.add_prompt(prompt, sampling)
        while not engine.scheduler.is_finished():
            engine.step()
        if not captured_logits:
            raise RuntimeError("the engine did not produce logits")
        mini_logits_all = captured_logits[0]
        mini_logits = mini_logits_all[-1]
        mini_hidden = captured_hidden[0][-1] if captured_hidden else None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        reference = AutoModelForCausalLM.from_pretrained(
            str(Path(reference_path).expanduser()),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device).eval()
        tokenizer = engine.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            reference_output = reference(**inputs, output_hidden_states=True, use_cache=False)
        ref_logits_all = reference_output.logits[0].float().cpu()
        compare_tokens = min(mini_logits_all.shape[0], ref_logits_all.shape[0])
        mini_logits_all = mini_logits_all[-compare_tokens:]
        ref_logits_all = ref_logits_all[-compare_tokens:]
        ref_logits = ref_logits_all[-1]
        abs_error = (mini_logits_all - ref_logits_all).abs()
        denominator = ref_logits_all.abs().amax().clamp_min(1e-12)
        mini_top = mini_logits.argmax().item()
        ref_top = ref_logits.argmax().item()
        mini_logp = F.log_softmax(mini_logits, dim=-1)
        ref_prob = F.softmax(ref_logits, dim=-1)
        result.update({
            "status": "ok",
            "max_absolute_error": float(abs_error.max()),
            "relative_error": float(abs_error.max() / denominator),
            "top1_match": bool(mini_top == ref_top),
            "top1_match_rate": float((mini_logits_all.argmax(dim=-1) == ref_logits_all.argmax(dim=-1)).float().mean()),
            "mini_top1": int(mini_top),
            "reference_top1": int(ref_top),
            "kl_divergence_reference_to_mini": float(F.kl_div(mini_logp, ref_prob, reduction="sum")),
        })
        if mini_hidden is not None and getattr(reference_output, "hidden_states", None):
            ref_hidden = reference_output.hidden_states[-1][0, -1].float().cpu()
            hidden_error = (mini_hidden - ref_hidden).abs()
            result["hidden_max_absolute_error"] = float(hidden_error.max())
            result["hidden_relative_error"] = float(hidden_error.max() / ref_hidden.abs().amax().clamp_min(1e-12))
        del reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # reference backends vary by installation
        result["reason"] = f"reference comparison failed: {type(exc).__name__}: {exc}"
    finally:
        runner.run_model = original_run_model
        runner.model.forward = original_forward
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Qwen3 GPTQ-INT8 checkpoint directory")
    parser.add_argument("--reference-model-path", default=None, help="Optional Transformers/GPTQModel checkpoint for logits comparison")
    parser.add_argument("--output", default="results.json", help="JSON result path")
    parser.add_argument("--report", default=None, help="Markdown summary path (default: same stem as --output)")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations excluded from formal statistics")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-length", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-cached-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prompt", action="append", help="Prompt; repeat for multiple prompts")
    parser.add_argument("--prompt-file", help="UTF-8 file with one prompt per line")
    parser.add_argument("--chat-template", action="store_true", help="Apply the checkpoint chat template")
    parser.add_argument("--profile-projections", action="store_true", help="Record GPTQ dequantization and matmul CUDA event timings")
    parser.add_argument(
        "--gptq-backend",
        choices=("reference", "fused", "fused_decode", "fused_prefill"),
        default=os.environ.get("GPTQ_BACKEND", "reference"),
        help="GPTQLinear runtime backend",
    )
    parser.add_argument("--correctness", action="store_true", help="Compare one fixed prompt with Transformers/GPTQModel")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.batch_size <= 0 or args.max_tokens <= 0:
        raise ValueError("iterations, batch-size and max-tokens must be positive; warmup cannot be negative")
    if args.profile_projections and args.gptq_backend != "reference":
        raise ValueError("projection profiler currently measures the reference GPTQ backend only")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 GPTQ benchmark requires CUDA; use tests/test_gptq.py for CPU layout checks")
    torch.manual_seed(args.seed)
    if not Path(args.model_path).expanduser().is_dir():
        raise FileNotFoundError(args.model_path)

    load_start = time.perf_counter()
    engine = LLMEngine(build_config(args))
    load_end = time.perf_counter()
    attach_projection_names(engine)
    prompts = load_prompts(args, engine.tokenizer)
    sampling = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        max_model_length=args.max_model_length,
    )
    profiler = ProjectionProfiler(args.profile_projections)

    load_memory = memory_snapshot()
    warmup_runs = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        warmup_runs.append(run_one(engine, prompts, sampling, profiler)[0])
    warmup_memory = memory_snapshot()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    measured_runs = []
    projection_rows: list[dict[str, Any]] = []
    for _ in range(args.iterations):
        run, rows = run_one(engine, prompts, sampling, profiler)
        measured_runs.append(run)
        projection_rows.extend(rows)
    measured_memory = memory_snapshot()

    # Correctness must exercise the production forward directly; do not leave
    # the benchmark monkey-patch installed while loading/running the reference.
    profiler.uninstall()
    correctness = None
    if args.correctness:
        correctness = correctness_check(engine, prompts[0], args.reference_model_path)

    runner = engine.model_runner
    blocks = getattr(engine.scheduler.block_manager, "blocks", [])
    free_blocks = getattr(engine.scheduler.block_manager, "free_block_ids", [])
    report = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_revision": os.popen(f"git -C {REPO_ROOT} rev-parse HEAD 2>/dev/null").read().strip() or None,
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "dtype": "float16 activations/scales; int32 packed qweight",
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seed": args.seed,
            "gptq_backend": args.gptq_backend,
        },
        "input": {
            "prompts": prompts,
            "batch_size": len(prompts),
            "max_tokens": args.max_tokens,
            "max_model_length": args.max_model_length,
            "sampling": {"temperature": 0.6, "ignore_eos": True},
        },
        "loading": {
            "elapsed_ms": (load_end - load_start) * 1000.0,
            "model_runner_internal_warmup": True,
            "memory": load_memory,
        },
        "first_compile": {
            "measurement": "first explicit warmup iteration",
            "run": warmup_runs[0] if warmup_runs else None,
            "note": "The GPTQ model uses enforce_eager=True; this captures first-call compilation such as SamplerLayer torch.compile.",
        },
        "warmup": {"iterations": args.warmup, "runs": warmup_runs, "summary": aggregate_runs(warmup_runs), "memory": warmup_memory},
        "measured": {"iterations": args.iterations, "runs": measured_runs, "summary": aggregate_runs(measured_runs), "memory": measured_memory},
        "kv_cache": {
            "block_size": args.block_size,
            "total_blocks": len(blocks),
            "free_blocks_after_runs": len(free_blocks),
            "configured_max_cached_blocks": args.max_cached_blocks,
            "runtime_max_cached_blocks": runner.config.get("max_cached_blocks"),
        },
        "projection_profile": {
            "enabled": profiler.enabled,
            "summary": projection_summary(projection_rows),
            "by_type": projection_type_summary(projection_rows),
            "samples": projection_rows,
        },
        "correctness": correctness,
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    report_path = Path(args.report).expanduser() if args.report else output_path.with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.resolve()),
        "markdown_report": str(report_path.resolve()),
        "loading_ms": report["loading"]["elapsed_ms"],
        "prefill_tokens_per_second": report["measured"]["summary"]["prefill_tokens_per_second"],
        "decode_tokens_per_second": report["measured"]["summary"]["decode_tokens_per_second"],
        "ttft_ms": report["measured"]["summary"]["ttft_ms"],
        "peak_allocated_bytes": measured_memory["peak_allocated_bytes"],
        "kv_blocks": report["kv_cache"]["total_blocks"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
