#!/usr/bin/env python3
"""Profile one short fused Qwen3 request and write a GPU timeline report.

The run intentionally separates model loading/warmup from the profiled request.
The report contains stage-level wall and CUDA-event timings, peak memory, a
Chrome trace, and the hottest PyTorch profiler events. It is diagnostic data;
do not use its timings as the formal throughput baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate the repository root from profile_fused.py")


REPO_ROOT = _find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from myvllm.engine.llm_engine import LLMEngine  # noqa: E402
from myvllm.layers.gptq import GPTQLinear  # noqa: E402
from myvllm.sampling_parameters import SamplingParams  # noqa: E402


DEFAULT_PROMPT = "Explain why the sky appears blue in two concise paragraphs."
BACKENDS = ("fused", "fused_decode", "fused_prefill")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min_ms": min(values) if values else None,
        "median_ms": percentile(values, 0.5),
        "p90_ms": percentile(values, 0.9),
        "mean_ms": statistics.fmean(values) if values else None,
        "max_ms": max(values) if values else None,
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


def load_prompt(args: argparse.Namespace, tokenizer) -> str:
    if args.prompt_file:
        prompts = [line.rstrip("\n") for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError("--prompt-file contains no non-empty prompt")
        prompt = prompts[0]
    else:
        prompt = args.prompt or DEFAULT_PROMPT
    if args.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name_or_path": str(Path(args.model_path).expanduser().resolve()),
        "quantization": "gptq",
        "gptq_backend": args.backend,
        "world_size": 1,
        "enforce_eager": True,
        "max_num_sequences": 1,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_cached_blocks": args.max_cached_blocks,
        "block_size": args.block_size,
        "max_model_length": args.max_model_length,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def timed_step(engine: LLMEngine, phase_hint: str) -> dict[str, Any]:
    """Measure host wall time and CUDA-stream elapsed time for one engine step."""
    sync_cuda()
    start_cpu = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.profiler.record_function(f"minivllm.{phase_hint}.engine_step"):
        _, processed_tokens, is_prefill = engine.step()
    end_event.record()
    sync_cuda()
    end_cpu = time.perf_counter()
    return {
        "phase": "prefill" if is_prefill else "decode",
        "phase_hint": phase_hint,
        "processed_tokens": int(processed_tokens),
        "wall_ms": (end_cpu - start_cpu) * 1000.0,
        # CUDA events measure elapsed time on the stream, including any idle
        # gaps between queued work. This is not a sum of kernel busy times.
        "stream_elapsed_ms": float(start_event.elapsed_time(end_event)),
    }


def run_request(
    engine: LLMEngine,
    prompt: str,
    prompt_tokens: int,
    args: argparse.Namespace,
    profiler: Any | None = None,
    projection_recorder: "ProjectionEventRecorder | None" = None,
) -> list[dict[str, Any]]:
    sampling = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        max_model_length=args.max_model_length,
    )
    engine.add_prompt(prompt, sampling)
    records: list[dict[str, Any]] = []
    while not engine.scheduler.is_finished():
        phase_hint = "prefill" if engine.scheduler.waiting else "decode"
        if projection_recorder is not None:
            projection_recorder.phase = phase_hint
        record = timed_step(engine, phase_hint)
        # LLMEngine.step() counts the sampled token after scheduler.postprocess
        # in its prefill total. For model input accounting, prefill consumes
        # only the prompt; subsequent decode steps each consume one token.
        record["input_tokens"] = prompt_tokens if record["phase"] == "prefill" else record["processed_tokens"]
        records.append(record)
        if profiler is not None:
            profiler.step()
    return records


def _event_value(event: Any, name: str) -> float:
    try:
        return float(getattr(event, name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def profiler_events(profiler: Any, limit: int = 60, *, device_time_available: bool) -> list[dict[str, Any]]:
    events = list(profiler.key_averages(group_by_input_shape=True))
    events.sort(
        key=lambda event: (
            _event_value(event, "self_device_time_total"),
            _event_value(event, "self_cpu_time_total"),
        ),
        reverse=True,
    )
    rows = []
    for event in events[:limit]:
        rows.append(
            {
                "name": str(getattr(event, "key", "")),
                "calls": int(getattr(event, "count", 0)),
                "self_cpu_ms": _event_value(event, "self_cpu_time_total") / 1000.0,
                "cpu_total_ms": _event_value(event, "cpu_time_total") / 1000.0,
                "self_device_ms": _event_value(event, "self_device_time_total") / 1000.0 if device_time_available else None,
                "device_total_ms": _event_value(event, "device_time_total") / 1000.0 if device_time_available else None,
                "input_shapes": getattr(event, "input_shapes", []),
            }
        )
    return rows


def trace_device_activity(trace_path: Path) -> dict[str, Any]:
    """Report whether an exported Chrome trace contains CUDA device work."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    device_categories = {"kernel", "cuda_kernel", "gpu_memcpy", "gpu_memset"}
    device_events = []
    for event in trace.get("traceEvents", []):
        categories = {part.strip().lower() for part in str(event.get("cat", "")).split(",")}
        if categories & device_categories:
            device_events.append(event)
    return {
        "cuda_device_activity_available": bool(device_events),
        "cuda_device_event_count": len(device_events),
        "trace_event_categories": sorted({str(event.get("cat", "")) for event in trace.get("traceEvents", [])}),
    }


class ProjectionEventRecorder:
    """Record module-named CUDA-event spans around each fused GPTQ projection."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.phase = "unknown"
        self.records: list[dict[str, Any]] = []
        self._original = GPTQLinear.forward
        self._installed = False

    def install(self, engine: LLMEngine) -> None:
        if not self.enabled or self._installed:
            return
        recorder = self
        for name, module in engine.model_runner.model.named_modules():
            if isinstance(module, GPTQLinear):
                module._system_profile_name = name

        def profiled_forward(layer: GPTQLinear, x: torch.Tensor) -> torch.Tensor:
            name = getattr(layer, "_system_profile_name", "GPTQLinear")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function(f"minivllm.gptq_projection.{name}"):
                output = recorder._original(layer, x)
            end.record()
            recorder.records.append({
                "phase": recorder.phase,
                "projection": name,
                "input_shape": list(x.shape),
                "start_event": start,
                "end_event": end,
            })
            return output

        GPTQLinear.forward = profiled_forward
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            GPTQLinear.forward = self._original
            self._installed = False

    def resolve(self) -> list[dict[str, Any]]:
        """Resolve CUDA event spans after the profiled request has synchronized."""
        resolved = []
        for record in self.records:
            start = record.pop("start_event")
            end = record.pop("end_event")
            record["stream_elapsed_ms"] = float(start.elapsed_time(end))
            resolved.append(record)
        self.records = []
        return resolved


def summarize_projection_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for record in records:
        key = (str(record["phase"]), str(record["projection"]), json.dumps(record["input_shape"]))
        grouped.setdefault(key, []).append(float(record["stream_elapsed_ms"]))
    rows = []
    for (phase, projection, shape), values in grouped.items():
        rows.append({
            "phase": phase,
            "projection": projection,
            "input_shape": json.loads(shape),
            "calls": len(values),
            "stream_elapsed_ms": summarize(values),
        })
    return sorted(rows, key=lambda row: (row["phase"], -float(row["stream_elapsed_ms"]["mean_ms"] or 0.0)))


def summarize_projection_types(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate layer instances by projection type for a compact report."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for record in records:
        projection_type = re.sub(r"^model\.layers\.\d+\.", "", str(record["projection"]))
        key = (str(record["phase"]), projection_type, json.dumps(record["input_shape"]))
        grouped.setdefault(key, []).append(float(record["stream_elapsed_ms"]))
    rows = []
    for (phase, projection_type, shape), values in grouped.items():
        rows.append({
            "phase": phase,
            "projection_type": projection_type,
            "input_shape": json.loads(shape),
            "calls": len(values),
            "stream_elapsed_ms": summarize(values),
        })
    return sorted(rows, key=lambda row: (row["phase"], -float(row["stream_elapsed_ms"]["median_ms"] or 0.0)))


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def format_optional_ms(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Fused 路径整体 GPU Profile",
        "",
        f"- 模型：`{report['model_path']}`",
        f"- Backend：`{report['backend']}`",
        f"- Prompt tokens：{report['prompt_tokens']}",
        f"- Profile max completion tokens：{report['config']['max_tokens']}",
        "",
        "本报告包含 profiler instrumentation，只用于定位瓶颈；正式吞吐和延迟应以无 profiler 的 performance 报告为准。",
        (
            "\n⚠️ **当前 trace 未采集到 CUDA kernel activity。设备时间 unavailable；请勿把它解释为 0 ms。**\n"
            if not report["cuda_trace"]["cuda_device_activity_available"]
            else ""
        ),
        "",
        "## 阶段级耗时",
        "",
        "| 阶段 | 次数 | Wall P50 | Wall P90 | CUDA stream P50 | CUDA stream P90 | 输入 token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("prefill", "decode"):
        values = report["stage_summary"].get(phase)
        if not values:
            continue
        lines.append(
            f"| {phase} | {values['count']} | {values['wall_ms']['median_ms']:.3f} ms | "
            f"{values['wall_ms']['p90_ms']:.3f} ms | {values['stream_elapsed_ms']['median_ms']:.3f} ms | "
            f"{values['stream_elapsed_ms']['p90_ms']:.3f} ms | {values['input_tokens']} |"
        )
    lines += [
        "",
        "CUDA event 时间是 stream 上的经过时间，可能包含 kernel 间空隙，不等同于 GPU kernel busy time；不能用 Wall 减去此值直接推算 CPU 时间。",
        "",
        "## GPTQ 投影（CUDA event）",
        "",
        "按每个 fused GPTQ 投影调用的 CUDA event 时间分组。该数据不依赖 CUPTI kernel activity trace，但仍可能包含该投影范围内的 stream 空隙。",
        "",
        "| 阶段 | 投影类型 | 输入形状 | 调用次数（所有层） | 平均 stream ms | P50 ms | P90 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["projection_summary"]:
        values = row["stream_elapsed_ms"]
        lines.append(
            f"| {row['phase']} | `{row['projection_type']}` | `{row['input_shape']}` | {row['calls']} | "
            f"{values['mean_ms']:.3f} | {values['median_ms']:.3f} | {values['p90_ms']:.3f} |"
        )
    lines += [
        "",
        "## Profiler 热点",
        "",
        "按 self device time 排序。设备时间 unavailable 表示本次未取得 CUDA kernel activity；它不代表耗时为零。",
        "",
        "| Kernel/范围 | 调用次数 | Self CPU ms | CPU total ms | Self device ms | Device total ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for event in report["top_events"][:40]:
        self_device = format_optional_ms(event["self_device_ms"])
        device_total = format_optional_ms(event["device_total_ms"])
        lines.append(
            f"| `{event['name']}` | {event['calls']} | {event['self_cpu_ms']:.3f} | "
            f"{event['cpu_total_ms']:.3f} | {self_device} | {device_total} |"
        )
    lines += [
        "",
        "## 输出文件",
        "",
        f"- Chrome trace：`{report['trace_path']}`",
        f"- 原始 JSON：`{report['json_path']}`",
        f"- CUDA device activity：`{'可用' if report['cuda_trace']['cuda_device_activity_available'] else '缺失'}（{report['cuda_trace']['cuda_device_event_count']} 个 device events）`",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="fused")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=8, help="Short completion; 1 profiles prefill only, 2 includes one decode step")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-model-length", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-cached-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-events", type=int, default=60)
    parser.add_argument("--with-stack", action="store_true", help="Include Python stack traces; increases overhead")
    parser.add_argument("--no-record-shapes", action="store_true")
    parser.add_argument("--no-profile-memory", action="store_true")
    parser.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help="Skip PyTorch profiler; use with Nsight Systems to avoid CUPTI subscriber conflicts",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fused system profiling requires CUDA")
    if args.max_tokens <= 0 or args.warmup < 0 or args.top_events <= 0:
        raise ValueError("max-tokens and top-events must be positive; warmup cannot be negative")
    if not Path(args.model_path).expanduser().is_dir():
        raise FileNotFoundError(args.model_path)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    json_path = output_dir / "summary.json"
    report_path = output_dir / "summary.md"
    torch.manual_seed(1234)

    load_start = time.perf_counter()
    engine = LLMEngine(build_config(args))
    load_ms = (time.perf_counter() - load_start) * 1000.0
    prompt = load_prompt(args, engine.tokenizer)
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    warmup_records = []
    for _ in range(args.warmup):
        warmup_records.extend(run_request(engine, prompt, prompt_tokens, args))
    sync_cuda()
    torch.cuda.reset_peak_memory_stats()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    profile_records: list[dict[str, Any]] = []
    projection_recorder = ProjectionEventRecorder(enabled=True)
    projection_recorder.install(engine)
    profiler = None
    nsys_range = args.no_torch_profiler
    if nsys_range:
        torch.cuda.nvtx.range_push("minivllm.profile.request")
    try:
        if args.no_torch_profiler:
            profile_records.extend(
                run_request(engine, prompt, prompt_tokens, args, None, projection_recorder)
            )
        else:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=not args.no_record_shapes,
                profile_memory=not args.no_profile_memory,
                with_stack=args.with_stack,
            ) as profiler:
                profile_records.extend(
                    run_request(engine, prompt, prompt_tokens, args, profiler, projection_recorder)
                )
    finally:
        if nsys_range:
            torch.cuda.nvtx.range_pop()
        projection_recorder.uninstall()
    sync_cuda()

    if profiler is not None:
        profiler.export_chrome_trace(str(trace_path))
    else:
        trace_path.write_text(
            json.dumps(
                {"displayTimeUnit": "ms", "traceEvents": [], "profile_status": "torch_profiler_disabled"},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    projection_events = projection_recorder.resolve()
    cuda_trace = trace_device_activity(trace_path)
    top_events = (
        profiler_events(
            profiler,
            args.top_events,
            device_time_available=cuda_trace["cuda_device_activity_available"],
        )
        if profiler is not None
        else []
    )
    phase_summary: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        rows = [row for row in profile_records if row["phase"] == phase]
        if not rows:
            continue
        phase_summary[phase] = {
            "count": len(rows),
            "wall_ms": summarize([float(row["wall_ms"]) for row in rows]),
            "stream_elapsed_ms": summarize([float(row["stream_elapsed_ms"]) for row in rows]),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        }

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision(),
        "git_worktree_dirty": git_worktree_dirty(),
        "profile_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": platform.platform(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "backend": args.backend,
        "gpu": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        },
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids": engine.tokenizer.encode(prompt),
        "config": {
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "max_model_length": args.max_model_length,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_cached_blocks": args.max_cached_blocks,
            "block_size": args.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "record_shapes": not args.no_record_shapes,
            "profile_memory": not args.no_profile_memory,
            "with_stack": args.with_stack,
            "torch_profiler": not args.no_torch_profiler,
        },
        "loading_ms": load_ms,
        "warmup_records": warmup_records,
        "profile_records": profile_records,
        "stage_summary": phase_summary,
        "projection_events": projection_events,
        "projection_summary": summarize_projection_types(projection_events),
        "projection_layer_summary": summarize_projection_events(projection_events),
        "cuda_trace": cuda_trace,
        "profile_status": (
            "external_cuda_trace_requested"
            if args.no_torch_profiler
            else "complete" if cuda_trace["cuda_device_activity_available"] else "cuda_activity_unavailable"
        ),
        "external_cuda_trace_path": str((output_dir / "fused.nsys-rep").resolve()) if args.no_torch_profiler else None,
        "memory": memory_snapshot(),
        "top_events": top_events,
        "trace_path": str(trace_path.resolve()),
        "json_path": str(json_path.resolve()),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"summary": str(json_path.resolve()), "report": str(report_path.resolve()), "trace": str(trace_path.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
