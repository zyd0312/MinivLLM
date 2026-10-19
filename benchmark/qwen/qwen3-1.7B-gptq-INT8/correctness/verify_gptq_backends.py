#!/usr/bin/env python3
"""Compare GPTQ reference and fused backends on identical logits.

The two model instances run in separate processes so an 8 GB GPU does not
need to hold two Qwen3 models at once.  The reference worker chooses a fixed
greedy token prefix; the candidate worker receives that prefix and evaluates
the same prefill and decode states.  This avoids comparing two independent
random generations.

Example::

    python verify_gptq_backends.py \
        --model-path /path/to/Qwen3-1.7B-GPTQ-Int8 \
        --candidate-backend fused \
        --chat-template \
        --output results/correctness/fused/run-<timestamp>/backend-correctness.json \
        --report results/correctness/fused/run-<timestamp>/backend-correctness.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate the repository root from verify_gptq_backends.py")


REPO_ROOT = _find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BACKENDS = ("reference", "fused", "fused_decode", "fused_prefill")
DEFAULT_PROMPT = "Explain why the sky appears blue in two concise paragraphs."


def _finite_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    """Return numerical and probability-distribution metrics for two logits tensors."""
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "passed": False,
        }
    if reference.numel() == 0 or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {
            "status": "non_finite_or_empty",
            "shape": list(reference.shape),
            "passed": False,
        }

    delta = (candidate - reference).abs()
    tolerance = atol + rtol * reference.abs()
    within_tolerance = delta <= tolerance
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    delta_rms = delta.square().mean().sqrt()
    reference_abs_max = reference.abs().max().clamp_min(1e-12)

    reference_logp = F.log_softmax(reference, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    reference_prob = reference_logp.exp()
    candidate_prob = candidate_logp.exp()
    # KL(reference || candidate), one value per token position.
    kl = (reference_prob * (reference_logp - candidate_logp)).sum(dim=-1)
    total_variation = 0.5 * (reference_prob - candidate_prob).abs().sum(dim=-1)
    top1_match = reference.argmax(dim=-1).eq(candidate.argmax(dim=-1))

    return {
        "status": "ok",
        "shape": list(reference.shape),
        "max_absolute_error": _finite_float(delta.max()),
        "mean_absolute_error": _finite_float(delta.mean()),
        "rms_error": _finite_float(delta_rms),
        "relative_rms_error": _finite_float(delta_rms / reference_rms),
        "relative_max_error": _finite_float(delta.max() / reference_abs_max),
        "allclose_rate": _finite_float(within_tolerance.float().mean()),
        "strict_allclose": bool(within_tolerance.all()),
        "top1_match_rate": _finite_float(top1_match.float().mean()),
        "total_variation_mean": _finite_float(total_variation.mean()),
        "total_variation_p99": _finite_float(torch.quantile(total_variation, 0.99)),
        "kl_mean": _finite_float(kl.mean()),
        "kl_p99": _finite_float(torch.quantile(kl, 0.99)),
        "kl_max": _finite_float(kl.max()),
        "passed": True,
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    """Compare same-input projection outputs without treating them as probabilities."""
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "passed": False,
        }
    if reference.numel() == 0 or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {"status": "non_finite_or_empty", "shape": list(reference.shape), "passed": False}
    delta = (candidate - reference).abs()
    tolerance = atol + rtol * reference.abs()
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    delta_rms = delta.square().mean().sqrt()
    return {
        "status": "ok",
        "shape": list(reference.shape),
        "max_absolute_error": _finite_float(delta.max()),
        "mean_absolute_error": _finite_float(delta.mean()),
        "rms_error": _finite_float(delta_rms),
        "relative_rms_error": _finite_float(delta_rms / reference_rms),
        "allclose_rate": _finite_float((delta <= tolerance).float().mean()),
        "passed": True,
    }


def evaluate_result(metrics: dict[str, Any], *, min_allclose_rate: float, min_top1_rate: float, max_mean_kl: float) -> bool:
    """Apply explicit acceptance thresholds to one logits comparison."""
    return (
        metrics.get("status") == "ok"
        and float(metrics.get("allclose_rate", 0.0)) >= min_allclose_rate
        and float(metrics.get("top1_match_rate", 0.0)) >= min_top1_rate
        and float(metrics.get("kl_mean", math.inf)) <= max_mean_kl
    )


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompts = [
            line.rstrip("\n")
            for line in Path(args.prompt_file).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError("--prompt-file does not contain a non-empty prompt")
        return prompts[0]
    return args.prompt or DEFAULT_PROMPT


def _worker_config(args: argparse.Namespace) -> dict[str, Any]:
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


@torch.inference_mode()
def _run_worker(args: argparse.Namespace) -> None:
    """Run one backend and write CPU logits to a torch artifact."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPTQ backend correctness verification requires CUDA")

    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams
    from myvllm.layers.gptq import GPTQLinear
    from myvllm.utils import reset_context

    engine = LLMEngine(_worker_config(args))
    prompt = args.prompt or DEFAULT_PROMPT
    if args.chat_template:
        prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Match LLMEngine.add_prompt exactly; it calls tokenizer.encode(prompt)
    # with the tokenizer's configured default for special tokens.
    tokenized_prompt_ids = engine.tokenizer.encode(prompt)
    if not tokenized_prompt_ids:
        raise ValueError("The prompt encoded to zero tokens")
    if len(tokenized_prompt_ids) > args.max_num_batched_tokens:
        raise ValueError("prompt token count exceeds --max-num-batched-tokens")
    if len(tokenized_prompt_ids) + args.decode_steps >= args.max_model_length:
        raise ValueError("max_model_length must exceed prompt length plus decode steps")

    forced_tokens = []
    if args.forced_tokens:
        forced_tokens = [int(value) for value in args.forced_tokens.split(",") if value.strip()]
        if len(forced_tokens) != args.decode_steps:
            raise ValueError("--forced-tokens must contain exactly --decode-steps integers")

    sampling = SamplingParams(
        temperature=1.0,
        max_tokens=args.decode_steps + 1,
        ignore_eos=True,
        max_model_length=args.max_model_length,
    )
    engine.add_prompt(prompt, sampling)
    scheduled, is_prefill = engine.scheduler.schedule()
    if not scheduled or not is_prefill or len(scheduled) != 1:
        raise RuntimeError("correctness worker could not schedule a single prefill sequence")

    runner = engine.model_runner
    projection_checks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    projection_phase = {"name": "prefill"}
    hooks = []
    if args.backend == args.reference_backend:
        for name, module in runner.model.named_modules():
            if not isinstance(module, GPTQLinear):
                continue
            projection_checks[name] = {"prefill": [], "decode": []}
            comparing = {"active": False}

            def capture_projection(_module, inputs, output, *, _name=name, _comparing=comparing):
                if _comparing["active"]:
                    return
                phase = projection_phase["name"]
                sample_index = len(projection_checks[_name][phase])
                original_backend = _module.backend
                try:
                    # Compare immediately on the exact same activation. This
                    # bounds host memory even for long prefill sequences.
                    _comparing["active"] = True
                    _module.backend = args.candidate_backend
                    candidate_output = _module.forward(inputs[0])
                    metrics = compare_tensors(output, candidate_output, atol=args.atol, rtol=args.rtol)
                except Exception as exc:
                    metrics = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "passed": False}
                finally:
                    _module.backend = original_backend
                    _comparing["active"] = False
                metrics["sample"] = sample_index
                projection_checks[_name][phase].append(metrics)

            hooks.append(module.register_forward_hook(capture_projection))

    prompt_token_ids = list(scheduled[0].token_ids)
    if prompt_token_ids != tokenized_prompt_ids:
        raise RuntimeError("tokenizer prompt ids differ from the sequence scheduled by the engine")
    input_ids = runner.prepare_prefill(scheduled)
    prefill_logits = runner.run_model(input_ids, is_prefill=True).detach().float().cpu()
    reset_context()

    if not forced_tokens:
        next_token = int(prefill_logits[-1].argmax().item())
        forced_tokens = []
        for _ in range(args.decode_steps):
            forced_tokens.append(next_token)
            # The next token is selected after the decode logits below.
            next_token = None

    decode_logits = []
    for step in range(args.decode_steps):
        token = forced_tokens[step]
        engine.scheduler.postprocess(scheduled, [token])
        scheduled, is_prefill = engine.scheduler.schedule()
        if not scheduled or is_prefill or len(scheduled) != 1:
            raise RuntimeError("correctness worker could not schedule a single decode step")
        projection_phase["name"] = "decode"
        input_ids = runner.prepare_decode(scheduled)
        logits = runner.run_model(input_ids, is_prefill=False).detach().float().cpu()
        reset_context()
        decode_logits.append(logits[-1])
        if args.forced_tokens is None and step + 1 < args.decode_steps:
            # Greedy prefix selection makes both workers evaluate the same
            # autoregressive state without relying on random sampling.
            forced_tokens[step + 1] = int(decode_logits[-1].argmax().item())

    for hook in hooks:
        hook.remove()

    torch.save(
        {
            "backend": args.backend,
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "forced_tokens": forced_tokens,
            "prefill_logits": prefill_logits,
            "decode_logits": torch.stack(decode_logits) if decode_logits else torch.empty((0, prefill_logits.shape[-1])),
            "projection_checks": projection_checks,
        },
        args.artifact,
    )


def _run_subprocess(args: argparse.Namespace, *, backend: str, artifact: Path, forced_tokens: list[int] | None = None) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-path",
        str(args.model_path),
        "--backend",
        backend,
        "--reference-backend",
        args.reference_backend,
        "--candidate-backend",
        args.candidate_backend,
        "--artifact",
        str(artifact),
        "--prompt",
        args.prompt,
        "--decode-steps",
        str(args.decode_steps),
        "--max-model-length",
        str(args.max_model_length),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-cached-blocks",
        str(args.max_cached_blocks),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
    ]
    if args.chat_template:
        command.append("--chat-template")
    if forced_tokens is not None:
        command.extend(["--forced-tokens", ",".join(str(value) for value in forced_tokens)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} correctness worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def render_report(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "# GPTQ reference/fused 正确性报告",
        "",
        f"- 模型：`{report['model_path']}`",
        f"- Reference backend：`{report['reference_backend']}`",
        f"- Candidate backend：`{report['candidate_backend']}`",
        f"- Prompt tokens：{len(report['prompt_token_ids'])}",
        f"- Decode 对照步数：{config['decode_steps']}",
        f"- 强制 decode token IDs：`{report['forced_tokens']}`",
        "",
        "## 判定阈值",
        "",
        "| 指标 | 阈值 |",
        "|---|---:|",
        f"| 单元素 allclose 条件 | abs(error) <= {config['atol']:.6g} + {config['rtol']:.6g} * abs(reference) |",
        f"| logits allclose rate | >= {config['min_allclose_rate']:.6f} |",
        f"| 单投影 allclose rate | >= {config['min_projection_allclose_rate']:.6f} |",
        f"| Top-1 一致率 | >= {config['min_top1_rate']:.6f} |",
        f"| 平均 KL(reference \\|\\| candidate) | <= {config['max_mean_kl']:.6g} |",
        "",
        "## 结果",
        "",
        "| 阶段 | 状态 | Max abs | RMS 相对误差 | allclose | Top-1 | 平均 TV | 平均 KL | 通过 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for phase in ("prefill", "decode"):
        metrics = report[phase]
        if metrics.get("status") != "ok":
            lines.append(f"| {phase} | `{metrics.get('status')}` | - | - | - | - | - | - | 否 |")
            continue
        lines.append(
            f"| {phase} | ok | {metrics['max_absolute_error']:.6g} | "
            f"{metrics['relative_rms_error']:.6g} | {metrics['allclose_rate']:.6%} | "
            f"{metrics['top1_match_rate']:.6%} | {metrics['total_variation_mean']:.6g} | "
            f"{metrics['kl_mean']:.6g} | "
            f"{'是' if metrics['passed'] else '否'} |"
        )
    lines += [
        "",
        "## 单投影同输入对照",
        "",
        "reference 前向期间，候选 kernel 立即处理每个 GPTQ 投影的同一 activation。",
        "",
        "| 阶段 | 样本数 | 失败样本 | 最大绝对误差 | 最差 allclose | 最差相对 RMS | 通过 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for phase in ("prefill", "decode"):
        samples = [sample for phase_map in report.get("projection_checks", {}).values() for sample in phase_map.get(phase, [])]
        if not samples:
            lines.append(f"| {phase} | 0 | 0 | - | - | - | 否 |")
            continue
        valid_samples = [sample for sample in samples if sample.get("status") == "ok"]
        if not valid_samples:
            lines.append(f"| {phase} | {len(samples)} | {len(samples)} | - | - | - | 否 |")
            continue
        lines.append(
            f"| {phase} | {len(samples)} | {sum(not sample.get('passed', False) for sample in samples)} | "
            f"{max(sample['max_absolute_error'] for sample in valid_samples):.6g} | "
            f"{min(sample['allclose_rate'] for sample in valid_samples):.6%} | "
            f"{max(sample['relative_rms_error'] for sample in valid_samples):.6g} | "
            f"{'是' if all(sample['passed'] for sample in samples) else '否'} |"
        )
    lines += [
        "",
        "### 最大误差投影样本",
        "",
        "| 阶段 | 投影 | 样本 | Max abs | 相对 RMS | allclose | 通过 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    projection_samples = [
        (phase, layer_name, sample)
        for layer_name, phase_map in report.get("projection_checks", {}).items()
        for phase, samples in phase_map.items()
        for sample in samples if sample.get("status") == "ok"
    ]
    for phase, layer_name, sample in sorted(projection_samples, key=lambda row: row[2]["max_absolute_error"], reverse=True)[:10]:
        lines.append(
            f"| {phase} | `{layer_name}` | {sample['sample']} | {sample['max_absolute_error']:.6g} | "
            f"{sample['relative_rms_error']:.6g} | {sample['allclose_rate']:.6%} | "
            f"{'是' if sample['passed'] else '否'} |"
        )
    lines += ["", f"## 总体结论", "", f"**{'通过' if report['passed'] else '未通过'}**", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", required=True, help="Qwen3 GPTQ-INT8 checkpoint directory")
    parser.add_argument("--reference-backend", choices=BACKENDS, default="reference")
    parser.add_argument("--candidate-backend", choices=BACKENDS, default="fused")
    parser.add_argument("--prompt", default=None, help="One prompt to compare")
    parser.add_argument("--prompt-file", default=None, help="Read the first non-empty prompt from a UTF-8 file")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=4, help="Number of fixed-prefix decode logits to compare")
    parser.add_argument("--max-model-length", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-cached-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--atol", type=float, default=0.1, help="Absolute logit tolerance used by allclose")
    parser.add_argument("--rtol", type=float, default=0.02, help="Relative logit tolerance used by allclose")
    parser.add_argument("--min-allclose-rate", type=float, default=0.999)
    parser.add_argument("--min-top1-rate", type=float, default=0.99)
    parser.add_argument("--max-mean-kl", type=float, default=1e-3)
    parser.add_argument(
        "--min-projection-allclose-rate",
        type=float,
        default=None,
        help="Minimum allclose rate for same-input projection checks (defaults to --min-allclose-rate)",
    )
    parser.add_argument("--output", default="backend-correctness.json")
    parser.add_argument("--report", default=None)
    parser.add_argument("--keep-logits", action="store_true", help="Keep the intermediate torch logits artifact")
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--artifact", help=argparse.SUPPRESS)
    parser.add_argument("--forced-tokens", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        if not args.backend or not args.artifact:
            parser.error("worker mode requires --backend and --artifact")
        _run_worker(args)
        return 0
    if args.reference_backend == args.candidate_backend:
        raise ValueError("reference and candidate backend must be different")
    if args.decode_steps <= 0 or args.max_model_length <= 0:
        raise ValueError("decode-steps and max-model-length must be positive")
    if not 0.0 <= args.min_allclose_rate <= 1.0 or not 0.0 <= args.min_top1_rate <= 1.0:
        raise ValueError("minimum rates must be between zero and one")
    if args.min_projection_allclose_rate is None:
        args.min_projection_allclose_rate = args.min_allclose_rate
    if not 0.0 <= args.min_projection_allclose_rate <= 1.0:
        raise ValueError("min-projection-allclose-rate must be between zero and one")
    numeric_thresholds = (
        args.atol,
        args.rtol,
        args.max_mean_kl,
        args.min_allclose_rate,
        args.min_top1_rate,
        args.min_projection_allclose_rate,
        args.gpu_memory_utilization,
    )
    if not all(math.isfinite(value) for value in numeric_thresholds):
        raise ValueError("threshold and memory-utilization values must be finite")
    if args.atol < 0 or args.rtol < 0 or args.max_mean_kl < 0:
        raise ValueError("atol, rtol and max-mean-kl must be non-negative")
    if args.max_num_batched_tokens <= 0 or args.max_cached_blocks <= 0 or args.block_size <= 0:
        raise ValueError("token, cache-block and block-size limits must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu-memory-utilization must be greater than zero and at most one")
    if not Path(args.model_path).expanduser().is_dir():
        raise FileNotFoundError(args.model_path)
    args.prompt = _read_prompt(args)
    output_path = Path(args.output).expanduser()
    report_path = Path(args.report).expanduser() if args.report else output_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gptq-correctness-") as temp_dir:
        temp_dir = Path(temp_dir)
        reference_artifact = temp_dir / "reference.pt"
        candidate_artifact = temp_dir / "candidate.pt"
        _run_subprocess(args, backend=args.reference_backend, artifact=reference_artifact)
        reference = torch.load(reference_artifact, map_location="cpu", weights_only=True)
        forced_tokens = list(reference["forced_tokens"])
        _run_subprocess(args, backend=args.candidate_backend, artifact=candidate_artifact, forced_tokens=forced_tokens)
        candidate = torch.load(candidate_artifact, map_location="cpu", weights_only=True)

        if reference["prompt_token_ids"] != candidate["prompt_token_ids"]:
            raise RuntimeError("reference and candidate tokenizers produced different prompt token IDs")
        if list(candidate["forced_tokens"]) != forced_tokens:
            raise RuntimeError("candidate worker did not use the reference forced token prefix")

        prefill = compare_logits(reference["prefill_logits"], candidate["prefill_logits"], atol=args.atol, rtol=args.rtol)
        decode = compare_logits(reference["decode_logits"], candidate["decode_logits"], atol=args.atol, rtol=args.rtol)
        prefill["passed"] = evaluate_result(prefill, min_allclose_rate=args.min_allclose_rate, min_top1_rate=args.min_top1_rate, max_mean_kl=args.max_mean_kl)
        decode["passed"] = evaluate_result(decode, min_allclose_rate=args.min_allclose_rate, min_top1_rate=args.min_top1_rate, max_mean_kl=args.max_mean_kl)

        projection_checks = reference.get("projection_checks", {})
        projection_samples = []
        for layer_name, phase_map in projection_checks.items():
            for phase, samples in phase_map.items():
                for sample in samples:
                    sample["layer"] = layer_name
                    sample["phase"] = phase
                    sample["passed"] = (
                        sample.get("status") == "ok"
                        and sample.get("allclose_rate", 0.0) >= args.min_projection_allclose_rate
                    )
                    projection_samples.append(sample)
        projection_passed = bool(projection_samples) and all(sample["passed"] for sample in projection_samples)

        report = {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "reference_backend": args.reference_backend,
            "candidate_backend": args.candidate_backend,
            "prompt": reference["prompt"],
            "prompt_token_ids": list(reference["prompt_token_ids"]),
            "forced_tokens": forced_tokens,
            "config": {
                "decode_steps": args.decode_steps,
                "atol": args.atol,
                "rtol": args.rtol,
                "min_allclose_rate": args.min_allclose_rate,
                "min_projection_allclose_rate": args.min_projection_allclose_rate,
                "min_top1_rate": args.min_top1_rate,
                "max_mean_kl": args.max_mean_kl,
            },
            "prefill": prefill,
            "decode": decode,
            "projection_checks": projection_checks,
            "projection_passed": projection_passed,
            "passed": bool(prefill["passed"] and decode["passed"] and projection_passed),
        }
        if args.keep_logits:
            logits_path = output_path.with_suffix(".logits.pt")
            torch.save({"reference": reference, "candidate": candidate}, logits_path)
            report["logits_artifact"] = str(logits_path.resolve())

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"output": str(output_path.resolve()), "report": str(report_path.resolve()), "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
