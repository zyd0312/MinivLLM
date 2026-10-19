import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "benchmark/qwen/qwen3-1.7B-gptq-INT8/correctness/verify_gptq_backends.py"
SPEC = importlib.util.spec_from_file_location("verify_gptq_backends", MODULE_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def test_identical_logits_pass_probability_metrics():
    reference = torch.tensor([[3.0, 1.0, -2.0], [0.5, 2.0, 1.0]])
    metrics = VERIFY.compare_logits(reference, reference.clone(), atol=0.0, rtol=0.0)
    assert metrics["status"] == "ok"
    assert metrics["max_absolute_error"] == 0.0
    assert metrics["allclose_rate"] == 1.0
    assert metrics["top1_match_rate"] == 1.0
    assert metrics["total_variation_mean"] == 0.0
    assert metrics["kl_mean"] == 0.0
    assert VERIFY.evaluate_result(metrics, min_allclose_rate=1.0, min_top1_rate=1.0, max_mean_kl=1e-8)


def test_shape_mismatch_fails_without_probability_calculation():
    metrics = VERIFY.compare_logits(torch.zeros(2, 3), torch.zeros(1, 3), atol=0.1, rtol=0.02)
    assert metrics["status"] == "shape_mismatch"
    assert not metrics["passed"]


def test_probability_metrics_detect_changed_top1():
    reference = torch.tensor([[4.0, 3.0, 0.0]])
    candidate = torch.tensor([[1.0, 5.0, 0.0]])
    metrics = VERIFY.compare_logits(reference, candidate, atol=0.01, rtol=0.01)
    assert metrics["top1_match_rate"] == 0.0
    assert metrics["total_variation_mean"] > 0.0
    assert metrics["kl_mean"] > 0.0
    assert not VERIFY.evaluate_result(metrics, min_allclose_rate=1.0, min_top1_rate=1.0, max_mean_kl=1e-8)


def test_same_input_projection_comparison_uses_numeric_metrics():
    reference = torch.tensor([[1.0, -2.0], [0.25, 4.0]])
    metrics = VERIFY.compare_tensors(reference, reference.clone(), atol=0.0, rtol=0.0)
    assert metrics["status"] == "ok"
    assert metrics["allclose_rate"] == 1.0
    assert metrics["relative_rms_error"] == 0.0
    assert "kl_mean" not in metrics


def test_projection_shape_mismatch_is_reported():
    metrics = VERIFY.compare_tensors(torch.zeros(2, 4), torch.zeros(2, 3), atol=0.1, rtol=0.02)
    assert metrics["status"] == "shape_mismatch"
    assert not metrics["passed"]


def test_report_handles_projection_samples_that_failed():
    report = {
        "model_path": "/tmp/model",
        "reference_backend": "reference",
        "candidate_backend": "fused",
        "prompt_token_ids": [1, 2],
        "forced_tokens": [3],
        "config": {
            "decode_steps": 1,
            "atol": 0.1,
            "rtol": 0.02,
            "min_allclose_rate": 0.999,
            "min_projection_allclose_rate": 0.999,
            "min_top1_rate": 0.99,
            "max_mean_kl": 1e-3,
        },
        "prefill": {"status": "error", "passed": False},
        "decode": {"status": "error", "passed": False},
        "projection_checks": {
            "layer.q_proj": {
                "prefill": [
                    {"status": "error", "passed": False},
                    {
                        "status": "ok",
                        "max_absolute_error": 0.02,
                        "allclose_rate": 0.999,
                        "relative_rms_error": 0.001,
                        "sample": 1,
                        "passed": True,
                    },
                ],
                "decode": [],
            }
        },
        "passed": False,
    }
    markdown = VERIFY.render_report(report)
    assert "| prefill | 2 | 1 | 0.02 |" in markdown
    assert "**未通过**" in markdown
