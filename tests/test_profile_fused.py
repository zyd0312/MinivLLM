import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "benchmark/qwen/qwen3-1.7B-gptq-INT8/system-profile/profile_fused.py"
SPEC = importlib.util.spec_from_file_location("profile_fused", MODULE_PATH)
PROFILE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROFILE)


def test_trace_device_activity_reports_kernel_events(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}, {"cat": "kernel"}]}))

    result = PROFILE.trace_device_activity(trace)

    assert result["cuda_device_activity_available"] is True
    assert result["cuda_device_event_count"] == 1


def test_trace_device_activity_marks_cupti_missing(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}, {"cat": "user_annotation"}]}))

    result = PROFILE.trace_device_activity(trace)

    assert result["cuda_device_activity_available"] is False
    assert result["cuda_device_event_count"] == 0


def test_projection_summary_groups_by_phase_name_and_shape():
    rows = PROFILE.summarize_projection_events([
        {"phase": "decode", "projection": "layers.0.q_proj", "input_shape": [1, 2048], "stream_elapsed_ms": 0.2},
        {"phase": "decode", "projection": "layers.0.q_proj", "input_shape": [1, 2048], "stream_elapsed_ms": 0.4},
        {"phase": "prefill", "projection": "layers.0.q_proj", "input_shape": [20, 2048], "stream_elapsed_ms": 1.0},
    ])

    assert len(rows) == 2
    decode = next(row for row in rows if row["phase"] == "decode")
    assert decode["calls"] == 2
    assert abs(decode["stream_elapsed_ms"]["median_ms"] - 0.3) < 1e-9
    assert abs(decode["stream_elapsed_ms"]["p90_ms"] - 0.38) < 1e-9


def test_report_marks_missing_cuda_activity_as_unavailable():
    report = {
        "model_path": "/tmp/model",
        "backend": "fused",
        "prompt_tokens": 20,
        "config": {"max_tokens": 2},
        "stage_summary": {
            "prefill": {
                "count": 1,
                "wall_ms": {"median_ms": 1.0, "p90_ms": 1.0},
                "stream_elapsed_ms": {"median_ms": 0.9, "p90_ms": 0.9},
                "input_tokens": 20,
            }
        },
        "projection_summary": [],
        "top_events": [{
            "name": "aten::copy_", "calls": 1, "self_cpu_ms": 0.1, "cpu_total_ms": 0.1,
            "self_device_ms": None, "device_total_ms": None,
        }],
        "trace_path": "/tmp/trace.json",
        "json_path": "/tmp/summary.json",
        "cuda_trace": {"cuda_device_activity_available": False, "cuda_device_event_count": 0},
    }

    markdown = PROFILE.render_report(report)

    assert "trace 未采集到 CUDA kernel activity" in markdown
    assert "unavailable" in markdown
