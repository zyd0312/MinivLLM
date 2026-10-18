import sys
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from myvllm.layers.gptq import GPTQLinear, dequantize_gptq_int8
from myvllm.utils import set_context
from myvllm.utils.quant_config import GPTQConfig
from myvllm.config import Config


@pytest.fixture
def checkpoint_config(tmp_path):
    metadata = {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "quantization_config": asdict(GPTQConfig()),
        "torch_dtype": "float16", "tie_word_embeddings": True,
        "attention_bias": False, "hidden_act": "silu",
        "vocab_size": 151936, "hidden_size": 2048,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 6144, "num_hidden_layers": 28,
        "rope_theta": 1000000, "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6, "eos_token_id": 151645,
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645, 151643]})
    )
    return {"model_name_or_path": str(tmp_path), "world_size": 1,
            "enforce_eager": True, "max_model_length": 128}


def test_config_loads_quantization_metadata_only_when_selected(checkpoint_config):
    config = Config(
        checkpoint_config,
        quantization="gptq",
    )
    assert config["hidden_size"] == 2048
    assert config["quantization_config"] == GPTQConfig()
    assert config["eos"] == [151645, 151643]
    assert "hidden_size" not in checkpoint_config
    assert "quantization_config" not in checkpoint_config

    dense = Config({"model_name_or_path": "Qwen/Qwen3-0.6B"})
    assert dense["quantization"] is None
    assert "quantization_config" not in dense


def test_no_selector_does_not_read_checkpoint(checkpoint_config, monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("Dense Config must not open checkpoint metadata")
    monkeypatch.setattr(Path, "open", fail_open)
    config = Config(checkpoint_config)
    assert config["quantization"] is None
    assert "quantization_config" not in config


def test_explicit_none_overrides_selector(checkpoint_config):
    config = Config({**checkpoint_config, "quantization": "gptq",
                     "quantization_config": GPTQConfig()}, quantization=None)
    assert config["quantization"] is None
    assert "quantization_config" not in config


def test_config_worker_pickle_does_not_reread(checkpoint_config, monkeypatch):
    config = Config({**checkpoint_config, "quantization": "gptq"})
    def fail_open(*args, **kwargs):
        raise AssertionError("Worker deserialization must not reread metadata")
    monkeypatch.setattr(Path, "open", fail_open)
    restored = pickle.loads(pickle.dumps(config))
    assert isinstance(restored, Config)
    assert restored == config


def test_quantization_requires_config_file(tmp_path):
    with pytest.raises(ValueError, match="config.json"):
        Config({"model_name_or_path": str(tmp_path)}, quantization="gptq")


@pytest.mark.parametrize("override, message", [
    ({"world_size": 2}, "world_size=1"),
    ({"enforce_eager": False}, "enforce_eager=True"),
])
def test_invalid_runtime_mode(checkpoint_config, override, message):
    with pytest.raises(ValueError, match=message):
        Config({**checkpoint_config, **override}, quantization="gptq")


def test_gptq_unpack_low_byte_first():
    qweight = torch.full((32, 2), -2122219135, dtype=torch.int32)  # 0x81818181
    scales = torch.ones((1, 2), dtype=torch.float16)
    weight = dequantize_gptq_int8(qweight, scales)
    assert weight.shape == (2, 128)
    assert torch.equal(weight, torch.ones_like(weight))


def test_gptq_linear_requires_validated_weights():
    layer = GPTQLinear(128, 4, GPTQConfig())
    with pytest.raises(RuntimeError, match="not been loaded"):
        layer(torch.ones(1, 128, dtype=torch.float16))


def test_gptq_linear_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported GPTQ backend"):
        GPTQLinear(128, 4, GPTQConfig(), backend="unknown")


@pytest.mark.parametrize("is_prefill", [False, True])
def test_fused_backend_uses_explicit_context(monkeypatch, is_prefill):
    layer = GPTQLinear(128, 4, GPTQConfig(), backend="fused")
    layer._loaded = True
    calls = []

    def fake_forward(x, *, is_prefill):
        calls.append(is_prefill)
        return x[..., :4]

    monkeypatch.setattr(layer, "_forward_fused", fake_forward)
    set_context(is_prefill=is_prefill)
    try:
        output = layer(torch.ones(1, 128, dtype=torch.float16))
    finally:
        set_context(is_prefill=False)
    assert output.shape == (1, 4)
    assert calls == [is_prefill]


def test_gptq_linear_loads_checkpoint_group():
    layer = GPTQLinear(128, 4, GPTQConfig())
    tensors = {
        "qweight": torch.zeros((32, 4), dtype=torch.int32),
        "qzeros": torch.full((1, 1), 0x7F7F7F7F, dtype=torch.int32),
        "scales": torch.ones((1, 4), dtype=torch.float16),
        "g_idx": torch.arange(128, dtype=torch.int32) // 128,
    }
    layer.load_quantized_weights(tensors, "test")
    output = layer(torch.ones(2, 128, dtype=torch.float16))
    assert output.shape == (2, 4)
    assert torch.equal(output, torch.full_like(output, -128 * 128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused GPTQ tests require CUDA")
@pytest.mark.parametrize("backend, rows", [("fused_decode", 1), ("fused_prefill", 8)])
def test_fused_gptq_matches_reference(backend, rows):
    config = GPTQConfig()
    packed_bytes = torch.randint(0, 256, (32, 8, 4), dtype=torch.int64)
    byte_shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
    qweight = (packed_bytes << byte_shifts).sum(dim=-1).to(torch.int32)
    tensors = {
        "qweight": qweight,
        "qzeros": torch.full((1, 2), 0x7F7F7F7F, dtype=torch.int32),
        "scales": torch.rand((1, 8), dtype=torch.float16).clamp_min(0.01),
        "g_idx": torch.arange(128, dtype=torch.int32) // 128,
    }
    reference = GPTQLinear(128, 8, config, backend="reference")
    fused = GPTQLinear(128, 8, config, backend=backend)
    reference.load_quantized_weights(tensors, "reference")
    fused.load_quantized_weights(tensors, "fused")
    reference = reference.cuda()
    fused = fused.cuda()
    x = torch.randn(rows, 128, dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        expected = reference(x)
        actual = fused(x)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
