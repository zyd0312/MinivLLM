import sys
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from myvllm.layers.gptq import GPTQLinear, dequantize_gptq_int8
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
