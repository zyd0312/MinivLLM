"""Configuration for the supported local Qwen3 GPTQ checkpoint."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 8
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    pack_dtype: str = "int32"
    checkpoint_format: str = "gptq"
    quant_method: str = "gptq"
    lm_head: bool = False

    def __post_init__(self):
        expected = dict(bits=8, group_size=128, desc_act=False, sym=True,
                        pack_dtype="int32", checkpoint_format="gptq",
                        quant_method="gptq", lm_head=False)
        for name, value in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"Unsupported GPTQ {name}={actual!r}; expected {value!r}")

    @classmethod
    def from_dict(cls, metadata: dict):
        # Do not silently apply a per-module override that this implementation ignores.
        if metadata.get("dynamic"):
            raise ValueError("GPTQ dynamic per-module overrides are not supported")
        missing = set(cls.__dataclass_fields__) - metadata.keys()
        if missing:
            raise ValueError(f"Missing GPTQ configuration fields: {sorted(missing)}")
        return cls(**{key: metadata[key] for key in cls.__dataclass_fields__})


def _load_quantization_config(config: dict, quantization) -> dict:
    """Load and validate backend metadata for an explicitly selected quantizer.

    This function owns GPTQ-specific checkpoint knowledge. It returns only
    derived engine/model fields; the generic ``Config`` class simply merges
    those fields into its dictionary.
    """
    requested_config = None
    if isinstance(quantization, GPTQConfig):
        requested_config = quantization
        quantization = "gptq"
    elif isinstance(quantization, dict):
        requested_config = GPTQConfig.from_dict(quantization)
        quantization = "gptq"
    if str(quantization).lower() != "gptq":
        raise ValueError(f"Unsupported quantization backend: {quantization!r}")

    model_path = Path(config.get("model_name_or_path", "")).expanduser()
    metadata_path = model_path / "config.json"
    if not metadata_path.is_file():
        raise ValueError(
            "Quantized models require config.json under model_name_or_path: "
            f"{metadata_path}"
        )
    with metadata_path.open() as file:
        hf = json.load(file)
    metadata = hf.get("quantization_config")
    if metadata is None:
        raise ValueError(f"Missing quantization_config in {metadata_path}")
    quant_config = GPTQConfig.from_dict(metadata)
    if requested_config is not None and requested_config != quant_config:
        raise ValueError("Requested quantization does not match checkpoint config.json")

    if hf.get("model_type") != "qwen3" or hf.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("GPTQ currently supports Qwen3ForCausalLM only")
    if config.get("world_size", 1) != 1:
        raise ValueError("GPTQ currently requires world_size=1")
    if not config.get("enforce_eager", False):
        raise ValueError("GPTQ currently requires enforce_eager=True (CUDA Graph is not supported)")
    for name, expected in {
        "torch_dtype": "float16", "tie_word_embeddings": True,
        "attention_bias": False, "hidden_act": "silu",
    }.items():
        if hf.get(name) != expected:
            raise ValueError(f"Unsupported Qwen3 {name}={hf.get(name)!r}; expected {expected!r}")
    if hf.get("rope_scaling") is not None or hf.get("use_sliding_window", False):
        raise ValueError("GPTQ Qwen3 currently requires unscaled RoPE and full attention")

    max_model_length = config.get("max_model_length", min(32768, hf["max_position_embeddings"]))
    if not 0 < max_model_length <= min(32768, hf["max_position_embeddings"]):
        raise ValueError("GPTQ max_model_length must be positive and at most 32768 / max_position_embeddings")

    mapping = {
        "vocab_size": "vocab_size", "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads", "num_kv_heads": "num_key_value_heads",
        "head_dim": "head_dim", "intermediate_size": "intermediate_size",
        "num_layers": "num_hidden_layers", "base": "rope_theta",
        "max_position": "max_position_embeddings", "rms_norm_epsilon": "rms_norm_eps",
        "qkv_bias": "attention_bias", "tie_word_embeddings": "tie_word_embeddings",
    }
    derived = {target: hf[source] for target, source in mapping.items()}
    derived.update(
        model_name_or_path=str(model_path.resolve()), model_type="qwen3",
        quantization_config=quant_config, dtype="float16", ffn_bias=False,
        scale=1.0, max_model_length=max_model_length,
    )

    eos = hf["eos_token_id"]
    generation_path = model_path / "generation_config.json"
    if generation_path.is_file():
        with generation_path.open() as file:
            generation_eos = json.load(file).get("eos_token_id")
            if generation_eos is not None:
                eos = generation_eos
    derived["eos"] = eos
    return derived


