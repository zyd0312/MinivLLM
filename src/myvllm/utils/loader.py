import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig
import re
import json
from pathlib import Path
from myvllm.layers.gptq import GPTQLinear


def _resolve_checkpoint_path(model_name_or_path: str) -> str:
    from huggingface_hub import snapshot_download

    checkpoint_path = os.path.expanduser(model_name_or_path)
    if not os.path.isdir(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
            )
        except Exception as e:
            raise ValueError(f"Could not find or download model '{model_name_or_path}': {e}") from e
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")
    return checkpoint_path


@torch.no_grad()
def _load_gptq_checkpoint(model: nn.Module, checkpoint_path: str):
    """Stream a checkpoint and strictly load its float and GPTQ tensors."""
    from myvllm.utils.quant_config import GPTQConfig

    with open(os.path.join(checkpoint_path, "config.json")) as f:
        hf_config = json.load(f)
    checkpoint_quant = GPTQConfig.from_dict(hf_config.get("quantization_config", {}))
    if checkpoint_quant != model.quant_config:
        raise ValueError("Model GPTQ configuration does not match checkpoint config.json")
    if hf_config.get("tie_word_embeddings") is not True or hf_config.get("quantization_config", {}).get("lm_head") is not False:
        raise ValueError("Expected tied embeddings and a non-quantized lm_head")

    files = sorted(Path(checkpoint_path).glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    key_to_file = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in key_to_file:
                    raise ValueError(f"Duplicate tensor in checkpoint shards: {key}")
                key_to_file[key] = file_path

    float_params = dict(model.named_parameters())
    quant_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, GPTQLinear)
    }
    if len(quant_modules) != model.model.layers.__len__() * 7:
        raise ValueError(f"Expected 7 GPTQ projections per Qwen3 layer, found {len(quant_modules)}")

    expected_keys = set(float_params)
    for name, module in quant_modules.items():
        expected_keys.update(f"{name}.{suffix}" for suffix in module.checkpoint_shapes())
    actual_keys = set(key_to_file)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}" + ("..." if len(unexpected) > 8 else ""))
        raise ValueError("Checkpoint tensor set does not match this Qwen3 GPTQ model: " + "; ".join(details))

    loaded = set()
    file_handles = {}
    try:
        def read_tensor(name: str) -> torch.Tensor:
            file_path = key_to_file[name]
            handle = file_handles.get(file_path)
            if handle is None:
                handle = safe_open(str(file_path), framework="pt", device="cpu")
                file_handles[file_path] = handle
                handle.__enter__()
            return handle.get_tensor(name)

        for name, param in float_params.items():
            weight = read_tensor(name)
            if tuple(param.shape) != tuple(weight.shape):
                raise ValueError(f"{name}: expected shape {tuple(param.shape)}, got {tuple(weight.shape)}")
            if weight.dtype != torch.float16:
                raise ValueError(f"{name}: expected torch.float16, got {weight.dtype}")
            param.copy_(weight.to(device=param.device, dtype=param.dtype))
            loaded.add(name)

        for name, module in quant_modules.items():
            group = {suffix: read_tensor(f"{name}.{suffix}")
                     for suffix in module.checkpoint_shapes()}
            module.load_quantized_weights(group, name)
            loaded.update(f"{name}.{suffix}" for suffix in group)
            del group
    finally:
        for handle in file_handles.values():
            handle.__exit__(None, None, None)

    if len(loaded) != len(expected_keys):
        raise RuntimeError(f"Loaded {len(loaded)} of {len(expected_keys)} expected checkpoint tensors")
    print(f"Loaded and validated {len(float_params)} FP16 tensors and {len(quant_modules)} GPTQ projections")
    return loaded


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    if getattr(model, "quant_config", None) is not None:
        checkpoint_path = _resolve_checkpoint_path(model_name_or_path)
        return _load_gptq_checkpoint(model, checkpoint_path)

    from huggingface_hub import snapshot_download

    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None

    # First, try local paths
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Now map and load weights into custom model
    loaded_params = set()
    skipped_params = []

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        try:
            # 1. Handle QKV merge (q_proj + k_proj + v_proj → qkv_projection)
            if '.self_attn.q_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    k_name = hf_name.replace('q_proj', 'k_proj')
                    v_name = hf_name.replace('q_proj', 'v_proj')

                    if k_name in hf_weights and v_name in hf_weights:
                        q_weight = hf_weight
                        k_weight = hf_weights[k_name]
                        v_weight = hf_weights[v_name]

                        # Concatenate q, k, v along output dimension
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(qkv_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(k_name)
                            loaded_params.add(v_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 2. Handle gate_up merge (gate_proj + up_proj → gate_up)
            elif '.mlp.gate_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_name in hf_weights:
                        gate_weight = hf_weight
                        up_weight = hf_weights[up_name]

                        # Concatenate gate and up along output dimension
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 3. Handle gate_up merge for bias (if present)
            elif '.mlp.gate_proj.bias' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_bias_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_bias_name in hf_weights:
                        gate_bias = hf_weight
                        up_bias = hf_weights[up_bias_name]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_bias)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_bias_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 4. Skip k_proj, v_proj, up_proj (already merged)
            elif any(x in hf_name for x in ['.k_proj.', '.v_proj.', '.up_proj.']):
                if hf_name not in loaded_params:
                    skipped_params.append((hf_name, "Merged into qkv_projection or gate_up"))

            # 5. All other parameters: load directly (names match HF)
            else:
                try:
                    param = model.get_parameter(hf_name)
                    if param.shape != hf_weight.shape:
                        # Handle vocab size mismatch for embeddings/lm_head
                        if len(param.shape) > 0 and len(hf_weight.shape) > 0:
                            min_size = min(param.shape[0], hf_weight.shape[0])
                            param.data[:min_size].copy_(hf_weight[:min_size])
                        else:
                            param.data.copy_(hf_weight)
                    else:
                        param.data.copy_(hf_weight)
                    loaded_params.add(hf_name)
                except AttributeError:
                    skipped_params.append((hf_name, "Parameter not found"))

        except Exception as e:
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # Check for model parameters that weren't loaded
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")

    if unloaded_params:
        print(f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f"  ... and {len(unloaded_params) - 15} more")

    if skipped_params:
        # Group skipped by reason
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]

        if merged_skips:
            print(f"Skipped (merged into other weights): {len(merged_skips)}")
        if not_found_skips:
            print(f"Skipped (not found in model): {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:
                print(f"  - {name}")
            if len(not_found_skips) > 5:
                print(f"  ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:
            print(f"Skipped (no mapping rule): {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:
                print(f"  - {name}")
            if len(no_mapping_skips) > 5:
                print(f"  ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")
    return loaded_params
