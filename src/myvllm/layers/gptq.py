"""Single-device GPTQ INT8 backends for the validated checkpoint layout."""

import torch
from torch import nn
from torch.nn import functional as F

from myvllm.utils.quant_config import GPTQConfig
from myvllm.utils import get_context


SUPPORTED_GPTQ_BACKENDS = frozenset({"reference", "fused", "fused_decode", "fused_prefill"})


def dequantize_gptq_int8(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return FP16 [output, input] weights for the validated GPTQ v1 layout.

    Each int32 packs four input rows, low byte first. For this symmetric
    checkpoint qzeros stores (zero_point - 1) == 127 and groups contain 128 rows.
    The loader checks both assumptions before accepting a checkpoint.
    """
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=qweight.device)
    values = ((qweight[:, None, :] >> shifts[None, :, None]) & 255)
    values = values.reshape(-1, qweight.shape[1]).to(scales.dtype)
    values.sub_(128)
    # Broadcast by group without materializing an input-sized scale matrix.
    values = values.view(scales.shape[0], 128, scales.shape[1])
    values.mul_(scales[:, None, :])
    return values.reshape(-1, scales.shape[1]).t().contiguous()


class GPTQLinear(nn.Module):
    """Packed GPTQ weights with reference and optional fused execution backends."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: GPTQConfig,
        backend: str = "reference",
    ):
        super().__init__()
        if input_size <= 0 or input_size % quant_config.group_size or output_size <= 0 or output_size % 4:
            raise ValueError("GPTQ requires input_size divisible by 128 and output_size divisible by 4")
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        if backend not in SUPPORTED_GPTQ_BACKENDS:
            raise ValueError(f"Unsupported GPTQ backend {backend!r}; expected one of {sorted(SUPPORTED_GPTQ_BACKENDS)}")
        self.backend = backend
        self.register_buffer("qweight", torch.empty(input_size // 4, output_size, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(input_size // 128, output_size, dtype=torch.float16))
        self._loaded = False

    def checkpoint_shapes(self) -> dict:
        return {
            "qweight": (tuple(self.qweight.shape), torch.int32),
            "qzeros": ((self.input_size // 128, self.output_size // 4), torch.int32),
            "scales": (tuple(self.scales.shape), torch.float16),
            "g_idx": ((self.input_size,), torch.int32),
        }

    @torch.no_grad()
    def load_quantized_weights(self, tensors: dict[str, torch.Tensor], name: str):
        self._loaded = False
        for suffix, (shape, dtype) in self.checkpoint_shapes().items():
            if suffix not in tensors:
                raise ValueError(f"Missing GPTQ tensor: {name}.{suffix}")
            tensor = tensors[suffix]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name}.{suffix}: expected {shape} {dtype}, got {tuple(tensor.shape)} {tensor.dtype}")
        if not torch.all(tensors["qzeros"] == 0x7F7F7F7F).item():
            raise ValueError(f"{name}.qzeros: expected GPTQ v1 symmetric zero_point-1 bytes (0x7f)")
        g_idx = tensors["g_idx"]
        expected = torch.arange(self.input_size, dtype=torch.int32, device=g_idx.device) // 128
        if not torch.equal(g_idx, expected):
            raise ValueError(f"{name}.g_idx: expected contiguous groups of 128 inputs")
        scales = tensors["scales"]
        if not torch.all(torch.isfinite(scales) & (scales > 0)).item():
            raise ValueError(f"{name}.scales: expected finite positive scales")
        self.qweight.copy_(tensors["qweight"])
        self.scales.copy_(scales)
        self._loaded = True

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dequantize_gptq_int8(self.qweight, self.scales))

    def _forward_fused(self, x: torch.Tensor, *, is_prefill: bool) -> torch.Tensor:
        from myvllm.layers.gptq_kernels import fused_gptq_decode, fused_gptq_prefill

        if is_prefill:
            return fused_gptq_prefill(x, self.qweight, self.scales)
        return fused_gptq_decode(x, self.qweight, self.scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("GPTQLinear weights have not been loaded and validated")
        if x.dtype != torch.float16 or self.scales.dtype != torch.float16:
            raise ValueError("GPTQLinear requires FP16 activations and scales")
        if self.backend == "reference":
            return self._forward_reference(x)
        if self.backend == "fused_decode":
            return self._forward_fused(x, is_prefill=False)
        if self.backend == "fused_prefill":
            return self._forward_fused(x, is_prefill=True)
        return self._forward_fused(x, is_prefill=get_context().is_prefill)
