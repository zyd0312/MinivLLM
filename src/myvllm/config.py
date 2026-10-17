"""Runtime configuration shared by the engine and model workers."""

from myvllm.utils.quant_config import _load_quantization_config

_UNSET = object()


class Config(dict):
    """Dictionary-compatible engine configuration with optional quantization."""

    def __init__(self, values=None, *, quantization=_UNSET):
        super().__init__(values or {})
        if quantization is _UNSET:
            quantization = self.get("quantization")
        self["quantization"] = quantization
        # Derived metadata must never enable quantization without the selector.
        self.pop("quantization_config", None)
        if quantization is not None:
            self.update(_load_quantization_config(self, quantization))
