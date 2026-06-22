"""ONNX deployment package: export a small CNN and verify runtime parity."""

from .model import SmallCNN
from .export import (
    export_to_onnx,
    torch_forward,
    onnx_forward,
    max_abs_diff,
)

__all__ = [
    "SmallCNN",
    "export_to_onnx",
    "torch_forward",
    "onnx_forward",
    "max_abs_diff",
]
