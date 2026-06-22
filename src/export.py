"""Export a :class:`SmallCNN` to ONNX and verify parity with onnxruntime.

The export uses a dynamic batch axis so the same graph serves any batch size.
Parity is checked by comparing the torch forward pass against an onnxruntime
inference session on identical random input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import SmallCNN


def export_to_onnx(
    model: torch.nn.Module,
    path: str | Path,
    image_size: int,
    in_channels: int,
    opset: int = 17,
) -> str:
    """Export ``model`` to an ONNX file at ``path``.

    The model is switched to eval mode before tracing so that any
    stochastic layers behave deterministically. A dynamic batch axis is
    declared on both the input and the output.

    Returns:
        The string path the model was written to.
    """
    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, in_channels, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(path)))
    return str(path)


def torch_forward(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """Run a torch forward pass on a numpy input and return numpy logits."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    return out.numpy()


def onnx_forward(path: str | Path, x: np.ndarray) -> np.ndarray:
    """Run inference on the exported ONNX graph and return numpy logits."""
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x.astype(np.float32)})
    return outputs[0]


def max_abs_diff(
    model: torch.nn.Module,
    path: str | Path,
    x: np.ndarray,
) -> float:
    """Return the maximum absolute difference between torch and ONNX logits."""
    torch_out = torch_forward(model, x)
    onnx_out = onnx_forward(path, x)
    return float(np.max(np.abs(torch_out - onnx_out)))
