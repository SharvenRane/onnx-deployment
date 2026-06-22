"""Behavior tests for the ONNX export and torch parity check."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.model import SmallCNN
from src.export import export_to_onnx, torch_forward, onnx_forward, max_abs_diff
from src.cli import main as cli_main

TOL = 1e-4


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SmallCNN(in_channels=1, num_classes=10, image_size=28)


@pytest.fixture
def onnx_path(model, tmp_path):
    path = tmp_path / "model.onnx"
    return export_to_onnx(model, path, image_size=28, in_channels=1)


def test_model_output_shape(model):
    x = torch.randn(3, 1, 28, 28)
    out = model(x)
    assert out.shape == (3, 10)


def test_image_size_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        SmallCNN(image_size=30)


def test_export_creates_valid_file(onnx_path):
    import os

    assert os.path.exists(onnx_path)
    # onnxruntime can load and report the declared single input.
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert len(sess.get_inputs()) == 1
    assert len(sess.get_outputs()) == 1


def test_onnx_matches_torch_within_tolerance(model, onnx_path):
    rng = np.random.default_rng(123)
    x = rng.standard_normal((5, 1, 28, 28)).astype(np.float32)
    diff = max_abs_diff(model, onnx_path, x)
    assert diff < TOL


def test_parity_holds_across_multiple_random_inputs(model, onnx_path):
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.standard_normal((2, 1, 28, 28)).astype(np.float32)
        diff = max_abs_diff(model, onnx_path, x)
        assert diff < TOL


def test_dynamic_batch_axis(model, onnx_path):
    rng = np.random.default_rng(0)
    for batch in (1, 3, 8):
        x = rng.standard_normal((batch, 1, 28, 28)).astype(np.float32)
        out = onnx_forward(onnx_path, x)
        assert out.shape == (batch, 10)


def test_argmax_predictions_agree(model, onnx_path):
    rng = np.random.default_rng(99)
    x = rng.standard_normal((16, 1, 28, 28)).astype(np.float32)
    t_pred = torch_forward(model, x).argmax(axis=1)
    o_pred = onnx_forward(onnx_path, x).argmax(axis=1)
    assert np.array_equal(t_pred, o_pred)


def test_export_is_deterministic_for_fixed_seed(tmp_path):
    torch.manual_seed(0)
    m1 = SmallCNN(image_size=28)
    p1 = export_to_onnx(m1, tmp_path / "a.onnx", image_size=28, in_channels=1)

    torch.manual_seed(0)
    m2 = SmallCNN(image_size=28)
    p2 = export_to_onnx(m2, tmp_path / "b.onnx", image_size=28, in_channels=1)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)
    out1 = onnx_forward(p1, x)
    out2 = onnx_forward(p2, x)
    assert np.allclose(out1, out2, atol=1e-6)


def test_multichannel_and_custom_size(tmp_path):
    torch.manual_seed(1)
    model = SmallCNN(in_channels=3, num_classes=4, image_size=32)
    path = export_to_onnx(
        model, tmp_path / "rgb.onnx", image_size=32, in_channels=3
    )
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
    assert max_abs_diff(model, path, x) < TOL


def test_cli_export_reports_parity_ok(tmp_path, capsys):
    out = tmp_path / "cli.onnx"
    code = cli_main(["export", "--output", str(out), "--seed", "0"])
    captured = capsys.readouterr()
    assert code == 0
    assert "parity OK" in captured.out
    assert out.exists()


def test_cli_run_outputs_predictions(tmp_path, capsys):
    out = tmp_path / "cli.onnx"
    cli_main(["export", "--output", str(out)])
    capsys.readouterr()
    code = cli_main(["run", "--model", str(out), "--batch", "5"])
    captured = capsys.readouterr()
    assert code == 0
    assert "predictions" in captured.out
    assert "(5, 10)" in captured.out
