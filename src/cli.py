"""Command line interface for exporting and verifying the ONNX model.

Examples:
    Export a model and check parity::

        python -m src.cli export --output model.onnx

    Run inference on random input with the exported graph::

        python -m src.cli run --model model.onnx
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from .model import SmallCNN
from .export import export_to_onnx, onnx_forward, max_abs_diff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-deployment",
        description="Export a small CNN to ONNX and verify runtime parity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--in-channels", type=int, default=1)
    common.add_argument("--num-classes", type=int, default=10)
    common.add_argument("--image-size", type=int, default=28)
    common.add_argument("--seed", type=int, default=0)

    p_export = sub.add_parser(
        "export", parents=[common], help="Export to ONNX and check parity."
    )
    p_export.add_argument("--output", default="model.onnx")
    p_export.add_argument("--opset", type=int, default=17)
    p_export.add_argument("--tolerance", type=float, default=1e-4)

    p_run = sub.add_parser(
        "run", parents=[common], help="Run inference with an ONNX graph."
    )
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--batch", type=int, default=1)

    return parser


def _make_model(args: argparse.Namespace) -> SmallCNN:
    torch.manual_seed(args.seed)
    return SmallCNN(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.command == "export":
        model = _make_model(args)
        path = export_to_onnx(
            model,
            args.output,
            image_size=args.image_size,
            in_channels=args.in_channels,
            opset=args.opset,
        )
        x = rng.standard_normal(
            (4, args.in_channels, args.image_size, args.image_size),
            dtype=np.float32,
        )
        diff = max_abs_diff(model, path, x)
        print(f"exported: {path}")
        print(f"max_abs_diff: {diff:.3e}")
        if diff <= args.tolerance:
            print(f"parity OK (tolerance {args.tolerance:.1e})")
            return 0
        print(f"parity FAILED (tolerance {args.tolerance:.1e})", file=sys.stderr)
        return 1

    if args.command == "run":
        x = rng.standard_normal(
            (args.batch, args.in_channels, args.image_size, args.image_size),
            dtype=np.float32,
        )
        logits = onnx_forward(args.model, x)
        preds = logits.argmax(axis=1)
        print(f"logits shape: {logits.shape}")
        print(f"predictions: {preds.tolist()}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
