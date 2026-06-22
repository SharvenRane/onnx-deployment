# onnx-deployment

This project takes a small PyTorch convolutional classifier, exports it to the
ONNX format, and runs the exported graph with onnxruntime. The point is to show
that the deployed graph produces the same numbers as the original PyTorch model,
which is the property you actually care about when you move a model out of a
training framework and into a runtime.

## Why this matters

Training happens in PyTorch, but serving often happens somewhere else. ONNX is a
portable graph format that many runtimes can load, and onnxruntime is a fast CPU
and GPU engine for those graphs. The risk in any export step is silent numeric
drift: the graph loads, it runs, it returns plausible looking logits, and yet
the values have quietly diverged from the trained model. The tests here close
that gap by comparing the two side by side on random input and asserting the
maximum absolute difference stays below 1e-4.

## Layout

```
src/
  model.py    the SmallCNN architecture
  export.py   ONNX export plus torch and onnxruntime forward passes
  cli.py      command line entry point
tests/
  test_parity.py   behavior and parity tests
```

## The model

`SmallCNN` is a compact two block network. Each block is a 3x3 convolution
followed by a ReLU and a 2x2 max pool, so the spatial size is quartered before a
single linear head maps the flattened features to class logits. It takes single
channel square images by default and works on CPU in well under a second, which
keeps the tests fast and free of any dataset download. The channel count, class
count, and image size are all configurable.

## Install

```
pip install -r requirements.txt
```

## Export and verify

```
python -m src.cli export --output model.onnx
```

This builds the model, writes `model.onnx` with a dynamic batch axis, validates
the graph with the ONNX checker, then runs both the PyTorch model and the
onnxruntime session on the same random batch. It prints the maximum absolute
difference and reports whether parity holds within the tolerance. The command
exits non zero if the difference exceeds the tolerance, so it works as a gate in
a deployment pipeline.

## Run inference with the exported graph

```
python -m src.cli run --model model.onnx --batch 5
```

This loads the ONNX file in onnxruntime, feeds a random batch through it, and
prints the output shape and the predicted class per row. Because the export
declares a dynamic batch axis, any batch size loads against the same graph.

## Tests

```
python -m pytest tests/ -q
```

The suite checks output shape, that an invalid image size is rejected, that the
exported file loads in onnxruntime with one input and one output, and that the
exported graph accepts several different batch sizes through its dynamic axis.
The core checks compare PyTorch against onnxruntime: the maximum absolute
difference stays under 1e-4 across repeated random inputs and across a multi
channel configuration, the argmax predictions agree exactly, and the two
exporter calls under a fixed seed produce matching outputs. Two further tests
drive the command line interface end to end and confirm it reports parity and
prints predictions.

## Notes

The export pins the legacy TorchScript based exporter so the only dependencies
needed are onnx and onnxruntime. The newer torch.export based path additionally
requires onnxscript, which this project does not pull in.
