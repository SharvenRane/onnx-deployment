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
serve/
  app.py        FastAPI inference service with readiness, liveness and stats endpoints
  loadtest.py   in cluster correctness check and closed loop load generator
  Dockerfile    non root serving image with the model baked in
k8s/
  deployment.yaml      Deployment, Service and PodDisruptionBudget
  hpa.yaml             HorizontalPodAutoscaler on CPU
  kind-config.yaml     local cluster with the test data mounted
  run_experiments.sh   every measurement in k8s/results
  results/             raw measurements
src/
  model.py    the SmallCNN architecture
  export.py   ONNX export plus torch and onnxruntime forward passes
  cli.py      command line entry point
tests/
  test_parity.py   behavior and parity tests
  test_serve.py    service probes, predictions, bad input, and the client side resize identity
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

## Serving it on Kubernetes

`serve/` and `k8s/` take a real exported model all the way to a load balanced, autoscaled service,
and measure it. The model is the ResNet50 chest X-ray classifier from the
[tensorrt-optimization](https://github.com/SharvenRane/tensorrt-optimization) benchmark (NLM
Montgomery tuberculosis set), exported to a single 94 MB ONNX file and served with ONNX Runtime on
CPU.

**The service** (`serve/app.py`, FastAPI) takes a PNG, applies exactly the torchvision eval
transform the model was trained with, and returns probabilities, logits and the pod that answered.
`/readyz` only reports ready after the model has loaded and run a warm up inference, so no pod
receives traffic cold. The image runs as a non root user with the model baked in.

**The cluster** is a local kind cluster. The Deployment gives each pod exactly 1 CPU and one ONNX
Runtime thread, with startup, readiness and liveness probes, a rolling update that never drops
below the current replica count (`maxUnavailable: 0`), a `preStop` delay so endpoint removal
propagates before a pod stops, and a PodDisruptionBudget. A HorizontalPodAutoscaler targets 60% CPU
between 1 and 6 replicas. Load comes from `serve/loadtest.py` running as a Job inside the cluster,
so requests go through cluster DNS and kube-proxy like any in cluster client. It opens a fresh
connection per request, so the Service balances per request rather than pinning a worker to a pod.

### Results

**Correctness through the whole stack.** All 42 test radiographs were sent at full resolution through the
Service: accuracy 0.8571, the same prediction as PyTorch on CPU for 42 of 42, and a maximum logit
difference of 4.8e-05.

**Scaling with replicas** (16 concurrent clients, 45 s each, 0 failed requests in every run):

| replicas | throughput | p50 latency | p95 latency | speedup |
|---|---|---|---|---|
| 1 | 33.0 rps | 477 ms | 522 ms | 1.00x |
| 2 | 61.7 rps | 254 ms | 481 ms | 1.87x |
| 4 | 105.7 rps | 124 ms | 347 ms | 3.20x |
| 8 | 174.5 rps | 75 ms | 207 ms | 5.29x |

Requests spread evenly: at 8 replicas each pod served between 937 and 1,011 requests.

**Autoscaling from one idle pod under sustained load** (240 s, 26,947 requests, 0 failed). The
HPA started from a single settled pod at 0% CPU and scaled 1 to 2 replicas 47 s after the load Job
was submitted, to 4 at 78 s, and to its ceiling of 6 at 120 s, adding at most 2 pods per 15 s as
configured. The load generator's own 10 s windows show the same staircase (the one window that straddles
the step from 2 to 4 pods, 98.1 rps, is left out):

| load phase | throughput (10 s windows) | p95 latency |
|---|---|---|
| first 40 s | 33.7 to 34.3 rps | 477 to 494 ms |
| next 30 s | 61.0 to 64.3 rps | 450 to 466 ms |
| next 40 s | 111.7 to 124.6 rps | 301 to 362 ms |
| remaining 120 s | 138.3 to 159.4 rps | 223 to 322 ms |

**Rolling restart under load** (4 replicas, 16 concurrent clients, 90 s): every pod was replaced
in 12 s while 10,228 requests ran, with **0 failed requests**. Throughput stayed between 110.5 and
118.9 rps in every full 10 s window, and p95 rose from about 310 ms to 384 ms in the window where pods
were swapped.

Raw measurements, including the per 10 s windows and the HPA timeline, are in `k8s/results/`.

### What the numbers say

**Replicas buy throughput almost linearly up to 4, then less.** 1.87x at 2 and 3.20x at 4 replicas,
but 5.29x at 8. The load generator is a single Python process and the node shares 24 CPUs with
the pods, so the flattening at 8 is not attributed to the service itself; separating those would
need the load generated from another machine.

**Latency is queueing, not compute.** One replica runs one inference at a time, about 29 ms of
model time, so with 16 concurrent clients a single pod's p50 is 477 ms of waiting. Adding replicas
cuts that queue, which is why p50 falls roughly in proportion.

**Zero downtime needed three settings working together:** a readiness probe that waits for a real
inference, `maxUnavailable: 0`, and a `preStop` delay. The rollout test is what shows they do.

### Reproduce

```bash
python -m pip install onnx
python -c "import onnx; onnx.save_model(onnx.load('resnet50_montgomery.onnx'), 'models/model.onnx', save_as_external_data=False)"
docker build -f serve/Dockerfile -t classifier:local .
kind create cluster --name ml --config k8s/kind-config.yaml
kind load docker-image classifier:local --name ml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deployment metrics-server --type=json   -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl apply -f k8s/deployment.yaml
bash k8s/run_experiments.sh
```

Measured on a Ryzen 9 9900X under WSL2 (24 vCPUs visible to the kind node), kind v0.34, Kubernetes
client 1.37, ONNX Runtime 1.30. Every run is one run; the numbers are indicative of shape, not a
capacity guarantee for other hardware.

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
