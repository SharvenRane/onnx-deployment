"""HTTP inference service for an ONNX image classifier.

POST /predict   body: a PNG or JPEG image. Returns class probabilities, the predicted class,
                the raw logits and the pod that answered.
GET  /healthz   liveness: the process is up.
GET  /readyz    readiness: the model is loaded and has completed a warm up inference.
GET  /stats     request count and mean model latency for this replica.
GET  /metrics   Prometheus metrics: requests by handler and status, request latency, model
                latency, requests in flight and model readiness.

Preprocessing reproduces the torchvision eval transform the model was trained with: convert to
RGB, Pillow bilinear resize to 224 x 224, scale to [0, 1], ImageNet normalisation. Serving any
other resize would feed the network pixels it never saw in training.

Configuration by environment variable:
  MODEL_PATH       path to the .onnx file (default /models/model.onnx)
  ORT_THREADS      intra op threads per replica (default 1, one replica per CPU core)

Decoding and inference run on a single worker thread, so a replica still computes one request at
a time. With one ORT thread and one CPU per pod that is the honest capacity of a replica; scaling
is done with replicas. Keeping that work off the event loop means a request is accepted and its
latency timer starts as soon as it arrives, so the request latency histogram includes the time a
request waits for the worker, which is what a client actually experiences.
"""
from __future__ import annotations

import asyncio
import io
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
POD = socket.gethostname()

state = {"session": None, "input": None, "ready": False, "requests": 0, "model_ms": 0.0}
lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

HANDLERS = {"/predict", "/healthz", "/readyz", "/stats", "/metrics"}
REQUESTS = Counter("classifier_http_requests_total", "HTTP requests by handler, method and status code",
                   ["handler", "method", "status"])
REQUEST_SECONDS = Histogram(
    "classifier_http_request_duration_seconds",
    "Time from the request reaching the application to the response being sent",
    ["handler"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4,
             0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0),
)
IN_FLIGHT = Gauge("classifier_http_requests_in_flight", "Requests currently being handled, including queued ones")
MODEL_SECONDS = Histogram(
    "classifier_model_inference_seconds",
    "ONNX Runtime session.run time for one image",
    buckets=(0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.25, 0.5, 1.0),
)
MODEL_READY = Gauge("classifier_model_ready", "1 once the model is loaded and warmed up")


def preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return ((x - MEAN) / STD)[None].astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def load_model() -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    sess.run(None, {name: np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)})  # warm up before ready
    state.update(session=sess, input=name, ready=True)
    MODEL_READY.set(1)


class PrometheusMiddleware:
    """Counts and times every request at the ASGI boundary, before routing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        handler = scope["path"] if scope["path"] in HANDLERS else "other"
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        IN_FLIGHT.inc()
        t0 = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            REQUEST_SECONDS.labels(handler).observe(time.perf_counter() - t0)
            REQUESTS.labels(handler, scope["method"], str(status["code"])).inc()
            IN_FLIGHT.dec()


@asynccontextmanager
async def lifespan(_app):
    load_model()
    yield
    state["ready"] = False
    MODEL_READY.set(0)


app = FastAPI(title="onnx-image-classifier", lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "pod": POD}


@app.get("/readyz")
def readyz():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "pod": POD}


@app.get("/stats")
def stats():
    n = state["requests"]
    return {"pod": POD, "requests": n, "mean_model_ms": state["model_ms"] / n if n else None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def infer(body: bytes):
    """Decode and run one image. Runs on the single inference thread."""
    try:
        x = preprocess(body)
    except Exception as exc:  # malformed upload
        return None, f"could not decode image: {exc}", 0.0
    t0 = time.perf_counter()
    logits = state["session"].run(None, {state["input"]: x})[0][0]
    seconds = time.perf_counter() - t0
    MODEL_SECONDS.observe(seconds)
    return logits, None, seconds * 1000.0


@app.post("/predict")
async def predict(request: Request):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    body = await request.body()
    logits, error, ms = await asyncio.get_running_loop().run_in_executor(worker, infer, body)
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    with lock:
        state["requests"] += 1
        state["model_ms"] += ms
    probs = softmax(logits.astype(np.float64))
    return JSONResponse({
        "pod": POD,
        "predicted_class": int(np.argmax(logits)),
        "probabilities": [float(p) for p in probs],
        "logits": [float(v) for v in logits],
        "model_ms": ms,
    })
