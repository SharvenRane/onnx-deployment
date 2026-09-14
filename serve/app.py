"""HTTP inference service for an ONNX image classifier.

POST /predict   body: a PNG or JPEG image. Returns class probabilities, the predicted class,
                the raw logits and the pod that answered.
GET  /healthz   liveness: the process is up.
GET  /readyz    readiness: the model is loaded and has completed a warm up inference.
GET  /stats     request count and mean model latency for this replica.

Preprocessing reproduces the torchvision eval transform the model was trained with: convert to
RGB, Pillow bilinear resize to 224 x 224, scale to [0, 1], ImageNet normalisation. Serving any
other resize would feed the network pixels it never saw in training.

Configuration by environment variable:
  MODEL_PATH       path to the .onnx file (default /models/model.onnx)
  ORT_THREADS      intra op threads per replica (default 1, one replica per CPU core)

Inference runs on the event loop, so a replica serves one request at a time. With one ORT thread
and one CPU per pod that is the honest capacity of a replica; scaling is done with replicas.
"""
from __future__ import annotations

import io
import os
import socket
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
POD = socket.gethostname()

state = {"session": None, "input": None, "ready": False, "requests": 0, "model_ms": 0.0}
lock = threading.Lock()


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


@asynccontextmanager
async def lifespan(_app):
    load_model()
    yield
    state["ready"] = False


app = FastAPI(title="onnx-image-classifier", lifespan=lifespan)


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


@app.post("/predict")
async def predict(request: Request):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    body = await request.body()
    try:
        x = preprocess(body)
    except Exception as exc:  # malformed upload
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc
    t0 = time.perf_counter()
    logits = state["session"].run(None, {state["input"]: x})[0][0]
    ms = (time.perf_counter() - t0) * 1000.0
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
