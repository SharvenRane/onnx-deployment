"""Load generator and correctness check for the inference service.

Two modes:

  check   send every image in a split once, compare predictions and logits with a reference
          produced by PyTorch on CPU, and report agreement through the whole serving stack
  load    closed loop load: `concurrency` workers each send a request as soon as the previous
          one returns, for `duration` seconds, recording latency, status and answering pod

    python loadtest.py check --url http://classifier/predict --images /data --reference /data/reference.json
    python loadtest.py load  --url http://classifier/predict --images /data --concurrency 16 --duration 60

In load mode images are resized to 224 x 224 on the client first. Pillow's resize to the size an
image already has is an identity, so the server computes exactly the same tensor, and the
benchmark measures serving rather than uploading 5 MB radiographs.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def load_images(folder: str, names: list[str] | None, resize: int | None) -> list[tuple[str, bytes]]:
    paths = [Path(folder) / n for n in names] if names else sorted(Path(folder).glob("*.png"))
    out = []
    for p in paths:
        if resize:
            buf = io.BytesIO()
            Image.open(p).resize((resize, resize), Image.BILINEAR).save(buf, format="PNG")
            out.append((p.name, buf.getvalue()))
        else:
            out.append((p.name, p.read_bytes()))
    return out


def summarize(lat_ms: list[float]) -> dict:
    if not lat_ms:
        return {}
    a = np.array(lat_ms)
    return {k: float(np.percentile(a, q)) for k, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99))} | {
        "mean_ms": float(a.mean())}


async def run_check(args) -> dict:
    ref = json.loads(Path(args.reference).read_text())
    images = load_images(args.images, ref["files"], resize=None)
    agree, correct, max_diff, pods = 0, 0, 0.0, Counter()
    async with httpx.AsyncClient(timeout=60) as client:
        for (name, data), label, ref_logits in zip(images, ref["labels"], ref["logits_cpu_fp32"]):
            r = await client.post(args.url, content=data, headers={"content-type": "image/png"})
            r.raise_for_status()
            body = r.json()
            pods[body["pod"]] += 1
            agree += int(body["predicted_class"] == int(np.argmax(ref_logits)))
            correct += int(body["predicted_class"] == label)
            max_diff = max(max_diff, float(np.max(np.abs(np.array(body["logits"]) - np.array(ref_logits)))))
    return {"images": len(images), "accuracy": correct / len(images),
            "prediction_agreement_with_reference": agree / len(images),
            "max_abs_logit_diff": max_diff, "pods": dict(pods)}


async def run_load(args) -> dict:
    images = load_images(args.images, None, resize=224)
    lat, codes, pods, timeline = [], Counter(), Counter(), []
    stop = time.perf_counter() + args.duration
    start = time.perf_counter()

    async def worker(i: int, client: httpx.AsyncClient):
        k = i
        while time.perf_counter() < stop:
            _, data = images[k % len(images)]
            k += 1
            t0 = time.perf_counter()
            try:
                r = await client.post(args.url, content=data, headers={"content-type": "image/png"})
                code = r.status_code
                if code == 200:
                    pods[r.json()["pod"]] += 1
            except httpx.HTTPError as exc:
                code = type(exc).__name__
            t1 = time.perf_counter()
            codes[str(code)] += 1
            if code == 200:
                lat.append((t1 - t0) * 1000.0)
            timeline.append((round(t1 - start, 3), code == 200, round((t1 - t0) * 1000.0, 2)))

    # No keep alive: every request opens a new connection, so the Service spreads load across pods
    # per request rather than pinning each worker to one pod for the whole run.
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        await asyncio.gather(*(worker(i, client) for i in range(args.concurrency)))
    elapsed = time.perf_counter() - start
    ok = codes.get("200", 0)
    # per 10 s window, so a scale out or a rollout is visible over time
    windows = {}
    for t, good, ms in timeline:
        w = int(t // 10) * 10
        d = windows.setdefault(w, {"ok": 0, "failed": 0, "lat": []})
        d["ok" if good else "failed"] += 1
        if good:
            d["lat"].append(ms)
    per_window = [{"t": w, "ok": d["ok"], "failed": d["failed"], "rps": d["ok"] / 10,
                   "p95_ms": float(np.percentile(d["lat"], 95)) if d["lat"] else None}
                  for w, d in sorted(windows.items())]
    return {"concurrency": args.concurrency, "duration_s": elapsed, "requests": sum(codes.values()),
            "ok": ok, "failed": sum(codes.values()) - ok, "status_codes": dict(codes),
            "throughput_rps": ok / elapsed, "latency": summarize(lat), "pods": dict(pods),
            "per_10s": per_window}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "load"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = asyncio.run(run_check(args) if args.mode == "check" else run_load(args))
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "per_10s"}, indent=1))


if __name__ == "__main__":
    main()
