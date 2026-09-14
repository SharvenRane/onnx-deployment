"""Tests for the HTTP inference service in serve/app.py.

A small CNN is exported at the service's input shape (3 x 224 x 224) so the real app code runs
end to end without the production model file.
"""
import importlib
import io
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.export import export_to_onnx  # noqa: E402
from src.model import SmallCNN  # noqa: E402

SERVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "serve"))


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.onnx"
    torch.manual_seed(0)
    export_to_onnx(SmallCNN(in_channels=3, num_classes=2, image_size=224), path, image_size=224, in_channels=3)
    os.environ["MODEL_PATH"] = str(path)
    sys.path.insert(0, SERVE)
    app_module = importlib.import_module("app")
    with TestClient(app_module.app) as c:
        yield c, app_module


def test_probes_report_ready_after_startup(client):
    c, _ = client
    assert c.get("/healthz").status_code == 200
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_predict_returns_normalised_probabilities(client):
    c, _ = client
    img = np.random.default_rng(0).integers(0, 256, (300, 260), dtype=np.uint8)
    r = c.post("/predict", content=_png(img), headers={"content-type": "image/png"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in (0, 1)
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-6
    assert int(np.argmax(body["logits"])) == body["predicted_class"]


def test_garbage_upload_is_a_client_error(client):
    c, _ = client
    r = c.post("/predict", content=b"not an image", headers={"content-type": "image/png"})
    assert r.status_code == 400


def test_client_side_resize_gives_the_server_an_identical_tensor(client):
    """The load test resizes on the client; that is only valid if the server tensor is unchanged."""
    _, app_module = client
    big = np.random.default_rng(1).integers(0, 256, (1024, 900), dtype=np.uint8)
    small = Image.fromarray(big).resize((224, 224), Image.BILINEAR)
    direct = app_module.preprocess(_png(big))
    via_client = app_module.preprocess(_png(np.asarray(small)))
    assert np.array_equal(direct, via_client)


def test_stats_count_requests(client):
    c, _ = client
    before = c.get("/stats").json()["requests"]
    img = np.zeros((224, 224), dtype=np.uint8)
    c.post("/predict", content=_png(img), headers={"content-type": "image/png"})
    assert c.get("/stats").json()["requests"] == before + 1


def _samples(c) -> dict:
    """Scrape /metrics and index every sample by (name, sorted labels)."""
    from prometheus_client.parser import text_string_to_metric_families

    r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    out = {}
    for family in text_string_to_metric_families(r.text):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def test_metrics_count_requests_by_status_and_time_them(client):
    c, _ = client
    ok = (("handler", "/predict"), ("method", "POST"), ("status", "200"))
    bad = (("handler", "/predict"), ("method", "POST"), ("status", "400"))
    hist = (("handler", "/predict"),)
    before = _samples(c)
    img = np.zeros((224, 224), dtype=np.uint8)
    for _ in range(3):
        assert c.post("/predict", content=_png(img), headers={"content-type": "image/png"}).status_code == 200
    assert c.post("/predict", content=b"junk", headers={"content-type": "image/png"}).status_code == 400
    after = _samples(c)

    def delta(key):
        return after.get(key, 0.0) - before.get(key, 0.0)

    assert delta(("classifier_http_requests_total", ok)) == 3
    assert delta(("classifier_http_requests_total", bad)) == 1
    # every request, good or bad, lands in the latency histogram
    assert delta(("classifier_http_request_duration_seconds_count", hist)) == 4
    assert delta(("classifier_http_request_duration_seconds_sum", hist)) > 0
    # the model only runs for decodable images
    assert delta(("classifier_model_inference_seconds_count", ())) == 3
    assert after[("classifier_http_request_duration_seconds_bucket", hist + (("le", "+Inf"),))] == \
        after[("classifier_http_request_duration_seconds_count", hist)]
    assert after[("classifier_model_ready", ())] == 1.0
    assert after[("classifier_http_requests_in_flight", ())] == 1.0  # only the scrape being served


def test_unknown_paths_share_one_label(client):
    c, _ = client
    for path in ("/a", "/b/c", "/wp-login.php"):
        assert c.get(path).status_code == 404
    samples = _samples(c)
    handlers = {dict(labels).get("handler") for name, labels in samples if name == "classifier_http_requests_total"}
    assert "other" in handlers
    assert not {"/a", "/b/c", "/wp-login.php"} & handlers
    assert samples[("classifier_http_requests_total", (("handler", "other"), ("method", "GET"), ("status", "404")))] >= 3
