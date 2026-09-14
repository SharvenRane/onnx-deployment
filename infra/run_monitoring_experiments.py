#!/usr/bin/env python3
"""Checks that the monitoring stack tells the truth, and that the latency alert fires.

Runs against the stack created by infra/terraform, using kubectl only (Prometheus and Alertmanager
are reached through the API server's service proxy, so nothing is port forwarded). Standard library
only, so it runs on any machine with python3 and kubectl.

  1. p95 truthfulness: for fixed replica counts, a load Job runs serve/loadtest.py inside the cluster.
     Afterwards the p95 over exactly that run is computed by Prometheus with histogram_quantile and
     compared with the p95 the load generator measured from the client side.
  2. Alert: with the HPA active, load heavier than the HPA ceiling can absorb is held for 5 minutes
     while /api/v1/alerts and Alertmanager's /api/v2/alerts are polled. Then the load stops and
     polling continues until the alert resolves.
  3. Dashboard evidence: every dashboard panel query is evaluated with /api/v1/query_range over the
     alert run and written to JSON.

    KUBECONFIG=infra/terraform/kubeconfig python3 infra/run_monitoring_experiments.py

Test images are expected at /data/images inside the kind node (the extra mount in cluster.tf) and
results are written to /data/results, which is data_host_path/results on the machine running kind.
Pass a different host side results directory as the first argument.
"""
from __future__ import annotations

import calendar
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

NS = "serving"
MON = "monitoring"
PROM = f"/api/v1/namespaces/{MON}/services/http:kps-prometheus:http-web/proxy"
AM = f"/api/v1/namespaces/{MON}/services/http:kps-alertmanager:http-web/proxy"
RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/data/k8s/results")
SEL = f'namespace="{NS}", handler="/predict"'
ALERT = "ClassifierHighLatencyP95"


def kubectl(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kubectl", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def raw(path: str) -> dict:
    return json.loads(kubectl("get", "--raw", path))


def query(expr: str, at: float | None = None) -> list:
    params = {"query": expr}
    if at is not None:
        params["time"] = f"{at:.3f}"
    d = raw(f"{PROM}/api/v1/query?{urllib.parse.urlencode(params)}")
    assert d["status"] == "success", d
    return d["data"]["result"]


def scalar(expr: str, at: float | None = None) -> float | None:
    res = query(expr, at)
    return float(res[0]["value"][1]) if res else None


def query_range(expr: str, start: float, end: float, step: int = 10) -> list:
    params = {"query": expr, "start": f"{start:.0f}", "end": f"{end:.0f}", "step": str(step)}
    d = raw(f"{PROM}/api/v1/query_range?{urllib.parse.urlencode(params)}")
    assert d["status"] == "success", d
    return d["data"]["result"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def job_manifest(name: str, args: list[str]) -> str:
    return json.dumps({
        "apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": name, "namespace": NS},
        "spec": {"backoffLimit": 0, "template": {"spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "loadtest", "image": "classifier:local", "imagePullPolicy": "Never",
                "command": ["python", "/app/loadtest.py"], "args": args,
                "resources": {"requests": {"cpu": "2", "memory": "512Mi"}},
                "volumeMounts": [{"name": "data", "mountPath": "/data"}]}],
            "volumes": [{"name": "data", "hostPath": {"path": "/data", "type": "Directory"}}]}}}})


def start_load(name: str, concurrency: int, duration: int) -> str:
    out = f"/data/results/{name}.json"
    kubectl("-n", NS, "delete", "job", name, "--ignore-not-found", "--wait=true")
    args = ["load", "--url", f"http://classifier.{NS}.svc/predict", "--images", "/data/images",
            "--concurrency", str(concurrency), "--duration", str(duration), "--out", out]
    subprocess.run(["kubectl", "apply", "-f", "-"], input=job_manifest(name, args), text=True, check=True,
                   capture_output=True)
    return name


def wait_job(name: str, timeout: int = 1200) -> tuple[float, float]:
    kubectl("-n", NS, "wait", "--for=condition=complete", f"job/{name}", f"--timeout={timeout}s")
    pod = kubectl("-n", NS, "get", "pods", "-l", f"job-name={name}", "-o", "jsonpath={.items[0].metadata.name}")
    st = json.loads(kubectl("-n", NS, "get", "pod", pod, "-o", "json"))["status"]["containerStatuses"][0]["state"]
    term = st["terminated"]
    return iso(term["startedAt"]), iso(term["finishedAt"])


def iso(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def set_replicas(n: int):
    """Pin the HPA to n so a run measures a fixed replica count."""
    kubectl("-n", NS, "patch", "hpa", "classifier", "--type=merge", "-p",
            json.dumps({"spec": {"minReplicas": n, "maxReplicas": n}}))
    for _ in range(120):
        d = json.loads(kubectl("-n", NS, "get", "deploy", "classifier", "-o", "json"))
        if d["spec"]["replicas"] == n and d["status"].get("readyReplicas") == n and d["status"].get("replicas") == n:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"deployment did not settle at {n} replicas")
    wait_targets(n)


def wait_targets(n: int):
    for _ in range(60):
        up = scalar(f'sum(up{{namespace="{NS}", service="classifier"}})')
        if up == n:
            return
        time.sleep(2)
    raise RuntimeError(f"Prometheus does not see {n} classifier targets")


def hq(q: float, buckets: dict[float, float]) -> float | None:
    """histogram_quantile's interpolation, for the exact bucket deltas."""
    items = sorted(buckets.items())
    total = items[-1][1] if items else 0
    if total <= 0:
        return None
    rank, prev_le, prev_c = q * total, 0.0, 0.0
    for le, c in items:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c) if c > prev_c else le
        prev_le, prev_c = le, c
    return None


def bucket_snapshot(at: float) -> dict[tuple, float]:
    res = query(f"classifier_http_request_duration_seconds_bucket{{{SEL}}}", at)
    return {(r["metric"]["pod"], float(r["metric"]["le"])): float(r["value"][1]) for r in res}


def compare(name: str, t_before: float, started: float, finished: float) -> dict:
    """p95 over one run: Prometheus histogram_quantile vs the load generator."""
    t_after = finished + 20  # at least three scrapes after the last request
    while time.time() < t_after + 2:
        time.sleep(1)
    window = int(round(t_after - t_before))
    def promql(q: float) -> str:
        return (f"histogram_quantile({q}, sum by (le) (increase("
                f"classifier_http_request_duration_seconds_bucket{{{SEL}}}[{window}s])))")

    prom = {f"p{int(q * 100)}_ms": (scalar(promql(q), t_after) or float("nan")) * 1000 for q in (0.5, 0.95, 0.99)}
    # the same quantile from exact counter deltas, which avoids increase() extrapolation at the window edges
    before, after = bucket_snapshot(t_before), bucket_snapshot(t_after)
    delta: dict[float, float] = {}
    for (pod, le), v in after.items():
        delta[le] = delta.get(le, 0.0) + v - before.get((pod, le), 0.0)
    exact = {f"p{int(q * 100)}_ms": hq(q, delta) * 1000 for q in (0.5, 0.95, 0.99)}
    client = json.loads((RESULTS / f"{name}.json").read_text())
    return {
        "run": name, "loadtest_started_unix": started, "loadtest_finished_unix": finished,
        "prometheus_window_s": window, "promql_p95": promql(0.95),
        "loadtest": {k: client["latency"][k] for k in ("p50_ms", "p95_ms", "p99_ms")},
        "prometheus_histogram_quantile": prom,
        "prometheus_exact_bucket_delta": exact,
        "requests_loadtest_ok": client["ok"], "requests_loadtest_failed": client["failed"],
        "requests_counted_by_prometheus": delta.get(float("inf")),
        "throughput_rps": client["throughput_rps"],
        "p95_difference_ms": prom["p95_ms"] - client["latency"]["p95_ms"],
    }


def truthfulness(replicas: int, concurrency: int = 16, duration: int = 120) -> dict:
    name = f"prom-r{replicas}"
    log(f"truthfulness run: {replicas} replicas, {concurrency} clients, {duration} s")
    set_replicas(replicas)
    time.sleep(15)
    t_before = time.time()
    time.sleep(5)
    start_load(name, concurrency, duration)
    started, finished = wait_job(name)
    r = compare(name, t_before, started, finished) | {"replicas": replicas, "concurrency": concurrency}
    log(json.dumps({k: r[k] for k in ("loadtest", "prometheus_histogram_quantile", "prometheus_exact_bucket_delta",
                                      "requests_loadtest_ok", "requests_counted_by_prometheus")}))
    return r


def alerts_now() -> dict:
    prom = raw(f"{PROM}/api/v1/alerts")["data"]["alerts"]
    mine = [a for a in prom if a["labels"].get("alertname") == ALERT]
    am = raw(f"{AM}/api/v2/alerts?filter=" + urllib.parse.quote(f'alertname="{ALERT}"'))
    p95 = scalar("classifier:request_duration_seconds:p95_1m")
    ready = scalar(f'sum(kube_deployment_status_replicas_ready{{namespace="{NS}", deployment="classifier"}})')
    return {"t": time.time(), "prometheus_state": mine[0]["state"] if mine else "inactive",
            "active_at": mine[0].get("activeAt") if mine else None,
            "alertmanager_state": am[0]["status"]["state"] if am else None,
            "p95_1m_s": p95, "ready_replicas": ready}


def alert_run(concurrency: int = 48, duration: int = 300) -> dict:
    name = "prom-alert"
    log(f"alert run: HPA 1 to 6, {concurrency} clients, {duration} s")
    set_replicas(1)
    kubectl("-n", NS, "patch", "hpa", "classifier", "--type=merge", "-p",
            json.dumps({"spec": {"minReplicas": 1, "maxReplicas": 6}}))
    time.sleep(30)
    t_before = time.time()
    time.sleep(5)
    timeline = []
    start_load(name, concurrency, duration)
    while True:
        s = alerts_now()
        timeline.append(s)
        done = kubectl("-n", NS, "get", "job", name, "-o", "jsonpath={.status.succeeded}", check=False).strip() == "1"
        log(s["prometheus_state"], s["alertmanager_state"], s["p95_1m_s"], s["ready_replicas"])
        if done:
            break
        time.sleep(10)
    started, finished = wait_job(name)
    comparison = compare(name, t_before, started, finished)
    firing_rows = [s for s in timeline if s["prometheus_state"] == "firing"]
    # after the load: keep polling until the alert clears
    resolved_at = None
    for _ in range(60):
        s = alerts_now()
        timeline.append(s)
        log("after load", s["prometheus_state"], s["alertmanager_state"], s["p95_1m_s"])
        if s["prometheus_state"] == "inactive":
            resolved_at = s["t"]
            break
        time.sleep(10)
    end = time.time()
    firing_api = next((s for s in timeline if s["prometheus_state"] == "firing"), None)
    snapshot_prom = raw(f"{PROM}/api/v1/query?" + urllib.parse.urlencode(
        {"query": f'ALERTS{{alertname="{ALERT}"}}', "time": f"{firing_api['t']:.0f}"})) if firing_api else None
    first_breach = scalar(f"min_over_time(timestamp(classifier:request_duration_seconds:p95_1m > "
                          f"{threshold()})[{int(end - t_before)}s:5s])", end)
    return {
        "alert": ALERT, "threshold_s": threshold(), "for": "2m", "concurrency": concurrency, "duration_s": duration,
        "loadtest_started_unix": started, "loadtest_finished_unix": finished,
        "first_breach_unix": first_breach,
        "first_pending_unix": next((s["t"] for s in timeline if s["prometheus_state"] == "pending"), None),
        "first_firing_seen_unix": firing_api["t"] if firing_api else None,
        "prometheus_active_at": firing_api["active_at"] if firing_api else None,
        "first_alertmanager_active_unix": next((s["t"] for s in timeline if s["alertmanager_state"] == "active"), None),
        "resolved_seen_unix": resolved_at,
        "polls_firing": len(firing_rows),
        "alerts_series_at_first_firing_poll": snapshot_prom,
        "p95_comparison": comparison,
        "timeline": timeline,
        "window": [t_before, end],
    }


def threshold() -> float:
    groups = raw(f"{PROM}/api/v1/rules?type=alert")["data"]["groups"]
    for g in groups:
        for r in g["rules"]:
            if r["name"] == ALERT:
                return float(r["query"].split(">")[-1])
    raise RuntimeError("alert rule not loaded")


def dashboard_export(start: float, end: float) -> dict:
    """Evaluates every panel query of infra/grafana/classifier-dashboard.json over the run."""
    here = Path(__file__).resolve().parent
    dash = json.loads((here / "grafana" / "classifier-dashboard.json").read_text())
    out = {"start_unix": start, "end_unix": end, "step_s": 10, "rate_interval": "1m", "panels": []}
    for p in dash["panels"]:
        series = []
        for t in p["targets"]:
            expr = t["expr"].replace("$namespace", NS).replace("$__rate_interval", "1m")
            res = query_range(expr, start, end, 10)
            series.append({"legend": t["legendFormat"], "expr": expr,
                           "result": [{"metric": r["metric"], "values": r["values"]} for r in res]})
        out["panels"].append({"title": p["title"], "unit": p["fieldConfig"]["defaults"]["unit"], "series": series})
    return out


def rules_snapshot() -> dict:
    groups = raw(f"{PROM}/api/v1/rules")["data"]["groups"]
    return {"groups": [g for g in groups if g["name"].startswith("classifier.")]}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = raw(f"{PROM}/api/v1/targets?state=active")["data"]["activeTargets"]
    mine = [{"scrapeUrl": t["scrapeUrl"], "health": t["health"], "scrapeInterval": t["scrapeInterval"]}
            for t in targets if t["labels"].get("namespace") == NS]
    log("classifier targets", mine)
    runs = [truthfulness(1), truthfulness(4)]
    (RESULTS / "prometheus_vs_loadtest.json").write_text(json.dumps(runs, indent=1))
    alert = alert_run()
    (RESULTS / "alert_firing.json").write_text(json.dumps(alert | {"rules": rules_snapshot()}, indent=1))
    (RESULTS / "prometheus_dashboard_queries.json").write_text(
        json.dumps(dashboard_export(*alert["window"]), indent=1))
    log("done")


if __name__ == "__main__":
    main()
