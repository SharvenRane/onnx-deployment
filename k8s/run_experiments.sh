#!/bin/bash
# Runs every Kubernetes measurement in k8s/results against a kind cluster created from
# k8s/kind-config.yaml, with classifier:local loaded and k8s/deployment.yaml applied.
# Test images go in /opt/data/k8s/images and the PyTorch reference in /opt/data/k8s/reference.json.
set -u
cd /opt/work/onnx-deployment
R=/opt/data/k8s/results
mkdir -p $R && chmod 777 $R   # the service image runs as uid 10001

run_job() {  # name, then loadtest.py args
  local name=$1; shift
  local args=""; for a in "$@"; do args="$args\"$a\", "; done
  kubectl delete job $name --ignore-not-found >/dev/null
  cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: {name: $name}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: loadtest
          image: classifier:local
          imagePullPolicy: Never
          command: ["python", "/app/loadtest.py"]
          args: [${args%, }]
          resources: {requests: {cpu: "2", memory: 512Mi}}
          volumeMounts: [{name: data, mountPath: /data}]
      volumes: [{name: data, hostPath: {path: /data, type: Directory}}]
YAML
  kubectl wait --for=condition=complete job/$name --timeout=900s >/dev/null || { echo "job $name failed"; kubectl logs job/$name | tail -20; }
}

scale_to() {
  kubectl scale deployment/classifier --replicas=$1 >/dev/null
  kubectl rollout status deployment/classifier --timeout=300s >/dev/null
  kubectl wait --for=condition=ready pod -l app=classifier --timeout=300s >/dev/null
  sleep 3
}

echo "== correctness through the Service, 2 replicas"
kubectl delete hpa classifier --ignore-not-found >/dev/null
scale_to 2
run_job check check --url http://classifier/predict --images /data/images --reference /data/reference.json --out /data/results/check.json
cat $R/check.json

echo "== fixed replicas, concurrency 16, 45 s"
for n in 1 2 4 8; do
  scale_to $n
  run_job load-r$n load --url http://classifier/predict --images /data/images --concurrency 16 --duration 45 --out /data/results/replicas_$n.json
  python3 -c "import json;d=json.load(open('$R/replicas_$n.json'));print($n,'replicas',round(d['throughput_rps'],1),'rps p50',round(d['latency']['p50_ms'],1),'p95',round(d['latency']['p95_ms'],1),'failed',d['failed'],'pods',len(d['pods']))"
done

echo "== HPA from 1 settled replica under sustained load"
kubectl delete hpa classifier --ignore-not-found >/dev/null
kubectl scale deployment/classifier --replicas=1 >/dev/null
kubectl rollout status deployment/classifier --timeout=300s >/dev/null
# idle long enough that metrics-server's window holds no load from earlier runs
sleep 90
kubectl apply -f k8s/hpa.yaml >/dev/null
for i in $(seq 1 40); do
  u=$(kubectl get hpa classifier -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null)
  r=$(kubectl get deploy classifier -o jsonpath='{.status.replicas}')
  [ -n "$u" ] && [ "$u" -lt 10 ] && [ "$r" = "1" ] && break
  sleep 5
done
echo "before load: replicas $(kubectl get deploy classifier -o jsonpath='{.status.replicas}') cpu ${u}%"
( t0=$(date +%s); for i in $(seq 0 5 300); do now=$(( $(date +%s) - t0 )); echo "$now $(kubectl get deploy classifier -o jsonpath='{.status.replicas} {.status.readyReplicas}') $(kubectl get hpa classifier -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}')"; sleep 5; done ) > $R/hpa_timeline.txt &
TL=$!
kubectl delete job load-hpa --ignore-not-found >/dev/null
cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: {name: load-hpa}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: loadtest
          image: classifier:local
          imagePullPolicy: Never
          command: ["python", "/app/loadtest.py"]
          args: ["load", "--url", "http://classifier/predict", "--images", "/data/images", "--concurrency", "16", "--duration", "240", "--out", "/data/results/hpa.json"]
          resources: {requests: {cpu: "2", memory: 512Mi}}
          volumeMounts: [{name: data, mountPath: /data}]
      volumes: [{name: data, hostPath: {path: /data, type: Directory}}]
YAML
kubectl wait --for=condition=complete job/load-hpa --timeout=600s >/dev/null
wait $TL
cat $R/hpa_timeline.txt
python3 - <<PY
import json
d=json.load(open("$R/hpa.json"))
print("hpa run: ok", d["ok"], "failed", d["failed"], "rps", round(d["throughput_rps"],1), "pods", len(d["pods"]))
for w in d["per_10s"]: print(w["t"], round(w["rps"],1), w["p95_ms"] and round(w["p95_ms"]), w["failed"])
PY
kubectl delete hpa classifier >/dev/null

echo "== rolling restart under load, 4 replicas"
scale_to 4
( sleep 25; date +%s > $R/rollout_start.txt; kubectl rollout restart deployment/classifier >/dev/null; kubectl rollout status deployment/classifier --timeout=300s > $R/rollout_status.txt; date +%s > $R/rollout_end.txt ) &
RO=$!
JOBSTART=$(date +%s)
run_job load-rollout load --url http://classifier/predict --images /data/images --concurrency 16 --duration 90 --out /data/results/rollout.json
wait $RO
echo "job start $JOBSTART rollout start $(cat $R/rollout_start.txt) end $(cat $R/rollout_end.txt)"
python3 -c "import json;d=json.load(open('$R/rollout.json'));print('rollout: requests',d['requests'],'failed',d['failed'],d['status_codes'],'pods seen',len(d['pods']))"
kubectl get pods
