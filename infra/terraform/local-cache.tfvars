# The larger images used by the kube-prometheus-stack release at the pinned chart version. With
# -var-file=local-cache.tfvars they are pulled into the local docker cache once and copied into the
# kind node, so recreating the cluster does not download them again. kube-state-metrics and
# metrics-server are small and are left to the node to pull.
preload_images = [
  "docker.io/grafana/grafana-image-renderer:v5.12.3",
  "docker.io/grafana/grafana:13.2.1-distroless",
  "ghcr.io/jkroepke/kube-webhook-certgen:1.8.8",
  "quay.io/kiwigrid/k8s-sidecar:2.11.2",
  "quay.io/prometheus-operator/prometheus-config-reloader:v0.94.0",
  "quay.io/prometheus-operator/prometheus-operator:v0.94.0",
  "quay.io/prometheus/alertmanager:v0.34.0",
  "quay.io/prometheus/node-exporter:v1.12.1-distroless",
  "quay.io/prometheus/prometheus:v3.14.0-distroless",
]
