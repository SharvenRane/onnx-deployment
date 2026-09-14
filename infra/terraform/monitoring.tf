resource "kubernetes_namespace_v1" "monitoring" {
  metadata {
    name = "monitoring"
  }
}

# Resource metrics for the HPA. kind's kubelet serves a self signed certificate.
resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  namespace  = "kube-system"
  repository = "https://kubernetes-sigs.github.io/metrics-server/"
  chart      = "metrics-server"
  version    = var.metrics_server_version
  wait       = true
  timeout    = 600

  depends_on = [terraform_data.preload_images]

  values = [yamlencode({
    args = ["--kubelet-insecure-tls"]
  })]
}

# Prometheus, Alertmanager, Grafana, kube-state-metrics and node exporter.
resource "helm_release" "kube_prometheus_stack" {
  name       = "kps"
  namespace  = kubernetes_namespace_v1.monitoring.metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = var.kube_prometheus_stack_version
  wait       = true
  timeout    = 1800 # a cold node pulls roughly 1 GB of images for this release

  depends_on = [terraform_data.preload_images]

  values = [
    file("${path.module}/values/kube-prometheus-stack.yaml"),
    yamlencode({
      grafana = {
        adminPassword = var.grafana_admin_password
        imageRenderer = { enabled = var.grafana_image_renderer }
      }
    }),
  ]
}

# ServiceMonitor and PrometheusRule for the classifier. They are custom resources, so they ship as
# a local chart: Helm, unlike kubernetes_manifest, does not need the CRDs to exist at plan time.
resource "helm_release" "classifier_monitoring" {
  depends_on = [helm_release.kube_prometheus_stack]

  name      = "classifier-monitoring"
  namespace = kubernetes_namespace_v1.serving.metadata[0].name
  chart     = "${path.module}/charts/classifier-monitoring"
  wait      = true

  values = [yamlencode({
    selector                   = local.labels
    scrapeInterval             = "5s"
    latencyP95ThresholdSeconds = var.latency_p95_threshold_seconds
    errorRatioThreshold        = var.error_ratio_threshold
    alertFor                   = var.alert_for
  })]
}

# Grafana's dashboard sidecar loads any ConfigMap carrying the grafana_dashboard label.
resource "kubernetes_config_map_v1" "classifier_dashboard" {
  metadata {
    name      = "classifier-dashboard"
    namespace = kubernetes_namespace_v1.monitoring.metadata[0].name
    labels    = { grafana_dashboard = "1" }
  }

  data = {
    "classifier.json" = file("${path.module}/../grafana/classifier-dashboard.json")
  }
}
