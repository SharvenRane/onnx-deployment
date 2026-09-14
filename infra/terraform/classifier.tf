locals {
  labels = { app = "classifier" }
}

resource "kubernetes_namespace_v1" "serving" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_deployment_v1" "classifier" {
  depends_on = [terraform_data.load_image]

  metadata {
    name      = "classifier"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
    labels    = local.labels
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = local.labels
    }

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_unavailable = "0" # never drop below the current replica count during a rollout
        max_surge       = "1"
      }
    }

    template {
      metadata {
        labels = local.labels
      }

      spec {
        termination_grace_period_seconds = 30

        container {
          name              = "classifier"
          image             = var.image
          image_pull_policy = var.load_local_image ? "Never" : "IfNotPresent"

          port {
            name           = "http"
            container_port = 8000
          }

          env {
            name  = "ORT_THREADS"
            value = tostring(var.ort_threads)
          }
          env {
            name  = "MODEL_PATH"
            value = "/models/model.onnx"
          }

          resources {
            requests = {
              cpu    = var.resources.cpu_request
              memory = var.resources.memory_request
            }
            limits = {
              cpu    = var.resources.cpu_limit
              memory = var.resources.memory_limit
            }
          }

          startup_probe {
            http_get {
              path = "/readyz"
              port = 8000
            }
            period_seconds    = 1
            failure_threshold = 60
          }

          readiness_probe {
            http_get {
              path = "/readyz"
              port = 8000
            }
            period_seconds    = 2
            failure_threshold = 2
          }

          liveness_probe {
            http_get {
              path = "/healthz"
              port = 8000
            }
            period_seconds    = 10
            timeout_seconds   = 5
            failure_threshold = 3
          }

          lifecycle {
            pre_stop {
              # keep serving while the endpoint removal reaches kube-proxy
              exec {
                command = ["sleep", "5"]
              }
            }
          }
        }
      }
    }
  }

  lifecycle {
    # the HPA owns the replica count once it exists
    ignore_changes = [spec[0].replicas]
  }
}

resource "kubernetes_service_v1" "classifier" {
  metadata {
    name      = "classifier"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
    labels    = local.labels
  }

  spec {
    selector = local.labels

    port {
      name        = "http"
      port        = 80
      target_port = "http"
    }
  }
}

resource "kubernetes_horizontal_pod_autoscaler_v2" "classifier" {
  metadata {
    name      = "classifier"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
  }

  spec {
    min_replicas = var.hpa_min_replicas
    max_replicas = var.hpa_max_replicas

    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment_v1.classifier.metadata[0].name
    }

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_cpu_target_percent
        }
      }
    }

    behavior {
      scale_up {
        stabilization_window_seconds = 0
        select_policy                = "Max"
        policy {
          type           = "Pods"
          value          = 2
          period_seconds = 15
        }
      }
      scale_down {
        stabilization_window_seconds = 60
        select_policy                = "Max"
        policy {
          type           = "Percent"
          value          = 100
          period_seconds = 15
        }
      }
    }
  }
}

resource "kubernetes_pod_disruption_budget_v1" "classifier" {
  metadata {
    name      = "classifier"
    namespace = kubernetes_namespace_v1.serving.metadata[0].name
  }

  spec {
    min_available = "1"
    selector {
      match_labels = local.labels
    }
  }
}
