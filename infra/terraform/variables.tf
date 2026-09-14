variable "cluster_name" {
  description = "Name of the kind cluster."
  type        = string
  default     = "ml"
}

variable "node_image" {
  description = "kind node image. Pinned to the default of the kind library the provider is built on."
  type        = string
  default     = "kindest/node:v1.35.0@sha256:452d707d4862f52530247495d180205e029056831160e22870e37e3f6c1ac31f"
}

variable "data_host_path" {
  description = "Host directory mounted into the node at /data for load test images and results. Empty to skip."
  type        = string
  default     = "/opt/data/k8s"
}

variable "image" {
  description = "Serving image. Built locally and loaded into the kind node."
  type        = string
  default     = "classifier:local"
}

variable "load_local_image" {
  description = "Run `kind load docker-image` for var.image after the cluster exists. Needs the kind CLI and docker."
  type        = bool
  default     = true
}

variable "preload_images" {
  description = "Images to pull into the local docker cache once and load into the node before the Helm releases. Empty pulls everything from the registries."
  type        = list(string)
  default     = []
}

variable "namespace" {
  description = "Namespace for the classifier."
  type        = string
  default     = "serving"
}

variable "replicas" {
  description = "Initial replica count. The HPA owns the count after creation."
  type        = number
  default     = 1
}

variable "hpa_min_replicas" {
  type    = number
  default = 1
}

variable "hpa_max_replicas" {
  type    = number
  default = 6
}

variable "hpa_cpu_target_percent" {
  description = "Average CPU utilisation, as a percent of the request, the HPA holds."
  type        = number
  default     = 60
}

variable "ort_threads" {
  description = "ONNX Runtime intra op threads per pod. Keep equal to the CPU limit."
  type        = number
  default     = 1
}

variable "resources" {
  description = "Container requests and limits for the classifier."
  type = object({
    cpu_request    = string
    cpu_limit      = string
    memory_request = string
    memory_limit   = string
  })
  default = {
    cpu_request    = "1"
    cpu_limit      = "1"
    memory_request = "600Mi"
    memory_limit   = "1Gi"
  }
}

variable "latency_p95_threshold_seconds" {
  description = "Alert when p95 request latency on /predict stays above this."
  type        = number
  default     = 0.25
}

variable "error_ratio_threshold" {
  description = "Alert when the share of /predict responses that are 5xx stays above this."
  type        = number
  default     = 0.05
}

variable "alert_for" {
  description = "How long a condition must hold before the alert fires."
  type        = string
  default     = "2m"
}

variable "grafana_admin_password" {
  description = "Grafana admin password. The cluster is local; override for anything shared."
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "grafana_image_renderer" {
  description = "Deploy the Grafana image renderer so panels can be exported as PNG over the HTTP API."
  type        = bool
  default     = true
}

variable "kube_prometheus_stack_version" {
  type    = string
  default = "91.2.1"
}

variable "metrics_server_version" {
  type    = string
  default = "3.14.0"
}
