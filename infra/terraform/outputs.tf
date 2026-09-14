output "kubeconfig_path" {
  value = kind_cluster.this.kubeconfig_path
}

output "classifier_url_in_cluster" {
  value = "http://classifier.${var.namespace}.svc/predict"
}

output "grafana" {
  value = "kubectl -n monitoring port-forward svc/kps-grafana 3000:80, then http://localhost:3000 (user admin)"
}

output "prometheus" {
  value = "kubectl -n monitoring port-forward svc/kps-prometheus 9090:9090, then http://localhost:9090"
}
