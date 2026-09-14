resource "kind_cluster" "this" {
  name            = var.cluster_name
  node_image      = var.node_image
  wait_for_ready  = true
  kubeconfig_path = abspath("${path.module}/kubeconfig")

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      dynamic "extra_mounts" {
        for_each = var.data_host_path == "" ? [] : [var.data_host_path]
        content {
          host_path      = extra_mounts.value
          container_path = "/data"
        }
      }
    }
  }
}

# kind nodes cannot pull an image that only exists in the local docker daemon, so copy it in.
resource "terraform_data" "load_image" {
  count            = var.load_local_image ? 1 : 0
  triggers_replace = [kind_cluster.this.id, var.image]

  provisioner "local-exec" {
    command = "kind load docker-image ${var.image} --name ${var.cluster_name}"
  }
}

# Optional: copy third party images from the local docker cache into the node, so a rebuilt
# cluster does not pull the monitoring stack over the network again.
resource "terraform_data" "preload_images" {
  count            = length(var.preload_images) > 0 ? 1 : 0
  triggers_replace = [kind_cluster.this.id, var.preload_images]

  provisioner "local-exec" {
    # `kind load docker-image` fails on multi platform images when docker uses the containerd image
    # store (the other platforms' layers are not local), so export only linux/amd64 and import that.
    command     = <<-EOT
      set -eo pipefail
      for img in ${join(" ", var.preload_images)}; do
        docker image inspect "$img" >/dev/null 2>&1 || docker pull -q --platform linux/amd64 "$img"
        docker save --platform linux/amd64 "$img" | docker exec -i ${var.cluster_name}-control-plane \
          ctr -n k8s.io images import --platform linux/amd64 --digests --snapshotter overlayfs - >/dev/null
      done
    EOT
    interpreter = ["bash", "-c"]
  }
}
