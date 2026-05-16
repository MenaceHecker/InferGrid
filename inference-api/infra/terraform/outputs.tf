output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.infergrid.name
}

output "cluster_endpoint" {
  description = "GKE cluster API endpoint"
  value       = google_container_cluster.infergrid.endpoint
  sensitive   = true
}

output "inference_api_ip" {
  description = "Static IP for the inference API LoadBalancer"
  value       = google_compute_global_address.inference_api.address
}

output "artifact_registry_url" {
  description = "Docker image push URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/infergrid"
}
