variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "credentials_file" {
  description = "Path to the GCP service account key JSON file"
  type        = string
  default     = "~/.gcp/infergrid-terraform-key.json"
}

variable "node_count" {
  description = "Initial number of nodes in the inference node pool"
  type        = number
  default     = 1
}
