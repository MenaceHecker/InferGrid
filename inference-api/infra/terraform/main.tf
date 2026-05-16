terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state in GCS, this keeps state out of the repo.
  # Creating the bucket first:
  #   gcloud storage buckets create gs://infergrid-prod-tfstate-tm2508 --location=us-central1
  backend "gcs" {
    bucket = "infergrid-prod-tfstate-tm2508"
    prefix = "infergrid/state"
  }
}

provider "google" {
  project     = var.project_id
  region      = var.region
  credentials = file(var.credentials_file)
}

# GKE Cluster

resource "google_container_cluster" "infergrid" {
  name     = "infergrid"
  location = var.region

  # We manage the node pool separately below for easier upgrades
  remove_default_node_pool = true
  initial_node_count       = 1

  # Enables the Prometheus-compatible managed metrics collection
  # (useful in Phase 3 as an alternative to self-hosted Prometheus)
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "APISERVER"]
  }

  # Workload Identity — lets pods authenticate to GCP APIs without key files
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  deletion_protection = false   # set to true after the project is live
}

resource "google_container_node_pool" "inference" {
  name       = "inference-pool"
  cluster    = google_container_cluster.infergrid.name
  location   = var.region
  node_count = var.node_count

  node_config {
    # e2-standard-2: 2 vCPU, 8GB RAM for free-tier eligible, enough for 6 pods
    machine_type = "e2-standard-2"
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      project = "infergrid"
      layer   = "inference"
    }
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 3   # node-level ceiling; pod-level ceiling set in HPA
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Static IP for the inference API LoadBalancer


resource "google_compute_global_address" "inference_api" {
  name = "infergrid-inference-api-ip"
}


# Artifact Registry — stores Docker images


resource "google_artifact_registry_repository" "infergrid" {
  location      = var.region
  repository_id = "infergrid"
  format        = "DOCKER"
  description   = "InferGrid Docker images"
}
