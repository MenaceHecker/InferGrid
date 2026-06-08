# InferGrid

InferGrid is a production-grade ML inference platform built on GCP that serves text classification requests through a FastAPI inference API, routes traffic across live model versions via A/B testing, and handles burst load through a Kafka-backed async queue with WebSocket result delivery. It is instrumented end-to-end with Prometheus metrics, three Grafana dashboards, and an automated KS-test drift detector — all deployed on GKE and validated under 200 concurrent users with zero failures.

> **Live endpoint:** `http://35.255.145.27/predict`  
> **Stack:** Python · FastAPI · Kubernetes · Kafka · Redis · Prometheus · Grafana · PostgreSQL · GCP

---

## Architecture

```mermaid
graph TD
    Client["Client"]

    subgraph GKE — infergrid namespace
        API["Inference API\nFastAPI :8000\n/predict /health /ready /metrics"]
        Worker["Consumer Worker\nconfluent-kafka"]
        Tracker["Experiment Tracker\nFastAPI :8001"]
        Redis["Redis\njob results TTL=5min"]
        Kafka["Kafka\nmodel-a.requests\nmodel-b.requests"]
    end

    subgraph GKE — monitoring namespace
        Prometheus["Prometheus\n:9090"]
        Grafana["Grafana\n:3000"]
        Drift["Drift Detector\nKS-test every 60s"]
        Adapter["Prometheus Adapter\ncustom.metrics.k8s.io"]
        HPA["HPA\n1–6 replicas"]
    end

    subgraph GCP
        GKE_Control["GKE Control Plane"]
        CloudSQL["Cloud SQL\nPostgreSQL"]
        AR["Artifact Registry\nDocker images"]
    end

    Client -->|"POST /predict"| API
    API -->|"sync path\nconcurrent < N"| API
    API -->|"async path\nconcurrent >= N\n202 + job_id"| Kafka
    API -->|"GET /ab/active"| Tracker
    Kafka --> Worker
    Worker -->|"inference result"| Redis
    Worker -->|"runs inference"| Worker
    API -->|"GET /ws/result/{job_id}\nor GET /result/{job_id}"| Redis
    Tracker --> CloudSQL
    Prometheus -->|"scrape /metrics"| API
    Prometheus -->|"scrape /metrics"| Drift
    Grafana --> Prometheus
    Drift --> Prometheus
    Adapter --> Prometheus
    HPA --> Adapter
    HPA -->|"scales"| API
```

---

## Benchmark Results

Model: sklearn TF-IDF + LogReg · Endpoint: `http://35.255.145.27` · 60s per scenario

### Sync Path

| Users | p50 (ms) | p95 (ms) | p99 (ms) | RPS   | Failures |
|-------|----------|----------|----------|-------|----------|
| 10    | 45       | 61       | 240      | 14.2  | 0        |
| 50    | 46       | 90       | 230      | 70.5  | 0        |
| 100   | 48       | 120      | 230      | 139.5 | 0        |
| 200   | 57       | 230      | 390      | 272.2 | 0        |

### Async Path (Kafka + Redis + WebSocket)

| Users | p50 (ms) | p95 (ms) | p99 (ms) | RPS   | Failures |
|-------|----------|----------|----------|-------|----------|
| 200   | 190      | 360      | 450      | 699.0 | 0        |

Zero failures across 67,963 total requests. Async path sustains 2.6× higher throughput than sync at the same user count.

---

## Local Quickstart

**Prerequisites:** Docker, docker-compose, Python 3.11+

```bash
git clone https://github.com/YOUR_USERNAME/infergrid.git
cd infergrid

# Start the inference API + experiment tracker + PostgreSQL
docker-compose up -d

# Send a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new rocket toward the ISS today."}'

# Expected response
# {"prediction": "sci.space", "confidence": 0.91, "model_backend": "sklearn", "model_version": "primary"}

# Open Grafana dashboards
open http://localhost:3000   # admin / infergrid
```

> **From clone to first prediction in under 2 minutes.**

### Local kind cluster (Kubernetes)

```bash
# Create cluster and deploy all services
chmod +x inference-api/k8s/scripts/kind-up.sh
./inference-api/k8s/scripts/kind-up.sh

# Validate
curl http://localhost:30080/health
curl http://localhost:30080/ready
```

---

## GCP Deployment

### Prerequisites

- GCP project with billing enabled
- `gcloud` CLI authenticated
- `terraform` >= 1.5
- `helm` >= 3.0
- `kubectl` configured

### 1. Provision infrastructure

```bash
# Create Terraform state bucket
gsutil mb -l us-central1 gs://infergrid-prod-tfstate

# Provision GKE cluster, node pool, Artifact Registry
cd infra/terraform
terraform init
terraform apply   # ~10 minutes
```

### 2. Deploy core services

```bash
# Build and push inference API image
chmod +x inference-api/k8s/scripts/gke-deploy.sh
./inference-api/k8s/scripts/gke-deploy.sh

# Deploy Kafka (Bitnami Helm chart)
./async-queue/kafka/deploy.sh

# Deploy Redis
kubectl apply -f async-queue/redis/redis.yaml

# Deploy consumer worker
kubectl apply -f async-queue/consumer/k8s/deployment.yaml
```

### 3. Deploy observability stack

```bash
# Prometheus
./observability/prometheus/deploy.sh

# Grafana + dashboards
./observability/grafana/build-dashboard-configmap.sh

# Drift detector
# First record baseline
python observability/drift_detector/register_baseline.py \
  --endpoint http://35.255.145.27 \
  --samples 500 \
  --output baseline.npy

kubectl create configmap drift-baseline \
  --namespace monitoring \
  --from-file=baseline.npy=baseline.npy \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f observability/drift_detector/k8s/deployment.yaml
```

### 4. Deploy experiment tracker

```bash
# Apply PostgreSQL schema
kubectl apply -f experiment-tracker/k8s/

# Run migrations
kubectl exec -n infergrid deploy/experiment-tracker -- \
  alembic upgrade head
```

### 5. Validate

```bash
# Inference API
curl http://35.255.145.27/health
curl http://35.255.145.27/ready
curl -X POST http://35.255.145.27/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new rocket."}'

# Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000
open http://localhost:3000   # admin / infergrid

# Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090
open http://localhost:9090
```

---

## Repository Structure

```
infergrid/
├── inference-api/          # FastAPI inference service
│   ├── app/
│   │   ├── main.py         # Routes, middleware, lifespan
│   │   ├── ab_router.py    # A/B routing with Experiment Tracker
│   │   ├── producer.py     # Kafka producer for async queue
│   │   ├── websocket.py    # WebSocket + HTTP fallback for job results
│   │   ├── metrics.py      # Prometheus instrumentation
│   │   └── models/         # sklearn and ONNX loaders
│   ├── k8s/                # Deployment, Service, HPA, ConfigMap manifests
│   └── tests/              # Integration tests (pytest)
├── experiment-tracker/     # Model versioning + A/B config API
│   ├── api/                # FastAPI endpoints
│   └── db/                 # SQLAlchemy models + Alembic migrations
├── async-queue/
│   ├── kafka/              # Helm values + deploy script
│   ├── redis/              # Redis deployment manifest
│   └── consumer/           # Kafka consumer worker
├── observability/
│   ├── prometheus/         # Deployment + scrape config + alert rules
│   ├── grafana/            # Deployment + 3 dashboard JSON exports
│   └── drift_detector/     # KS-test drift detection worker
├── infra/
│   └── terraform/          # GKE cluster, node pool, Artifact Registry
└── benchmarks/             # Locust load tests + results
```

---

## Observability

See [docs/observability.md](docs/observability.md) for full documentation on:
- All 6 Prometheus metrics and their bucket configurations
- What each Grafana dashboard panel shows and why it matters
- How the drift detector works and how to register a new baseline
- Alert rule thresholds and escalation guidance

---

## Development

```bash
# Install dependencies
cd inference-api
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format and lint
ruff format .
ruff check .

# Run end-to-end tests (requires both services running)
pytest tests/test_ab_end_to_end.py -v -m e2e
```

### CI

GitHub Actions runs on every push to `main`:
1. `ruff format --check` — formatting
2. `pytest tests/ -v` — unit and integration tests (e2e excluded)
3. `docker build` — image builds successfully
4. Deploy to GKE on merge to `main`

---

## Interview Prep

Every component in InferGrid maps to a system design question:

| Component | Question it answers |
|-----------|-------------------|
| Inference API + HPA | Design a model serving system that scales under traffic spikes |
| Prometheus + Grafana | How would you instrument a service for on-call engineers? |
| Drift Detector | How do you know when a deployed model is behaving differently? |
| Experiment Tracker | How would you safely roll out a new model version? |
| A/B Router | How would you run a controlled experiment comparing two ML models? |
| Kafka Queue | How would you handle a 10× traffic spike without dropping requests? |
| WebSocket delivery | How would you notify a client when a long-running job completes? |

---
