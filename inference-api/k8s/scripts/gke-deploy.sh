#!/usr/bin/env bash
# scripts/gke-deploy.sh
# To Build, push, and deploy the inference-api to GKE.
# Run from the repo root: ./inference-api/k8s/scripts/gke-deploy.sh
#
# Prerequisites:
#   terraform apply complete (cluster exists)
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   kubectl context pointed at infergrid cluster

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-infergrid-prod}"
REGION="${REGION:-us-central1}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/infergrid"
IMAGE="${REGISTRY}/inference-api:${IMAGE_TAG}"
K8S_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GCLOUD_BIN_DIR="$(gcloud info --format='value(installation.sdk_root)' 2>/dev/null)/bin"

if [[ -d "$GCLOUD_BIN_DIR" ]]; then
  export PATH="$GCLOUD_BIN_DIR:$PATH"
fi

for command in gcloud docker kubectl gke-gcloud-auth-plugin; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    echo "Install it, then rerun this script." >&2
    exit 1
  fi
done

echo "==> Authenticating Docker with Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building image: $IMAGE"
docker buildx build \
  --platform linux/amd64 \
  --tag "$IMAGE" \
  --push \
  "$(cd "$K8S_DIR/.." && pwd)"

echo "==> Fetching GKE credentials"
gcloud container clusters get-credentials infergrid \
  --region "$REGION" \
  --project "$PROJECT_ID"

echo "==> Applying manifests"
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/configmap.yaml"

# Inject real API key from env or generate one
API_KEY="${API_KEY:-$(openssl rand -hex 32)}"
kubectl create secret generic inference-api-secret \
  --namespace infergrid \
  --from-literal=API_KEY="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply the deployment, then replace the placeholder image with this build.
kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl set image deployment/inference-api \
  inference-api="$IMAGE" \
  -n infergrid
kubectl rollout restart deployment/inference-api -n infergrid

# Use LoadBalancer service on GKE (not NodePort)
kubectl apply -f "$K8S_DIR/service-loadbalancer.yaml"
kubectl apply -f "$K8S_DIR/hpa.yaml"

echo "==> Waiting for rollout"
kubectl rollout status deployment/inference-api -n infergrid --timeout=180s

echo ""
EXTERNAL_IP=$(kubectl get svc inference-api -n infergrid \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "    Deployed image: $IMAGE"
echo "    External IP: $EXTERNAL_IP (might take a moment to assign)"
echo "    Test: curl http://${EXTERNAL_IP}/health"
