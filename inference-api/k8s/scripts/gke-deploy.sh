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

echo "==> Authenticating Docker with Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building image: $IMAGE"
docker build -t "$IMAGE" "$(cd "$K8S_DIR/.." && pwd)"

echo "==> Pushing image to Artifact Registry"
docker push "$IMAGE"

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

# Update the image in the deployment
kubectl set image deployment/inference-api \
  inference-api="$IMAGE" \
  -n infergrid 2>/dev/null || kubectl apply -f "$K8S_DIR/deployment.yaml"

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
