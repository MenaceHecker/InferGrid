#!/usr/bin/env bash
# scripts/kind-up.sh
# One-shot: create kind cluster, load the Docker image, apply all manifests.
# Run from the repo root: ./inference-api/scripts/kind-up.sh

set -euo pipefail

CLUSTER_NAME="infergrid"
IMAGE_NAME="inference-api:latest"
API_DIR="$(cd "$(dirname "$0")/.." && pwd)"
K8S_DIR="$API_DIR/k8s"

echo "==> Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$API_DIR"

echo "==> Creating kind cluster: $CLUSTER_NAME"
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "    Cluster already exists, skipping create."
else
  kind create cluster --config "$K8S_DIR/kind-config.yaml" --name "$CLUSTER_NAME"
fi

echo "==> Loading image into kind cluster (no registry push needed)"
kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"

echo "==> Applying manifests"
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/configmap.yaml"

# Create secret from env var or generate a random one for local dev
API_KEY="${API_KEY:-$(openssl rand -hex 32)}"
kubectl create secret generic inference-api-secret \
  --namespace infergrid \
  --from-literal=API_KEY="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl apply -f "$K8S_DIR/service-nodeport.yaml"

echo "==> Waiting for rollout..."
kubectl rollout status deployment/inference-api -n infergrid --timeout=120s

echo ""
echo "  Done. Inference API available at http://localhost:30080"
echo "    Test it: curl http://localhost:30080/health"
echo "    Logs:    kubectl logs -n infergrid -l app=inference-api -f"