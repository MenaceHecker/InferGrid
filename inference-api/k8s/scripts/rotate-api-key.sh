#!/usr/bin/env bash
# inference-api/k8s/scripts/rotate-api-key.sh
# Generate a new API key and apply it to the Kubernetes Secret.
# The deployment is restarted automatically to pick up the new value.
#
# Usage:
#   ./inference-api/k8s/scripts/rotate-api-key.sh
#   API_KEY=my-own-key ./inference-api/k8s/scripts/rotate-api-key.sh

set -euo pipefail

NAMESPACE="infergrid"
SECRET_NAME="inference-api-secret"

API_KEY="${API_KEY:-$(openssl rand -hex 32)}"

echo "==> Applying new API key to secret: $SECRET_NAME"
kubectl create secret generic "$SECRET_NAME" \
  --namespace "$NAMESPACE" \
  --from-literal=API_KEY="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Restarting deployment to pick up new key"
kubectl rollout restart deployment/inference-api -n "$NAMESPACE"
kubectl rollout status deployment/inference-api -n "$NAMESPACE" --timeout=120s

echo ""
echo "Key rotated. Store this key securely, it will not be shown again:"
echo ""
echo "  X-API-Key: $API_KEY"
echo ""
echo "Test it:"
echo "  curl -H 'X-API-Key: $API_KEY' http://35.255.145.27/health"