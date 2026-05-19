#!/usr/bin/env bash
# observability/prometheus/deploy.sh
# To deploy Prometheus to GKE and validate inference-api metrics are flowing.
#
# To run from repo root: ./observability/prometheus/deploy.sh

set -euo pipefail

NAMESPACE="monitoring"
INFERENCE_ENDPOINT="${ENDPOINT:-http://35.255.145.27}"

echo "==> Applying Prometheus manifests"
kubectl apply -f observability/prometheus/prometheus.yaml

echo "==> Waiting for Prometheus rollout"
kubectl rollout status deployment/prometheus -n "$NAMESPACE" --timeout=120s

echo "==> Generating some inference traffic so metrics are populated"
for i in $(seq 1 10); do
  curl -s -o /dev/null -X POST "${INFERENCE_ENDPOINT}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "NASA launched a new rocket toward the ISS today."}'
done
echo "    10 requests sent."

echo "==> Port-forwarding Prometheus to localhost:9090 for validation"
echo "    (runs for 30 seconds, then exits)"
kubectl port-forward -n "$NAMESPACE" svc/prometheus 9090:9090 &
PF_PID=$!
sleep 3   # wait for port-forward to be ready

echo "==> Checking http_requests_total is visible in Prometheus"
RESULT=$(curl -s "http://localhost:9090/api/v1/query" \
  --data-urlencode 'query=http_requests_total' | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(len(d['data']['result']))")

kill "$PF_PID" 2>/dev/null || true

if [ "$RESULT" -gt 0 ]; then
  echo "  Prometheus is scraping inference-api. Found $RESULT time series."
else
  echo "  No http_requests_total series found."
  echo "  Check pod annotations and scrape config."
  exit 1
fi

echo ""
echo "To browse Prometheus UI:"
echo "  kubectl port-forward -n monitoring svc/prometheus 9090:9090"
echo "  open http://localhost:9090"