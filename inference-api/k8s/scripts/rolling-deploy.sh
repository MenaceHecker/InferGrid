set -euo pipefail

PROJECT_ID="${PROJECT_ID:-infergrid-prod}"
REGION="${REGION:-us-central1}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/infergrid"
IMAGE="${REGISTRY}/inference-api:${IMAGE_TAG}"
ENDPOINT="${ENDPOINT:-http://35.255.145.27}"   # replace with your static IP
GCLOUD_BIN_DIR="$(gcloud info --format='value(installation.sdk_root)' 2>/dev/null)/bin"

if [[ -d "$GCLOUD_BIN_DIR" ]]; then
  export PATH="$GCLOUD_BIN_DIR:$PATH"
fi

for command in gcloud docker kubectl gke-gcloud-auth-plugin curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    echo "Install it, then rerun this script." >&2
    exit 1
  fi
done

echo "==> Rolling deploy: $IMAGE"
echo "    Endpoint under test: $ENDPOINT"

# 1. Build and push the new image

echo "==> Building and pushing image"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker buildx build \
  --platform linux/amd64 \
  --tag "$IMAGE" \
  --push \
  inference-api/

# 2. Start background health checker so logs any non-200 during rollout

HEALTH_LOG=$(mktemp)
echo "==> Starting continuous health checks (logging to $HEALTH_LOG)"

(
  FAILURES=0
  CHECKS=0
  while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}/health" 2>/dev/null || echo "000")
    CHECKS=$((CHECKS + 1))
    if [ "$STATUS" != "200" ]; then
      FAILURES=$((FAILURES + 1))
      echo "[$(date -u +%H:%M:%S)] FAIL status=$STATUS (failure $FAILURES)" | tee -a "$HEALTH_LOG"
    else
      echo "[$(date -u +%H:%M:%S)] OK $STATUS (check $CHECKS)" >> "$HEALTH_LOG"
    fi
    sleep 0.5
  done
) &
CHECKER_PID=$!
cleanup() {
  kill "$CHECKER_PID" 2>/dev/null || true
  wait "$CHECKER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 3. Trigger the rolling update

echo "==> Updating image in deployment"
kubectl set image deployment/inference-api \
  inference-api="$IMAGE" \
  --namespace infergrid

echo "==> Waiting for rollout to complete"
kubectl rollout status deployment/inference-api \
  --namespace infergrid \
  --timeout=180s

# 4. Stop health checker and report results

cleanup
trap - EXIT

TOTAL_CHECKS=$(wc -l < "$HEALTH_LOG" | tr -d ' ')
FAILURES=$(grep -c "FAIL" "$HEALTH_LOG" 2>/dev/null || true)

echo ""
echo "=============================="
echo "  Rolling deploy complete"
echo "=============================="
echo "  Image:         $IMAGE"
echo "  Total checks:  $TOTAL_CHECKS"
echo "  Failures:       $FAILURES"
echo "  Zero downtime: $([ "$FAILURES" -eq 0 ] && echo YES || echo NO)"
echo ""

if [ "$FAILURES" -gt 0 ]; then
  echo "  Failed requests:"
  grep "FAIL" "$HEALTH_LOG"
fi

rm -f "$HEALTH_LOG"

# Fail the script if any requests dropped during rollout
[ "$FAILURES" -eq 0 ]
