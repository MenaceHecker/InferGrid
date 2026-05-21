#!/usr/bin/env bash
# observability/grafana/build-dashboard-configmap.sh
# Packages the three dashboard JSON files into a Kubernetes ConfigMap
# so Grafana loads them automatically on startup via the provisioning provider.
#
# Run from repo root: ./observability/grafana/build-dashboard-configmap.sh

set -euo pipefail

GRAFANA_DIR="observability/grafana"

echo "==> Building grafana-dashboards ConfigMap"
kubectl create configmap grafana-dashboards \
  --namespace monitoring \
  --from-file="system-health.json=${GRAFANA_DIR}/system-health.json" \
  --from-file="model-performance.json=${GRAFANA_DIR}/model-performance.json" \
  --from-file="drift-monitor.json=${GRAFANA_DIR}/drift-monitor.json" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Applying Grafana deployment"
kubectl apply -f "${GRAFANA_DIR}/grafana.yaml"

echo "==> Waiting for Grafana rollout"
kubectl rollout status deployment/grafana -n monitoring --timeout=120s

echo ""
echo "✅  Grafana is live."
echo ""
echo "To open the UI:"
echo "  kubectl port-forward -n monitoring svc/grafana 3000:3000"
echo "  open http://localhost:3000"
echo "  Login: admin / infergrid"
echo ""
echo "Dashboards are pre-loaded under the 'InferGrid' folder."