#!/usr/bin/env bash

set -euo pipefail
 
TRACKER_URL="${TRACKER_URL:-http://localhost:8001}"
RUN_ID="${RUN_ID:-$(date +%s)-$$}"
DATASET_HASH=$(echo -n "20newsgroups-test-v1" | sha256sum | awk '{print $1}')
 
echo "==> Registering model A (newsgroups-sklearn-v1)"
MODEL_A=$(curl -sf -X POST "${TRACKER_URL}/models/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"newsgroups-sklearn-v1\",
    \"version_hash\": \"$(echo -n "sklearn-v1-${RUN_ID}" | sha256sum | awk '{print $1}')\",
    \"status\": \"active\"
  }")
MODEL_A_ID=$(echo "$MODEL_A" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "    model_a id=$MODEL_A_ID"
 
echo "==> Registering model B (newsgroups-onnx-v1)"
MODEL_B=$(curl -sf -X POST "${TRACKER_URL}/models/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"newsgroups-onnx-v1\",
    \"version_hash\": \"$(echo -n "onnx-v1-${RUN_ID}" | sha256sum | awk '{print $1}')\",
    \"status\": \"staging\"
  }")
MODEL_B_ID=$(echo "$MODEL_B" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "    model_b id=$MODEL_B_ID"
 
echo "==> Seeding evaluation metrics for model A"
for metric in "accuracy:0.912" "f1:0.897" "latency_p95_ms:42.3"; do
  NAME="${metric%%:*}"
  VALUE="${metric##*:}"
  curl -sf -X POST "${TRACKER_URL}/models/${MODEL_A_ID}/evaluations" \
    -H "Content-Type: application/json" \
    -d "{\"metric_name\": \"${NAME}\", \"metric_value\": ${VALUE}, \"dataset_hash\": \"${DATASET_HASH}\"}" \
    > /dev/null
done
echo "    accuracy=0.912, f1=0.897, latency_p95_ms=42.3"
 
echo "==> Seeding evaluation metrics for model B"
for metric in "accuracy:0.934" "f1:0.921" "latency_p95_ms:38.7"; do
  NAME="${metric%%:*}"
  VALUE="${metric##*:}"
  curl -sf -X POST "${TRACKER_URL}/models/${MODEL_B_ID}/evaluations" \
    -H "Content-Type: application/json" \
    -d "{\"metric_name\": \"${NAME}\", \"metric_value\": ${VALUE}, \"dataset_hash\": \"${DATASET_HASH}\"}" \
    > /dev/null
done
echo "    accuracy=0.934, f1=0.921, latency_p95_ms=38.7"
 
echo "==> Configuring A/B split (70% model A, 30% model B)"
curl -sf -X POST "${TRACKER_URL}/ab/configure" \
  -H "Content-Type: application/json" \
  -d "{\"model_a_id\": ${MODEL_A_ID}, \"model_b_id\": ${MODEL_B_ID}, \"split_weight\": 0.7}" \
  > /dev/null
 
echo "==> Calling /ab/compare"
COMPARE=$(curl -sf "${TRACKER_URL}/ab/compare")
echo "$COMPARE" | python3 -m json.tool
 
echo ""
echo "==> Validating results"
python3 - << PYEOF
import json, sys
 
data = json.loads('''${COMPARE}''')
 
assert data["split_weight"] == 0.7, f"Expected split_weight=0.7, got {data['split_weight']}"
assert data["model_a"]["metrics"]["accuracy"] == 0.912, "model_a accuracy mismatch"
assert data["model_b"]["metrics"]["accuracy"] == 0.934, "model_b accuracy mismatch"
assert data["model_b"]["metrics"]["accuracy"] > data["model_a"]["metrics"]["accuracy"], \
    "model_b should have higher accuracy"
 
print("All assertions passed.")
print(f"    model_a accuracy: {data['model_a']['metrics']['accuracy']}")
print(f"    model_b accuracy: {data['model_b']['metrics']['accuracy']}")
print(f"    Winner by accuracy: model_b (+{data['model_b']['metrics']['accuracy'] - data['model_a']['metrics']['accuracy']:.3f})")
PYEOF
