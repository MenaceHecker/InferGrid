#!/usr/bin/env bash
# benchmarks/run.sh
# Run the full benchmark suite: sync path at 10/50/100/200 RPS,
# then async path at 200 concurrent users.
# Results saved to benchmarks/results/ as CSV files.
#
# Usage: ENDPOINT=http://35.255.145.27 ./benchmarks/run.sh

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://35.255.145.27}"
RESULTS_DIR="benchmarks/results"
LOCUSTFILE="benchmarks/locustfile.py"
RUN_TIME="${RUN_TIME:-60s}"

mkdir -p "$RESULTS_DIR"

echo "=============================="
echo " InferGrid Benchmark Suite"
echo " Endpoint: $ENDPOINT"
echo " Run time: $RUN_TIME per level"
echo "=============================="

run_sync() {
  local users=$1
  local spawn_rate=$2
  local label="sync_u${users}"
  echo ""
  echo "==> Sync path: $users users (spawn rate $spawn_rate/s)"
  locust -f "$LOCUSTFILE" SyncUser \
    --headless \
    --users "$users" \
    --spawn-rate "$spawn_rate" \
    --host "$ENDPOINT" \
    --run-time "$RUN_TIME" \
    --csv "${RESULTS_DIR}/${label}" \
    --html "${RESULTS_DIR}/${label}.html" \
    2>&1 | tail -5
  echo "    Results: ${RESULTS_DIR}/${label}_stats.csv"
}

run_async() {
  local users=$1
  local spawn_rate=$2
  local label="async_u${users}"
  echo ""
  echo "==> Async path: $users users (spawn rate $spawn_rate/s)"
  locust -f "$LOCUSTFILE" AsyncUser \
    --headless \
    --users "$users" \
    --spawn-rate "$spawn_rate" \
    --host "$ENDPOINT" \
    --run-time "$RUN_TIME" \
    --csv "${RESULTS_DIR}/${label}" \
    --html "${RESULTS_DIR}/${label}.html" \
    2>&1 | tail -5
  echo "    Results: ${RESULTS_DIR}/${label}_stats.csv"
}

# Sync path at increasing load
run_sync 10 5
run_sync 50 10
run_sync 100 20
run_sync 200 40

# Async path at peak load
run_async 200 40

echo ""
echo "=============================="
echo " All benchmarks complete."
echo " Generating results.md ..."
echo "=============================="

python3 - << 'PYEOF'
import csv
import glob
import os

results_dir = "benchmarks/results"
rows = []

for stats_file in sorted(glob.glob(f"{results_dir}/*_stats.csv")):
    label = os.path.basename(stats_file).replace("_stats.csv", "")
    with open(stats_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") in ("/predict [sync]", "/predict [async]"):
                rows.append({
                    "label": label,
                    "endpoint": row["Name"],
                    "requests": row["Request Count"],
                    "failures": row["Failure Count"],
                    "p50": row["50%"],
                    "p95": row["95%"],
                    "p99": row["99%"],
                    "rps": row["Requests/s"],
                })

with open("benchmarks/results.md", "w") as f:
    f.write("# InferGrid Benchmark Results\n\n")
    f.write("Endpoint: `http://35.255.145.27`  \n")
    f.write("Model: sklearn TF-IDF + LogReg (20 Newsgroups)\n\n")
    f.write("| Scenario | Endpoint | Users | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | RPS |\n")
    f.write("|----------|----------|-------|----------|----------|----------|----------|----------|-----|\n")
    for r in rows:
        f.write(
            f"| {r['label']} | {r['endpoint']} | - | {r['requests']} | "
            f"{r['failures']} | {r['p50']} | {r['p95']} | {r['p99']} | {r['rps']} |\n"
        )

print("Written: benchmarks/results.md")
PYEOF

cat benchmarks/results.md