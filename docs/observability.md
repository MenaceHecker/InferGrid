# Observability

InferGrid's observability stack runs in the `monitoring` namespace on GKE and
consists of three components: a Prometheus server that scrapes inference metrics,
a Grafana instance with three pre-provisioned dashboards, and a standalone drift
detector worker that runs a KS-test every 60 seconds.

## Architecture

```
inference-api pods
  └── /metrics (prometheus_client text format)
        └── scraped every 15s by Prometheus
              └── queried by Grafana (3 dashboards)
              └── queried by drift-detector (every 60s)
                    └── emits model_drift_ks_statistic → Prometheus
                          └── alert rule fires if KS > 0.15
```

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests by method, endpoint, and status code. Used by HPA via Prometheus Adapter. |
| `request_latency_seconds` | Histogram | End-to-end latency per endpoint. Buckets sized for p50/p95/p99 resolution. |
| `prediction_confidence` | Histogram | Distribution of model confidence scores (0.0–1.0). A healthy model clusters near 1.0. |
| `model_load_time_seconds` | Gauge | Startup time to load the model. Set once at process start. |
| `model_drift_ks_statistic` | Gauge | KS-test statistic comparing live vs baseline confidence distribution. Emitted by the drift detector. |
| `model_drift_window_size` | Gauge | Number of confidence samples in the current drift detection window. |

## Dashboards

Three dashboards are pre-provisioned in Grafana under the **InferGrid** folder.
Dashboard JSON exports are committed to `observability/grafana/` and loaded
automatically via the Grafana provisioning ConfigMap.

### Dashboard 1 — System Health

**Purpose:** First stop during an incident. Answers: is the service up, how much
traffic is it handling, and is anything broken?

| Panel | Query | Why it matters |
|-------|-------|----------------|
| Request Rate | `sum(rate(http_requests_total[2m]))` | Baseline traffic signal. Sudden drop = something is wrong upstream. |
| Error Rate | 5xx rate / total rate | Threshold alerts at 1% (yellow) and 5% (red). |
| Pod Count | `count(up{app='inference-api'} == 1)` | Confirms HPA is scaling as expected under load. |
| CPU per pod | `container_cpu_usage_seconds_total` | Identifies hot pods before they hit limits. |
| Memory per pod | `container_memory_working_set_bytes` | Catches memory leaks in long-running inference processes. |

### Dashboard 2 — Model Performance

**Purpose:** Answers whether the model is serving correctly and within latency
budget, independent of infrastructure health.

| Panel | Query | Why it matters |
|-------|-------|----------------|
| Latency p50/p95/p99 | `histogram_quantile` on `request_latency_seconds_bucket` | p95 above 500ms triggers the readiness probe and the HighLatency alert. |
| p95 Gauge | Same query, stat panel | At-a-glance SLO indicator. Green < 100ms, yellow < 500ms, red above. |
| Throughput | `rate(http_requests_total{status_code='200'}[2m])` | Successful predictions per second. Compare against RPS to see error ratio. |
| Confidence Distribution | `rate(prediction_confidence_bucket[5m])` | A healthy model peaks in the 0.8–1.0 buckets. Spreading toward lower buckets is an early drift signal before the KS-test fires. |
| Model Load Time | `model_load_time_seconds` | Cold-start cost. Relevant for understanding pod startup latency during HPA scale-up. |

### Dashboard 3 — Drift Monitor

**Purpose:** Detects when the model's output distribution has shifted from the
baseline recorded at registration time. This catches silent failures — cases
where the model is running but producing systematically different predictions
than it did in evaluation.

| Panel | Query | Why it matters |
|-------|-------|----------------|
| KS-Statistic over time | `model_drift_ks_statistic` | The primary signal. Values above 0.15 indicate the live distribution is significantly different from baseline. |
| Drift threshold line | `vector(0.15)` | Dashed red reference line. Crossing it triggers the `ModelDriftDetected` alert. |
| Drift Status badge | Threshold mappings on KS gauge | NORMAL / WARNING / DRIFT DETECTED. Green below 0.10, yellow 0.10–0.15, red above. |
| Confidence Distribution | `increase(prediction_confidence_bucket[5m])` | Bar chart of the live distribution. Compare visually against expected baseline shape. |
| Drift Events | `model_drift_ks_statistic > bool 0.15` | Binary signal: 1 when drifting, 0 when normal. Useful for counting drift events over a shift. |

## Drift Detection

The drift detector (`observability/drift_detector/detector.py`) is a standalone
Python worker that runs every 60 seconds. It:

1. Pulls the last 3600 seconds of `prediction_confidence` observations from
   Prometheus (up to 500 samples).
2. Runs `scipy.stats.ks_2samp` against the baseline distribution recorded at
   model registration time.
3. Emits `model_drift_ks_statistic` and `model_drift_window_size` on its own
   `/metrics` endpoint (port 9091), scraped by Prometheus.

**Baseline registration** — run once after deploying a new model version:

```bash
python observability/drift_detector/register_baseline.py \
  --endpoint http://35.255.145.27 \
  --samples 500 \
  --output baseline.npy

kubectl create configmap drift-baseline \
  --namespace monitoring \
  --from-file=baseline.npy=baseline.npy \
  --dry-run=client -o yaml | kubectl apply -f -
```

The KS-statistic threshold of 0.15 was chosen based on the 20 Newsgroups
dataset characteristics. A value of 0.0 means identical distributions; 1.0
means completely non-overlapping. In practice, natural variance in the
production traffic keeps the statistic below 0.05 on a healthy model.

## Accessing Grafana

```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
open http://localhost:3000
# Login: admin / infergrid
```

## Alert Rules

Three alert rules are defined in `observability/prometheus/prometheus.yaml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HighErrorRate` | 5xx rate > 5% for 2 minutes | warning |
| `HighLatency` | p95 latency > 500ms for 2 minutes | warning |
| `ModelDriftDetected` | KS-statistic > 0.15 for 1 minute | critical |