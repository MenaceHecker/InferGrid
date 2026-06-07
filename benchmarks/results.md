# InferGrid Benchmark Results

Endpoint: `http://35.255.145.27`  
Model: sklearn TF-IDF + LogReg (20 Newsgroups)  
Run time: 60s per scenario  
Date: to be filled after running `benchmarks/run.sh`

## Sync Path

| Users | p50 (ms) | p95 (ms) | p99 (ms) | RPS | Error Rate |
|-------|----------|----------|----------|-----|------------|
| 10    | —        | —        | —        | —   | —          |
| 50    | —        | —        | —        | —   | —          |
| 100   | —        | —        | —        | —   | —          |
| 200   | —        | —        | —        | —   | —          |

## Async Path

| Users | p50 enqueue (ms) | p95 total (ms) | RPS | Queue depth peak |
|-------|-----------------|----------------|-----|-----------------|
| 200   | —               | —              | —   | —               |

## Notes

- p95 latency target: < 100ms (sync), < 500ms (async end-to-end)
- Run `benchmarks/run.sh` to populate this table with real numbers
- Results CSV files are in `benchmarks/results/` (gitignored)
- HTML reports are in `benchmarks/results/*.html`

## How to Run

```bash
pip install locust
ENDPOINT=http://35.255.145.27 ./benchmarks/run.sh
```