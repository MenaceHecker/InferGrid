"""
drift_detector/register_baseline.py

To run this once after deploying a new model version to record the
baseline confidence distribution. The drift detector compares all
future live distributions against this baseline.

Usage:
    python register_baseline.py \
        --endpoint http://35.255.145.27 \
        --samples 1000 \
        --output /app/baseline.npy

The script sends real inference requests to the live endpoint,
collects the confidence scores, and saves them as a numpy array.
"""

import argparse
import json
import time

import numpy as np
import requests

# 20 Newsgroups sample texts: representative of normal traffic
SAMPLE_TEXTS = [
    "NASA launched a new rocket toward the International Space Station today.",
    "The hockey team scored three goals in the final period to win.",
    "Congress is debating new gun control legislation this week.",
    "Scientists discovered a new exoplanet in a nearby star system.",
    "The stock market fell sharply on concerns about inflation.",
    "A new graphics card was released with improved ray tracing performance.",
    "The motorcycle race at the track was postponed due to rain.",
    "Researchers published findings on a potential new cancer treatment.",
    "The baseball team won the championship after a 12-inning game.",
    "An atheist group filed a lawsuit challenging a local ordinance.",
    "The new operating system update includes security patches.",
    "Astronomers observed a rare alignment of planets this weekend.",
    "The car manufacturer recalled vehicles due to a brake defect.",
    "A new encryption algorithm was proposed by cryptographers.",
    "The religious organization announced plans for a new community center.",
]


def collect_baseline(
    endpoint: str,
    n_samples: int,
    api_key: str | None,
) -> np.ndarray:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    scores = []
    texts = SAMPLE_TEXTS * (n_samples // len(SAMPLE_TEXTS) + 1)

    print(f"Collecting {n_samples} baseline samples from {endpoint}/predict ...")
    for i, text in enumerate(texts[:n_samples]):
        try:
            resp = requests.post(
                f"{endpoint}/predict",
                headers=headers,
                json={"text": text},
                timeout=5,
            )
            resp.raise_for_status()
            confidence = resp.json()["confidence"]
            scores.append(float(confidence))

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_samples} collected...")

        except requests.RequestException as exc:
            print(f"  WARNING: request {i} failed: {exc}")

        time.sleep(0.05)  # 50ms between requests so don't hammer the API

    return np.array(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record baseline confidence distribution")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", default="baseline.npy")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    baseline = collect_baseline(args.endpoint, args.samples, args.api_key)

    if len(baseline) < 30:
        print(f"ERROR: Only collected {len(baseline)} samples. Need at least 30.")
        raise SystemExit(1)

    np.save(args.output, baseline)
    print(f"\nBaseline saved: {args.output}")
    print(f"  Samples : {len(baseline)}")
    print(f"  Mean    : {baseline.mean():.4f}")
    print(f"  Std     : {baseline.std():.4f}")
    print(f"  Min/Max : {baseline.min():.4f} / {baseline.max():.4f}")


if __name__ == "__main__":
    main()
