"""
benchmarks/locustfile.py

Load test for InferGrid inference API covering both sync and async paths.

Usage:
  # Headless — sync path only
  locust -f benchmarks/locustfile.py SyncUser \
    --headless -u 50 -r 10 \
    --host http://35.255.145.27 \
    --run-time 60s \
    --csv benchmarks/results/sync

  # Headless — async path (forces enqueue by saturating concurrency)
  locust -f benchmarks/locustfile.py AsyncUser \
    --headless -u 200 -r 20 \
    --host http://35.255.145.27 \
    --run-time 60s \
    --csv benchmarks/results/async

  # Both user classes together
  locust -f benchmarks/locustfile.py \
    --headless -u 200 -r 20 \
    --host http://35.255.145.27 \
    --run-time 60s \
    --csv benchmarks/results/combined
"""

import random
import time

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

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
    "The new operating system update includes important security patches.",
    "Astronomers observed a rare alignment of planets this weekend.",
    "The car manufacturer recalled vehicles due to a brake defect.",
    "A new encryption algorithm was proposed by cryptographers.",
    "The religious organization announced plans for a new community center.",
]

# Track async job results for latency measurement
_async_results: dict[str, float] = {}  # job_id -> enqueue_time


class SyncUser(HttpUser):
    """
    Simulates a client using the synchronous /predict path.
    Represents normal traffic below the MAX_CONCURRENT_REQUESTS threshold.
    """

    wait_time = between(0.1, 0.5)
    weight = 1

    @task
    def predict(self) -> None:
        text = random.choice(SAMPLE_TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="/predict [sync]",
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "prediction" not in body:
                    response.failure("Missing prediction in response")
            elif response.status_code == 200 and "job_id" in response.json():
                # Unexpectedly got async, will still mark success
                response.success()
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")


class AsyncUser(HttpUser):
    """
    Simulates high-concurrency traffic that triggers the async queue path.
    Sends predict requests rapidly to saturate MAX_CONCURRENT_REQUESTS,
    then polls for results via the HTTP fallback endpoint.
    """

    wait_time = between(0.05, 0.1)   # faster than SyncUser to force enqueue
    weight = 3

    @task(4)
    def predict_async(self) -> None:
        text = random.choice(SAMPLE_TEXTS)
        enqueue_start = time.monotonic()

        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="/predict [async]",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}")
                return

            body = response.json()

            # Sync response : record as success
            if "prediction" in body:
                response.success()
                return

            # Async response : poll for result
            job_id = body.get("job_id")
            if not job_id:
                response.failure("No job_id in async response")
                return

            response.success()
            _async_results[job_id] = enqueue_start

    @task(1)
    def poll_result(self) -> None:
        """Poll a random pending job result."""
        if not _async_results:
            return

        job_id = random.choice(list(_async_results.keys()))
        enqueue_time = _async_results[job_id]

        with self.client.get(
            f"/result/{job_id}",
            catch_response=True,
            name="/result/{job_id} [poll]",
        ) as response:
            if response.status_code == 200:
                total_ms = (time.monotonic() - enqueue_time) * 1000
                _async_results.pop(job_id, None)
                response.success()
            elif response.status_code == 404:
                # Still pending — check if we've been waiting too long
                waited_s = time.monotonic() - enqueue_time
                if waited_s > 30:
                    _async_results.pop(job_id, None)
                    response.failure(f"Job {job_id} timed out after {waited_s:.1f}s")
                else:
                    response.success()  # still pending, not a failure
            else:
                response.failure(f"Unexpected status {response.status_code}")

# Custom event so it would print summary at test end


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:  # type: ignore[type-arg]
    stats = environment.stats
    print("\n" + "=" * 60)
    print("INFERGRID LOAD TEST SUMMARY")
    print("=" * 60)

    for name, entry in stats.entries.items():
        if entry.num_requests == 0:
            continue
        print(f"\n  {name[1]}")
        print(f"    Requests : {entry.num_requests}")
        print(f"    Failures : {entry.num_failures}")
        print(f"    p50      : {entry.get_response_time_percentile(0.50):.0f}ms")
        print(f"    p95      : {entry.get_response_time_percentile(0.95):.0f}ms")
        print(f"    p99      : {entry.get_response_time_percentile(0.99):.0f}ms")
        print(f"    RPS      : {entry.current_rps:.1f}")

    pending = len(_async_results)
    if pending:
        print(f"\n  NOTE: {pending} async jobs still pending at test end")
    print("=" * 60)