"""
async-queue/consumer/worker.py

Kafka consumer worker for async inference.

Pulls jobs from infergrid.model-a.requests and infergrid.model-b.requests,
runs inference using the same sklearn/ONNX model as the inference API,
and writes results to Redis at key job:{job_id}:result with a 5-minute TTL.

Environment variables:
  KAFKA_BROKER        Kafka bootstrap server (default: kafka.infergrid.svc.cluster.local:9092)
  KAFKA_GROUP_ID      Consumer group ID (default: infergrid-workers)
  REDIS_URL           Redis connection URL (default: redis://redis.infergrid.svc.cluster.local:6379)
  MODEL_PATH          Path to the model file (default: /app/models/classifier.pkl)
  RESULT_TTL_SECONDS  Redis TTL for job results (default: 300)
  WORKER_ID           Optional identifier for this worker instance
"""

import json
import logging
import os
import signal
import sys
import time
from typing import Any

import redis
from confluent_kafka import Consumer, KafkaError, KafkaException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker:%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# Configuration

KAFKA_BROKER = os.environ.get(
    "KAFKA_BROKER",
    "kafka.infergrid.svc.cluster.local:9092",
)
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "infergrid-workers")
REDIS_URL = os.environ.get(
    "REDIS_URL",
    "redis://redis.infergrid.svc.cluster.local:6379",
)
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/classifier.pkl")
RESULT_TTL = int(os.environ.get("RESULT_TTL_SECONDS", "300"))
WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")

TOPICS = [
    "infergrid.model-a.requests",
    "infergrid.model-b.requests",
]

# Model loading


def load_model(model_path: str) -> Any:
    """Load sklearn or ONNX model — same logic as inference API."""
    if model_path.endswith(".onnx"):
        from app.models.onnx_loader import OnnxLoader

        return OnnxLoader(model_path, classes=[])
    else:
        from app.models.sklearn_loader import SklearnLoader

        return SklearnLoader(model_path)


# ---------------------------------------------------------------------------
# Redis result writer
# ---------------------------------------------------------------------------

RESULT_KEY_PREFIX = "job:"
RESULT_KEY_SUFFIX = ":result"


def result_key(job_id: str) -> str:
    return f"{RESULT_KEY_PREFIX}{job_id}{RESULT_KEY_SUFFIX}"


def write_result(
    redis_client: redis.Redis,
    job_id: str,
    prediction: str,
    confidence: float,
    model_version: str,
    elapsed_ms: float,
) -> None:
    payload = json.dumps(
        {
            "job_id": job_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "elapsed_ms": round(elapsed_ms, 2),
            "completed_at": time.time(),
        }
    )
    redis_client.setex(result_key(job_id), RESULT_TTL, payload)
    log.debug("Wrote result for job %s (TTL=%ds)", job_id, RESULT_TTL)


# Job processing


def process_message(
    raw_value: bytes,
    model: Any,
    redis_client: redis.Redis,
) -> None:
    """Deserialize a Kafka message, run inference, write result to Redis."""
    try:
        job = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        log.error("Malformed message skipping: %s", exc)
        return

    job_id = job.get("job_id")
    text = job.get("text")
    model_version = job.get("model_version", "primary")

    if not job_id or not text:
        log.error("Message missing job_id or text skipping: %s", job)
        return

    start = time.monotonic()
    try:
        prediction, confidence = model.predict(text)
        elapsed_ms = (time.monotonic() - start) * 1000
        write_result(redis_client, job_id, prediction, confidence, model_version, elapsed_ms)
        log.info(
            "job=%s model_version=%s prediction=%s confidence=%.3f elapsed=%.1fms",
            job_id,
            model_version,
            prediction,
            confidence,
            elapsed_ms,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - start) * 1000
        # Write error result so the WebSocket endpoint can return it
        error_payload = json.dumps(
            {
                "job_id": job_id,
                "error": str(exc),
                "completed_at": time.time(),
            }
        )
        redis_client.setex(result_key(job_id), RESULT_TTL, error_payload)
        log.error("Inference failed for job %s: %s", job_id, exc)


# Consumer loop


def run_worker(model: Any, redis_client: redis.Redis) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "group.id": KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 1000,
            "session.timeout.ms": 10000,
        }
    )

    consumer.subscribe(TOPICS)
    log.info("[%s] Subscribed to %s via %s", WORKER_ID, TOPICS, KAFKA_BROKER)

    # Graceful shutdown on SIGTERM / SIGINT
    running = True

    def _shutdown(signum, frame):  # type: ignore[type-arg]
        nonlocal running
        log.info("[%s] Shutdown signal received", WORKER_ID)
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while running:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    log.debug("Reached end of partition %s", msg.partition())
                else:
                    raise KafkaException(msg.error())
                continue

            process_message(msg.value(), model, redis_client)

    finally:
        consumer.close()
        log.info("[%s] Consumer closed", WORKER_ID)


# Entrypoint


if __name__ == "__main__":
    log.info("[%s] Loading model from %s", WORKER_ID, MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        sys.exit(1)

    log.info("[%s] Connecting to Redis at %s", WORKER_ID, REDIS_URL)
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()

    run_worker(model, redis_client)
