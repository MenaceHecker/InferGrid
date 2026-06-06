"""
inference-api/app/producer.py

Kafka producer for enqueuing async inference requests.
Used by /predict when concurrent request count exceeds MAX_CONCURRENT_REQUESTS.

Each message is a JSON-encoded dict:
  {
    "job_id":      str,   # UUID for result lookup
    "text":        str,   # input text to classify
    "model_version": str, # "model_a" | "model_b" | "primary"
    "enqueued_at": float  # unix timestamp
  }

Topic routing:
  model_a or primary → infergrid.model-a.requests
  model_b            → infergrid.model-b.requests
"""

import json
import logging
import os
import time

from confluent_kafka import Producer as KafkaProducer

log = logging.getLogger(__name__)

KAFKA_BROKER = os.environ.get(
    "KAFKA_BROKER",
    "kafka.infergrid.svc.cluster.local:9092",
)

TOPIC_MODEL_A = "infergrid.model-a.requests"
TOPIC_MODEL_B = "infergrid.model-b.requests"

_producer: KafkaProducer | None = None


def get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        _producer = KafkaProducer({
            "bootstrap.servers": KAFKA_BROKER,
            "acks": "1",                  # leader ack only so low latency
            "retries": 3,
            "retry.backoff.ms": 100,
            "delivery.timeout.ms": 5000,
        })
    return _producer


def _topic_for(model_version: str) -> str:
    return TOPIC_MODEL_B if model_version == "model_b" else TOPIC_MODEL_A


def _delivery_callback(err, msg) -> None:  # type: ignore[type-arg]
    if err:
        log.error("Kafka delivery failed: %s", err)
    else:
        log.debug(
            "Delivered job %s to %s [%d]",
            msg.key().decode(),
            msg.topic(),
            msg.partition(),
        )


def enqueue_job(job_id: str, text: str, model_version: str) -> str:
    """
    Publish an inference job to the appropriate Kafka topic.
    Returns the topic the message was sent to.
    Raises RuntimeError if the producer cannot deliver within timeout.
    """
    topic = _topic_for(model_version)
    payload = json.dumps({
        "job_id": job_id,
        "text": text,
        "model_version": model_version,
        "enqueued_at": time.time(),
    })

    producer = get_producer()
    producer.produce(
        topic,
        key=job_id,
        value=payload,
        callback=_delivery_callback,
    )
    producer.poll(0)  # trigger delivery callbacks without blocking
    return topic