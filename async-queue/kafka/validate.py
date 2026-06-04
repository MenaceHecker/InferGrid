"""
async-queue/kafka/validate.py
Validates Kafka is working by producing a message to each topic
and consuming it back. Run after deploy.sh succeeds.

Usage:
  KAFKA_BROKER=localhost:9092 python async-queue/kafka/validate.py
  (port-forward first: kubectl port-forward -n infergrid svc/kafka 9092:9092)
"""

import json
import os
import uuid

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient

BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPICS = ["infergrid.model-a.requests", "infergrid.model-b.requests"]


def check_topics_exist() -> None:
    admin = AdminClient({"bootstrap.servers": BROKER})
    metadata = admin.list_topics(timeout=10)
    existing = set(metadata.topics.keys())
    for topic in TOPICS:
        status = "OK" if topic in existing else "ERROR"
        print(f"  {status} {topic}")
    missing = [t for t in TOPICS if t not in existing]
    if missing:
        raise SystemExit(f"Missing topics: {missing}")


def produce_and_consume(topic: str) -> None:
    job_id = str(uuid.uuid4())
    message = json.dumps({"job_id": job_id, "text": "NASA launched a rocket today."})

    # Produce
    producer = Producer({"bootstrap.servers": BROKER})
    producer.produce(topic, key=job_id, value=message)
    producer.flush(timeout=5)

    # Consume
    consumer = Consumer({
        "bootstrap.servers": BROKER,
        "group.id": f"validate-{uuid.uuid4()}",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([topic])

    for _ in range(20):  # poll up to 20 times
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            raise RuntimeError(f"Consumer error: {msg.error()}")
        received = json.loads(msg.value())
        assert received["job_id"] == job_id, "job_id mismatch"
        consumer.close()
        return

    consumer.close()
    raise RuntimeError(f"Did not receive message from {topic} within timeout")


def main() -> None:
    print(f"==> Connecting to Kafka at {BROKER}")

    print("==> Checking topics exist")
    check_topics_exist()

    for topic in TOPICS:
        print(f"==> Testing produce/consume on {topic}")
        produce_and_consume(topic)
        print(f"      Round-trip OK")

    print("\n  Kafka validation passed. Both topics are healthy.")


if __name__ == "__main__":
    main()