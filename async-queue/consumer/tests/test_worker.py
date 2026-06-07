"""
Tests for the Kafka consumer worker.
No real Kafka or Redis so all external calls are monkeypatched.
"""

import json
import time

import pytest
from async_queue.consumer.worker import (
    RESULT_TTL,
    process_message,
    result_key,
    write_result,
)

# Fixtures


class MockModel:
    def predict(self, text: str) -> tuple[str, float]:
        return "sci.space", 0.91


class BrokenModel:
    def predict(self, text: str) -> tuple[str, float]:
        raise RuntimeError("ONNX runtime failure")


class MockRedis:
    def __init__(self):
        self.store: dict[str, tuple[str, int]] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = (value, ttl)

    def get(self, key: str) -> str | None:
        entry = self.store.get(key)
        return entry[0] if entry else None

    def ping(self) -> bool:
        return True


# result_key


def test_result_key_format():
    key = result_key("abc-123")
    assert key == "job:abc-123:result"


def test_result_key_includes_job_id():
    job_id = "test-job-id"
    assert job_id in result_key(job_id)


# write_result


def test_write_result_stores_in_redis():
    r = MockRedis()
    write_result(r, "job-1", "sci.space", 0.91, "model_a", 42.0)
    assert result_key("job-1") in r.store


def test_write_result_correct_ttl():
    r = MockRedis()
    write_result(r, "job-1", "sci.space", 0.91, "model_a", 42.0)
    _, ttl = r.store[result_key("job-1")]
    assert ttl == RESULT_TTL


def test_write_result_payload_shape():
    r = MockRedis()
    write_result(r, "job-1", "sci.space", 0.91, "model_a", 42.0)
    payload = json.loads(r.store[result_key("job-1")][0])
    assert payload["job_id"] == "job-1"
    assert payload["prediction"] == "sci.space"
    assert payload["confidence"] == pytest.approx(0.91)
    assert payload["model_version"] == "model_a"
    assert payload["elapsed_ms"] == pytest.approx(42.0)
    assert "completed_at" in payload


# process_message happy path


def test_process_message_writes_result():
    r = MockRedis()
    msg = json.dumps(
        {
            "job_id": "job-abc",
            "text": "NASA launched a rocket.",
            "model_version": "model_a",
            "enqueued_at": time.time(),
        }
    ).encode()
    process_message(msg, MockModel(), r)
    assert result_key("job-abc") in r.store


def test_process_message_correct_prediction():
    r = MockRedis()
    msg = json.dumps(
        {
            "job_id": "job-abc",
            "text": "NASA launched a rocket.",
            "model_version": "model_a",
            "enqueued_at": time.time(),
        }
    ).encode()
    process_message(msg, MockModel(), r)
    payload = json.loads(r.store[result_key("job-abc")][0])
    assert payload["prediction"] == "sci.space"
    assert payload["confidence"] == pytest.approx(0.91)


def test_process_message_default_model_version():
    r = MockRedis()
    # No model_version in message so should default to "primary"
    msg = json.dumps(
        {
            "job_id": "job-xyz",
            "text": "Some text",
        }
    ).encode()
    process_message(msg, MockModel(), r)
    payload = json.loads(r.store[result_key("job-xyz")][0])
    assert payload["model_version"] == "primary"


# process_message error cases


def test_process_message_malformed_json_does_not_raise():
    r = MockRedis()
    process_message(b"not valid json", MockModel(), r)
    assert len(r.store) == 0


def test_process_message_missing_job_id_does_not_raise():
    r = MockRedis()
    msg = json.dumps({"text": "some text"}).encode()
    process_message(msg, MockModel(), r)
    assert len(r.store) == 0


def test_process_message_missing_text_does_not_raise():
    r = MockRedis()
    msg = json.dumps({"job_id": "job-1"}).encode()
    process_message(msg, MockModel(), r)
    assert len(r.store) == 0


def test_process_message_inference_failure_writes_error_result():
    r = MockRedis()
    msg = json.dumps(
        {
            "job_id": "job-broken",
            "text": "Some text",
            "model_version": "model_a",
        }
    ).encode()
    process_message(msg, BrokenModel(), r)
    # Error result should still be written so the WebSocket endpoint can return it
    assert result_key("job-broken") in r.store
    payload = json.loads(r.store[result_key("job-broken")][0])
    assert "error" in payload
    assert payload["job_id"] == "job-broken"
