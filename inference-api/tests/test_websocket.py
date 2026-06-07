"""
Tests for the WebSocket result endpoint and HTTP fallback.
Redis is monkeypatched so no real Redis connection needed.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

SAMPLE_RESULT = json.dumps(
    {
        "job_id": "test-job-123",
        "prediction": "sci.space",
        "confidence": 0.91,
        "model_version": "model_a",
        "elapsed_ms": 42.0,
        "completed_at": 1234567890.0,
    }
)


# HTTP fallback : GET /result/{job_id}


def test_http_result_returns_404_when_pending():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.aclose = AsyncMock()

    with patch("app.websocket._get_redis", return_value=mock_redis):
        response = client.get("/result/test-job-123")

    assert response.status_code == 404
    assert response.json()["status"] == "pending"


def test_http_result_returns_200_when_ready():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=SAMPLE_RESULT)
    mock_redis.aclose = AsyncMock()

    with patch("app.websocket._get_redis", return_value=mock_redis):
        response = client.get("/result/test-job-123")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "sci.space"
    assert body["job_id"] == "test-job-123"


def test_http_result_includes_confidence():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=SAMPLE_RESULT)
    mock_redis.aclose = AsyncMock()

    with patch("app.websocket._get_redis", return_value=mock_redis):
        body = client.get("/result/test-job-123").json()

    assert body["confidence"] == pytest.approx(0.91)


def test_http_result_404_body_includes_job_id():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.aclose = AsyncMock()

    with patch("app.websocket._get_redis", return_value=mock_redis):
        body = client.get("/result/missing-job").json()

    assert body["job_id"] == "missing-job"


# WebSocket endpoint


def test_websocket_sends_result_when_ready():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=SAMPLE_RESULT)
    mock_redis.aclose = AsyncMock()

    with (
        patch("app.websocket._get_redis", return_value=mock_redis),
        client.websocket_connect("/ws/result/test-job-123") as ws,
    ):
        data = ws.receive_json()

    assert data["prediction"] == "sci.space"
    assert data["job_id"] == "test-job-123"


def test_websocket_sends_pending_then_result():
    call_count = 0

    async def get_side_effect(key: str) -> str | None:
        nonlocal call_count
        call_count += 1
        # First call returns None (pending), second returns the result
        return None if call_count < 2 else SAMPLE_RESULT

    mock_redis = AsyncMock()
    mock_redis.get = get_side_effect
    mock_redis.aclose = AsyncMock()

    with (
        patch("app.websocket._get_redis", return_value=mock_redis),
        patch("app.websocket.POLL_INTERVAL", 0.0),
        client.websocket_connect("/ws/result/test-job-123") as ws,
    ):
        first = ws.receive_json()
        second = ws.receive_json()

    assert first["status"] == "pending"
    assert second["prediction"] == "sci.space"


def test_websocket_sends_timeout_when_result_never_arrives():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.aclose = AsyncMock()

    with (
        patch("app.websocket._get_redis", return_value=mock_redis),
        patch("app.websocket.POLL_TIMEOUT", 0.0),
        client.websocket_connect("/ws/result/missing-job") as ws,
    ):
        data = ws.receive_json()

    assert data["status"] == "timeout"
    assert data["job_id"] == "missing-job"


def test_websocket_result_includes_model_version():
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=SAMPLE_RESULT)
    mock_redis.aclose = AsyncMock()

    with (
        patch("app.websocket._get_redis", return_value=mock_redis),
        client.websocket_connect("/ws/result/test-job-123") as ws,
    ):
        data = ws.receive_json()

    assert data["model_version"] == "model_a"
