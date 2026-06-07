"""
inference-api/app/websocket.py

WebSocket endpoint for async inference result delivery.

Flow:
  1. Client connects to GET /ws/result/{job_id}
  2. Server polls Redis for key job:{job_id}:result
  3. When result appears, server pushes JSON and closes the connection
  4. If result does not appear within POLL_TIMEOUT_SECONDS, server sends
     a timeout error and closes

Polling strategy:
  - Poll every POLL_INTERVAL_SECONDS (default 0.5s)
  - Give up after POLL_TIMEOUT_SECONDS (default 30s)
  - Each poll sends a {"status": "pending"} keepalive so the client
    knows the connection is alive and load balancers don't time out

Fallback for clients that don't support WebSockets:
  GET /result/{job_id} : HTTP long-poll, returns 200 when ready or 404 if expired
"""

import json
import logging
import os

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

REDIS_URL = os.environ.get(
    "REDIS_URL",
    "redis://redis.infergrid.svc.cluster.local:6379",
)
POLL_INTERVAL = float(os.environ.get("WS_POLL_INTERVAL_SECONDS", "0.5"))
POLL_TIMEOUT = float(os.environ.get("WS_POLL_TIMEOUT_SECONDS", "30"))

router = APIRouter()

RESULT_KEY_PREFIX = "job:"
RESULT_KEY_SUFFIX = ":result"


def _result_key(job_id: str) -> str:
    return f"{RESULT_KEY_PREFIX}{job_id}{RESULT_KEY_SUFFIX}"


def _get_redis() -> aioredis.Redis:
    return aioredis.from_url(REDIS_URL, decode_responses=True)


# WebSocket endpoint


@router.websocket("/ws/result/{job_id}")
async def ws_result(websocket: WebSocket, job_id: str) -> None:
    """
    Stream the inference result for a given job_id over WebSocket.
    Sends periodic {"status": "pending"} keepalives while waiting.
    Sends the full result payload and closes when the result is ready.
    Sends {"status": "timeout"} and closes if result does not arrive in time.
    """
    await websocket.accept()
    log.info("WebSocket connected for job %s", job_id)

    redis = _get_redis()
    elapsed = 0.0

    try:
        while elapsed < POLL_TIMEOUT:
            result = await redis.get(_result_key(job_id))

            if result is not None:
                payload = json.loads(result)
                await websocket.send_json(payload)
                log.info("Pushed result for job %s after %.1fs", job_id, elapsed)
                return

            # Send keepalive so client knows we're still waiting
            try:
                await websocket.send_json(
                    {
                        "status": "pending",
                        "job_id": job_id,
                        "elapsed_s": round(elapsed, 1),
                    }
                )
            except WebSocketDisconnect:
                log.info("Client disconnected for job %s", job_id)
                return

            import asyncio

            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        # Timeout
        await websocket.send_json(
            {
                "status": "timeout",
                "job_id": job_id,
                "message": f"Result not available after {POLL_TIMEOUT}s",
            }
        )
        log.warning("Timeout waiting for job %s result", job_id)

    except WebSocketDisconnect:
        log.info("Client disconnected for job %s", job_id)
    finally:
        await redis.aclose()
        await websocket.close()


# HTTP fallback for clients that don't support WebSockets


@router.get("/result/{job_id}")
async def http_result(job_id: str) -> JSONResponse:
    """
    HTTP fallback for WebSocket-unsupported clients.
    Returns 200 with the result if available, 404 if not yet ready or expired.
    """
    redis = _get_redis()
    try:
        result = await redis.get(_result_key(job_id))
        if result is None:
            return JSONResponse(
                status_code=404,
                content={"status": "pending", "job_id": job_id},
            )
        return JSONResponse(status_code=200, content=json.loads(result))
    finally:
        await redis.aclose()
