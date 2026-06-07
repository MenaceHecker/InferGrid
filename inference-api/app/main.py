import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response

from app.ab_router import get_active_ab_config, make_routing_decision
from app.metrics import (
    AB_ROUTING_COUNT,
    MODEL_LOAD_TIME,
    PREDICTION_CONFIDENCE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_app,
)
from app.models.onnx_loader import OnnxLoader
from app.models.sklearn_loader import SklearnLoader
from app.producer import enqueue_job
from app.schemas import EnqueuedResponse, PredictRequest, PredictResponse
from app.websocket import router as ws_router

log = logging.getLogger(__name__)

# App state

_model: OnnxLoader | SklearnLoader | None = None
_model_backend: str = "none"
_concurrent_requests: int = 0

_READINESS_PROBE_TEXT = "science space rocket"
_READINESS_TIMEOUT_MS = 500
_MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))


def _load_model(model_path: str) -> tuple[OnnxLoader | SklearnLoader, str]:
    start = time.monotonic()
    if model_path.endswith(".onnx"):
        classes = _get_classes(model_path)
        loader = OnnxLoader(model_path, classes)
        backend = "onnx"
    else:
        loader = SklearnLoader(model_path)
        backend = "sklearn"
    MODEL_LOAD_TIME.set(time.monotonic() - start)
    return loader, backend


def _get_classes(onnx_path: str) -> list[str]:
    pkl_path = Path(onnx_path).with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        return list(payload["classes"])
    return [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
        "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
        "sci.space", "soc.religion.christian", "talk.politics.guns",
        "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _model, _model_backend
    model_path = os.environ.get("MODEL_PATH", "models/classifier.onnx")
    _model, _model_backend = _load_model(model_path)
    log.info("[startup] %s model loaded from %s", _model_backend, model_path)
    yield
    _model = None
    _model_backend = "none"
    log.info("[shutdown] model released")


app = FastAPI(
    title="InferGrid Inference API",
    description="Production ML inference service",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/metrics", metrics_app)
app.include_router(ws_router)


# Middleware


@app.middleware("http")
async def track_requests(request: Request, call_next: Any) -> Response:
    start = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
    return response


# Routes


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_backend": _model_backend,
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.monotonic()
    try:
        _model.predict(_READINESS_PROBE_TEXT)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference error: {exc}") from exc
    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > _READINESS_TIMEOUT_MS:
        raise HTTPException(
            status_code=503,
            detail=f"Inference too slow: {elapsed_ms:.1f}ms > {_READINESS_TIMEOUT_MS}ms",
        )
    return {
        "status": "ready",
        "model_backend": _model_backend,
        "inference_ms": round(elapsed_ms, 2),
    }


@app.post("/predict")
async def predict(body: PredictRequest) -> PredictResponse | EnqueuedResponse:
    global _concurrent_requests

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = get_active_ab_config()
    decision = make_routing_decision(config)
    AB_ROUTING_COUNT.labels(model_version=decision.model_version).inc()

    if _concurrent_requests >= _MAX_CONCURRENT:
        job_id = str(uuid.uuid4())
        topic = enqueue_job(job_id, body.text, decision.model_version)
        log.info(
            "Enqueued job %s to %s (concurrent=%d, max=%d)",
            job_id, topic, _concurrent_requests, _MAX_CONCURRENT,
        )
        return EnqueuedResponse(job_id=job_id, status="enqueued")

    _concurrent_requests += 1
    try:
        prediction, confidence = _model.predict(body.text)
        PREDICTION_CONFIDENCE.observe(confidence)
        log.info(
            "predict routed to %s (model_id=%s)",
            decision.model_version,
            decision.model_id,
        )
        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            model_backend=_model_backend,
            model_version=decision.model_version,
        )
    finally:
        _concurrent_requests -= 1