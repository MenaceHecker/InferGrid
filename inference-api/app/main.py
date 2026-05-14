import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from app.models.onnx_loader import OnnxLoader
from app.models.sklearn_loader import SklearnLoader
from app.schemas import PredictRequest, PredictResponse

# App state

_model: OnnxLoader | SklearnLoader | None = None
_model_backend: str = "none"

# Warm-up probe text, hort and unambiguous, fast to classify.
_READINESS_PROBE_TEXT = "science space rocket"

# /ready will return 503 if inference takes longer than this.
_READINESS_TIMEOUT_MS = 500


def _load_model(model_path: str) -> tuple[OnnxLoader | SklearnLoader, str]:
    """
    Load ONNX model if path ends in .onnx, otherwise fall back to sklearn.
    Returns (loader_instance, backend_name).
    """
    if model_path.endswith(".onnx"):
        classes = _get_classes(model_path)
        loader = OnnxLoader(model_path, classes)
        return loader, "onnx"
    else:
        loader = SklearnLoader(model_path)
        return loader, "sklearn"


def _get_classes(onnx_path: str) -> list[str]:
    """
    Resolve class labels for the ONNX model.
    Looks for a sibling .pkl file first; falls back to 20 Newsgroups order.
    """
    pkl_path = Path(onnx_path).with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        return list(payload["classes"])

    return [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load model at startup, release at shutdown."""
    global _model, _model_backend
    model_path = os.environ.get("MODEL_PATH", "models/classifier.onnx")
    _model, _model_backend = _load_model(model_path)
    print(f"[startup] {_model_backend} model loaded from {model_path}")
    yield
    _model = None
    _model_backend = "none"
    print("[shutdown] model released")


app = FastAPI(
    title="InferGrid Inference API",
    description="Production ML inference service",
    version="0.1.0",
    lifespan=lifespan,
)


# Routes


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Liveness probe. Returns 200 as long as the process is alive.
    Does NOT verify model performance, that is /ready's job.
    Kubernetes restarts the pod if this stops returning 200.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_backend": _model_backend,
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """
    Readiness probe. Returns 200 only when:
      1. The model is loaded.
      2. A test inference completes within _READINESS_TIMEOUT_MS.

    Kubernetes will not route traffic to this pod until this passes.
    If inference degrades past the timeout, the pod is pulled from rotation
    without being restarted (unlike a failed liveness probe).
    """
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


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Placeholder, replaced by prometheus-client exposition in Phase 3."""
    return {"metrics": "not yet instrumented"}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    """
    Classify text using the loaded model (ONNX by default).
    Returns predicted class label and confidence score.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prediction, confidence = _model.predict(body.text)
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        model_backend=_model_backend,
    )