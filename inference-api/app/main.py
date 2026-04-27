import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.models.onnx_loader import OnnxLoader
from app.models.sklearn_loader import SklearnLoader
from app.schemas import PredictRequest, PredictResponse


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
_model: OnnxLoader | SklearnLoader | None = None
_model_backend: str = "none"


def _load_model(model_path: str) -> tuple[OnnxLoader | SklearnLoader, str]:
    """
    Load ONNX model if path ends in .onnx, otherwise fall back to sklearn.
    Returns (loader_instance, backend_name).
    """
    if model_path.endswith(".onnx"):
        # ONNX needs the class list, load from paired .pkl if present,
        # or use the standard 20 Newsgroups class order as default.
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
    import pickle
    from pathlib import Path

    pkl_path = Path(onnx_path).with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        return list(payload["classes"])

    # Standard 20 Newsgroups target_names order (sklearn default sort)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe, it returns 200 when the process is alive."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_backend": _model_backend,
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