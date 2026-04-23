from fastapi import FastAPI

app = FastAPI(
    title="InferGrid Inference API",
    description="Production ML inference service",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict:
    """Placeholder — will be replaced by prometheus-client exposition in Phase 3."""
    return {"metrics": "not yet instrumented"}


@app.post("/predict")
async def predict(body: dict) -> dict:
    """Placeholder — model loading wired in Day 3."""
    return {"prediction": None, "message": "model not yet loaded"}