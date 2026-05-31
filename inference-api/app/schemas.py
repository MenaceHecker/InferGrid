"""
inference-api/app/schemas.py
Pydantic request and response models for the inference API.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_backend: str
    # Which model version served this request: "model_a" | "model_b" | "primary"
    model_version: str = "primary"
