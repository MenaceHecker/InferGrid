"""
inference-api/app/schemas.py
Pydantic request and response models for the inference API.
"""

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_backend: str
    model_version: str = "primary"


class EnqueuedResponse(BaseModel):
    job_id: str
    status: str = "enqueued"
    result_url: str | None = None

    def __init__(self, **data: Any) -> None:
        if "result_url" not in data and "job_id" in data:
            data["result_url"] = f"/ws/result/{data['job_id']}"
        super().__init__(**data)
