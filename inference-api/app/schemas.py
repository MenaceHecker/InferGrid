from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Text to classify")


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of top class")
    model_backend: str = Field(..., description="Backend used: 'sklearn' or 'onnx'")
