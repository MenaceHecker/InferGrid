from datetime import datetime
 
from pydantic import BaseModel, Field, field_validator
 
 
# Models
 
 
class ModelRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    version_hash: str = Field(..., min_length=64, max_length=64)
    status: str = Field(default="staging", pattern="^(staging|active|retired)$")
 
 
class ModelResponse(BaseModel):
    id: int
    name: str
    version_hash: str
    status: str
    created_at: datetime
 
    model_config = {"from_attributes": True}
 
# Evaluations
 
 
class EvaluationAddRequest(BaseModel):
    metric_name: str = Field(..., min_length=1, max_length=64)
    metric_value: float
    dataset_hash: str = Field(..., min_length=64, max_length=64)
 
 
class EvaluationResponse(BaseModel):
    id: int
    model_id: int
    metric_name: str
    metric_value: float
    dataset_hash: str
    recorded_at: datetime
 
    model_config = {"from_attributes": True}
 
 
class ModelMetricsResponse(BaseModel):
    model: ModelResponse
    evaluations: list[EvaluationResponse]
 
# A/B configs
 
 
class ABConfigureRequest(BaseModel):
    model_a_id: int
    model_b_id: int
    split_weight: float = Field(default=0.5, ge=0.0, le=1.0)
 
    @field_validator("model_b_id")
    @classmethod
    def models_must_differ(cls, v: int, info: any) -> int:
        if "model_a_id" in info.data and v == info.data["model_a_id"]:
            raise ValueError("model_a_id and model_b_id must be different")
        return v
 
 
class ABConfigResponse(BaseModel):
    id: int
    model_a_id: int
    model_b_id: int
    split_weight: float
    active: bool
    created_at: datetime
 
    model_config = {"from_attributes": True}
 
# A/B compare
 
 
class ModelCompareMetrics(BaseModel):
    model_id: int
    model_name: str
    version_hash: str
    metrics: dict[str, float]
 
 
class ABCompareResponse(BaseModel):
    model_a: ModelCompareMetrics
    model_b: ModelCompareMetrics
    split_weight: float
 