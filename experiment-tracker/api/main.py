from typing import Annotated

from api.schemas import (
    ABCompareResponse,
    ABConfigResponse,
    ABConfigureRequest,
    EvaluationAddRequest,
    EvaluationResponse,
    ModelCompareMetrics,
    ModelMetricsResponse,
    ModelRegisterRequest,
    ModelResponse,
)
from db.models import ABConfig, EvaluationRecord, ModelRecord, ModelStatus
from db.session import get_db
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

DbSession = Annotated[Session, Depends(get_db)]

app = FastAPI(
    title="InferGrid Experiment Tracker",
    description="Model versioning, evaluation tracking, and A/B routing configuration",
    version="0.1.0",
)


# POST /models/register


@app.post("/models/register", response_model=ModelResponse, status_code=201)
def register_model(
    body: ModelRegisterRequest,
    db: DbSession,
) -> ModelRecord:
    """
    Register a new model version.
    Returns 409 if a model with the same version_hash already exists.
    """
    existing = db.query(ModelRecord).filter(ModelRecord.version_hash == body.version_hash).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Model with version_hash {body.version_hash!r} already registered (id={existing.id})",
        )

    model = ModelRecord(
        name=body.name,
        version_hash=body.version_hash,
        status=ModelStatus(body.status),
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


# POST /models/{id}/evaluations


@app.post("/models/{model_id}/evaluations", response_model=EvaluationResponse, status_code=201)
def add_evaluation(
    model_id: int,
    body: EvaluationAddRequest,
    db: DbSession,
) -> EvaluationRecord:
    """
    Add an evaluation metric for a model.
    Returns 404 if the model does not exist.
    Returns 409 if a metric with the same name and dataset_hash already exists.
    """
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    existing = (
        db.query(EvaluationRecord)
        .filter(
            EvaluationRecord.model_id == model_id,
            EvaluationRecord.metric_name == body.metric_name,
            EvaluationRecord.dataset_hash == body.dataset_hash,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Metric {body.metric_name!r} for dataset {body.dataset_hash!r} already exists",
        )

    record = EvaluationRecord(
        model_id=model_id,
        metric_name=body.metric_name,
        metric_value=body.metric_value,
        dataset_hash=body.dataset_hash,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# GET /models/{id}/metrics


@app.get("/models/{model_id}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(
    model_id: int,
    db: DbSession,
) -> ModelMetricsResponse:
    """
    Fetch a model record and all its evaluation metrics.
    Returns 404 if the model does not exist.
    """
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelMetricsResponse(
        model=ModelResponse.model_validate(model),
        evaluations=[
            EvaluationResponse.model_validate(e)
            for e in model.evaluations  # type: ignore[name-defined]
        ],
    )


# POST /ab/configure


@app.post("/ab/configure", response_model=ABConfigResponse, status_code=201)
def configure_ab(
    body: ABConfigureRequest,
    db: DbSession,
) -> ABConfig:
    """
    Create a new A/B configuration.
    Deactivates any previously active config so only one can be active at a time.
    Returns 404 if either model does not exist.
    """
    model_a = db.query(ModelRecord).filter(ModelRecord.id == body.model_a_id).first()
    if not model_a:
        raise HTTPException(status_code=404, detail=f"Model A (id={body.model_a_id}) not found")

    model_b = db.query(ModelRecord).filter(ModelRecord.id == body.model_b_id).first()
    if not model_b:
        raise HTTPException(status_code=404, detail=f"Model B (id={body.model_b_id}) not found")

    # Deactivate all existing active configs
    db.query(ABConfig).filter(ABConfig.active.is_(True)).update({"active": False})

    config = ABConfig(
        model_a_id=body.model_a_id,
        model_b_id=body.model_b_id,
        split_weight=body.split_weight,
        active=True,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


# GET /ab/active


@app.get("/ab/active", response_model=ABConfigResponse)
def get_active_ab(db: DbSession) -> ABConfig:
    """
    Return the currently active A/B configuration.
    Returns 404 if no active config exists.
    """
    config = db.query(ABConfig).filter(ABConfig.active.is_(True)).first()
    if not config:
        raise HTTPException(status_code=404, detail="No active A/B configuration found")
    return config


# GET /ab/compare


@app.get("/ab/compare", response_model=ABCompareResponse)
def compare_ab(db: DbSession) -> ABCompareResponse:
    """
    Return side-by-side evaluation metrics for the two models in the active A/B config.
    Returns 404 if no active config exists.
    """
    config = db.query(ABConfig).filter(ABConfig.active.is_(True)).first()
    if not config:
        raise HTTPException(status_code=404, detail="No active A/B configuration found")

    def _metrics_for(model: ModelRecord) -> ModelCompareMetrics:
        metrics = {
            e.metric_name: e.metric_value
            for e in db.query(EvaluationRecord).filter(EvaluationRecord.model_id == model.id).all()
        }
        return ModelCompareMetrics(
            model_id=model.id,
            model_name=model.name,
            version_hash=model.version_hash,
            metrics=metrics,
        )

    return ABCompareResponse(
        model_a=_metrics_for(config.model_a),
        model_b=_metrics_for(config.model_b),
        split_weight=config.split_weight,
    )
