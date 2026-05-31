"""
Integration tests for the experiment tracker API.
Uses SQLite in-memory so no real PostgreSQL needed in CI.
"""

import pytest
from api.main import app
from db.models import Base, EvaluationRecord, ModelRecord, ModelStatus
from db.session import get_db
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Fixtures


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def two_models(db_session):
    """Seed two model records and return (model_a, model_b)."""
    model_a = ModelRecord(name="newsgroups-v1", version_hash="a" * 64, status=ModelStatus.active)
    model_b = ModelRecord(name="newsgroups-v2", version_hash="b" * 64, status=ModelStatus.staging)
    db_session.add_all([model_a, model_b])
    db_session.commit()
    db_session.refresh(model_a)
    db_session.refresh(model_b)
    return model_a, model_b


@pytest.fixture
def two_models_with_metrics(db_session, two_models):
    """Seed evaluation metrics for both models."""
    model_a, model_b = two_models
    db_session.add_all(
        [
            EvaluationRecord(
                model_id=model_a.id,
                metric_name="accuracy",
                metric_value=0.91,
                dataset_hash="d" * 64,
            ),
            EvaluationRecord(
                model_id=model_a.id, metric_name="f1", metric_value=0.89, dataset_hash="d" * 64
            ),
            EvaluationRecord(
                model_id=model_b.id,
                metric_name="accuracy",
                metric_value=0.94,
                dataset_hash="d" * 64,
            ),
            EvaluationRecord(
                model_id=model_b.id, metric_name="f1", metric_value=0.92, dataset_hash="d" * 64
            ),
        ]
    )
    db_session.commit()
    return model_a, model_b


# POST /models/register


def test_register_model_returns_201(client):
    response = client.post(
        "/models/register",
        json={
            "name": "newsgroups-v1",
            "version_hash": "a" * 64,
        },
    )
    assert response.status_code == 201


def test_register_model_response_shape(client):
    response = client.post(
        "/models/register",
        json={
            "name": "newsgroups-v1",
            "version_hash": "a" * 64,
        },
    )
    body = response.json()
    assert "id" in body
    assert "name" in body
    assert "version_hash" in body
    assert "status" in body
    assert "created_at" in body


def test_register_model_default_status_is_staging(client):
    response = client.post(
        "/models/register",
        json={
            "name": "newsgroups-v1",
            "version_hash": "a" * 64,
        },
    )
    assert response.json()["status"] == "staging"


def test_register_model_duplicate_hash_returns_409(client):
    payload = {"name": "m", "version_hash": "a" * 64}
    client.post("/models/register", json=payload)
    response = client.post("/models/register", json=payload)
    assert response.status_code == 409


def test_register_model_missing_name_returns_422(client):
    response = client.post("/models/register", json={"version_hash": "a" * 64})
    assert response.status_code == 422


def test_register_model_short_hash_returns_422(client):
    response = client.post(
        "/models/register",
        json={
            "name": "m",
            "version_hash": "abc",
        },
    )
    assert response.status_code == 422


# GET /models/{id}/metrics


def test_get_model_metrics_returns_200(client, two_models_with_metrics):
    model_a, _ = two_models_with_metrics
    response = client.get(f"/models/{model_a.id}/metrics")
    assert response.status_code == 200


def test_get_model_metrics_response_shape(client, two_models_with_metrics):
    model_a, _ = two_models_with_metrics
    body = client.get(f"/models/{model_a.id}/metrics").json()
    assert "model" in body
    assert "evaluations" in body
    assert isinstance(body["evaluations"], list)


def test_get_model_metrics_returns_all_evaluations(client, two_models_with_metrics):
    model_a, _ = two_models_with_metrics
    body = client.get(f"/models/{model_a.id}/metrics").json()
    assert len(body["evaluations"]) == 2


def test_get_model_metrics_404_for_unknown_model(client):
    response = client.get("/models/99999/metrics")
    assert response.status_code == 404


# POST /ab/configure


def test_configure_ab_returns_201(client, two_models):
    model_a, model_b = two_models
    response = client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": model_b.id,
            "split_weight": 0.7,
        },
    )
    assert response.status_code == 201


def test_configure_ab_is_active(client, two_models):
    model_a, model_b = two_models
    response = client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": model_b.id,
        },
    )
    assert response.json()["active"] is True


def test_configure_ab_deactivates_previous(client, two_models):
    model_a, model_b = two_models
    client.post("/ab/configure", json={"model_a_id": model_a.id, "model_b_id": model_b.id})
    client.post("/ab/configure", json={"model_a_id": model_b.id, "model_b_id": model_a.id})
    active = client.get("/ab/active").json()
    # Only one active config so the model ids are swapped in the second call
    assert active["model_a_id"] == model_b.id


def test_configure_ab_same_model_ids_returns_422(client, two_models):
    model_a, _ = two_models
    response = client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": model_a.id,
        },
    )
    assert response.status_code == 422


def test_configure_ab_unknown_model_returns_404(client, two_models):
    model_a, _ = two_models
    response = client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": 99999,
        },
    )
    assert response.status_code == 404


def test_configure_ab_split_weight_out_of_range_returns_422(client, two_models):
    model_a, model_b = two_models
    response = client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": model_b.id,
            "split_weight": 1.5,
        },
    )
    assert response.status_code == 422


# GET /ab/active


def test_get_active_ab_returns_200(client, two_models):
    model_a, model_b = two_models
    client.post("/ab/configure", json={"model_a_id": model_a.id, "model_b_id": model_b.id})
    response = client.get("/ab/active")
    assert response.status_code == 200


def test_get_active_ab_404_when_none_configured(client):
    response = client.get("/ab/active")
    assert response.status_code == 404


def test_get_active_ab_returns_correct_split(client, two_models):
    model_a, model_b = two_models
    client.post(
        "/ab/configure",
        json={
            "model_a_id": model_a.id,
            "model_b_id": model_b.id,
            "split_weight": 0.3,
        },
    )
    body = client.get("/ab/active").json()
    assert body["split_weight"] == pytest.approx(0.3)


# GET /ab/compare


def test_compare_ab_returns_200(client, two_models_with_metrics):
    model_a, model_b = two_models_with_metrics
    client.post("/ab/configure", json={"model_a_id": model_a.id, "model_b_id": model_b.id})
    response = client.get("/ab/compare")
    assert response.status_code == 200


def test_compare_ab_response_shape(client, two_models_with_metrics):
    model_a, model_b = two_models_with_metrics
    client.post("/ab/configure", json={"model_a_id": model_a.id, "model_b_id": model_b.id})
    body = client.get("/ab/compare").json()
    assert "model_a" in body
    assert "model_b" in body
    assert "split_weight" in body
    assert "metrics" in body["model_a"]
    assert "metrics" in body["model_b"]


def test_compare_ab_returns_correct_metrics(client, two_models_with_metrics):
    model_a, model_b = two_models_with_metrics
    client.post("/ab/configure", json={"model_a_id": model_a.id, "model_b_id": model_b.id})
    body = client.get("/ab/compare").json()
    assert body["model_a"]["metrics"]["accuracy"] == pytest.approx(0.91)
    assert body["model_b"]["metrics"]["accuracy"] == pytest.approx(0.94)


def test_compare_ab_404_when_no_active_config(client):
    response = client.get("/ab/compare")
    assert response.status_code == 404
