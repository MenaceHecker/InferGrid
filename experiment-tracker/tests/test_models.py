"""
Tests for the experiment tracker database models.
Uses SQLite in-memory so no real PostgreSQL is needed in CI.
"""

import pytest
from db.models import ABConfig, Base, EvaluationRecord, ModelRecord, ModelStatus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


# ModelRecord


def test_create_model_record(db):
    model = ModelRecord(
        name="newsgroups-v1",
        version_hash="abc123" * 10,
        status=ModelStatus.staging,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    assert model.id is not None


def test_model_default_status_is_staging(db):
    model = ModelRecord(name="m", version_hash="a" * 64)
    db.add(model)
    db.commit()
    db.refresh(model)
    assert model.status == ModelStatus.staging


def test_model_version_hash_is_unique(db):
    from sqlalchemy.exc import IntegrityError

    db.add(ModelRecord(name="m1", version_hash="x" * 64))
    db.commit()
    db.add(ModelRecord(name="m2", version_hash="x" * 64))
    with pytest.raises(IntegrityError):
        db.commit()


def test_model_status_transitions(db):
    model = ModelRecord(name="m", version_hash="b" * 64, status=ModelStatus.staging)
    db.add(model)
    db.commit()

    model.status = ModelStatus.active
    db.commit()
    db.refresh(model)
    assert model.status == ModelStatus.active

    model.status = ModelStatus.retired
    db.commit()
    db.refresh(model)
    assert model.status == ModelStatus.retired


# EvaluationRecord


def test_create_evaluation(db):
    model = ModelRecord(name="m", version_hash="c" * 64)
    db.add(model)
    db.commit()

    eval_record = EvaluationRecord(
        model_id=model.id,
        metric_name="accuracy",
        metric_value=0.923,
        dataset_hash="d" * 64,
    )
    db.add(eval_record)
    db.commit()
    db.refresh(eval_record)
    assert eval_record.id is not None
    assert eval_record.metric_value == pytest.approx(0.923)


def test_evaluation_unique_constraint(db):
    from sqlalchemy.exc import IntegrityError

    model = ModelRecord(name="m", version_hash="e" * 64)
    db.add(model)
    db.commit()

    db.add(
        EvaluationRecord(
            model_id=model.id,
            metric_name="accuracy",
            metric_value=0.9,
            dataset_hash="f" * 64,
        )
    )
    db.commit()

    db.add(
        EvaluationRecord(
            model_id=model.id,
            metric_name="accuracy",
            metric_value=0.95,
            dataset_hash="f" * 64,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()


def test_evaluation_cascade_delete(db):
    model = ModelRecord(name="m", version_hash="g" * 64)
    db.add(model)
    db.commit()

    db.add(
        EvaluationRecord(
            model_id=model.id,
            metric_name="f1",
            metric_value=0.88,
            dataset_hash="h" * 64,
        )
    )
    db.commit()

    db.delete(model)
    db.commit()
    assert db.query(EvaluationRecord).count() == 0


# ABConfig


def test_create_ab_config(db):
    model_a = ModelRecord(name="a", version_hash="i" * 64)
    model_b = ModelRecord(name="b", version_hash="j" * 64)
    db.add_all([model_a, model_b])
    db.commit()

    config = ABConfig(
        model_a_id=model_a.id,
        model_b_id=model_b.id,
        split_weight=0.7,
        active=True,
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    assert config.id is not None
    assert config.split_weight == pytest.approx(0.7)
    assert config.active is True


def test_ab_config_split_weight_defaults_to_half(db):
    model_a = ModelRecord(name="a", version_hash="k" * 64)
    model_b = ModelRecord(name="b", version_hash="l" * 64)
    db.add_all([model_a, model_b])
    db.commit()

    config = ABConfig(model_a_id=model_a.id, model_b_id=model_b.id, active=True)
    db.add(config)
    db.commit()
    db.refresh(config)

    assert config.split_weight == pytest.approx(0.5)


def test_ab_config_relationships(db):
    model_a = ModelRecord(name="a", version_hash="m" * 64)
    model_b = ModelRecord(name="b", version_hash="n" * 64)
    db.add_all([model_a, model_b])
    db.commit()

    config = ABConfig(model_a_id=model_a.id, model_b_id=model_b.id, active=True)
    db.add(config)
    db.commit()
    db.refresh(config)

    assert config.model_a.name == "a"
    assert config.model_b.name == "b"
