"""
experiment-tracker/db/models.py
SQLAlchemy ORM models for the experiment tracker.

Three tables:
  - models      — registered model versions
  - evaluations — metric snapshots per model per dataset
  - ab_configs  — active A/B traffic split configuration
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass

# models


class ModelStatus(str, enum.Enum):
    staging = "staging"      # registered but not yet serving traffic
    active = "active"        # currently serving production traffic
    retired = "retired"      # no longer in use


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    # SHA-256 hash of the model artifact — used to detect duplicate registrations
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), nullable=False, default=ModelStatus.staging
    )

    evaluations: Mapped[list["EvaluationRecord"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ModelRecord id={self.id} name={self.name!r} status={self.status}>"


# evaluations


class EvaluationRecord(Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        # One metric value per (model, metric_name, dataset) combination
        UniqueConstraint("model_id", "metric_name", "dataset_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    # SHA-256 hash of the evaluation dataset — identifies which dataset produced this metric
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model: Mapped["ModelRecord"] = relationship(back_populates="evaluations")

    def __repr__(self) -> str:
        return (
            f"<EvaluationRecord model_id={self.model_id} "
            f"{self.metric_name}={self.metric_value:.4f}>"
        )

# ab_configs


class ABConfig(Base):
    __tablename__ = "ab_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    model_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )

    # Fraction of traffic routed to model_a (0.0–1.0).
    # model_b receives (1 - split_weight).
    split_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Only one ABConfig should be active at a time.
    # Enforced at the application layer in the API.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model_a: Mapped["ModelRecord"] = relationship(foreign_keys=[model_a_id])
    model_b: Mapped["ModelRecord"] = relationship(foreign_keys=[model_b_id])

    def __repr__(self) -> str:
        return (
            f"<ABConfig model_a={self.model_a_id} model_b={self.model_b_id} "
            f"split={self.split_weight} active={self.active}>"
        )