"""
src/database.py
===============
SQLite database schema and session management using SQLAlchemy ORM.

Tables created:
  - datasets
  - subjects
  - processing_runs
  - feature_runs
  - model_runs
  - predictions
  - fuzzy_rules

The database stores metadata and experiment tracking information only.
Raw EEG signals are NOT stored here — they remain on the filesystem.

Usage:
    from src.database import init_db, get_session, Dataset

    init_db()                          # creates tables
    session = get_session()
    ds = Dataset(name="ds007169", ...)
    session.add(ds)
    session.commit()
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Engine setup
# ------------------------------------------------------------------ #

_engine = None
_SessionLocal = None


def _get_db_path() -> Path:
    """Return absolute path to the SQLite database file."""
    cfg = get_config()
    return resolve_path(cfg.paths.database)


def get_engine():
    """Return (and lazily create) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_path = _get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        logger.debug("Database engine created: %s", db_url)
    return _engine


def get_session_factory():
    """Return (and lazily create) the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,   # keep objects usable after session closes
            bind=get_engine(),
        )
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that provides a database session with automatic
    commit/rollback handling.

    Usage:
        with get_session() as session:
            session.add(some_record)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ------------------------------------------------------------------ #
# ORM Models
# ------------------------------------------------------------------ #

class Base(DeclarativeBase):
    pass


class Dataset(Base):
    """
    Tracks dataset metadata.
    One record per dataset used in the project.
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    dataset_id = Column(String(50), nullable=False, unique=True)
    source = Column(String(500))
    version = Column(String(50))
    license = Column(String(100))
    description = Column(Text)
    subjects = Column(Integer)
    channels = Column(Integer)
    sampling_rate = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    subjects_list = relationship("Subject", back_populates="dataset", cascade="all, delete-orphan")
    processing_runs = relationship("ProcessingRun", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} subjects={self.subjects}>"


class Subject(Base):
    """
    Tracks individual participants within a dataset.
    """
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    subject_code = Column(String(50), nullable=False)
    session = Column(String(50), default="ses-01")
    notes = Column(Text)

    dataset = relationship("Dataset", back_populates="subjects_list")

    def __repr__(self) -> str:
        return f"<Subject id={self.id} code={self.subject_code!r}>"


class ProcessingRun(Base):
    """
    Records one complete preprocessing run with its parameters.
    Allows reproducing the same preprocessing later.
    """
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    filter_low = Column(Float, nullable=False)
    filter_high = Column(Float, nullable=False)
    notch_frequency = Column(Float)
    window_seconds = Column(Float)
    overlap_seconds = Column(Float)
    preprocessing_version = Column(String(50))
    n_subjects_processed = Column(Integer)
    n_windows_total = Column(Integer)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="processing_runs")
    feature_runs = relationship("FeatureRun", back_populates="processing_run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<ProcessingRun id={self.id} "
            f"bp=[{self.filter_low},{self.filter_high}]Hz "
            f"win={self.window_seconds}s>"
        )


class FeatureRun(Base):
    """
    Records one feature extraction run and its parameters.
    """
    __tablename__ = "feature_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    processing_run_id = Column(Integer, ForeignKey("processing_runs.id"), nullable=False)
    feature_version = Column(String(50))
    number_of_features = Column(Integer)
    feature_names = Column(Text)   # JSON list of feature column names
    n_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    processing_run = relationship("ProcessingRun", back_populates="feature_runs")
    model_runs = relationship("ModelRun", back_populates="feature_run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<FeatureRun id={self.id} "
            f"n_features={self.number_of_features} "
            f"n_samples={self.n_samples}>"
        )


class ModelRun(Base):
    """
    Records one model training run and its evaluation metrics.
    """
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    feature_run_id = Column(Integer, ForeignKey("feature_runs.id"), nullable=False)
    model_version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    balanced_accuracy = Column(Float)
    n_train_samples = Column(Integer)
    n_test_samples = Column(Integer)
    n_folds = Column(Integer)
    model_path = Column(String(500))
    parameters = Column(Text)    # JSON of model hyperparameters
    created_at = Column(DateTime, default=datetime.utcnow)

    feature_run = relationship("FeatureRun", back_populates="model_runs")
    predictions = relationship("Prediction", back_populates="model_run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<ModelRun id={self.id} "
            f"model={self.model_name!r} "
            f"acc={self.accuracy:.3f if self.accuracy else 'N/A'}>"
        )


class Prediction(Base):
    """
    Stores individual window-level predictions for a model run.
    Enables detailed inspection of per-sample results.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False)
    subject_code = Column(String(50))
    window_id = Column(Integer)
    true_class = Column(String(20))
    predicted_class = Column(String(20))
    confidence = Column(Float)
    fuzzy_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_run = relationship("ModelRun", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} "
            f"subject={self.subject_code!r} "
            f"true={self.true_class!r} "
            f"pred={self.predicted_class!r}>"
        )


class FuzzyRule(Base):
    """
    Stores the fuzzy inference rules used by the fuzzy classifier.
    """
    __tablename__ = "fuzzy_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_number = Column(Integer, nullable=False)
    rule_text = Column(Text, nullable=False)
    antecedent = Column(Text)     # JSON representation of antecedent
    consequent = Column(String(50))
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<FuzzyRule id={self.id} no={self.rule_number} -> {self.consequent!r}>"


# ------------------------------------------------------------------ #
# Database initialisation
# ------------------------------------------------------------------ #

def init_db() -> None:
    """
    Create all tables in the SQLite database.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised: %s", _get_db_path())


def get_or_create_dataset(
    name: str,
    dataset_id: str,
    source: str,
    version: str,
    license_: str,
    description: str,
    subjects: int,
    channels: int,
    sampling_rate: int,
) -> Dataset:
    """
    Return existing dataset record or create a new one.
    Idempotent — safe to call on every run.
    """
    with get_session() as session:
        existing = (
            session.query(Dataset).filter_by(dataset_id=dataset_id).first()
        )
        if existing:
            logger.info(
                "Dataset '%s' already registered (id=%d).", dataset_id, existing.id
            )
            return existing

        ds = Dataset(
            name=name,
            dataset_id=dataset_id,
            source=source,
            version=version,
            license=license_,
            description=description,
            subjects=subjects,
            channels=channels,
            sampling_rate=sampling_rate,
        )
        session.add(ds)
        session.flush()   # get id before commit
        logger.info("Dataset '%s' registered (id=%d).", dataset_id, ds.id)
        return ds


def record_processing_run(
    dataset_db_id: int,
    filter_low: float,
    filter_high: float,
    notch_frequency: float,
    window_seconds: float,
    overlap_seconds: float,
    preprocessing_version: str,
    n_subjects: int = 0,
    n_windows: int = 0,
    notes: str = "",
) -> int:
    """Insert a new processing_run record and return its id."""
    with get_session() as session:
        run = ProcessingRun(
            dataset_id=dataset_db_id,
            filter_low=filter_low,
            filter_high=filter_high,
            notch_frequency=notch_frequency,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            preprocessing_version=preprocessing_version,
            n_subjects_processed=n_subjects,
            n_windows_total=n_windows,
            notes=notes,
        )
        session.add(run)
        session.flush()
        logger.info("Processing run recorded (id=%d).", run.id)
        return run.id


def record_feature_run(
    processing_run_id: int,
    feature_version: str,
    number_of_features: int,
    feature_names: str,
    n_samples: int,
) -> int:
    """Insert a new feature_run record and return its id."""
    with get_session() as session:
        run = FeatureRun(
            processing_run_id=processing_run_id,
            feature_version=feature_version,
            number_of_features=number_of_features,
            feature_names=feature_names,
            n_samples=n_samples,
        )
        session.add(run)
        session.flush()
        logger.info("Feature run recorded (id=%d, %d features, %d samples).",
                    run.id, number_of_features, n_samples)
        return run.id


def record_model_run(
    model_name: str,
    feature_run_id: int,
    model_version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    balanced_accuracy: float,
    n_train: int,
    n_test: int,
    n_folds: int,
    model_path: str = "",
    parameters: str = "{}",
) -> int:
    """Insert a new model_run record and return its id."""
    with get_session() as session:
        run = ModelRun(
            model_name=model_name,
            feature_run_id=feature_run_id,
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            balanced_accuracy=balanced_accuracy,
            n_train_samples=n_train,
            n_test_samples=n_test,
            n_folds=n_folds,
            model_path=model_path,
            parameters=parameters,
        )
        session.add(run)
        session.flush()
        logger.info(
            "Model run recorded — %s: acc=%.3f, f1=%.3f (id=%d).",
            model_name, accuracy, f1, run.id,
        )
        return run.id


def get_latest_model_runs(n: int = 5):
    """Return the n most recent model runs for dashboard display."""
    with get_session() as session:
        runs = (
            session.query(ModelRun)
            .order_by(ModelRun.created_at.desc())
            .limit(n)
            .all()
        )
        # Detach from session for use outside context
        session.expunge_all()
        return runs


def store_fuzzy_rules(rules: list) -> None:
    """
    Store fuzzy rules in the database.
    ``rules`` is a list of dicts with keys:
        rule_number, rule_text, antecedent, consequent
    """
    with get_session() as session:
        # Clear existing rules
        session.query(FuzzyRule).delete()
        for r in rules:
            fr = FuzzyRule(
                rule_number=r["rule_number"],
                rule_text=r["rule_text"],
                antecedent=r.get("antecedent", ""),
                consequent=r.get("consequent", ""),
                enabled=r.get("enabled", True),
            )
            session.add(fr)
        logger.info("Stored %d fuzzy rules in database.", len(rules))


def get_fuzzy_rules() -> list:
    """Return all enabled fuzzy rules from the database."""
    with get_session() as session:
        rules = (
            session.query(FuzzyRule)
            .filter_by(enabled=True)
            .order_by(FuzzyRule.rule_number)
            .all()
        )
        session.expunge_all()
        return rules


def get_all_experiment_history_df():
    """
    Fetch all historical model runs from the database as a pandas DataFrame.
    """
    import pandas as pd
    with get_session() as session:
        runs = (
            session.query(ModelRun)
            .order_by(ModelRun.created_at.desc())
            .all()
        )
        if not runs:
            return None

        records = []
        for r in runs:
            records.append({
                "Run ID": r.id,
                "Model": r.model_name,
                "Version": r.model_version or "1.0",
                "Accuracy": round(r.accuracy, 4) if r.accuracy is not None else None,
                "F1 (Macro)": round(r.f1_score, 4) if r.f1_score is not None else None,
                "Balanced Acc": round(r.balanced_accuracy, 4) if r.balanced_accuracy is not None else None,
                "Precision": round(r.precision, 4) if r.precision is not None else None,
                "Recall": round(r.recall, 4) if r.recall is not None else None,
                "Folds": r.n_folds or 5,
                "Test Samples": r.n_test_samples,
                "Date & Time (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "N/A",
            })
        return pd.DataFrame(records)

