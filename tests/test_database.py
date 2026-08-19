"""tests/test_database.py — Tests for database module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_db_initialises(tmp_path, monkeypatch):
    """Database should initialise and create all expected tables."""
    import sqlalchemy as sa
    from src import database as db_module

    # Point database to a temp file
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    original_get_db_path = db_module._get_db_path
    monkeypatch.setattr(db_module, "_get_db_path", lambda: db_path)

    db_module.init_db()
    engine = db_module.get_engine()

    inspector = sa.inspect(engine)
    tables = inspector.get_table_names()

    expected_tables = {
        "datasets", "subjects", "processing_runs",
        "feature_runs", "model_runs", "predictions", "fuzzy_rules",
    }
    for t in expected_tables:
        assert t in tables, f"Table '{t}' missing from database"


def test_dataset_insert_and_retrieve(tmp_path, monkeypatch):
    """Dataset records should be insertable and retrievable."""
    from src import database as db_module

    db_path = tmp_path / "test2.db"
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(db_module, "_get_db_path", lambda: db_path)

    db_module.init_db()

    ds = db_module.get_or_create_dataset(
        name="Test Dataset",
        dataset_id="test001",
        source="http://example.com",
        version="1.0",
        license_="CC0",
        description="Test",
        subjects=5,
        channels=19,
        sampling_rate=250,
    )
    assert ds is not None
    assert ds.dataset_id == "test001"

    # Calling again should return same record (idempotent)
    ds2 = db_module.get_or_create_dataset(
        name="Test Dataset", dataset_id="test001",
        source="http://example.com", version="1.0",
        license_="CC0", description="Test",
        subjects=5, channels=19, sampling_rate=250,
    )
    assert ds2 is not None


def test_model_run_record(tmp_path, monkeypatch):
    """Model run records should be storable."""
    from src import database as db_module

    db_path = tmp_path / "test3.db"
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(db_module, "_get_db_path", lambda: db_path)

    db_module.init_db()

    # Create prerequisite records
    ds = db_module.get_or_create_dataset(
        "DS", "ds_x", "http://x.com", "1.0", "CC0", "desc", 5, 19, 250
    )
    pr_id = db_module.record_processing_run(
        ds.id, 1.0, 40.0, 50.0, 4.0, 2.0, "1.0", 5, 100
    )
    fr_id = db_module.record_feature_run(pr_id, "1.0", 22, "[]", 100)
    mr_id = db_module.record_model_run(
        "Random Forest", fr_id, "1.0",
        accuracy=0.75, precision=0.74, recall=0.73,
        f1=0.73, balanced_accuracy=0.72,
        n_train=80, n_test=20, n_folds=5,
    )
    assert isinstance(mr_id, int)
    assert mr_id > 0


def test_fuzzy_rules_storage(tmp_path, monkeypatch):
    """Fuzzy rules should store and retrieve correctly."""
    from src import database as db_module

    db_path = tmp_path / "test4.db"
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)
    monkeypatch.setattr(db_module, "_get_db_path", lambda: db_path)

    db_module.init_db()

    rules = [
        {"rule_number": 1, "rule_text": "IF theta is HIGH THEN HIGH", "consequent": "HIGH", "enabled": True},
        {"rule_number": 2, "rule_text": "IF theta is LOW THEN LOW", "consequent": "LOW", "enabled": True},
    ]
    db_module.store_fuzzy_rules(rules)

    retrieved = db_module.get_fuzzy_rules()
    assert len(retrieved) == 2
    assert retrieved[0].rule_number == 1
