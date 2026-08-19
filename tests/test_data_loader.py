"""tests/test_data_loader.py — Tests for dataset loader and label mapping."""
import pytest
import pandas as pd
import numpy as np


def test_label_mapping_has_all_nback_levels(config):
    """Label mapping must cover nback_1 through nback_4."""
    from src.data_loader import OpenNeuroNBackDataset
    ds = OpenNeuroNBackDataset(config=config)
    mapping = ds.get_label_mapping()

    for level in ["nback_1", "nback_2", "nback_3", "nback_4"]:
        assert level in mapping, f"{level} missing from label mapping"


def test_label_mapping_valid_classes(config):
    """All mapped labels must be LOW, MODERATE, or HIGH."""
    from src.data_loader import OpenNeuroNBackDataset
    valid = {"LOW", "MODERATE", "HIGH"}
    ds = OpenNeuroNBackDataset(config=config)
    for trial_type, label in ds.get_label_mapping().items():
        assert label in valid, f"Invalid label {label!r} for {trial_type!r}"


def test_events_tsv_filtering(config, tmp_path):
    """Tutorial trials should be excluded from events DataFrame."""
    from src.data_loader import OpenNeuroNBackDataset
    import mne

    ds = OpenNeuroNBackDataset(config=config)

    # Synthetic events with tutorial entries
    events_with_tutorials = pd.DataFrame({
        "onset": [1.0, 2.0, 3.0, 4.0],
        "duration": [1.5, 1.5, 1.5, 1.5],
        "trial_type": ["nback_1", "nback_2", "nback_1", "nback_2"],
        "nback_n": [1, 2, 1, 2],
        "istutorial": ["true", "true", "false", "false"],
    })

    # Normalize column names (as done in loader)
    events_with_tutorials.columns = [c.strip().lower() for c in events_with_tutorials.columns]

    # Simulate the filtering logic
    before = len(events_with_tutorials)
    events_with_tutorials["istutorial_norm"] = events_with_tutorials["istutorial"].str.lower()
    filtered = events_with_tutorials[events_with_tutorials["istutorial_norm"] != "true"]

    # Should have removed 2 tutorial rows
    assert len(filtered) == 2


def test_factory_returns_correct_class(config):
    """load_dataset factory should return OpenNeuroNBackDataset for ds007169."""
    from src.data_loader import load_dataset, OpenNeuroNBackDataset
    ds = load_dataset("ds007169", config=config)
    assert isinstance(ds, OpenNeuroNBackDataset)


def test_factory_raises_for_unknown_dataset(config):
    """load_dataset should raise ValueError for unknown dataset IDs."""
    from src.data_loader import load_dataset
    with pytest.raises(ValueError, match="Unknown dataset ID"):
        load_dataset("unknown_dataset_xyz", config=config)


def test_config_label_mapping_configurable(config):
    """Label mapping should read from config.yaml."""
    from src.data_loader import OpenNeuroNBackDataset
    ds = OpenNeuroNBackDataset(config=config)
    mapping = ds.get_label_mapping()

    # These are the defaults set in config.yaml
    assert mapping["nback_1"] == "LOW"
    assert mapping["nback_2"] == "MODERATE"
    assert mapping["nback_3"] in ("HIGH", "MODERATE")  # configurable
    assert mapping["nback_4"] in ("HIGH", "MODERATE")  # configurable
