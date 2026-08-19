"""
tests/conftest.py
=================
Pytest fixtures shared across all test modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    from src.config import get_config
    return get_config()


# ── Synthetic EEG data ────────────────────────────────────────────────────

@pytest.fixture
def synthetic_raw():
    """Return a minimal MNE Raw object with synthetic EEG data."""
    import mne
    sfreq = 250
    n_channels = 19
    duration_s = 30
    n_samples = sfreq * duration_s

    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2",
    ]
    ch_types = ["eeg"] * n_channels

    rng = np.random.default_rng(42)
    data = rng.normal(0, 10e-6, size=(n_channels, n_samples))  # in Volts

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_events_df():
    """Minimal events DataFrame simulating nback task events."""
    return pd.DataFrame({
        "onset": [5.0, 6.5, 8.0, 9.5, 15.0, 16.5, 18.0, 19.5],
        "duration": [1.5] * 8,
        "trial_type": ["nback_1", "nback_1", "nback_2", "nback_2",
                       "nback_3", "nback_3", "nback_4", "nback_4"],
        "nback_n": [1, 1, 2, 2, 3, 3, 4, 4],
        "label": ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH", "HIGH", "HIGH"],
        "istutorial": [False] * 8,
    })


@pytest.fixture
def synthetic_segments(synthetic_raw, synthetic_events_df, config):
    """Return a list of synthetic segment dicts."""
    from src.segmentation import segment_subject
    segs = segment_subject("sub-test", synthetic_raw, synthetic_events_df, config=config)
    if not segs:
        # Create manually if segmentation returns empty
        data = synthetic_raw.get_data()[:, :500]   # smaller = faster sample entropy
        segs = [
            {
                "subject_id": "sub-test",
                "window_id": i,
                "start_time": float(i * 2),
                "end_time": float(i * 2 + 2),
                "trial_type": ["nback_1", "nback_2", "nback_3"][i % 3],
                "label": ["LOW", "MODERATE", "HIGH"][i % 3],
                "data": data,
                "channel_names": synthetic_raw.ch_names,
                "sfreq": 250.0,
                "n_channels": 19,
                "n_samples": 500,
            }
            for i in range(9)
        ]
    return segs


@pytest.fixture
def synthetic_features_df(synthetic_segments, config):
    """Return a feature DataFrame from synthetic segments."""
    from src.feature_extraction import extract_features_from_segments
    return extract_features_from_segments(synthetic_segments, config=config, save=False)


@pytest.fixture
def synthetic_X_y(synthetic_features_df):
    """Return X, y, groups arrays."""
    from src.feature_extraction import get_feature_columns
    from sklearn.preprocessing import LabelEncoder
    df = synthetic_features_df
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    groups = df["subject_id"].values
    return X, y, groups, feature_cols
