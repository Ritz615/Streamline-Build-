"""tests/test_features.py — Tests for feature extraction module."""
import numpy as np
import pandas as pd
import pytest


def test_feature_extraction_runs(synthetic_segments, config):
    """Feature extraction should run without errors."""
    from src.feature_extraction import extract_features_from_segments
    df = extract_features_from_segments(synthetic_segments, config=config, save=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_expected_columns_exist(synthetic_features_df):
    """Feature DataFrame should contain all expected columns."""
    expected = [
        "subject_id", "window_id", "label",
        "theta_power", "alpha_power", "beta_power",
        "theta_relative", "alpha_relative", "beta_relative",
        "theta_alpha_ratio", "theta_beta_ratio", "beta_alpha_ratio",
        "mean", "std", "variance", "rms", "skewness", "kurtosis",
        "spectral_entropy", "sample_entropy",
    ]
    for col in expected:
        assert col in synthetic_features_df.columns, f"Missing column: {col}"


def test_feature_values_are_numeric(synthetic_features_df):
    """All feature columns should contain numeric values."""
    from src.feature_extraction import get_feature_columns
    feat_cols = get_feature_columns(synthetic_features_df)
    for col in feat_cols:
        assert pd.api.types.is_numeric_dtype(synthetic_features_df[col]), \
            f"Column {col!r} is not numeric"


def test_no_nan_in_features(synthetic_features_df):
    """Feature DataFrame should not contain NaN after extraction."""
    from src.feature_extraction import get_feature_columns
    feat_cols = get_feature_columns(synthetic_features_df)
    nan_count = synthetic_features_df[feat_cols].isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in features"


def test_no_inf_in_features(synthetic_features_df):
    """Feature DataFrame should not contain infinite values."""
    from src.feature_extraction import get_feature_columns
    feat_cols = get_feature_columns(synthetic_features_df)
    inf_count = np.isinf(synthetic_features_df[feat_cols].values).sum()
    assert inf_count == 0, f"Found {inf_count} Inf values in features"


def test_labels_valid(synthetic_features_df):
    """Labels should only contain valid class values."""
    valid_labels = {"LOW", "MODERATE", "HIGH"}
    labels = set(synthetic_features_df["label"].unique())
    assert labels.issubset(valid_labels), f"Invalid labels found: {labels - valid_labels}"


def test_band_powers_non_negative(synthetic_features_df):
    """Band power values must be non-negative."""
    for col in ["theta_power", "alpha_power", "beta_power"]:
        if col in synthetic_features_df.columns:
            assert (synthetic_features_df[col] >= 0).all(), \
                f"{col} has negative values"


def test_relative_power_sums_to_one(synthetic_features_df):
    """Relative powers should sum to approximately 1.0 per window."""
    rel_cols = ["theta_relative", "alpha_relative", "beta_relative"]
    if all(c in synthetic_features_df.columns for c in rel_cols):
        total = synthetic_features_df[rel_cols].sum(axis=1)
        assert (total >= 0.99).all() and (total <= 1.01).all(), \
            "Relative powers do not sum to 1.0"


def test_spectral_entropy_positive(synthetic_features_df):
    """Spectral entropy must be positive."""
    if "spectral_entropy" in synthetic_features_df.columns:
        assert (synthetic_features_df["spectral_entropy"] >= 0).all()


def test_feature_count_reasonable(synthetic_features_df):
    """Feature count should be within expected range."""
    from src.feature_extraction import get_feature_columns
    feat_cols = get_feature_columns(synthetic_features_df)
    assert 15 <= len(feat_cols) <= 30, \
        f"Unexpected feature count: {len(feat_cols)}"
