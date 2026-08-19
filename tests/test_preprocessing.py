"""tests/test_preprocessing.py — Tests for EEG preprocessing pipeline."""
import numpy as np
import pytest


def test_preprocess_subject_runs(synthetic_raw, config):
    """Preprocessing should complete without errors."""
    from src.preprocessing import preprocess_subject
    raw_out, report = preprocess_subject("sub-test", synthetic_raw, config=config, save=False)

    assert raw_out is not None
    assert isinstance(report, dict)
    assert "steps_applied" in report
    assert len(report["steps_applied"]) > 0


def test_output_has_eeg_channels(synthetic_raw, config):
    """Preprocessed output should retain EEG channels."""
    from src.preprocessing import preprocess_subject
    raw_out, report = preprocess_subject("sub-test", synthetic_raw, config=config, save=False)

    assert len(raw_out.ch_names) > 0
    assert report["final_n_channels"] > 0


def test_no_nan_after_preprocessing(synthetic_raw, config):
    """Preprocessed data should contain no NaN values."""
    from src.preprocessing import preprocess_subject
    raw_out, _ = preprocess_subject("sub-test", synthetic_raw, config=config, save=False)

    data = raw_out.get_data()
    assert not np.any(np.isnan(data)), "NaN found in preprocessed data"


def test_no_inf_after_preprocessing(synthetic_raw, config):
    """Preprocessed data should contain no infinite values."""
    from src.preprocessing import preprocess_subject
    raw_out, _ = preprocess_subject("sub-test", synthetic_raw, config=config, save=False)

    data = raw_out.get_data()
    assert not np.any(np.isinf(data)), "Inf found in preprocessed data"


def test_correct_sampling_rate_preserved(synthetic_raw, config):
    """Sampling rate should be unchanged after preprocessing."""
    from src.preprocessing import preprocess_subject
    original_sfreq = synthetic_raw.info["sfreq"]
    raw_out, _ = preprocess_subject("sub-test", synthetic_raw, config=config, save=False)

    assert raw_out.info["sfreq"] == original_sfreq


def test_band_power_computed(synthetic_raw, config):
    """Band power function should return positive finite values."""
    from src.feature_extraction import compute_band_power
    data = synthetic_raw.get_data()
    power = compute_band_power(data, sfreq=250.0, low=4.0, high=8.0)

    assert power >= 0
    assert np.isfinite(power)


def test_bad_channel_detection(synthetic_raw, config):
    """Flat channel should be detected as bad."""
    import mne
    from src.preprocessing import _detect_bad_channels

    # Make one channel flat
    raw_copy = synthetic_raw.copy()
    data = raw_copy.get_data()
    data[0, :] = 0.0  # make Fp1 flat
    info = raw_copy.info
    raw_flat = mne.io.RawArray(data, info, verbose=False)

    bads = _detect_bad_channels(raw_flat, "sub-test", flat_threshold_uv=0.5)
    assert "Fp1" in bads
