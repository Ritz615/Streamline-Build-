"""
src/feature_extraction.py
=========================
Extracts cognitive-load-relevant features from each EEG window.

Features per window:
  ─────────────────────────────────────────────────
  Frequency band features (averaged across channels):
    theta_power     : absolute theta band power (4–8 Hz)
    alpha_power     : absolute alpha band power (8–13 Hz)
    beta_power      : absolute beta band power  (13–30 Hz)
    theta_relative  : theta / total band power
    alpha_relative  : alpha / total band power
    beta_relative   : beta / total band power

  Band ratios:
    theta_alpha_ratio  : theta / alpha
    theta_beta_ratio   : theta / beta
    beta_alpha_ratio   : beta / alpha

  Statistical features (averaged across channels):
    mean, std, variance, rms, skewness, kurtosis

  Entropy features (averaged across channels):
    spectral_entropy  : frequency-domain entropy
    sample_entropy    : nonlinear complexity measure

  Metadata columns (not features):
    subject_id, window_id, start_time, end_time, trial_type, label
  ─────────────────────────────────────────────────

Usage:
    from src.feature_extraction import extract_features_from_segments
    df = extract_features_from_segments(segments, config)
    df.to_csv("data/features/features.csv", index=False)
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal, stats

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Band power helper
# ------------------------------------------------------------------ #

def compute_band_power(
    data: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    method: str = "welch",
) -> float:
    """
    Compute mean absolute band power across all channels.

    Parameters:
        data  : shape (n_channels, n_samples) — single window
        sfreq : sampling frequency (Hz)
        low   : band low cutoff (Hz)
        high  : band high cutoff (Hz)

    Returns:
        Mean band power (µV²/Hz) averaged across channels.
    """
    n_channels, n_samples = data.shape
    powers = []

    for ch_idx in range(n_channels):
        ch_data = data[ch_idx, :]
        # Compute power spectral density with Welch method
        freqs, psd = signal.welch(
            ch_data,
            fs=sfreq,
            nperseg=min(n_samples, int(sfreq * 2)),  # 2-second segments
            noverlap=None,
            scaling="density",
        )
        # Select frequency band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        if not np.any(idx_band):
            powers.append(0.0)
            continue
        # Integrate using trapezoidal rule
        band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
        powers.append(float(band_power))

    return float(np.mean(powers))


def compute_all_band_powers(
    data: np.ndarray, sfreq: float, bands: dict
) -> Dict[str, float]:
    """
    Compute absolute power for all bands.

    Returns dict: {band_name: power, ...}
    """
    result = {}
    for band_name, band_cfg in bands.items():
        power = compute_band_power(
            data, sfreq,
            float(band_cfg["low"]),
            float(band_cfg["high"]),
        )
        result[f"{band_name}_power"] = power
    return result


# ------------------------------------------------------------------ #
# Relative power and ratios
# ------------------------------------------------------------------ #

def compute_relative_powers(band_powers: Dict[str, float], bands: list) -> Dict[str, float]:
    """
    Compute relative power for each band.
    relative = band_power / total_power (sum of specified bands)
    """
    total = sum(band_powers.get(f"{b}_power", 0.0) for b in bands)
    result = {}
    if total <= 0:
        for b in bands:
            result[f"{b}_relative"] = 0.0
        return result

    for b in bands:
        result[f"{b}_relative"] = band_powers.get(f"{b}_power", 0.0) / total

    return result


def compute_band_ratios(band_powers: Dict[str, float]) -> Dict[str, float]:
    """Compute band power ratios relevant to cognitive load."""
    theta = band_powers.get("theta_power", 0.0)
    alpha = band_powers.get("alpha_power", 1e-10)  # avoid division by zero
    beta = band_powers.get("beta_power", 1e-10)

    alpha = alpha if alpha > 0 else 1e-10
    beta = beta if beta > 0 else 1e-10

    return {
        "theta_alpha_ratio": theta / alpha,
        "theta_beta_ratio": theta / beta,
        "beta_alpha_ratio": beta / alpha,
    }


# ------------------------------------------------------------------ #
# Statistical features
# ------------------------------------------------------------------ #

def compute_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain statistical features averaged across channels.

    data: shape (n_channels, n_samples)
    """
    flat = data.flatten()

    # Per-channel stats, then average
    per_channel_mean = np.mean(data, axis=1)
    per_channel_std = np.std(data, axis=1)
    per_channel_var = np.var(data, axis=1)
    per_channel_rms = np.sqrt(np.mean(data ** 2, axis=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_channel_skew = stats.skew(data, axis=1)
        per_channel_kurt = stats.kurtosis(data, axis=1)

    return {
        "mean": float(np.mean(per_channel_mean)),
        "std": float(np.mean(per_channel_std)),
        "variance": float(np.mean(per_channel_var)),
        "rms": float(np.mean(per_channel_rms)),
        "skewness": float(np.mean(per_channel_skew)),
        "kurtosis": float(np.mean(per_channel_kurt)),
    }


# ------------------------------------------------------------------ #
# Entropy features
# ------------------------------------------------------------------ #

def compute_spectral_entropy(
    data: np.ndarray,
    sfreq: float,
    low: float = 1.0,
    high: float = 40.0,
) -> float:
    """
    Spectral entropy — measures the uniformity of the power spectrum.
    Higher entropy = more uniform / complex signal.

    Averaged across channels.
    """
    entropies = []
    for ch_idx in range(data.shape[0]):
        ch_data = data[ch_idx, :]
        n = len(ch_data)
        freqs, psd = signal.welch(
            ch_data,
            fs=sfreq,
            nperseg=min(n, int(sfreq * 2)),
        )
        idx = np.logical_and(freqs >= low, freqs <= high)
        psd_band = psd[idx]
        if psd_band.sum() <= 0:
            entropies.append(0.0)
            continue
        psd_norm = psd_band / psd_band.sum()
        # Shannon entropy
        ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        entropies.append(float(ent))

    return float(np.mean(entropies))


def compute_sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
    max_channels: int = 5,
) -> float:
    """
    Sample entropy — nonlinear complexity measure.
    Computationally expensive; limited to first max_channels channels.

    Higher values = more complex / less predictable signal.
    """
    # Limit channels for performance
    n_ch = min(data.shape[0], max_channels)
    entropies = []

    for ch_idx in range(n_ch):
        ch_data = data[ch_idx, :]
        if np.std(ch_data) <= 0:
            entropies.append(0.0)
            continue
        se = _sample_entropy_1d(ch_data, m=m, r_factor=r_factor)
        entropies.append(se)

    return float(np.mean(entropies)) if entropies else 0.0


def _sample_entropy_1d(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Fast sample entropy for 1D signal using vectorised NumPy.

    Reference:
      Richman & Moorman (2000). Am J Physiol Heart Circ Physiol.
    """
    N = len(x)
    if N < m + 2:
        return 0.0

    # Cap length for performance (sample entropy is O(N²))
    max_n = 300
    if N > max_n:
        x = x[:max_n]
        N = max_n

    def _count_matches_vec(template_len: int) -> int:
        """Vectorised template matching using NumPy broadcasting."""
        # Build template matrix: shape (N-template_len, template_len)
        indices = np.arange(N - template_len)
        templates = np.array([x[i:i + template_len] for i in indices])  # (M, L)
        # Compute pairwise Chebyshev distances
        diff = np.abs(templates[:, None, :] - templates[None, :, :])  # (M, M, L)
        max_diff = diff.max(axis=2)  # (M, M)
        # Count upper triangle (i < j) where distance < r
        upper = np.triu(max_diff < r, k=1)
        return int(upper.sum())

    r = r_factor * float(np.std(x)) if hasattr(x, '__len__') else r_factor
    A = _count_matches_vec(m + 1)
    B = _count_matches_vec(m)

    if B == 0:
        return 0.0
    if A == 0:
        return float(np.log(B + 1))  # approximate

    return float(-np.log(A / B))


# ------------------------------------------------------------------ #
# Per-window feature extraction
# ------------------------------------------------------------------ #

def extract_features_from_window(
    segment: Dict,
    config=None,
) -> Optional[Dict]:
    """
    Extract all features from one segment dict.

    Parameters:
        segment : dict from segmentation.py with keys:
                  subject_id, window_id, start_time, end_time,
                  trial_type, label, data, sfreq

    Returns:
        feature dict or None if extraction fails
    """
    cfg = config or get_config()
    bands_cfg = cfg.bands.as_dict() if hasattr(cfg.bands, "as_dict") else dict(cfg.bands)
    feat_cfg = cfg.features
    primary_bands = list(cfg.primary_bands)

    data = segment["data"]       # (n_channels, n_samples)
    sfreq = float(segment["sfreq"])
    subject_id = segment["subject_id"]
    window_id = segment["window_id"]

    if data is None or data.size == 0:
        return None

    # ---- Check for NaN / Inf ----
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.warning("[%s] Window %d: NaN/Inf in data, skipping.", subject_id, window_id)
        return None

    try:
        features = {
            "subject_id": subject_id,
            "window_id": window_id,
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "trial_type": segment["trial_type"],
            "label": segment["label"],
        }

        # ---- Absolute band powers ----
        primary_band_cfg = {b: bands_cfg[b] for b in primary_bands if b in bands_cfg}
        band_powers = compute_all_band_powers(data, sfreq, primary_band_cfg)
        features.update(band_powers)

        # ---- Relative powers ----
        rel_powers = compute_relative_powers(band_powers, primary_bands)
        features.update(rel_powers)

        # ---- Band ratios ----
        ratios = compute_band_ratios(band_powers)
        features.update(ratios)

        # ---- Statistical features ----
        stat_feats = compute_statistical_features(data)
        features.update(stat_feats)

        # ---- Entropy features ----
        se_cfg = feat_cfg
        features["spectral_entropy"] = compute_spectral_entropy(data, sfreq)
        m = int(se_cfg.get("sample_entropy_m", 2))
        r_fac = float(se_cfg.get("sample_entropy_r_factor", 0.2))
        features["sample_entropy"] = compute_sample_entropy(data, m=m, r_factor=r_fac)

        return features

    except Exception as e:
        logger.error(
            "[%s] Window %d: feature extraction error: %s",
            subject_id, window_id, e,
        )
        return None


# ------------------------------------------------------------------ #
# Batch feature extraction
# ------------------------------------------------------------------ #

def extract_features_from_segments(
    segments: List[Dict],
    config=None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Extract features from all segments and return a DataFrame.

    Parameters:
        segments : list of segment dicts from segmentation.py
        config   : Config object
        save     : whether to save features.csv

    Returns:
        DataFrame with one row per window
    """
    cfg = config or get_config()
    logger.info("Extracting features from %d windows...", len(segments))

    rows = []
    n_failed = 0

    for i, seg in enumerate(segments):
        if i % 100 == 0 and i > 0:
            logger.info("  Progress: %d/%d windows", i, len(segments))
        feats = extract_features_from_window(seg, config=cfg)
        if feats is not None:
            rows.append(feats)
        else:
            n_failed += 1

    if not rows:
        raise RuntimeError(
            "Feature extraction produced zero valid rows. "
            "Check segmentation and preprocessing steps."
        )

    df = pd.DataFrame(rows)

    # ---- Final validation ----
    numeric_cols = [c for c in df.columns if c not in
                    ("subject_id", "window_id", "trial_type", "label")]
    nan_count = df[numeric_cols].isna().sum().sum()
    inf_count = np.isinf(df[numeric_cols].values).sum()

    if nan_count > 0:
        logger.warning("Feature DataFrame: %d NaN values detected. Filling with column median.", nan_count)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if inf_count > 0:
        logger.warning("Feature DataFrame: %d Inf values detected. Replacing with large finite value.", inf_count)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    logger.info(
        "Feature extraction complete: %d windows, %d features, %d failed.",
        len(df), len(numeric_cols), n_failed,
    )

    if save:
        _save_features(df, cfg)

    return df


def _save_features(df: pd.DataFrame, config) -> None:
    """Save feature DataFrame to CSV."""
    out_path = resolve_path(config.paths.feature_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Features saved to: %s (%d rows × %d cols)", out_path, len(df), len(df.columns))


def load_features(config=None) -> pd.DataFrame:
    """Load the saved features.csv into a DataFrame."""
    cfg = config or get_config()
    fpath = resolve_path(cfg.paths.feature_file)
    if not fpath.exists():
        raise FileNotFoundError(
            f"Features file not found: {fpath}\n"
            "Please run:  python main.py --features"
        )
    df = pd.read_csv(fpath)
    logger.info("Features loaded: %d rows × %d cols from %s", len(df), len(df.columns), fpath)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature column names (excludes metadata)."""
    meta_cols = {"subject_id", "window_id", "start_time", "end_time",
                 "trial_type", "label", "session"}
    return [c for c in df.columns if c not in meta_cols]
