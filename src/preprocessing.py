"""
src/preprocessing.py
====================
EEG preprocessing pipeline.

Steps applied in order:
  1. Pick EEG channels (drop ECG, EOG, misc)
  2. Detect obviously bad channels (flat signal)
  3. Band-pass filter (default: 1–40 Hz, FIR)
  4. Notch filter (default: 50 Hz)
  5. Re-reference to average
  6. Amplitude-based artifact rejection (±150 µV threshold)
  7. Save processed output to data/processed/

Note on ICA:
  ICA is NOT applied in this MVP.
  Reason: ICA requires reliable identification of artifact components
  (typically eye blinks / muscle) which benefits from manual inspection
  or enough epochs.  For this academic prototype, amplitude thresholding
  provides reasonable artifact removal without the complexity of ICA.
  ICA can be added later via the apply_ica flag in config.yaml.

Usage:
    from src.preprocessing import preprocess_subject
    processed_path = preprocess_subject("sub-01", raw, config)
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Main preprocessing entry point
# ------------------------------------------------------------------ #

def preprocess_subject(
    subject_id: str,
    raw: mne.io.BaseRaw,
    config=None,
    save: bool = True,
) -> Tuple[mne.io.BaseRaw, dict]:
    """
    Run the full preprocessing pipeline on one subject's raw EEG.

    Parameters:
        subject_id : e.g. "sub-01"
        raw        : MNE Raw object (loaded but not yet preprocessed)
        config     : Config object (defaults to config.yaml)
        save       : whether to save the preprocessed .fif file

    Returns:
        raw_clean  : preprocessed MNE Raw
        report     : dict with preprocessing metadata
    """
    cfg = config or get_config()
    pp = cfg.preprocessing

    report = {
        "subject_id": subject_id,
        "original_n_channels": len(raw.ch_names),
        "original_duration_s": raw.times[-1] if len(raw.times) > 0 else 0,
        "sampling_rate_hz": raw.info["sfreq"],
        "steps_applied": [],
        "bad_channels": [],
        "n_bad_channels": 0,
        "artifacts_rejected": False,
        "preprocessing_version": pp.get("version", "1.0"),
    }

    logger.info("[%s] Starting preprocessing (%.1f s EEG)", subject_id, report["original_duration_s"])

    # ---- Step 1: Pick EEG channels ----
    raw = _pick_eeg_channels(raw, subject_id)
    report["n_eeg_channels"] = len(raw.ch_names)
    report["steps_applied"].append("pick_eeg_channels")

    if len(raw.ch_names) == 0:
        raise ValueError(f"[{subject_id}] No EEG channels found after channel selection.")

    # ---- Step 2: Detect flat / bad channels ----
    bad_channels = _detect_bad_channels(raw, subject_id)
    if bad_channels:
        raw.info["bads"] = bad_channels
        logger.warning("[%s] Bad channels marked: %s", subject_id, bad_channels)
    report["bad_channels"] = bad_channels
    report["n_bad_channels"] = len(bad_channels)
    report["steps_applied"].append("detect_bad_channels")

    # ---- Step 3: Band-pass filter ----
    l_freq = float(pp.get("low_frequency", 1.0))
    h_freq = float(pp.get("high_frequency", 40.0))
    logger.info("[%s] Band-pass filtering: %.1f–%.1f Hz", subject_id, l_freq, h_freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
    report["filter_low_hz"] = l_freq
    report["filter_high_hz"] = h_freq
    report["steps_applied"].append(f"bandpass_{l_freq}_{h_freq}Hz")

    # ---- Step 4: Notch filter ----
    notch_freq = float(pp.get("notch_frequency", 50.0))
    logger.info("[%s] Notch filter: %.1f Hz", subject_id, notch_freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = raw.notch_filter(
            freqs=[notch_freq],
            method="fir",
            verbose=False,
        )
    report["notch_hz"] = notch_freq
    report["steps_applied"].append(f"notch_{notch_freq}Hz")

    # ---- Step 5: Average reference ----
    ref = pp.get("reference", "average")
    if ref == "average":
        logger.info("[%s] Setting average reference", subject_id)
        raw = raw.set_eeg_reference("average", projection=False, verbose=False)
        report["reference"] = "average"
        report["steps_applied"].append("average_reference")

    # ---- Step 6: Artifact rejection (amplitude threshold) ----
    threshold_uv = float(pp.get("artifact_threshold_uv", 150.0))
    raw, n_artifacts = _mark_artifacts(raw, subject_id, threshold_uv)
    report["artifact_threshold_uv"] = threshold_uv
    report["n_artifact_annotations"] = n_artifacts
    report["steps_applied"].append(f"artifact_threshold_{threshold_uv}uV")
    if n_artifacts > 0:
        report["artifacts_rejected"] = True

    # ---- Step 7: Save ----
    save_path = None
    if save:
        save_path = _save_preprocessed(raw, subject_id, cfg)
        report["saved_path"] = str(save_path)

    report["final_n_channels"] = len(raw.ch_names)
    report["final_duration_s"] = raw.times[-1] if len(raw.times) > 0 else 0

    logger.info(
        "[%s] Preprocessing complete. %d channels, %.1f s. Steps: %s",
        subject_id,
        report["final_n_channels"],
        report["final_duration_s"],
        report["steps_applied"],
    )
    return raw, report


# ------------------------------------------------------------------ #
# Step helpers
# ------------------------------------------------------------------ #

def _pick_eeg_channels(
    raw: mne.io.BaseRaw, subject_id: str
) -> mne.io.BaseRaw:
    """
    Keep only EEG channels.
    Drops ECG, EOG, STIM, and any misc channels.
    """
    try:
        raw.pick("eeg", verbose=False)
        logger.debug(
            "[%s] Picked %d EEG channels: %s",
            subject_id,
            len(raw.ch_names),
            raw.ch_names[:5],
        )
    except Exception as e:
        logger.warning("[%s] Error picking EEG channels: %s", subject_id, e)
    return raw


def _detect_bad_channels(
    raw: mne.io.BaseRaw,
    subject_id: str,
    flat_threshold_uv: float = 0.5,
    high_var_factor: float = 5.0,
) -> List[str]:
    """
    Simple bad channel detection:
      - Flat channels: std < flat_threshold_uv
      - Excessively noisy channels: std > high_var_factor * median(std)

    Returns list of bad channel names.
    """
    data = raw.get_data()  # shape (n_channels, n_times)
    # Robustly handle units: convert to uV if currently in Volts
    data_uv = data * 1e6 if np.std(data) < 1e-3 else data
    channel_stds = np.std(data_uv, axis=1)
    median_std = np.median(channel_stds)

    bad_channels = []
    for i, ch in enumerate(raw.ch_names):
        if channel_stds[i] < flat_threshold_uv:
            logger.warning(
                "[%s] Flat channel detected: %s (std=%.3f µV)",
                subject_id, ch, channel_stds[i],
            )
            bad_channels.append(ch)
        elif median_std > 0 and channel_stds[i] > high_var_factor * median_std:
            logger.warning(
                "[%s] High-variance channel: %s (std=%.1f µV vs median=%.1f µV)",
                subject_id, ch, channel_stds[i], median_std,
            )
            bad_channels.append(ch)

    return bad_channels


def _mark_artifacts(
    raw: mne.io.BaseRaw,
    subject_id: str,
    threshold_uv: float = 150.0,
) -> Tuple[mne.io.BaseRaw, int]:
    """
    Mark time segments exceeding amplitude threshold as 'BAD_artifact'.
    These annotations are respected by MNE when creating epochs.

    Uses a 0.5-second window to scan for threshold crossings.
    """
    data = raw.get_data()  # (n_channels, n_times)
    data_uv = data * 1e6 if np.std(data) < 1e-3 else data
    sfreq = raw.info["sfreq"]
    window_samples = int(0.5 * sfreq)  # 0.5-second windows
    n_samples = data_uv.shape[1]

    artifact_onsets = []
    artifact_durations = []

    for start in range(0, n_samples - window_samples, window_samples):
        segment = data_uv[:, start : start + window_samples]
        max_amp = np.max(np.abs(segment))
        if max_amp > threshold_uv:
            onset_s = start / sfreq
            duration_s = window_samples / sfreq
            artifact_onsets.append(onset_s)
            artifact_durations.append(duration_s)

    if artifact_onsets:
        annotations = mne.Annotations(
            onset=artifact_onsets,
            duration=artifact_durations,
            description=["BAD_artifact"] * len(artifact_onsets),
        )
        existing = raw.annotations
        raw.set_annotations(existing + annotations)
        logger.info(
            "[%s] Marked %d artifact segments (threshold=%.1f µV)",
            subject_id, len(artifact_onsets), threshold_uv,
        )
    else:
        logger.info("[%s] No artifacts detected (threshold=%.1f µV)", subject_id, threshold_uv)

    return raw, len(artifact_onsets)


def _save_preprocessed(
    raw: mne.io.BaseRaw,
    subject_id: str,
    config,
) -> Path:
    """Save preprocessed Raw as .fif file."""
    processed_dir = resolve_path(config.paths.processed_data) / subject_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / f"{subject_id}_preprocessed_raw.fif"
    raw.save(str(out_path), overwrite=True, verbose=False)
    logger.info("[%s] Preprocessed EEG saved: %s", subject_id, out_path)
    return out_path


# ------------------------------------------------------------------ #
# Batch preprocessing
# ------------------------------------------------------------------ #

def preprocess_all_subjects(
    dataset_loader,
    config=None,
    subjects: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> List[dict]:
    """
    Preprocess all (or specified) subjects.

    Parameters:
        dataset_loader : OpenNeuroNBackDataset instance
        config         : Config object
        subjects       : list of subject IDs to process (None = all)
        skip_existing  : skip if preprocessed .fif already exists

    Returns:
        List of preprocessing report dicts (one per subject)
    """
    cfg = config or get_config()
    if subjects is None:
        subjects = dataset_loader.get_subject_list()

    reports = []
    processed_dir_base = resolve_path(cfg.paths.processed_data)

    for sub_id in subjects:
        out_path = processed_dir_base / sub_id / f"{sub_id}_preprocessed_raw.fif"

        if skip_existing and out_path.exists():
            logger.info("[%s] Already preprocessed, skipping.", sub_id)
            reports.append({
                "subject_id": sub_id,
                "status": "skipped",
                "saved_path": str(out_path),
            })
            continue

        try:
            raw, events_df, _ = dataset_loader.load_subject(sub_id)
            _, report = preprocess_subject(sub_id, raw, config=cfg, save=True)
            report["status"] = "success"
            reports.append(report)
        except Exception as e:
            logger.error("[%s] Preprocessing failed: %s", sub_id, e)
            reports.append({
                "subject_id": sub_id,
                "status": "failed",
                "error": str(e),
            })

    success = sum(1 for r in reports if r.get("status") == "success")
    skipped = sum(1 for r in reports if r.get("status") == "skipped")
    failed = sum(1 for r in reports if r.get("status") == "failed")
    logger.info(
        "Preprocessing complete: %d success, %d skipped, %d failed",
        success, skipped, failed,
    )
    return reports


def load_preprocessed(subject_id: str, config=None) -> Optional[mne.io.BaseRaw]:
    """
    Load a previously preprocessed .fif file.
    Returns None if the file does not exist.
    """
    cfg = config or get_config()
    fif_path = (
        resolve_path(cfg.paths.processed_data)
        / subject_id
        / f"{subject_id}_preprocessed_raw.fif"
    )
    if not fif_path.exists():
        logger.warning("Preprocessed file not found: %s", fif_path)
        return None

    logger.info("[%s] Loading preprocessed EEG from %s", subject_id, fif_path)
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    return raw
