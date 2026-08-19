"""
src/segmentation.py
===================
Segments continuous preprocessed EEG into fixed-length windows.

Default:
  Window  = 4 seconds
  Overlap = 2 seconds  (step = 2 s)

Every segment (window) retains:
  - subject_id
  - window_id
  - start_time (s)
  - end_time   (s)
  - trial_type (nback_1 / nback_2 / nback_3 / nback_4)
  - label      (LOW / MODERATE / HIGH)

IMPORTANT:
  Labels are assigned based on WHICH task was running during
  each window.  A window is only labeled if it falls within
  a known n-back trial period (derived from events.tsv onset times).

Usage:
    from src.segmentation import segment_subject
    segments = segment_subject("sub-01", raw_preprocessed, events_df, config)
    # Returns list of dicts, each with {'data': np.ndarray, 'label': ..., ...}
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


def segment_subject(
    subject_id: str,
    raw,          # mne.io.BaseRaw
    events_df: pd.DataFrame,
    config=None,
) -> List[Dict]:
    """
    Segment preprocessed EEG into labeled windows.

    Strategy:
      1. Use a sliding window over the full recording.
      2. For each window, determine the dominant workload label
         by checking which n-back trials overlap with the window.
      3. Keep windows that have an unambiguous label.
      4. Discard windows that span multiple different labels
         (mixed-label windows are discarded for label clarity).
      5. Discard windows overlapping 'BAD_artifact' annotations.

    Parameters:
        subject_id : e.g. "sub-01"
        raw        : preprocessed MNE Raw
        events_df  : DataFrame from data_loader (onset, duration, trial_type, label)
        config     : Config object

    Returns:
        List of segment dicts:
        {
            'subject_id': str,
            'window_id':  int,
            'start_time': float,   # seconds from recording start
            'end_time':   float,
            'trial_type': str,     # e.g. 'nback_2'
            'label':      str,     # LOW / MODERATE / HIGH
            'data':       np.ndarray,  # shape (n_channels, n_samples)
            'channel_names': list,
            'sfreq':      float,
        }
    """
    cfg = config or get_config()
    seg_cfg = cfg.segmentation
    window_s = float(seg_cfg.get("window_seconds", 4.0))
    overlap_s = float(seg_cfg.get("overlap_seconds", 2.0))
    step_s = window_s - overlap_s

    sfreq = raw.info["sfreq"]
    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    data_array = raw.get_data()  # (n_channels, n_times)
    n_channels, n_times = data_array.shape
    ch_names = raw.ch_names
    total_duration_s = n_times / sfreq

    logger.info(
        "[%s] Segmenting %.1f s EEG → window=%.1f s, step=%.1f s",
        subject_id, total_duration_s, window_s, step_s,
    )

    # ---- Build label timeline ----
    # For each sample, assign a label (None if no task running)
    label_timeline = _build_label_timeline(
        events_df, n_times, sfreq, window_s
    )

    # ---- Build artifact mask ----
    artifact_mask = _build_artifact_mask(raw, n_times)

    # ---- Slide windows ----
    segments = []
    window_id = 0

    for start_sample in range(0, n_times - window_samples + 1, step_samples):
        end_sample = start_sample + window_samples
        start_s = start_sample / sfreq
        end_s = end_sample / sfreq

        # Check for artifacts in this window
        if np.any(artifact_mask[start_sample:end_sample]):
            continue   # skip artifact-contaminated windows

        # Determine window label
        window_labels = label_timeline[start_sample:end_sample]
        label, trial_type = _resolve_window_label(window_labels)

        if label is None:
            continue   # no task running, discard

        segment_data = data_array[:, start_sample:end_sample]

        # Basic validity check
        if np.any(np.isnan(segment_data)) or np.any(np.isinf(segment_data)):
            logger.warning(
                "[%s] Window %d has NaN/Inf values, skipping.", subject_id, window_id
            )
            continue

        segments.append({
            "subject_id": subject_id,
            "window_id": window_id,
            "start_time": round(start_s, 4),
            "end_time": round(end_s, 4),
            "trial_type": trial_type,
            "label": label,
            "data": segment_data,
            "channel_names": ch_names,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "n_samples": window_samples,
        })
        window_id += 1

    logger.info(
        "[%s] Segmentation: %d valid windows extracted (label counts: %s)",
        subject_id,
        len(segments),
        _count_labels(segments),
    )
    return segments


# ------------------------------------------------------------------ #
# Label timeline helpers
# ------------------------------------------------------------------ #

def _build_label_timeline(
    events_df: pd.DataFrame,
    n_times: int,
    sfreq: float,
    window_s: float,
) -> np.ndarray:
    """
    Build a sample-level array of (trial_type, label) tuples.
    Each sample gets the label of the task running at that moment.

    Strategy: For each n-back trial event:
      - onset_sample = onset * sfreq
      - We mark samples from onset to onset + window_s as that trial's label.
        (Because individual trial events in events.tsv are per-stimulus,
         we extend each event's label by window_s to create continuous blocks.)

    Returns ndarray of objects (tuples or None) of shape (n_times,).
    """
    # Initialize with None
    timeline = np.full(n_times, None, dtype=object)

    if events_df.empty or "onset" not in events_df.columns:
        logger.warning("Events DataFrame is empty. No labels assigned.")
        return timeline

    # Group by onset blocks
    # The events.tsv contains per-stimulus events.
    # We treat each nback_N stimulus event as belonging to the same block
    # as long as the trial_type remains the same.
    # We label each sample with the closest preceding event's trial_type.

    valid_events = events_df[
        events_df["trial_type"].notna() & events_df["label"].notna()
    ].copy()

    if valid_events.empty:
        return timeline

    # Sort by onset
    valid_events = valid_events.sort_values("onset").reset_index(drop=True)

    for i, row in valid_events.iterrows():
        onset_s = row["onset"]
        trial_type = row["trial_type"]
        label = row["label"]

        onset_sample = int(onset_s * sfreq)
        if onset_sample < 0 or onset_sample >= n_times:
            continue

        # Duration from events.tsv; if transient pulse (< 1.0s), extend to 2.0s stimulus block
        dur_s = float(row.get("duration", 0) or 2.0)
        if dur_s < 1.0:
            dur_s = 2.0  # default stimulus duration for n-back block

        end_sample = min(int((onset_s + dur_s) * sfreq), n_times)
        cell = (trial_type, label)
        for idx in range(onset_sample, end_sample):
            timeline[idx] = cell

    return timeline


def _build_artifact_mask(raw, n_times: int) -> np.ndarray:
    """
    Return a boolean mask of shape (n_times,).
    True = sample is within a 'BAD_artifact' annotation.
    """
    mask = np.zeros(n_times, dtype=bool)
    sfreq = raw.info["sfreq"]

    for annot in raw.annotations:
        if "BAD" in annot["description"].upper():
            onset_sample = int(annot["onset"] * sfreq)
            end_sample = int((annot["onset"] + annot["duration"]) * sfreq)
            onset_sample = max(0, onset_sample)
            end_sample = min(n_times, end_sample)
            mask[onset_sample:end_sample] = True

    return mask


def _resolve_window_label(window_labels: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the label for a window from its sample-level labels.

    Returns (label, trial_type) or (None, None) if:
      - No labeled samples in the window
      - Multiple conflicting labels (mixed-label window)
    """
    # Extract non-None tuples
    valid = [v for v in window_labels if v is not None]
    if not valid:
        return None, None

    # Count unique labels
    labels_seen = set(v[1] for v in valid)  # workload class
    types_seen = set(v[0] for v in valid)   # trial_type

    if len(labels_seen) > 1:
        # Ambiguous window — more than one workload class
        return None, None

    label = labels_seen.pop()
    trial_type = types_seen.pop() if len(types_seen) == 1 else "mixed"

    # Require at least 50% of samples to be labeled
    coverage = len(valid) / len(window_labels)
    if coverage < 0.5:
        return None, None

    return label, trial_type


def _count_labels(segments: List[Dict]) -> Dict[str, int]:
    """Count occurrences of each label in a list of segments."""
    counts: Dict[str, int] = {}
    for seg in segments:
        lbl = seg.get("label", "UNKNOWN")
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# ------------------------------------------------------------------ #
# Batch segmentation
# ------------------------------------------------------------------ #

def segment_all_subjects(
    dataset_loader,
    preprocessed_loader,
    config=None,
) -> List[Dict]:
    """
    Segment all preprocessed subjects.

    Parameters:
        dataset_loader      : OpenNeuroNBackDataset instance
        preprocessed_loader : function(subject_id) → MNE Raw
        config              : Config object

    Returns:
        Flat list of all segment dicts across all subjects.
    """
    cfg = config or get_config()
    subjects = dataset_loader.get_subject_list()
    all_segments = []

    for sub_id in subjects:
        raw = preprocessed_loader(sub_id)
        if raw is None:
            logger.warning("[%s] No preprocessed data found, skipping.", sub_id)
            continue

        _, events_df, _ = dataset_loader.load_subject(sub_id)

        segs = segment_subject(sub_id, raw, events_df, config=cfg)
        all_segments.extend(segs)
        logger.info("[%s] %d segments added. Total so far: %d", sub_id, len(segs), len(all_segments))

    logger.info("Total segments across all subjects: %d", len(all_segments))
    return all_segments
