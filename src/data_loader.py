"""
src/data_loader.py
==================
Dataset abstraction layer.

Architecture:
    BaseDataset (abstract)
        └── OpenNeuroNBackDataset   ← ds007169, primary dataset
        └── STEWDataset             ← future, optional

OpenNeuroNBackDataset:
  - Locates BrainVision (.vhdr) files
  - Reads events.tsv to extract workload labels
  - Filters out tutorial trials
  - Applies configurable label mapping
  - Returns per-subject raw MNE objects and label information

Usage:
    from src.data_loader import OpenNeuroNBackDataset
    ds = OpenNeuroNBackDataset()
    subjects = ds.get_subject_list()
    raw, events, label_map = ds.load_subject("sub-01")
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Base class
# ------------------------------------------------------------------ #

class BaseDataset(ABC):
    """
    Abstract base class for EEG dataset loaders.

    Subclass this to add support for a new dataset.
    Implement: get_subject_list(), load_subject(), get_label_mapping().
    """

    def __init__(self, config=None):
        self.config = config or get_config()

    @abstractmethod
    def get_subject_list(self) -> List[str]:
        """Return list of subject IDs available in the dataset."""
        ...

    @abstractmethod
    def load_subject(
        self, subject_id: str
    ) -> Tuple[mne.io.BaseRaw, pd.DataFrame, Dict[str, str]]:
        """
        Load EEG data for one subject.

        Returns:
            raw         : MNE Raw object (continuous EEG)
            events_df   : DataFrame of events with columns
                          [onset, duration, trial_type, nback_n, ...]
            label_map   : mapping from trial_type → workload class
        """
        ...

    @abstractmethod
    def get_label_mapping(self) -> Dict[str, str]:
        """Return trial_type → workload class mapping from config."""
        ...

    def validate_data_directory(self) -> bool:
        """Check whether the raw data directory exists and has expected files."""
        raise NotImplementedError("Subclass must implement validate_data_directory()")


# ------------------------------------------------------------------ #
# OpenNeuro ds007169 — Cognitive Workload 5-level n-back
# ------------------------------------------------------------------ #

class OpenNeuroNBackDataset(BaseDataset):
    """
    Loader for OpenNeuro ds007169:
      'Cognitive Workload 5-level n-back'

    Dataset characteristics (verified from source):
      - 18 subjects
      - 19 EEG channels (10-20: Fp1, Fp2, F7, F3, Fz, F4, F8,
                         T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2)
      - 250 Hz sampling rate
      - BrainVision format (.vhdr / .eeg / .vmrk)
      - Tasks: nback_1, nback_2, nback_3, nback_4
      - No rest/baseline condition
      - Tutorial trials excluded via istutorial flag

    Label mapping (configurable in config.yaml):
      nback_1 → LOW
      nback_2 → MODERATE
      nback_3 → HIGH
      nback_4 → HIGH

    IMPORTANT:
      This mapping is an OPERATIONAL RESEARCH CATEGORIZATION
      based on task difficulty. It is not a medically validated
      cognitive load scale. The dataset contains no baseline/rest.
    """

    DATASET_ID = "ds007169"
    TASK_NAME = "nback"
    EXPECTED_CHANNELS = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2",
    ]
    SAMPLING_RATE = 250

    def __init__(self, config=None):
        super().__init__(config)
        raw_base = resolve_path(self.config.paths.raw_data)
        self.data_dir = raw_base / self.DATASET_ID
        self._label_map = self._build_label_map()
        logger.info("OpenNeuroNBackDataset initialised. Data dir: %s", self.data_dir)

    # ---------------------------------------------------------------- #
    # Label mapping
    # ---------------------------------------------------------------- #

    def _build_label_map(self) -> Dict[str, str]:
        """Build trial_type → workload class map from config.yaml."""
        cfg_map = self.config.label_mapping
        l1 = cfg_map.get("nback_1", "LOW")
        l2 = cfg_map.get("nback_2", "MODERATE")
        l3 = cfg_map.get("nback_3", "HIGH")
        l4 = cfg_map.get("nback_4", "HIGH")

        mapping = {
            "nback_1": l1,
            "nback_2": l2,
            "nback_3": l3,
            "nback_4": l4,
            "1-back": l1,
            "2-back": l2,
            "3-back": l3,
            "4-back": l4,
        }
        logger.debug("Label mapping: %s", mapping)
        return mapping

    def get_label_mapping(self) -> Dict[str, str]:
        return dict(self._label_map)

    # ---------------------------------------------------------------- #
    # Subject list
    # ---------------------------------------------------------------- #

    def get_subject_list(self) -> List[str]:
        """
        Return sorted list of subject IDs found in the data directory.
        Only includes subjects with a valid .vhdr file.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                "Please run:  python main.py --download"
            )

        subjects = []
        for sub_dir in sorted(self.data_dir.glob("sub-*")):
            if sub_dir.is_dir():
                vhdr_files = list(sub_dir.glob("eeg/*.vhdr"))
                if vhdr_files:
                    subjects.append(sub_dir.name)
                else:
                    logger.warning(
                        "Subject %s: no .vhdr file found, skipping.", sub_dir.name
                    )
        if not subjects:
            raise RuntimeError(
                f"No subjects with valid EEG files found in: {self.data_dir}\n"
                "Run:  python main.py --download"
            )
        logger.info("Found %d subjects: %s", len(subjects), subjects[:5])
        return subjects

    # ---------------------------------------------------------------- #
    # Load single subject
    # ---------------------------------------------------------------- #

    def load_subject(
        self, subject_id: str
    ) -> Tuple[mne.io.BaseRaw, pd.DataFrame, Dict[str, str]]:
        """
        Load EEG data and events for one subject.

        Parameters:
            subject_id : e.g. "sub-01"

        Returns:
            raw        : MNE Raw (all EEG channels, unchanged)
            events_df  : cleaned events DataFrame with workload labels
            label_map  : trial_type → class string
        """
        subject_dir = self.data_dir / subject_id / "eeg"
        if not subject_dir.exists():
            raise FileNotFoundError(
                f"Subject directory not found: {subject_dir}"
            )

        # Find .vhdr file
        vhdr_files = sorted(subject_dir.glob("*.vhdr"))
        if not vhdr_files:
            raise FileNotFoundError(
                f"No .vhdr file for {subject_id} in {subject_dir}"
            )
        vhdr_path = vhdr_files[0]

        # Load EEG with MNE
        logger.info("[%s] Loading EEG: %s", subject_id, vhdr_path.name)
        raw = mne.io.read_raw_brainvision(
            str(vhdr_path),
            preload=True,
            verbose=False,
        )

        # Validate sampling rate
        sfreq = raw.info["sfreq"]
        if sfreq != self.SAMPLING_RATE:
            logger.warning(
                "[%s] Unexpected sampling rate: %.1f Hz (expected %d Hz).",
                subject_id, sfreq, self.SAMPLING_RATE,
            )

        # Load events from events.tsv
        events_df = self._load_events_tsv(subject_id, subject_dir)

        # Set EEG channel types where needed
        raw = self._set_channel_types(raw)

        # Validate channels
        self._validate_channels(raw, subject_id)

        logger.info(
            "[%s] Loaded %.1f s EEG, %d events.",
            subject_id,
            raw.times[-1],
            len(events_df),
        )
        return raw, events_df, self._label_map

    # ---------------------------------------------------------------- #
    # Events
    # ---------------------------------------------------------------- #

    def _load_events_tsv(
        self, subject_id: str, subject_dir: Path
    ) -> pd.DataFrame:
        """
        Load and clean events.tsv for a subject.

        Filters:
          - Removes tutorial trials (istutorial == True/true)
          - Keeps only task-relevant trial_types (nback_1..4)
          - Adds 'label' column with workload class
        """
        tsv_files = sorted(subject_dir.glob(f"*_{self.TASK_NAME}_events.tsv"))
        if not tsv_files:
            # Try any events.tsv in the folder
            tsv_files = sorted(subject_dir.glob("*events.tsv"))

        if not tsv_files:
            logger.warning("[%s] No events.tsv found. Returning empty events.", subject_id)
            return pd.DataFrame(columns=["onset", "duration", "trial_type", "label"])

        tsv_path = tsv_files[0]
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

        logger.debug("[%s] Raw events: %d rows, columns: %s", subject_id, len(df), list(df.columns))

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # ---- Filter tutorial trials ----
        cfg_map = self.config.label_mapping
        exclude_tutorial = cfg_map.get("exclude_tutorial", True)
        if exclude_tutorial and "istutorial" in df.columns:
            before = len(df)
            # Handle string and bool representations
            df["istutorial_norm"] = df["istutorial"].astype(str).str.lower()
            df = df[df["istutorial_norm"] != "true"].copy()
            df = df.drop(columns=["istutorial_norm"])
            logger.debug(
                "[%s] Removed %d tutorial trials.", subject_id, before - len(df)
            )

        # ---- Keep only relevant trial_types ----
        valid_types = list(self._label_map.keys())
        if "trial_type" in df.columns:
            df = df[df["trial_type"].isin(valid_types)].copy()
        else:
            logger.warning("[%s] 'trial_type' column not found in events.tsv.", subject_id)
            return pd.DataFrame(columns=["onset", "duration", "trial_type", "label"])

        # ---- Add workload label column ----
        df["label"] = df["trial_type"].map(self._label_map)

        # ---- Ensure onset is numeric ----
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        df["duration"] = pd.to_numeric(df.get("duration", 0), errors="coerce").fillna(0)
        df = df.dropna(subset=["onset"])
        df = df.sort_values("onset").reset_index(drop=True)

        # ---- Log class distribution ----
        if "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
            logger.info("[%s] Events after filtering: %s", subject_id, counts)

        return df

    # ---------------------------------------------------------------- #
    # Channel handling
    # ---------------------------------------------------------------- #

    def _set_channel_types(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Ensure proper channel type assignments.
        BrainVision files may include ECG channels that need explicit typing.
        """
        ch_types_map = {}
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if "ECG" in ch_upper or ch_upper in ("A1", "A2"):
                ch_types_map[ch] = "ecg"
            elif "EOG" in ch_upper:
                ch_types_map[ch] = "eog"
        if ch_types_map:
            raw.set_channel_types(ch_types_map)
            logger.debug("Channel types set: %s", ch_types_map)
        return raw

    def _validate_channels(self, raw: mne.io.BaseRaw, subject_id: str) -> None:
        """Warn if expected EEG channels are missing."""
        eeg_chs = [c for c in raw.ch_names if raw.get_channel_types([c])[0] == "eeg"]
        missing = [c for c in self.EXPECTED_CHANNELS if c not in eeg_chs]
        if missing:
            logger.warning(
                "[%s] Missing expected EEG channels: %s", subject_id, missing
            )
        extra = [c for c in eeg_chs if c not in self.EXPECTED_CHANNELS]
        if extra:
            logger.debug("[%s] Extra channels (will be ignored if not EEG): %s", subject_id, extra)

    # ---------------------------------------------------------------- #
    # Validation
    # ---------------------------------------------------------------- #

    def validate_data_directory(self) -> bool:
        """
        Check that the dataset directory exists and contains at least
        one valid subject directory.
        """
        if not self.data_dir.exists():
            return False
        try:
            subs = self.get_subject_list()
            return len(subs) > 0
        except Exception:
            return False

    def get_dataset_info(self) -> dict:
        """Return a summary dict of dataset metadata for display."""
        try:
            subjects = self.get_subject_list()
            n_subjects = len(subjects)
            available = True
        except Exception:
            n_subjects = 0
            available = False

        return {
            "dataset_id": self.DATASET_ID,
            "name": "Cognitive Workload 5-level n-back",
            "source": "https://openneuro.org/datasets/ds007169",
            "license": "CC0",
            "subjects_available": n_subjects,
            "subjects_expected": 18,
            "channels": 19,
            "sampling_rate_hz": self.SAMPLING_RATE,
            "format": "BrainVision (.vhdr/.eeg/.vmrk)",
            "tasks": ["nback_1", "nback_2", "nback_3", "nback_4"],
            "label_mapping": self._label_map,
            "data_available": available,
        }


# ------------------------------------------------------------------ #
# STEW Dataset stub (future support)
# ------------------------------------------------------------------ #

class STEWDataset(BaseDataset):
    """
    Stub for STEW (Simultaneous Task EEG Workload) dataset.

    STEW characteristics:
      - 48 subjects
      - 14 EEG channels
      - 128 Hz
      - Rest and multitasking conditions
      - Perceived workload ratings (SWAT scale)

    NOT required for MVP. Implement when ready to extend.
    """

    DATASET_ID = "STEW"

    def get_subject_list(self) -> List[str]:
        raise NotImplementedError(
            "STEW dataset support is not implemented in this version. "
            "See PROJECT_GUIDE.md for extension instructions."
        )

    def load_subject(self, subject_id: str):
        raise NotImplementedError("STEW loader not implemented.")

    def get_label_mapping(self):
        raise NotImplementedError("STEW label mapping not implemented.")

    def validate_data_directory(self) -> bool:
        return False


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

DATASET_REGISTRY = {
    "ds007169": OpenNeuroNBackDataset,
    "STEW": STEWDataset,
}


def load_dataset(dataset_id: Optional[str] = None, config=None) -> BaseDataset:
    """
    Factory: return the correct dataset loader.

    Parameters:
        dataset_id : override the dataset ID (default: from config.yaml)
        config     : Config instance (default: load from config.yaml)

    Returns:
        BaseDataset subclass instance
    """
    if config is None:
        config = get_config()
    if dataset_id is None:
        dataset_id = config.dataset.id

    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset ID: {dataset_id!r}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    cls = DATASET_REGISTRY[dataset_id]
    return cls(config=config)
