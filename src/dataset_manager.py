"""
src/dataset_manager.py
======================
High-level dataset management — registration, validation,
subject inventory.  Bridges data_loader.py with database.py.

Usage:
    from src.dataset_manager import DatasetManager
    manager = DatasetManager()
    manager.register_dataset()
    info = manager.get_dataset_info()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import get_config
from src.data_loader import load_dataset, OpenNeuroNBackDataset
from src.database import (
    init_db,
    get_or_create_dataset,
    get_session,
    Subject,
    Dataset,
)

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages dataset registration and subject inventory in the database.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.dataset_loader = load_dataset(config=self.config)

    # ---------------------------------------------------------------- #
    # Registration
    # ---------------------------------------------------------------- #

    def register_dataset(self) -> int:
        """
        Register the dataset in the database (idempotent).
        Returns the database id of the dataset record.
        """
        init_db()
        cfg = self.config
        info = self.dataset_loader.get_dataset_info()

        ds = get_or_create_dataset(
            name=cfg.dataset.name,
            dataset_id=cfg.dataset.id,
            source=cfg.dataset.source,
            version=cfg.dataset.version,
            license_=cfg.dataset.license,
            description=cfg.dataset.description,
            subjects=info.get("subjects_available", 0) or cfg.dataset.subjects,
            channels=cfg.dataset.channels,
            sampling_rate=cfg.dataset.sampling_rate,
        )
        return ds.id

    def register_subjects(self, dataset_db_id: int) -> int:
        """
        Scan the filesystem for available subjects and register them
        in the subjects table.  Skips already-registered subjects.

        Returns number of newly registered subjects.
        """
        try:
            subject_list = self.dataset_loader.get_subject_list()
        except FileNotFoundError as e:
            logger.warning("Cannot register subjects: %s", e)
            return 0

        newly_added = 0
        with get_session() as session:
            for sub_id in subject_list:
                existing = (
                    session.query(Subject)
                    .filter_by(dataset_id=dataset_db_id, subject_code=sub_id)
                    .first()
                )
                if not existing:
                    sub = Subject(
                        dataset_id=dataset_db_id,
                        subject_code=sub_id,
                        session="ses-01",
                        notes="",
                    )
                    session.add(sub)
                    newly_added += 1

        logger.info(
            "Subjects registered: %d new / %d total",
            newly_added,
            len(subject_list),
        )
        return newly_added

    # ---------------------------------------------------------------- #
    # Information
    # ---------------------------------------------------------------- #

    def get_dataset_info(self) -> dict:
        """Return dataset metadata as a dict (for dashboard display)."""
        return self.dataset_loader.get_dataset_info()

    def get_subject_list(self) -> List[str]:
        """Return list of available subject IDs."""
        return self.dataset_loader.get_subject_list()

    def is_data_available(self) -> bool:
        """Return True if the dataset directory exists and has valid files."""
        return self.dataset_loader.validate_data_directory()

    def get_registered_subjects_from_db(self, dataset_db_id: int) -> List[str]:
        """Return subject codes registered in the database."""
        with get_session() as session:
            subjects = (
                session.query(Subject)
                .filter_by(dataset_id=dataset_db_id)
                .all()
            )
            codes = [s.subject_code for s in subjects]
        return codes

    def full_setup(self) -> Dict[str, int]:
        """
        Convenience: initialise DB, register dataset and subjects.
        Returns {'dataset_id': <int>, 'n_subjects': <int>}
        """
        db_id = self.register_dataset()
        n_new = self.register_subjects(db_id)
        return {"dataset_db_id": db_id, "n_subjects_new": n_new}
