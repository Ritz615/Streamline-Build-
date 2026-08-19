#!/usr/bin/env python
"""
scripts/download_dataset.py
===========================
Downloads the OpenNeuro ds007169 dataset.

Primary method:  openneuro-py Python package
Fallback method: AWS S3 via boto3 (OpenNeuro uses public S3)
Manual fallback: Printed instructions

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --force    # re-download even if exists
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, setup_logging, resolve_path
from src.database import init_db, get_or_create_dataset

logger = logging.getLogger(__name__)

DATASET_ID = "ds007169"
DATASET_VERSION = "1.0.0"
SNAPSHOT = "1.0.0"
S3_BUCKET = "openneuro.org"
S3_PREFIX = f"ds007169"


def check_existing(target_dir: Path) -> bool:
    """Return True if dataset already exists with at least one valid EEG file."""
    if not target_dir.exists():
        return False
    vhdr_files = list(target_dir.rglob("*.vhdr"))
    if vhdr_files:
        logger.info(
            "Dataset already found: %d .vhdr files in %s", len(vhdr_files), target_dir
        )
        return True
    return False


def attempt_openneuro_py(target_dir: Path) -> bool:
    """Download using the openneuro-py package."""
    try:
        import openneuro
        logger.info("Attempting download via openneuro-py...")
        openneuro.download(
            dataset=DATASET_ID,
            version=SNAPSHOT,
            target_dir=str(target_dir),
        )
        logger.info("openneuro-py download complete.")
        return True
    except ImportError:
        logger.warning("openneuro-py not installed. Trying next method.")
        return False
    except Exception as e:
        logger.warning("openneuro-py download failed: %s", e)
        return False


def attempt_aws_s3(target_dir: Path) -> bool:
    """Download using AWS CLI (boto3) — OpenNeuro datasets are public on S3."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from tqdm import tqdm

        logger.info("Attempting download via AWS S3...")
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
            region_name="us-east-1",
        )

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/")

        objects = []
        for page in pages:
            for item in page.get("Contents", []):
                key = item["Key"]
                rel_path = key.replace(f"{S3_PREFIX}/", "", 1)
                # Only download EEG modality files and root metadata
                if "sourcedata/" in rel_path or "/ecg/" in rel_path or "/pupil/" in rel_path or "/motion/" in rel_path:
                    continue
                objects.append(item)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Found %d relevant EEG & metadata files to download.", len(objects))
        target_dir.mkdir(parents=True, exist_ok=True)

        def _download_one(obj_item):
            k = obj_item["Key"]
            r_path = k.replace(f"{S3_PREFIX}/", "", 1)
            if not r_path:
                return
            l_path = target_dir / r_path
            l_path.parent.mkdir(parents=True, exist_ok=True)
            if l_path.exists() and l_path.stat().st_size > 0:
                return  # already downloaded
            # New thread-safe client per thread
            thread_s3 = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED, max_pool_connections=10),
                region_name="us-east-1",
            )
            thread_s3.download_file(S3_BUCKET, k, str(l_path))

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_one, obj) for obj in objects]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading EEG Dataset", unit="files"):
                pass

        logger.info("S3 parallel download complete.")
        return True

    except ImportError:
        logger.warning("boto3 not installed. Trying next method.")
        return False
    except Exception as e:
        logger.warning("S3 download failed: %s", e)
        return False


def print_manual_instructions(target_dir: Path) -> None:
    """Print manual download instructions."""
    print("\n" + "=" * 70)
    print("  MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print(f"""
The automatic download failed. Please download manually using one of
these methods:

METHOD 1 — OpenNeuro CLI (recommended):
  pip install openneuro-py
  openneuro-py download --dataset=ds007169 --target={target_dir}

METHOD 2 — DataLad:
  datalad install https://github.com/OpenNeuroDatasets/ds007169
  datalad get -r .

METHOD 3 — Manual website download:
  1. Visit: https://openneuro.org/datasets/ds007169
  2. Click "Download"
  3. Extract to: {target_dir}

After downloading, verify with:
  python main.py --status

Expected structure:
  {target_dir}/
  ├── dataset_description.json
  ├── participants.tsv
  ├── sub-01/
  │   └── eeg/
  │       ├── sub-01_task-nback_eeg.vhdr
  │       ├── sub-01_task-nback_eeg.eeg
  │       ├── sub-01_task-nback_eeg.vmrk
  │       └── sub-01_task-nback_events.tsv
  └── ...
""")
    print("=" * 70 + "\n")


def validate_download(target_dir: Path) -> bool:
    """Validate the downloaded dataset structure."""
    issues = []

    if not target_dir.exists():
        issues.append(f"Directory not found: {target_dir}")
        return False

    desc = target_dir / "dataset_description.json"
    if not desc.exists():
        issues.append("dataset_description.json missing")

    vhdr_files = list(target_dir.rglob("*.vhdr"))
    if not vhdr_files:
        issues.append("No .vhdr EEG files found")

    events_files = list(target_dir.rglob("*events.tsv"))
    if not events_files:
        issues.append("No events.tsv files found")

    if issues:
        logger.warning("Download validation issues: %s", "; ".join(issues))
        return False

    logger.info(
        "Download validation passed: %d subjects, %d EEG files",
        len([d for d in target_dir.glob("sub-*") if d.is_dir()]),
        len(vhdr_files),
    )
    return True


def register_in_db(config) -> None:
    """Register the dataset in SQLite."""
    init_db()
    get_or_create_dataset(
        name=config.dataset.name,
        dataset_id=config.dataset.id,
        source=config.dataset.source,
        version=config.dataset.version,
        license_=config.dataset.license,
        description=config.dataset.description,
        subjects=config.dataset.subjects,
        channels=config.dataset.channels,
        sampling_rate=config.dataset.sampling_rate,
    )


def download_dataset(config=None, force: bool = False) -> bool:
    """
    Main download function. Returns True if dataset is available.

    Called by pipeline.py and directly by the script.
    """
    cfg = config or get_config()
    raw_dir = resolve_path(cfg.paths.raw_data)
    target_dir = raw_dir / DATASET_ID

    logger.info("Dataset target directory: %s", target_dir)

    # Check if already downloaded
    if not force and check_existing(target_dir):
        register_in_db(cfg)
        return True

    if force:
        logger.info("Force re-download requested.")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Try download methods in order
    success = (
        attempt_openneuro_py(target_dir) or
        attempt_aws_s3(target_dir)
    )

    if success:
        valid = validate_download(target_dir)
        if valid:
            register_in_db(cfg)
            logger.info("✓ Dataset downloaded and validated successfully.")
            return True
        else:
            logger.warning("Download completed but validation found issues.")
            register_in_db(cfg)
            return True  # Return True even with warnings — user can proceed
    else:
        print_manual_instructions(target_dir)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OpenNeuro ds007169 dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config)

    success = download_dataset(config=config, force=args.force)
    sys.exit(0 if success else 1)
