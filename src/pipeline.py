"""
src/pipeline.py
===============
Orchestrates the complete EEG cognitive load research pipeline.

Stages:
  1. download    — check / download dataset
  2. preprocess  — EEG preprocessing (filter, reference, artifact rejection)
  3. features    — segment + extract features → features.csv
  4. train       — train fuzzy + Random Forest models
  5. evaluate    — cross-validated evaluation + confusion matrices
  6. report      — generate final experiment report

Each stage can be run independently or as part of --all.

Usage:
    from src.pipeline import Pipeline
    p = Pipeline()
    p.run_all()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.database import (
    init_db,
    record_feature_run,
    record_model_run,
    record_processing_run,
    store_fuzzy_rules,
)
from src.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main research pipeline.

    Usage:
        p = Pipeline()
        p.run_download()
        p.run_preprocess()
        p.run_features()
        p.run_train()
        p.run_evaluate()
        p.run_all()
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self._dataset_db_id: Optional[int] = None
        self._processing_run_id: Optional[int] = None
        self._feature_run_id: Optional[int] = None

    # ---------------------------------------------------------------- #
    # Stage 1: Download
    # ---------------------------------------------------------------- #

    def run_download(self) -> bool:
        """Check dataset availability and download if needed."""
        logger.info("=" * 60)
        logger.info("STAGE 1: DATASET")
        logger.info("=" * 60)

        init_db()
        manager = DatasetManager(config=self.config)

        if manager.is_data_available():
            logger.info("Dataset already available. Skipping download.")
            result = manager.full_setup()
            self._dataset_db_id = result["dataset_db_id"]
            return True

        logger.info("Dataset not found. Attempting download...")

        # Import the download script logic
        try:
            from scripts.download_dataset import download_dataset
            success = download_dataset(config=self.config)
        except ImportError:
            logger.warning("download_dataset script not importable. Trying inline download.")
            success = self._attempt_download()

        if success:
            result = manager.full_setup()
            self._dataset_db_id = result["dataset_db_id"]
            logger.info("Dataset stage complete.")
        else:
            logger.error(
                "Dataset download failed.\n"
                "Please download manually:\n"
                "  openneuro-py download --dataset=ds007169 --target=data/raw/ds007169\n"
                "Or visit: https://openneuro.org/datasets/ds007169"
            )

        return success

    def _attempt_download(self) -> bool:
        """Inline download attempt using openneuro-py."""
        try:
            import openneuro as on
            raw_dir = resolve_path(self.config.paths.raw_data)
            target = raw_dir / "ds007169"
            target.mkdir(parents=True, exist_ok=True)
            on.download(dataset="ds007169", target_dir=str(target))
            return True
        except Exception as e:
            logger.error("openneuro-py download failed: %s", e)
            return False

    # ---------------------------------------------------------------- #
    # Stage 2: Preprocess
    # ---------------------------------------------------------------- #

    def run_preprocess(self) -> List[Dict]:
        """Preprocess all subjects."""
        logger.info("=" * 60)
        logger.info("STAGE 2: PREPROCESSING")
        logger.info("=" * 60)

        from src.data_loader import load_dataset
        from src.preprocessing import preprocess_all_subjects

        dataset_loader = load_dataset(config=self.config)

        # Ensure dataset is registered
        if self._dataset_db_id is None:
            manager = DatasetManager(config=self.config)
            result = manager.full_setup()
            self._dataset_db_id = result["dataset_db_id"]

        reports = preprocess_all_subjects(dataset_loader, config=self.config)

        # Record in database
        n_success = sum(1 for r in reports if r.get("status") in ("success", "skipped"))
        pp = self.config.preprocessing
        seg = self.config.segmentation

        self._processing_run_id = record_processing_run(
            dataset_db_id=self._dataset_db_id,
            filter_low=float(pp.get("low_frequency", 1.0)),
            filter_high=float(pp.get("high_frequency", 40.0)),
            notch_frequency=float(pp.get("notch_frequency", 50.0)),
            window_seconds=float(seg.get("window_seconds", 4.0)),
            overlap_seconds=float(seg.get("overlap_seconds", 2.0)),
            preprocessing_version=str(pp.get("version", "1.0")),
            n_subjects=n_success,
        )

        logger.info("Preprocessing stage complete. %d subjects processed.", n_success)
        return reports

    # ---------------------------------------------------------------- #
    # Stage 3: Features
    # ---------------------------------------------------------------- #

    def run_features(self) -> pd.DataFrame:
        """Segment EEG and extract features."""
        logger.info("=" * 60)
        logger.info("STAGE 3: FEATURE EXTRACTION")
        logger.info("=" * 60)

        from src.data_loader import load_dataset
        from src.preprocessing import load_preprocessed
        from src.segmentation import segment_subject
        from src.feature_extraction import extract_features_from_segments
        from src.feature_selection import select_features
        from src.visualization import run_eda, plot_feature_importance

        dataset_loader = load_dataset(config=self.config)
        subjects = dataset_loader.get_subject_list()

        all_segments = []
        for sub_id in subjects:
            raw = load_preprocessed(sub_id, config=self.config)
            if raw is None:
                logger.warning("[%s] No preprocessed data. Run --preprocess first.", sub_id)
                continue

            _, events_df, _ = dataset_loader.load_subject(sub_id)
            segs = segment_subject(sub_id, raw, events_df, config=self.config)
            all_segments.extend(segs)

        if not all_segments:
            raise RuntimeError(
                "No segments extracted. Ensure preprocessing completed successfully."
            )

        logger.info("Total segments: %d. Extracting features...", len(all_segments))
        df = extract_features_from_segments(all_segments, config=self.config, save=True)

        # Feature selection
        from src.feature_extraction import get_feature_columns
        feature_cols = get_feature_columns(df)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df["label"].values)
        X = df[feature_cols].values
        selected, importance_df = select_features(df, config=self.config, save=True)

        # EDA
        logger.info("Running EDA...")
        run_eda(df, config=self.config)
        plot_feature_importance(importance_df, config=self.config)

        # Record in database
        if self._processing_run_id is None:
            self._processing_run_id = 1  # fallback

        self._feature_run_id = record_feature_run(
            processing_run_id=self._processing_run_id,
            feature_version=str(self.config.features.get("version", "1.0")),
            number_of_features=len(feature_cols),
            feature_names=json.dumps(feature_cols),
            n_samples=len(df),
        )

        logger.info(
            "Feature stage complete. %d features, %d windows.", len(feature_cols), len(df)
        )
        return df

    # ---------------------------------------------------------------- #
    # Stage 4: Train
    # ---------------------------------------------------------------- #

    def run_train(self) -> Dict:
        """Train fuzzy and Random Forest models."""
        logger.info("=" * 60)
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("=" * 60)

        from src.feature_extraction import load_features, get_feature_columns
        from src.feature_selection import get_top_features
        from src.fuzzy_classifier import FuzzyClassifier, get_fuzzy_rules_for_display
        from src.random_forest import RandomForestModel
        from sklearn.preprocessing import LabelEncoder

        df = load_features(config=self.config)
        feature_cols = get_feature_columns(df)
        X_all = df[feature_cols].values

        le = LabelEncoder()
        y_all = le.fit_transform(df["label"].values)
        groups = df["subject_id"].values

        # ---- Top features for fuzzy ----
        try:
            fuzzy_features = get_top_features(n=5, config=self.config)
        except FileNotFoundError:
            from src.feature_selection import select_features
            fuzzy_features, _ = select_features(df, config=self.config, n_top=5)

        fuzzy_feature_idx = [feature_cols.index(f) for f in fuzzy_features if f in feature_cols]
        X_fuzzy = X_all[:, fuzzy_feature_idx]
        actual_fuzzy_features = [feature_cols[i] for i in fuzzy_feature_idx]

        # ---- Fuzzy Classifier ----
        logger.info("Training Fuzzy Classifier...")
        logger.info("Fuzzy features: %s", actual_fuzzy_features)
        fuzzy_clf = FuzzyClassifier(config=self.config)
        fuzzy_clf.train(X_fuzzy, y_all, actual_fuzzy_features)
        fuzzy_path = fuzzy_clf.save(config=self.config)

        # Store rules in DB
        rules = get_fuzzy_rules_for_display(fuzzy_clf)
        store_fuzzy_rules(rules)
        logger.info("Fuzzy rules stored in database.")

        # ---- Random Forest ----
        logger.info("Training Random Forest...")
        rf_model = RandomForestModel(config=self.config)
        rf_model.train_final(X_all, y_all, feature_cols)
        rf_path = rf_model.save(config=self.config)

        logger.info("Training stage complete.")
        return {
            "fuzzy_model_path": fuzzy_path,
            "rf_model_path": rf_path,
            "fuzzy_features": actual_fuzzy_features,
            "all_features": feature_cols,
        }

    # ---------------------------------------------------------------- #
    # Stage 5: Evaluate
    # ---------------------------------------------------------------- #

    def run_evaluate(self) -> Dict:
        """Cross-validate both models and generate result files."""
        logger.info("=" * 60)
        logger.info("STAGE 5: EVALUATION")
        logger.info("=" * 60)

        from src.feature_extraction import load_features, get_feature_columns
        from src.feature_selection import get_top_features
        from src.fuzzy_classifier import FuzzyClassifier
        from src.random_forest import RandomForestModel
        from src.evaluation import (
            evaluate_fuzzy_cv,
            evaluate_rf_cv,
            save_comparison_csv,
            plot_confusion_matrix,
            plot_model_comparison,
        )
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        df = load_features(config=self.config)
        feature_cols = get_feature_columns(df)
        X_all = df[feature_cols].values

        le = LabelEncoder()
        y_all = le.fit_transform(df["label"].values)
        groups = df["subject_id"].values

        # Fuzzy features
        try:
            fuzzy_features = get_top_features(n=5, config=self.config)
        except FileNotFoundError:
            from src.feature_selection import select_features
            fuzzy_features, _ = select_features(df, config=self.config, n_top=5)

        fuzzy_feature_idx = [feature_cols.index(f) for f in fuzzy_features if f in feature_cols]
        X_fuzzy = X_all[:, fuzzy_feature_idx]
        actual_fuzzy_features = [feature_cols[i] for i in fuzzy_feature_idx]

        n_splits = int(self.config.validation.get("n_splits", 5))

        # ---- Evaluate Random Forest ----
        logger.info("Evaluating Random Forest (subject-wise CV)...")
        rf_model = RandomForestModel(config=self.config)
        rf_metrics = evaluate_rf_cv(
            rf_model, X_all, y_all, groups, feature_cols, n_splits=n_splits
        )

        # ---- Evaluate Fuzzy ----
        logger.info("Evaluating Fuzzy Classifier (subject-wise CV)...")
        fuzzy_clf = FuzzyClassifier(config=self.config)
        fuzzy_metrics = evaluate_fuzzy_cv(
            fuzzy_clf, X_fuzzy, y_all, groups,
            actual_fuzzy_features, n_splits=n_splits, config=self.config,
        )

        # ---- Save comparison ----
        save_comparison_csv(fuzzy_metrics, rf_metrics, config=self.config)

        # ---- Confusion matrices ----
        if fuzzy_metrics.get("confusion_matrix"):
            cm_fuzzy = np.array(fuzzy_metrics["confusion_matrix"])
            plot_confusion_matrix(cm_fuzzy, "Fuzzy Classifier", config=self.config)

        if rf_metrics.get("confusion_matrix"):
            cm_rf = np.array(rf_metrics["confusion_matrix"])
            plot_confusion_matrix(cm_rf, "Random Forest", config=self.config)

        plot_model_comparison(fuzzy_metrics, rf_metrics, config=self.config)

        # ---- Record in database ----
        feature_run_id = self._feature_run_id or 1

        record_model_run(
            model_name="Fuzzy Classifier",
            feature_run_id=feature_run_id,
            model_version=str(self.config.models.fuzzy.get("version", "1.0")),
            accuracy=fuzzy_metrics.get("accuracy", 0),
            precision=fuzzy_metrics.get("precision_macro", 0),
            recall=fuzzy_metrics.get("recall_macro", 0),
            f1=fuzzy_metrics.get("f1_macro", 0),
            balanced_accuracy=fuzzy_metrics.get("balanced_accuracy", 0),
            n_train=0,
            n_test=fuzzy_metrics.get("n_samples", 0),
            n_folds=n_splits,
            model_path=str(resolve_path(self.config.paths.models) / "fuzzy" / "fuzzy_classifier.joblib"),
        )
        record_model_run(
            model_name="Random Forest",
            feature_run_id=feature_run_id,
            model_version="1.0",
            accuracy=rf_metrics.get("accuracy", 0),
            precision=rf_metrics.get("precision_macro", 0),
            recall=rf_metrics.get("recall_macro", 0),
            f1=rf_metrics.get("f1_macro", 0),
            balanced_accuracy=rf_metrics.get("balanced_accuracy", 0),
            n_train=0,
            n_test=rf_metrics.get("n_samples", len(df)),
            n_folds=n_splits,
            model_path=str(resolve_path(self.config.paths.models) / "random_forest" / "random_forest.joblib"),
        )

        f_acc = float(fuzzy_metrics.get("accuracy", 0.0))
        f_f1 = float(fuzzy_metrics.get("f1_macro", 0.0))
        f_bal = float(fuzzy_metrics.get("balanced_accuracy", 0.0))
        r_acc = float(rf_metrics.get("accuracy", 0.0))
        r_f1 = float(rf_metrics.get("f1_macro", 0.0))
        r_bal = float(rf_metrics.get("balanced_accuracy", 0.0))

        logger.info(
            f"\n{'='*60}\n"
            f"  EVALUATION RESULTS\n"
            f"{'='*60}\n"
            f"  Fuzzy Classifier:\n"
            f"    Accuracy:  {f_acc:.3f}\n"
            f"    F1 Macro:  {f_f1:.3f}\n"
            f"    Bal. Acc.: {f_bal:.3f}\n\n"
            f"  Random Forest:\n"
            f"    Accuracy:  {r_acc:.3f}\n"
            f"    F1 Macro:  {r_f1:.3f}\n"
            f"    Bal. Acc.: {r_bal:.3f}\n"
            f"{'='*60}"
        )

        return {"fuzzy": fuzzy_metrics, "rf": rf_metrics}

    # ---------------------------------------------------------------- #
    # Stage 6: Report
    # ---------------------------------------------------------------- #

    def run_report(self) -> str:
        """Generate the final experiment report."""
        logger.info("Generating final experiment report...")

        from src.evaluation import load_comparison_csv
        from src.feature_selection import load_feature_importance
        from src.feature_extraction import load_features

        cfg = self.config
        reports_dir = resolve_path(cfg.paths.reports)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "final_experiment_report.md"

        lines = _build_report(cfg)
        report_text = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Report saved: %s", report_path)
        return str(report_path)

    # ---------------------------------------------------------------- #
    # Run all stages
    # ---------------------------------------------------------------- #

    def run_all(self) -> None:
        """Execute the complete pipeline."""
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  EEG COGNITIVE LOAD CLASSIFICATION PIPELINE            ║")
        logger.info("╚" + "═" * 58 + "╝")
        start = datetime.now()

        try:
            ok = self.run_download()
            if not ok:
                raise RuntimeError("Dataset not available. Cannot continue.")

            self.run_preprocess()
            self.run_features()
            self.run_train()
            results = self.run_evaluate()
            self.run_report()

        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            raise

        elapsed = (datetime.now() - start).total_seconds()
        logger.info("Pipeline complete in %.1f s", elapsed)


# ------------------------------------------------------------------ #
# Report builder
# ------------------------------------------------------------------ #

def _build_report(cfg) -> List[str]:
    """Build the markdown experiment report."""
    from src.evaluation import load_comparison_csv

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Final Experiment Report",
        f"**Generated:** {now}",
        "",
        "## Disclaimer",
        "> This is a **research prototype** using public anonymized EEG data.",
        "> It is **NOT** a medical device, clinical tool, or diagnostic system.",
        "> All results are for academic research purposes only.",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        f"- **Dataset ID:** {cfg.dataset.id}",
        f"- **Name:** {cfg.dataset.name}",
        f"- **Source:** {cfg.dataset.source}",
        f"- **License:** {cfg.dataset.license}",
        f"- **Subjects:** {cfg.dataset.subjects}",
        f"- **EEG Channels:** {cfg.dataset.channels} (10-20 montage)",
        f"- **Sampling Rate:** {cfg.dataset.sampling_rate} Hz",
        f"- **Format:** BrainVision (.vhdr/.eeg/.vmrk)",
        "",
        "## 2. Workload Label Mapping",
        "",
        "The following mapping was applied to n-back difficulty levels.",
        "**This is an operational research categorization based on task difficulty.**",
        "It is NOT a medically validated cognitive load scale.",
        "The dataset contains no rest/baseline condition.",
        "",
        "| Trial Type | Workload Class |",
        "|------------|----------------|",
        "| nback_1    | LOW            |",
        "| nback_2    | MODERATE       |",
        "| nback_3    | HIGH           |",
        "| nback_4    | HIGH           |",
        "",
        "## 3. Preprocessing",
        "",
        f"- Band-pass filter: {cfg.preprocessing.low_frequency}–{cfg.preprocessing.high_frequency} Hz",
        f"- Notch filter: {cfg.preprocessing.notch_frequency} Hz",
        f"- Re-reference: {cfg.preprocessing.reference}",
        f"- Artifact rejection: ±{cfg.preprocessing.artifact_threshold_uv} µV threshold",
        "- ICA: NOT applied (documented rationale in PROJECT_GUIDE.md)",
        "",
        "## 4. Segmentation",
        "",
        f"- Window length: {cfg.segmentation.window_seconds} s",
        f"- Overlap: {cfg.segmentation.overlap_seconds} s",
        "",
        "## 5. Features Extracted",
        "",
        "- Theta, alpha, beta absolute band power",
        "- Theta, alpha, beta relative band power",
        "- Theta/alpha, theta/beta, beta/alpha ratios",
        "- Statistical: mean, std, variance, RMS, skewness, kurtosis",
        "- Entropy: spectral entropy, sample entropy",
        "",
        "## 6. Model Validation",
        "",
        "**Subject-wise cross-validation** (StratifiedGroupKFold) was used.",
        "Windows from the same subject do NOT appear in both train and test folds.",
        "This prevents data leakage and tests generalization to unseen subjects.",
        "",
        "## 7. Results",
        "",
    ]

    # Try to add actual results
    try:
        comp_df = load_comparison_csv(cfg)
        if comp_df is not None:
            lines.append("| Model | Accuracy | F1 (macro) | Precision | Recall |")
            lines.append("|-------|----------|------------|-----------|--------|")
            for _, row in comp_df.iterrows():
                lines.append(
                    f"| {row['model']} | {row['accuracy']:.3f} | {row['f1_macro']:.3f} | "
                    f"{row['precision_macro']:.3f} | {row['recall_macro']:.3f} |"
                )
    except Exception:
        lines.append("*Run `python main.py --evaluate` to populate results.*")

    lines += [
        "",
        "## 8. Limitations",
        "",
        "- Dataset contains only 18 subjects (limited generalizability)",
        "- No rest/baseline condition; 1-back used as LOW proxy",
        "- Three-class labels are derived from task difficulty, not direct brain state measurement",
        "- EEG is noisy and varies significantly between individuals",
        "- Results may not generalize to populations outside this dataset",
        "- Model is a research prototype, NOT validated for clinical use",
        "- Performance may be affected by class imbalance (nback_3 and nback_4 both map to HIGH)",
        "",
        "## 9. Conclusion",
        "",
        "This prototype demonstrates an end-to-end EEG cognitive load classification",
        "pipeline combining classical signal processing, fuzzy logic, and Random Forest.",
        "The fuzzy classifier provides transparent, rule-based explanations for each",
        "prediction, making the system interpretable for academic analysis.",
        "",
        "---",
        "*Generated by EEG Cognitive Load Classification Research Prototype v1.0*",
    ]

    return lines
