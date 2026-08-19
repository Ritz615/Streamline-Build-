"""
src/random_forest.py
====================
Random Forest baseline model for EEG cognitive load classification.

Key design decisions:
  - Subject-wise cross-validation (StratifiedGroupKFold) is MANDATORY
  - Windows from the same subject must not appear in both train and test
  - This prevents data leakage and tests generalization to new subjects
  - class_weight='balanced' to handle class imbalance

Usage:
    from src.random_forest import RandomForestModel
    rf = RandomForestModel()
    results = rf.cross_validate(X, y, groups, feature_names)
    rf.train_final(X, y, feature_names)
    rf.save()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

CLASS_NAMES = ["LOW", "MODERATE", "HIGH"]


class RandomForestModel:
    """
    Random Forest classifier with subject-wise cross-validation.

    Attributes:
        model        : trained sklearn RandomForestClassifier
        feature_names: list of feature column names
        label_encoder: fitted LabelEncoder
        cv_results   : list of per-fold metric dicts
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.label_encoder = LabelEncoder()
        self.cv_results: List[Dict] = []
        self.is_trained = False

    def _make_model(self) -> RandomForestClassifier:
        """Create a RandomForestClassifier from config settings."""
        rf_cfg = self.config.models.random_forest
        return RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth") or None,
            min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            class_weight=str(rf_cfg.get("class_weight", "balanced")),
            random_state=int(rf_cfg.get("random_state", 42)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
        )

    # ---------------------------------------------------------------- #
    # Cross-validation (MANDATORY subject-wise)
    # ---------------------------------------------------------------- #

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5,
    ) -> Dict:
        """
        Subject-wise stratified cross-validation.

        IMPORTANT:
          We use StratifiedGroupKFold to ensure:
          1. Each fold has balanced class distribution (stratified)
          2. No subject appears in both train and test (group-based)

          This is the scientifically correct way to evaluate EEG models.
          Random window splitting would inflate accuracy by leaking
          subject-specific EEG patterns from train to test.

        Parameters:
            X            : feature matrix
            y            : encoded integer labels
            groups       : subject ID array (same subject = same group)
            feature_names: feature column names
            n_splits     : number of CV folds

        Returns:
            dict with mean/std of all metrics across folds
        """
        self.feature_names = feature_names
        self.label_encoder.fit(y)

        logger.info(
            "Starting %d-fold subject-wise cross-validation. "
            "%d samples, %d features, %d subjects",
            n_splits, len(X), len(feature_names), len(np.unique(groups)),
        )
        logger.info(
            "⚠️  Subject-wise splitting: subjects in test set are NOT seen during training."
        )

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_subjects = np.unique(groups[train_idx])
            test_subjects = np.unique(groups[test_idx])

            # Verify no subject overlap
            overlap = set(train_subjects) & set(test_subjects)
            if overlap:
                logger.error("DATA LEAKAGE: subjects %s in both train and test!", overlap)

            logger.info(
                "Fold %d: train=%d samples (%d subjects), test=%d samples (%d subjects)",
                fold_idx + 1,
                len(X_train), len(train_subjects),
                len(X_test), len(test_subjects),
            )

            # Train
            fold_model = self._make_model()
            fold_model.fit(X_train, y_train)

            # Predict
            y_pred = fold_model.predict(X_test)
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

            # Metrics for this fold
            metrics = self._compute_metrics(y_test, y_pred, fold=fold_idx + 1)
            fold_metrics.append(metrics)
            logger.info(
                "Fold %d: acc=%.3f, f1=%.3f, bal_acc=%.3f",
                fold_idx + 1, metrics["accuracy"], metrics["f1_macro"], metrics["balanced_accuracy"],
            )

        # Aggregate across folds
        agg = self._aggregate_metrics(fold_metrics)
        agg["confusion_matrix_all"] = confusion_matrix(all_y_true, all_y_pred).tolist()
        agg["classification_report"] = classification_report(
            all_y_true, all_y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        self.cv_results = fold_metrics
        logger.info(
            "Cross-validation complete:\n"
            "  Accuracy:          %.3f ± %.3f\n"
            "  Balanced Accuracy: %.3f ± %.3f\n"
            "  F1 (macro):        %.3f ± %.3f\n"
            "  Precision (macro): %.3f ± %.3f\n"
            "  Recall (macro):    %.3f ± %.3f",
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["balanced_accuracy_mean"], agg["balanced_accuracy_std"],
            agg["f1_macro_mean"], agg["f1_macro_std"],
            agg["precision_macro_mean"], agg["precision_macro_std"],
            agg["recall_macro_mean"], agg["recall_macro_std"],
        )
        return agg

    # ---------------------------------------------------------------- #
    # Final model training (on all data)
    # ---------------------------------------------------------------- #

    def train_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """
        Train final Random Forest on all data (after CV evaluation).

        This model is used for prediction in the dashboard.
        NOTE: The performance metrics come from cross-validation,
              not from this final model's training data.
        """
        self.feature_names = feature_names
        logger.info("Training final Random Forest on %d samples...", len(X))
        self.model = self._make_model()
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Final Random Forest trained. OOB score not applicable (oob_score=False).")

    # ---------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (integers)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_final() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (n_samples × 3)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_final() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a DataFrame, sorted descending."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ---------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------- #

    def _compute_metrics(self, y_true, y_pred, fold: int = 0) -> Dict:
        return {
            "fold": fold,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def _aggregate_metrics(self, fold_metrics: List[Dict]) -> Dict:
        result = {}
        keys = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        for k in keys:
            vals = [m[k] for m in fold_metrics]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"] = float(np.std(vals))
        return result

    # ---------------------------------------------------------------- #
    # Save / Load
    # ---------------------------------------------------------------- #

    def save(self, config=None) -> str:
        """Save model and metadata to disk."""
        cfg = config or self.config
        save_dir = resolve_path(cfg.paths.models) / "random_forest"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(save_dir / "random_forest.joblib")
        meta_path = save_dir / "random_forest_meta.json"

        joblib.dump(self.model, model_path)

        meta = {
            "feature_names": self.feature_names,
            "cv_results": self.cv_results,
            "model_params": self.model.get_params() if self.model else {},
            "class_names": CLASS_NAMES,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Random Forest saved: %s", model_path)
        return model_path

    def load(self, config=None) -> None:
        """Load model and metadata from disk."""
        cfg = config or self.config
        save_dir = resolve_path(cfg.paths.models) / "random_forest"
        model_path = save_dir / "random_forest.joblib"
        meta_path = save_dir / "random_forest_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Random Forest model not found: {model_path}\n"
                "Please run:  python main.py --train"
            )

        self.model = joblib.load(str(model_path))

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.cv_results = meta.get("cv_results", [])

        self.is_trained = True
        logger.info("Random Forest loaded from: %s", model_path)
