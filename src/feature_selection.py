"""
src/feature_selection.py
========================
Selects the most discriminative features for the fuzzy classifier.

Methods used:
  1. Kruskal-Wallis H-test  — non-parametric ANOVA between classes
  2. Mutual Information      — information-theoretic importance
  3. Combined ranking        — average rank of both methods

Output:
  results/metrics/feature_importance.csv
  (feature, kruskal_stat, kruskal_pvalue, mutual_info, combined_rank)

The top N features (default 5) are selected as fuzzy classifier inputs.

IMPORTANT:
  Feature importance shows which features are discriminative for this
  model on this dataset. It does NOT imply causality between EEG
  features and cognitive load.

Usage:
    from src.feature_selection import select_features
    selected, importance_df = select_features(df, config)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from src.config import get_config, resolve_path
from src.feature_extraction import get_feature_columns

logger = logging.getLogger(__name__)


def select_features(
    df: pd.DataFrame,
    config=None,
    n_top: Optional[int] = None,
    save: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Compute feature importance and return top N feature names.

    Parameters:
        df     : features DataFrame (from feature_extraction.py)
        config : Config object
        n_top  : override number of top features (default: from config)
        save   : save importance CSV

    Returns:
        selected_features : list of top feature column names
        importance_df     : DataFrame with all features ranked
    """
    cfg = config or get_config()
    fs_cfg = cfg.feature_selection
    n_top = n_top or int(fs_cfg.get("n_top_features", 5))
    random_state = int(fs_cfg.get("random_state", 42))

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y_str = df["label"].values

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    logger.info(
        "Feature selection: %d features, %d samples, %d classes",
        len(feature_cols), len(y), len(le.classes_),
    )

    # ---- Kruskal-Wallis ----
    kruskal_stats, kruskal_pvals = _run_kruskal_wallis(X, y, feature_cols, le.classes_)

    # ---- Mutual Information ----
    mi_scores = _run_mutual_info(X, y, random_state)

    # ---- Combined ranking ----
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "kruskal_stat": kruskal_stats,
        "kruskal_pvalue": kruskal_pvals,
        "mutual_info": mi_scores,
    })

    # Rank each method (1 = best)
    importance_df["rank_kruskal"] = importance_df["kruskal_stat"].rank(ascending=False)
    importance_df["rank_mi"] = importance_df["mutual_info"].rank(ascending=False)
    importance_df["combined_rank"] = (
        importance_df["rank_kruskal"] + importance_df["rank_mi"]
    ) / 2.0

    importance_df = importance_df.sort_values("combined_rank").reset_index(drop=True)
    importance_df["final_rank"] = range(1, len(importance_df) + 1)

    selected_features = importance_df["feature"].head(n_top).tolist()

    logger.info("Top %d features selected: %s", n_top, selected_features)
    logger.info(
        "Feature importance (top 10):\n%s",
        importance_df[["feature", "kruskal_stat", "mutual_info", "combined_rank"]]
        .head(10)
        .to_string(index=False),
    )

    if save:
        _save_importance(importance_df, cfg)

    return selected_features, importance_df


# ------------------------------------------------------------------ #
# Kruskal-Wallis
# ------------------------------------------------------------------ #

def _run_kruskal_wallis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    classes,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Kruskal-Wallis H-test for each feature.
    Tests whether distributions differ significantly across classes.
    """
    kruskal_stats = []
    kruskal_pvals = []

    for i, feat in enumerate(feature_names):
        groups = [X[y == c, i] for c in range(len(classes))]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            kruskal_stats.append(0.0)
            kruskal_pvals.append(1.0)
            continue

        try:
            stat, pval = stats.kruskal(*groups)
            kruskal_stats.append(float(stat))
            kruskal_pvals.append(float(pval))
        except Exception:
            kruskal_stats.append(0.0)
            kruskal_pvals.append(1.0)

    return np.array(kruskal_stats), np.array(kruskal_pvals)


# ------------------------------------------------------------------ #
# Mutual Information
# ------------------------------------------------------------------ #

def _run_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Compute mutual information between each feature and the target."""
    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    return mi


# ------------------------------------------------------------------ #
# Save
# ------------------------------------------------------------------ #

def _save_importance(df: pd.DataFrame, config) -> None:
    out_path = resolve_path(config.paths.metrics) / "feature_importance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Feature importance saved: %s", out_path)


def load_feature_importance(config=None) -> pd.DataFrame:
    """Load previously computed feature importance CSV."""
    cfg = config or get_config()
    fpath = resolve_path(cfg.paths.metrics) / "feature_importance.csv"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Feature importance file not found: {fpath}\n"
            "Please run:  python main.py --features"
        )
    return pd.read_csv(fpath)


def get_top_features(n: int = 5, config=None) -> List[str]:
    """Load feature importance and return top N feature names."""
    df = load_feature_importance(config)
    return df["feature"].head(n).tolist()
