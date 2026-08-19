"""
src/visualization.py
====================
Automated EDA and result visualization.

Generated figures:
  EDA:
    eda_class_distribution.png
    eda_theta_by_class.png
    eda_alpha_by_class.png
    eda_beta_by_class.png
    eda_theta_beta_ratio_by_class.png
    eda_spectral_entropy_by_class.png
    eda_correlation_matrix.png
    eda_feature_distributions.png

  Feature analysis:
    feature_importance_bar.png

All saved to results/figures/

Usage:
    from src.visualization import run_eda
    run_eda(features_df, config)
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# Consistent color palette for the 3 classes
CLASS_COLORS = {
    "LOW": "#2E86AB",
    "MODERATE": "#F6AE2D",
    "HIGH": "#E84855",
}
CLASS_ORDER = ["LOW", "MODERATE", "HIGH"]


# ------------------------------------------------------------------ #
# Main EDA runner
# ------------------------------------------------------------------ #

def run_eda(df: pd.DataFrame, config=None) -> List[str]:
    """
    Run full EDA — generate all plots.

    Parameters:
        df     : features DataFrame
        config : Config object

    Returns:
        List of saved file paths
    """
    cfg = config or get_config()
    figures_dir = resolve_path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting EDA — generating figures...")
    saved = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        saved.append(_plot_class_distribution(df, figures_dir))
        saved.append(_plot_band_by_class(df, "theta_power", "Theta Power", figures_dir))
        saved.append(_plot_band_by_class(df, "alpha_power", "Alpha Power", figures_dir))
        saved.append(_plot_band_by_class(df, "beta_power", "Beta Power", figures_dir))
        saved.append(_plot_band_by_class(df, "theta_beta_ratio", "Theta/Beta Ratio", figures_dir))
        saved.append(_plot_band_by_class(df, "spectral_entropy", "Spectral Entropy", figures_dir))
        saved.append(_plot_correlation_matrix(df, figures_dir))
        saved.append(_plot_feature_distributions(df, figures_dir))

    saved = [s for s in saved if s is not None]
    logger.info("EDA complete. %d figures saved to %s", len(saved), figures_dir)
    return saved


# ------------------------------------------------------------------ #
# Individual plots
# ------------------------------------------------------------------ #

def _plot_class_distribution(df: pd.DataFrame, figures_dir: Path) -> Optional[str]:
    """Bar chart of class distribution."""
    if "label" not in df.columns:
        return None

    counts = df["label"].value_counts()
    # Re-order
    counts = counts.reindex([c for c in CLASS_ORDER if c in counts.index], fill_value=0)
    colors = [CLASS_COLORS.get(c, "gray") for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_title("Class Distribution — Cognitive Load Labels", fontsize=13, pad=10)
    ax.set_xlabel("Workload Class", fontsize=12)
    ax.set_ylabel("Number of Windows", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()

    out = figures_dir / "eda_class_distribution.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _plot_band_by_class(
    df: pd.DataFrame,
    feature: str,
    feature_label: str,
    figures_dir: Path,
) -> Optional[str]:
    """Box + strip plot of a feature by class."""
    if feature not in df.columns or "label" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    # Box plot
    data_by_class = [
        df[df["label"] == cls][feature].dropna().values
        for cls in CLASS_ORDER if cls in df["label"].unique()
    ]
    present_classes = [c for c in CLASS_ORDER if c in df["label"].unique()]
    colors_list = [CLASS_COLORS[c] for c in present_classes]

    bp = ax.boxplot(
        data_by_class,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Individual points
    for i, (cls, clr) in enumerate(zip(present_classes, colors_list)):
        subset = df[df["label"] == cls][feature].dropna().values
        x_jitter = np.random.normal(i + 1, 0.06, size=len(subset))
        ax.scatter(x_jitter, subset, alpha=0.3, s=8, color=clr)

    ax.set_xticks(range(1, len(present_classes) + 1))
    ax.set_xticklabels(present_classes, fontsize=12)
    ax.set_title(f"{feature_label} by Cognitive Load Class", fontsize=13, pad=10)
    ax.set_xlabel("Workload Class", fontsize=12)
    ax.set_ylabel(feature_label, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe_name = feature.replace("/", "_").replace(" ", "_")
    out = figures_dir / f"eda_{safe_name}_by_class.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _plot_correlation_matrix(df: pd.DataFrame, figures_dir: Path) -> Optional[str]:
    """Correlation heatmap of numeric features."""
    meta_cols = {"subject_id", "window_id", "start_time", "end_time", "trial_type", "label"}
    num_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=10)
    plt.tight_layout()

    out = figures_dir / "eda_correlation_matrix.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _plot_feature_distributions(df: pd.DataFrame, figures_dir: Path) -> Optional[str]:
    """Grid of feature distribution histograms by class."""
    key_features = [
        "theta_power", "alpha_power", "beta_power",
        "theta_relative", "alpha_relative", "beta_relative",
        "theta_beta_ratio", "spectral_entropy", "sample_entropy",
    ]
    available = [f for f in key_features if f in df.columns]
    if not available:
        return None

    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    present_classes = [c for c in CLASS_ORDER if c in df["label"].unique()]

    for i, feat in enumerate(available):
        ax = axes[i]
        for cls in present_classes:
            vals = df[df["label"] == cls][feat].dropna()
            ax.hist(vals, bins=30, alpha=0.5, label=cls, color=CLASS_COLORS[cls], density=True)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Cognitive Load Class", fontsize=13, y=1.01)
    plt.tight_layout()

    out = figures_dir / "eda_feature_distributions.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------------ #
# Feature importance plot
# ------------------------------------------------------------------ #

def plot_feature_importance(importance_df: pd.DataFrame, config=None) -> str:
    """Horizontal bar chart of feature importance."""
    cfg = config or get_config()
    figures_dir = resolve_path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    top_n = min(15, len(importance_df))
    plot_df = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))

    bars = ax.barh(
        plot_df["feature"],
        plot_df.get("mutual_info", plot_df.get("importance", 0)),
        color="#4C72B0",
        alpha=0.8,
    )

    ax.set_xlabel("Importance Score (Mutual Information)", fontsize=11)
    ax.set_title("Feature Importance — Top Features", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = figures_dir / "feature_importance_bar.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved: %s", out)
    return str(out)


# ------------------------------------------------------------------ #
# EEG waveform plot (for dashboard)
# ------------------------------------------------------------------ #

def plot_eeg_waveform(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    n_channels: int = 5,
    title: str = "EEG Waveform",
) -> plt.Figure:
    """
    Plot a multi-channel EEG waveform.
    Returns a matplotlib Figure for Streamlit display.
    """
    n_show = min(n_channels, len(ch_names), data.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(12, n_show * 1.8), sharex=True)
    if n_show == 1:
        axes = [axes]

    t = np.arange(data.shape[1]) / sfreq
    offsets = 0
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))

    for i in range(n_show):
        ax = axes[i]
        ax.plot(t, data[i, :] * 1e6, color=colors[i], linewidth=0.8)
        ax.set_ylabel(ch_names[i], fontsize=8, rotation=0, ha="right", labelpad=30)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_power_spectrum(
    data: np.ndarray,
    sfreq: float,
    title: str = "Power Spectrum",
) -> plt.Figure:
    """Plot mean PSD across channels."""
    from scipy import signal as scipy_signal

    fig, ax = plt.subplots(figsize=(9, 4))
    freqs = None

    for ch_idx in range(data.shape[0]):
        f, psd = scipy_signal.welch(data[ch_idx, :], fs=sfreq, nperseg=min(len(data[ch_idx]), int(sfreq * 2)))
        if freqs is None:
            freqs = f
            mean_psd = psd
        else:
            mean_psd += psd

    mean_psd /= data.shape[0]

    # Band shading
    band_colors = {
        "Delta (1–4 Hz)": (1, 4, "#a8d8ea"),
        "Theta (4–8 Hz)": (4, 8, "#aa96da"),
        "Alpha (8–13 Hz)": (8, 13, "#fcbad3"),
        "Beta (13–30 Hz)": (13, 30, "#ffffd2"),
    }
    for label, (lo, hi, color) in band_colors.items():
        ax.axvspan(lo, hi, alpha=0.2, color=color, label=label)

    ax.semilogy(freqs, mean_psd, "b-", linewidth=1.5)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
