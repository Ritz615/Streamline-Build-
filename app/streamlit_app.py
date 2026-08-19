"""
app/streamlit_app.py
====================
Streamlit dashboard for EEG Cognitive Load Classification.

Pages:
  🏠 Dashboard          — project overview + model status
  📊 EEG Analysis       — waveform + spectrum visualisation
  🔬 Feature Analysis   — band power + entropy + importance
  🧠 Cognitive Load     — prediction + fuzzy explanation
  📈 Model Comparison   — metrics + confusion matrices
  ℹ️  About             — disclaimer + methodology + citation

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from streamlit import session_state as ss

from src.config import get_config, setup_logging, PROJECT_ROOT

# ── page config (must be first st call) ──────────────────────────────────
st.set_page_config(
    page_title="EEG Cognitive Load Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)  # suppress MNE noise in dashboard
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_config():
    return get_config()

cfg = _get_config()

# ── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* metric cards */
  [data-testid="metric-container"] {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 10px;
      padding: 12px 16px;
  }
  /* prediction box */
  .pred-box {
      padding: 24px;
      border-radius: 14px;
      text-align: center;
      font-size: 2.4rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin: 16px 0;
  }
  .pred-LOW      { background: #1a4e6e; color: #7ecef4; border: 2px solid #2E86AB; }
  .pred-MODERATE { background: #6e5010; color: #f6d86d; border: 2px solid #F6AE2D; }
  .pred-HIGH     { background: #6e1a1a; color: #f47a80; border: 2px solid #E84855; }
  /* disclaimer banner */
  .disclaimer {
      background: rgba(255, 165, 0, 0.15);
      border-left: 4px solid orange;
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 0.85rem;
  }
  /* sidebar separator */
  hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def _load_features():
    from src.feature_extraction import load_features
    try:
        return load_features(config=cfg)
    except FileNotFoundError:
        return None

@st.cache_data
def _load_comparison():
    from src.evaluation import load_comparison_csv
    return load_comparison_csv(config=cfg)

@st.cache_data
def _load_importance():
    from src.feature_selection import load_feature_importance
    try:
        return load_feature_importance(config=cfg)
    except FileNotFoundError:
        return None

@st.cache_resource
def _load_fuzzy_model():
    from src.fuzzy_classifier import FuzzyClassifier
    clf = FuzzyClassifier(config=cfg)
    try:
        clf.load(config=cfg)
        return clf
    except FileNotFoundError:
        return None

@st.cache_resource
def _load_rf_model():
    from src.random_forest import RandomForestModel
    rf = RandomForestModel(config=cfg)
    try:
        rf.load(config=cfg)
        return rf
    except FileNotFoundError:
        return None

def _data_status():
    """Return dict of what's available."""
    return {
        "dataset": (PROJECT_ROOT / "data" / "raw" / "ds007169").exists(),
        "preprocessed": any((PROJECT_ROOT / "data" / "processed").rglob("*.fif")),
        "features": (PROJECT_ROOT / "data" / "features" / "features.csv").exists(),
        "fuzzy_model": (PROJECT_ROOT / "models" / "fuzzy" / "fuzzy_classifier.joblib").exists(),
        "rf_model": (PROJECT_ROOT / "models" / "random_forest" / "random_forest.joblib").exists(),
        "results": (PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv").exists(),
    }

def _status_icon(ok: bool) -> str:
    return "✅" if ok else "❌"

def _disclaimer_banner():
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Research Prototype Only.</strong> '
        'This dashboard is for academic demonstration. It is <strong>NOT</strong> a medical device, '
        'clinical tool, or psychological assessment system. Do not use for diagnosis.</div>',
        unsafe_allow_html=True,
    )

# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 EEG Cognitive Load")
    st.caption("Research Prototype v1.0")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 EEG Analysis", "🔬 Feature Analysis",
         "🧠 Cognitive Load Prediction", "📈 Model Comparison", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    status = _data_status()
    st.markdown("**Pipeline Status**")
    for key, label in [
        ("dataset", "Dataset"), ("preprocessed", "Preprocessed"),
        ("features", "Features"), ("fuzzy_model", "Fuzzy Model"),
        ("rf_model", "RF Model"), ("results", "Results"),
    ]:
        st.markdown(f"{_status_icon(status[key])} {label}")

    st.markdown("---")
    st.caption("Dataset: ds007169 (CC0)\nOpenNeuro.org")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🧠 EEG Cognitive Load Research Dashboard")
    _disclaimer_banner()
    st.markdown("---")

    # Top metrics
    df = _load_features()
    comp = _load_comparison()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", cfg.dataset.id, help="OpenNeuro dataset identifier")
    with col2:
        st.metric("Subjects", cfg.dataset.subjects)
    with col3:
        st.metric("EEG Channels", cfg.dataset.channels)
    with col4:
        st.metric("Sampling Rate", f"{cfg.dataset.sampling_rate} Hz")

    st.markdown("---")

    # Feature / window summary
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📦 Dataset Overview")
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Dataset Name | {cfg.dataset.name} |
| Source | [OpenNeuro]({cfg.dataset.source}) |
| License | {cfg.dataset.license} |
| Format | BrainVision (.vhdr/.eeg/.vmrk) |
| Tasks | n-back levels 1–4 |
| Label mapping | nback_1→LOW, nback_2→MODERATE, nback_3/4→HIGH |
""")
        st.info("The label mapping is an operational research categorization "
                "based on task difficulty, not a validated cognitive scale.")

    with col_r:
        st.subheader("📊 Data & Model Status")
        if df is not None:
            n_subjects = df["subject_id"].nunique()
            n_windows = len(df)
            st.metric("Total Windows (segments)", n_windows)
            st.metric("Subjects in features", n_subjects)
            counts = df["label"].value_counts()
            st.bar_chart(counts)
        else:
            st.warning("Features not extracted yet.\n\nRun:  `python main.py --features`")

    st.markdown("---")

    # Latest experiment results
    st.subheader("📈 Latest Experiment Results")
    if comp is not None:
        for _, row in comp.iterrows():
            cols = st.columns(5)
            cols[0].metric("Model", row["model"])
            cols[1].metric("Accuracy", f"{row['accuracy']:.3f}")
            cols[2].metric("F1 (macro)", f"{row['f1_macro']:.3f}")
            cols[3].metric("Precision", f"{row['precision_macro']:.3f}")
            cols[4].metric("Recall", f"{row['recall_macro']:.3f}")
            st.markdown("---")
    else:
        st.info("No results yet. Run the full pipeline:\n```\npython main.py --all\n```")

    st.subheader("🚀 Quick Start Commands")
    st.code("""# Complete pipeline (recommended first run)
python main.py --all

# Individual stages
python main.py --download     # Step 1: Get dataset
python main.py --preprocess   # Step 2: Clean EEG
python main.py --features     # Step 3: Extract features
python main.py --train        # Step 4: Train models
python main.py --evaluate     # Step 5: Evaluate

# Run tests
pytest tests/ -v

# Check status
python main.py --status
""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: EEG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 EEG Analysis":
    st.title("📊 EEG Signal Analysis")
    _disclaimer_banner()
    st.markdown("---")

    st.info("This page visualises preprocessed EEG data for a selected subject.")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    subject_dirs = sorted([d.name for d in processed_dir.glob("sub-*") if d.is_dir()]) if processed_dir.exists() else []

    if not subject_dirs:
        st.warning(
            "No preprocessed EEG found.\n\n"
            "Run:  `python main.py --preprocess`"
        )
        st.stop()

    col1, col2 = st.columns([1, 3])
    with col1:
        sub_id = st.selectbox("Select Subject", subject_dirs)
        n_channels_show = st.slider("Channels to Display", 1, 10, 5)
        time_start = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
        time_end = st.number_input("End time (s)", min_value=1.0, value=10.0, step=1.0)

    with col2:
        @st.cache_resource
        def _load_preprocessed_raw(sub_id):
            from src.preprocessing import load_preprocessed
            return load_preprocessed(sub_id, config=cfg)

        with st.spinner(f"Loading {sub_id}..."):
            raw = _load_preprocessed_raw(sub_id)

        if raw is None:
            st.error(f"Could not load preprocessed data for {sub_id}.")
        else:
            sfreq = raw.info["sfreq"]
            total_dur = raw.times[-1]
            st.success(f"Loaded: {len(raw.ch_names)} channels, {total_dur:.1f} s @ {sfreq} Hz")

            start_s = max(0, time_start)
            end_s = min(total_dur, time_end)
            start_samp = int(start_s * sfreq)
            end_samp = int(end_s * sfreq)

            data_slice = raw.get_data()[:, start_samp:end_samp]

            # Waveform
            st.subheader("EEG Waveform")
            from src.visualization import plot_eeg_waveform
            fig_wv = plot_eeg_waveform(
                data_slice, sfreq, raw.ch_names,
                n_channels=n_channels_show,
                title=f"{sub_id} — EEG Waveform ({start_s:.1f}–{end_s:.1f} s)",
            )
            st.pyplot(fig_wv, use_container_width=True)
            plt.close("all")

            # Spectrum
            st.subheader("Power Spectrum (mean PSD)")
            from src.visualization import plot_power_spectrum
            fig_psd = plot_power_spectrum(data_slice, sfreq, title=f"{sub_id} — Mean PSD")
            st.pyplot(fig_psd, use_container_width=True)
            plt.close("all")

            # Info table
            st.subheader("Channel Information")
            ch_df = pd.DataFrame({
                "Channel": raw.ch_names,
                "Type": [raw.get_channel_types([c])[0] for c in raw.ch_names],
            })
            st.dataframe(ch_df, use_container_width=True, height=200)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    st.title("🔬 EEG Feature Analysis")
    _disclaimer_banner()
    st.markdown("---")

    df = _load_features()
    if df is None:
        st.warning("Features not extracted yet.\n\nRun: `python main.py --features`")
        st.stop()

    from src.feature_extraction import get_feature_columns
    feature_cols = get_feature_columns(df)
    CLASS_COLORS = {"LOW": "#2E86AB", "MODERATE": "#F6AE2D", "HIGH": "#E84855"}

    # ── Stats overview ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Windows", len(df))
    col2.metric("Subjects", df["subject_id"].nunique())
    col3.metric("Features", len(feature_cols))
    col4.metric("Classes", df["label"].nunique())

    st.markdown("---")

    # ── Class distribution ─────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Class Distribution")
        counts = df["label"].value_counts().reindex(["LOW", "MODERATE", "HIGH"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [CLASS_COLORS.get(c, "gray") for c in counts.index]
        ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
        for i, (c, v) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, v + 5, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Workload Class Distribution")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close("all")

    with col_r:
        st.subheader("Windows per Subject")
        sub_counts = df.groupby("subject_id").size().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(range(len(sub_counts)), sub_counts.values, color="#4C72B0", alpha=0.8)
        ax2.set_xticks(range(len(sub_counts)))
        ax2.set_xticklabels(sub_counts.index, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Windows")
        ax2.set_title("Windows per Subject")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close("all")

    st.markdown("---")

    # ── Feature box plots ─────────────────────────────────────────────────
    st.subheader("Feature Distributions by Class")
    selected_feat = st.selectbox(
        "Select feature to inspect",
        feature_cols,
        index=feature_cols.index("theta_beta_ratio") if "theta_beta_ratio" in feature_cols else 0,
    )

    present_classes = [c for c in ["LOW", "MODERATE", "HIGH"] if c in df["label"].unique()]
    data_by_class = [df[df["label"] == c][selected_feat].dropna().values for c in present_classes]
    colors_list = [CLASS_COLORS[c] for c in present_classes]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    bp = ax3.boxplot(data_by_class, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (cls, clr) in enumerate(zip(present_classes, colors_list)):
        subset = df[df["label"] == cls][selected_feat].dropna().values
        jitter = np.random.normal(i + 1, 0.05, size=min(len(subset), 500))
        sample = subset[:500]
        ax3.scatter(jitter, sample, alpha=0.25, s=6, color=clr)
    ax3.set_xticks(range(1, len(present_classes) + 1))
    ax3.set_xticklabels(present_classes, fontsize=11)
    ax3.set_ylabel(selected_feat)
    ax3.set_title(f"{selected_feat} by Cognitive Load Class")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3, use_container_width=True)
    plt.close("all")

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────
    st.subheader("Feature Importance (Kruskal-Wallis + Mutual Information)")
    imp_df = _load_importance()
    if imp_df is not None:
        fig4, ax4 = plt.subplots(figsize=(8, max(5, len(imp_df) * 0.35)))
        plot_df = imp_df.head(15).iloc[::-1]
        ax4.barh(plot_df["feature"], plot_df.get("mutual_info", 0), color="#4C72B0", alpha=0.8)
        ax4.set_xlabel("Mutual Information Score")
        ax4.set_title("Feature Importance (top 15)")
        ax4.grid(axis="x", alpha=0.3)
        st.pyplot(fig4, use_container_width=True)
        plt.close("all")
        with st.expander("View full importance table"):
            st.dataframe(imp_df.style.format({"kruskal_stat": "{:.2f}",
                                               "kruskal_pvalue": "{:.4f}",
                                               "mutual_info": "{:.4f}"}),
                         use_container_width=True)
    else:
        st.info("Feature importance not computed yet. Run `python main.py --features`")

    # ── Raw feature table ─────────────────────────────────────────────────
    with st.expander("📋 View Feature Data (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────
    with st.expander("📊 Feature Summary Statistics"):
        st.dataframe(
            df[feature_cols].describe().round(4).T,
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: COGNITIVE LOAD PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive Load Prediction":
    st.title("🧠 Cognitive Load Prediction")
    _disclaimer_banner()
    st.markdown("---")

    df = _load_features()
    fuzzy_clf = _load_fuzzy_model()
    rf_model = _load_rf_model()

    if df is None:
        st.warning("Features not available. Run: `python main.py --features`")
        st.stop()
    if fuzzy_clf is None:
        st.warning("Fuzzy model not trained. Run: `python main.py --train`")
        st.stop()

    from src.feature_extraction import get_feature_columns
    from src.explainability import explain_fuzzy, explain_rf, format_prediction_for_display

    feature_cols = get_feature_columns(df)

    st.subheader("Select or Enter EEG Feature Values")

    mode = st.radio(
        "Input mode",
        ["Select from dataset", "Enter values manually"],
        horizontal=True,
    )

    if mode == "Select from dataset":
        subjects = df["subject_id"].unique().tolist()
        sel_subject = st.selectbox("Subject", subjects)
        sub_df = df[df["subject_id"] == sel_subject]
        sel_window = st.slider("Window ID", int(sub_df["window_id"].min()),
                               int(sub_df["window_id"].max()), int(sub_df["window_id"].iloc[0]))
        row = sub_df[sub_df["window_id"] == sel_window]
        if row.empty:
            row = sub_df.iloc[[0]]
        row = row.iloc[0]
        true_label = row.get("label", "UNKNOWN")
        x_all = row[feature_cols].values.astype(float)
        st.info(f"True label: **{true_label}** | Trial type: {row.get('trial_type','N/A')}")
    else:
        st.markdown("Adjust feature values using the sliders:")
        # Show sliders for fuzzy features only
        fuzzy_features = fuzzy_clf.feature_names
        x_all_dict = {}
        # Set defaults to dataset means
        means = df[feature_cols].mean()
        cols_sl = st.columns(3)
        for i, feat in enumerate(feature_cols):
            mean_val = float(means.get(feat, 0.0))
            feat_min = float(df[feat].quantile(0.01))
            feat_max = float(df[feat].quantile(0.99))
            with cols_sl[i % 3]:
                x_all_dict[feat] = st.slider(
                    feat, feat_min, feat_max, mean_val,
                    step=(feat_max - feat_min) / 200,
                    format="%.4f",
                )
        x_all = np.array([x_all_dict[f] for f in feature_cols])
        true_label = "N/A (manual)"

    st.markdown("---")

    # ── Run prediction ─────────────────────────────────────────────────────
    if st.button("🔍 Predict Cognitive Load", type="primary", use_container_width=True):

        # Fuzzy prediction
        fuzzy_feature_idx = [feature_cols.index(f) for f in fuzzy_clf.feature_names if f in feature_cols]
        x_fuzzy = x_all[fuzzy_feature_idx]

        with st.spinner("Running fuzzy inference..."):
            try:
                fuzzy_result = fuzzy_clf.predict_single(x_fuzzy)
            except Exception as e:
                st.error(f"Fuzzy prediction error: {e}")
                fuzzy_result = None

        col_fuzzy, col_rf = st.columns(2)

        with col_fuzzy:
            st.subheader("🔶 Fuzzy Classifier")
            if fuzzy_result:
                pred = fuzzy_result["predicted_class"]
                score = fuzzy_result["fuzzy_score"]
                conf = fuzzy_result["confidence"]
                st.markdown(
                    f'<div class="pred-box pred-{pred}">{pred}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Fuzzy Score", f"{score:.1f} / 100")
                m2.metric("Confidence", f"{conf:.1%}")

                # Memberships
                st.markdown("**Feature Memberships:**")
                for feat, levels in fuzzy_result.get("memberships", {}).items():
                    dom = max(levels, key=lambda k: levels[k])
                    st.markdown(f"- **{feat}**: LOW={levels.get('LOW',0):.2f}  "
                                f"MEDIUM={levels.get('MEDIUM',0):.2f}  "
                                f"HIGH={levels.get('HIGH',0):.2f}  → **{dom}**")

                # Activated rules
                rules = fuzzy_result.get("activated_rules", [])
                if rules:
                    st.markdown("**Activated Rules:**")
                    for r in rules[:5]:
                        st.markdown(
                            f"- Rule {r['rule_number']} (strength={r['strength']:.2f}): {r['rule_text']}"
                        )

                # Full explanation
                with st.expander("📄 Full Explanation Text"):
                    explanation = explain_fuzzy(fuzzy_result, x_fuzzy, fuzzy_clf.feature_names)
                    st.text(explanation)
            else:
                st.error("Fuzzy prediction failed.")

        with col_rf:
            st.subheader("🌲 Random Forest")
            if rf_model and rf_model.is_trained:
                try:
                    proba = rf_model.predict_proba(x_all.reshape(1, -1))[0]
                    pred_int = int(np.argmax(proba))
                    class_names = ["LOW", "MODERATE", "HIGH"]
                    pred_rf = class_names[pred_int]

                    st.markdown(
                        f'<div class="pred-box pred-{pred_rf}">{pred_rf}</div>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Confidence", f"{proba[pred_int]:.1%}")

                    st.markdown("**Class Probabilities:**")
                    for cls, p in zip(class_names, proba):
                        bar_len = int(p * 20)
                        st.markdown(f"- **{cls}**: {'█' * bar_len}{'░' * (20 - bar_len)} {p:.1%}")

                    # Feature importance
                    fi_df = rf_model.get_feature_importance().head(5)
                    st.markdown("**Top Features (model-wide):**")
                    for _, row_fi in fi_df.iterrows():
                        st.markdown(f"- {row_fi['feature']}: {row_fi['importance']:.4f}")

                    with st.expander("📄 Full RF Explanation"):
                        rf_explanation = explain_rf(rf_model, x_all, feature_cols)
                        st.text(rf_explanation)

                except Exception as e:
                    st.error(f"RF prediction error: {e}")
            else:
                st.warning("Random Forest model not trained. Run: `python main.py --train`")

        # Comparison
        st.markdown("---")
        if true_label != "N/A (manual)" and fuzzy_result:
            cols = st.columns(3)
            cols[0].metric("True Label", true_label)
            cols[1].metric("Fuzzy Prediction", fuzzy_result["predicted_class"])
            if rf_model and rf_model.is_trained:
                cols[2].metric("RF Prediction", pred_rf)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    _disclaimer_banner()
    st.markdown("---")

    comp = _load_comparison()
    if comp is None:
        st.warning("No evaluation results found.\n\nRun: `python main.py --evaluate`")
        st.stop()

    st.info("All metrics are from **subject-wise cross-validation** "
            "(StratifiedGroupKFold). Subjects in the test set were never seen "
            "during training.")

    # Metrics table
    st.subheader("Performance Metrics")
    display_cols = ["model", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    available_cols = [c for c in display_cols if c in comp.columns]
    styled = comp[available_cols].style.format(
        {c: "{:.3f}" for c in available_cols if c != "model"}
    ).highlight_max(subset=[c for c in available_cols if c != "model"], color="#1a4e6e")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Comparison bar chart
    st.subheader("Visual Comparison")
    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
                   if c in comp.columns]
    metric_labels = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Acc",
                     "f1_macro": "F1 (macro)", "precision_macro": "Precision", "recall_macro": "Recall"}

    fig, ax = plt.subplots(figsize=(10, 5))
    n_metrics = len(metric_cols)
    x = np.arange(n_metrics)
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    for i, (_, row) in enumerate(comp.iterrows()):
        vals = [row.get(m, 0) for m in metric_cols]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=row["model"], color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metric_cols], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Subject-wise Cross-validated Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close("all")

    st.markdown("---")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    figures_dir = PROJECT_ROOT / "results" / "figures"
    cm_col1, cm_col2 = st.columns(2)

    for col, name, fname in [
        (cm_col1, "Fuzzy Classifier", "fuzzy_classifier_confusion_matrix.png"),
        (cm_col2, "Random Forest", "random_forest_confusion_matrix.png"),
    ]:
        img_path = figures_dir / fname
        with col:
            st.markdown(f"**{name}**")
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info(f"Confusion matrix not found. Run: `python main.py --evaluate`")

    st.markdown("---")

    # Fuzzy rules
    st.subheader("Fuzzy Rules (from database)")
    try:
        from src.database import get_fuzzy_rules, init_db
        init_db()
        rules = get_fuzzy_rules()
        if rules:
            rules_df = pd.DataFrame([
                {"Rule #": r.rule_number, "Rule": r.rule_text, "Consequence": r.consequent}
                for r in rules
            ])
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
        else:
            st.info("No fuzzy rules stored. Run `python main.py --train`")
    except Exception as e:
        st.warning(f"Could not load rules: {e}")

    st.markdown("---")

    # RF feature importance
    st.subheader("Random Forest — Feature Importance")
    rf_model = _load_rf_model()
    if rf_model and rf_model.is_trained:
        fi_df = rf_model.get_feature_importance()
        fig_fi, ax_fi = plt.subplots(figsize=(8, max(4, len(fi_df) * 0.35)))
        plot_df = fi_df.head(15).iloc[::-1]
        ax_fi.barh(plot_df["feature"], plot_df["importance"], color="#DD8452", alpha=0.85)
        ax_fi.set_xlabel("Feature Importance")
        ax_fi.set_title("Random Forest — Feature Importances (Gini)")
        ax_fi.grid(axis="x", alpha=0.3)
        st.pyplot(fig_fi, use_container_width=True)
        plt.close("all")
        st.caption("Feature importance shows how much each feature contributes to the model's "
                   "decisions. It does NOT imply causality.")
    else:
        st.info("RF model not loaded.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.error("""
**⚠️ IMPORTANT DISCLAIMER**

This is an **academic research prototype** built for a BE (Bachelor of Engineering) final year project.

- It is **NOT** a medical device
- It is **NOT** a clinical diagnostic tool
- It is **NOT** a psychological assessment system
- It does **NOT** provide medical, clinical, or psychological advice
- Results should **NOT** be used for any medical or clinical purpose

All EEG data used is **public**, **anonymized**, and licensed under **CC0**.
""")

    st.markdown("---")

    st.subheader("📋 Project Overview")
    st.markdown("""
This system demonstrates an end-to-end pipeline for EEG-based cognitive load
classification using:

1. **Signal Processing** — MNE-Python for EEG preprocessing
2. **Feature Engineering** — Theta/alpha/beta band power, spectral entropy, statistical features
3. **Fuzzy Logic** — Interpretable rule-based classification (scikit-fuzzy)
4. **Machine Learning** — Random Forest baseline (scikit-learn)
5. **Explainability** — Activated rule display and feature membership analysis

The system classifies EEG windows into three operational research categories:
- **LOW** — n-back level 1 (simplest task)
- **MODERATE** — n-back level 2
- **HIGH** — n-back levels 3 & 4 (most demanding tasks)
""")

    st.subheader("🗃️ Dataset")
    st.markdown(f"""
| Property | Value |
|----------|-------|
| Dataset ID | {cfg.dataset.id} |
| Name | {cfg.dataset.name} |
| Source | {cfg.dataset.source} |
| DOI | doi:10.18112/openneuro.ds007169.v1.0.0 |
| License | {cfg.dataset.license} (public domain) |
| Subjects | 18 |
| EEG Channels | 19 (10-20 montage) |
| Sampling Rate | 250 Hz |
| Format | BrainVision (.vhdr/.eeg/.vmrk) |
""")

    st.subheader("🏗️ System Architecture")
    st.code("""
PUBLIC EEG DATASET (ds007169)
        ↓
  MNE-Python: Band-pass + notch filter
  Average reference + artifact rejection
        ↓
  Sliding window segmentation (4s, 2s overlap)
        ↓
  Feature extraction:
    • Theta / Alpha / Beta power
    • Relative power + ratios
    • Statistical (mean, std, skewness, kurtosis)
    • Spectral entropy + Sample entropy
        ↓
  Feature selection (Kruskal-Wallis + MI)
        ↓
    ┌────────────────────┐
    │                    │
    ▼                    ▼
 FUZZY SYSTEM      RANDOM FOREST
 (scikit-fuzzy)    (scikit-learn)
    │                    │
    ▼                    ▼
 LOW/MOD/HIGH      LOW/MOD/HIGH
    │                    │
    └──────────┬─────────┘
               ▼
    Subject-wise CV evaluation
    Confusion matrices + F1
               ▼
    Explainability output
               ▼
    Streamlit Dashboard + SQLite DB
""", language="text")

    st.subheader("⚠️ Limitations")
    st.markdown("""
- Only 18 subjects (limited statistical power)
- No rest/baseline condition — 1-back task used as "LOW" proxy
- THREE-CLASS labels are derived from task difficulty, NOT direct brain state measurements
- EEG is noisy and highly individual — models may not generalise
- Results have not been clinically validated
- This prototype is for educational and research demonstration only
""")

    st.subheader("📚 Citation")
    st.code("""Booth, L., & Barras, M. (2024).
Cognitive Workload 5-level n-back [Data set].
OpenNeuro. doi:10.18112/openneuro.ds007169.v1.0.0
License: CC0 (Public Domain)""")

    st.subheader("🛠️ Technology Stack")
    st.markdown("""
| Component | Library |
|-----------|---------|
| EEG Processing | MNE-Python |
| Fuzzy Logic | scikit-fuzzy |
| Machine Learning | scikit-learn |
| Data | NumPy, Pandas, SciPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Database | SQLite + SQLAlchemy |
| Configuration | PyYAML |
""")
