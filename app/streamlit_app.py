"""
app/streamlit_app.py
====================
EEG Cognitive Load Research Dashboard — Professional UI/UX Design
Industry-standard dark glassmorphism interface with Plotly interactive charts.

Pages:
  🏠 Dashboard          — project overview + live metrics
  📊 EEG Analysis       — waveform + PSD visualisation
  🔬 Feature Analysis   — distributions + importance
  🧠 Cognitive Load     — dual-model prediction + explainability
  📈 Model Comparison   — cross-validated metrics + confusion matrices
  📜 Experiment History — SQLite database with export
  ℹ️  Methodology       — architecture + viva guide

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

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

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Cognitive Load · Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@st.cache_resource
def _get_config():
    return get_config()

cfg = _get_config()

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — Professional Light Theme CSS
# ─────────────────────────────────────────────────────────────────────────────
DESIGN_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  /* ── Root tokens — Premium Light Theme ── */
  :root {
    --bg:            #eef1f8;
    --surface-1:     #ffffff;
    --surface-2:     #f7f9fc;
    --surface-3:     #edf0f8;
    --border:        rgba(79, 70, 229, 0.13);
    --border-bright: rgba(79, 70, 229, 0.30);
    --accent-blue:   #4f46e5;
    --accent-purple: #7c3aed;
    --accent-teal:   #0d9488;
    --low-color:     #2563eb;
    --mod-color:     #d97706;
    --high-color:    #dc2626;
    --text-1:        #0f172a;
    --text-2:        #374151;
    --text-3:        #9ca3af;
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     24px;
    --radius-pill:   999px;
    --shadow-sm:     0 1px 3px rgba(15,23,42,0.05), 0 4px 14px rgba(79,70,229,0.06);
    --shadow-md:     0 4px 24px rgba(15,23,42,0.08), 0 12px 40px rgba(79,70,229,0.08);
    --shadow-lg:     0 16px 56px rgba(15,23,42,0.12), 0 4px 16px rgba(79,70,229,0.10);
  }

  /* ── Global reset ── */
  html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
  }

  /* ── App background ── */
  .stApp, .main {
    background: var(--bg) !important;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(79, 70, 229, 0.05) 0%, transparent 70%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(124, 58, 237, 0.04) 0%, transparent 70%) !important;
    color: var(--text-1) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(99,102,241,0.1) !important;
    box-shadow: 2px 0 12px rgba(15,23,42,0.05) !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] span {
    color: var(--text-2) !important;
    font-size: 13px !important;
  }
  [data-testid="stSidebar"] .stRadio label {
    color: var(--text-1) !important;
    font-size: 13.5px !important;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px 22px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  [data-testid="metric-container"]:hover {
    border-color: var(--border-bright) !important;
    box-shadow: var(--shadow-md) !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-2) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-1) !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
  }

  /* ── Dataframes & Tables ── */
  [data-testid="stDataFrame"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
  }
  [data-testid="stDataFrame"] table {
    background: transparent !important;
  }

  /* ── Primary button ── */
  .stButton > button[kind="primary"],
  .stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.01em !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
  }

  /* ── Selectbox / Input ── */
  .stSelectbox > div > div,
  .stTextInput > div > div,
  .stNumberInput > div > div {
    background: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
  }
  .stSelectbox > div > div:focus-within,
  .stTextInput > div > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
  }

  /* ── Slider ── */
  [data-testid="stSlider"] [data-testid="stTickBar"] {
    color: var(--text-3) !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: var(--radius-md) !important;
    margin: 8px 0 !important;
    box-shadow: var(--shadow-sm) !important;
  }
  [data-testid="stExpander"] summary {
    color: var(--text-1) !important;
    font-weight: 500 !important;
  }

  /* ── Code blocks ── */
  .stCodeBlock, pre, code {
    background: #f1f5f9 !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: var(--radius-sm) !important;
    color: #4f46e5 !important;
  }

  /* ── Download button ── */
  [data-testid="stDownloadButton"] button {
    background: #ffffff !important;
    color: var(--text-1) !important;
    border: 1px solid rgba(79, 70, 229, 0.3) !important;
    border-radius: var(--radius-pill) !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
  }
  [data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent-blue) !important;
    background: rgba(79, 70, 229, 0.06) !important;
  }

  /* ── Tabs ── */
  [data-baseweb="tab-list"] {
    background: #eef2f8 !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
  }
  [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-2) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
  }
  [aria-selected="true"][data-baseweb="tab"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.1) !important;
  }

  /* ── Alert/info boxes ── */
  [data-testid="stInfo"],
  [data-testid="stWarning"],
  [data-testid="stError"],
  [data-testid="stSuccess"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid !important;
  }
  [data-testid="stInfo"] {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    color: #1e40af !important;
  }
  [data-testid="stWarning"] {
    background: #fffbeb !important;
    border-color: #fde68a !important;
    color: #92400e !important;
  }
  [data-testid="stError"] {
    background: #fef2f2 !important;
    border-color: #fecaca !important;
    color: #991b1b !important;
  }
  [data-testid="stSuccess"] {
    background: #f0fdfa !important;
    border-color: #99f6e4 !important;
    color: #134e4a !important;
  }

  /* ── Horizontal rule ── */
  hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
  }

  /* ── Page title ── */
  h1 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.03em !important; }
  h2 { color: #1e293b !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
  h3 { color: #334155 !important; font-weight: 600 !important; }
  p, li { color: #475569 !important; line-height: 1.7 !important; }

  /* ── Custom cards ── */
  .kpi-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }
  .kpi-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-2px);
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-1);
    letter-spacing: -0.03em;
    line-height: 1;
  }
  .kpi-sub {
    font-size: 12px;
    color: var(--text-3);
    margin-top: 6px;
  }

  /* ── Header / Nav ── */
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid rgba(79, 70, 229, 0.12);
    border-radius: var(--radius-lg);
    padding: 12px 24px;
    margin-bottom: 28px;
    box-shadow: 0 1px 6px rgba(15,23,42,0.06), 0 4px 16px rgba(15,23,42,0.04);
  }
  .app-header-brand {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .app-header-logo {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 20px rgba(59, 82, 246, 0.4);
  }
  .app-header-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
  }
  .app-header-sub {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 400;
    margin-top: 2px;
  }
  .app-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: var(--radius-pill);
  }
  .app-header-badge::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse-dot 1.8s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
  }

  /* ── Section headers ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
  }
  .section-header-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
  }
  .section-header-text {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
  }

  /* ── Prediction result box ── */
  .pred-box {
    padding: 24px 20px;
    border-radius: var(--radius-md);
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 16px 0;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
  }
  .pred-box::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    opacity: 0.06;
    background: currentColor;
  }
  .pred-LOW      { color: #1d4ed8; border-color: #bfdbfe; background: #eff6ff; }
  .pred-MODERATE { color: #b45309; border-color: #fde68a; background: #fffbeb; }
  .pred-HIGH     { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }

  /* ── Pipeline status ── */
  .status-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px; color: #475569;
  }
  .status-dot-ok   { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
  .status-dot-fail { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink: 0; }

  /* ── Divider ── */
  .divider { height: 1px; background: #e2e8f0; margin: 24px 0; }

  /* ── Disclaimer ── */
  .disclaimer-banner {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    padding: 12px 18px;
    border-radius: var(--radius-sm);
    font-size: 12.5px;
    color: #78350f;
    margin-bottom: 20px;
  }

  /* ── Fuzzy rule row ── */
  .rule-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 13px;
    color: #475569;
  }
  .rule-num {
    font-size: 10px; font-weight: 700; color: #64748b;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    padding: 3px 8px; border-radius: 4px;
    white-space: nowrap; flex-shrink: 0;
  }
  .rule-badge-low      { background: #dbeafe; color: #1d4ed8; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
  .rule-badge-moderate { background: #fef3c7; color: #b45309; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
  .rule-badge-high     { background: #fee2e2; color: #b91c1c; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; }

  /* ── Probability bar ── */
  .prob-track {
    background: #e2e8f0;
    border-radius: var(--radius-pill);
    height: 8px;
    overflow: hidden;
    margin: 4px 0 12px;
  }
  .prob-fill-low  { height: 100%; background: linear-gradient(90deg, #1d4ed8, #3b82f6); border-radius: inherit; transition: width 0.6s ease; }
  .prob-fill-mod  { height: 100%; background: linear-gradient(90deg, #b45309, #f59e0b); border-radius: inherit; transition: width 0.6s ease; }
  .prob-fill-high { height: 100%; background: linear-gradient(90deg, #b91c1c, #ef4444); border-radius: inherit; transition: width 0.6s ease; }

  /* ── Membership bar ── */
  .mem-row { margin-bottom: 12px; }
  .mem-label { font-size: 12px; color: #475569; margin-bottom: 4px; display: flex; justify-content: space-between; }
  .mem-track { height: 6px; background: #e2e8f0; border-radius: 3px; }
  .mem-fill { height: 100%; border-radius: 3px; }

  /* ── History table search row ── */
  .hist-controls { display: flex; gap: 12px; margin-bottom: 16px; }

  /* Matplotlib dark style overrides */
  .stPlotlyChart { border-radius: var(--radius-md) !important; overflow: hidden !important; }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY LIGHT THEME — consistent with design tokens
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, system-ui, sans-serif", color="#475569", size=12),
    title_font=dict(color="#0f172a", size=15, family="Inter"),
    xaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickcolor="#94a3b8",
        zerolinecolor="#e2e8f0",
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickcolor="#94a3b8",
        zerolinecolor="#e2e8f0",
    ),
    legend=dict(bgcolor="rgba(255,255,255,0)", bordercolor="#e2e8f0", borderwidth=1),
    margin=dict(l=48, r=24, t=48, b=40),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#c7d2fe",
        font_color="#0f172a",
    ),
)
CLASS_COLORS = {"LOW": "#3b82f6", "MODERATE": "#f59e0b", "HIGH": "#ef4444"}

# Apply light theme to matplotlib (all values must be valid matplotlib hex/named colors)
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#cbd5e1",
    "text.color":       "#475569",
    "axes.labelcolor":  "#475569",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#e2e8f0",
    "axes.titlecolor":  "#0f172a",
    "grid.alpha":       0.7,
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _load_features():
    from src.feature_extraction import load_features
    try: return load_features(config=cfg)
    except FileNotFoundError: return None

@st.cache_data
def _load_comparison():
    from src.evaluation import load_comparison_csv
    return load_comparison_csv(config=cfg)

@st.cache_data
def _load_importance():
    from src.feature_selection import load_feature_importance
    try: return load_feature_importance(config=cfg)
    except FileNotFoundError: return None

@st.cache_resource
def _load_fuzzy_model():
    from src.fuzzy_classifier import FuzzyClassifier
    clf = FuzzyClassifier(config=cfg)
    try: clf.load(config=cfg); return clf
    except FileNotFoundError: return None

@st.cache_resource
def _load_rf_model():
    from src.random_forest import RandomForestModel
    rf = RandomForestModel(config=cfg)
    try: rf.load(config=cfg); return rf
    except FileNotFoundError: return None

def _data_status():
    return {
        "dataset":    (PROJECT_ROOT / "data" / "raw" / "ds007169").exists(),
        "preprocessed": any((PROJECT_ROOT / "data" / "processed").rglob("*.fif")),
        "features":   (PROJECT_ROOT / "data" / "features" / "features.csv").exists(),
        "fuzzy_model":(PROJECT_ROOT / "models" / "fuzzy" / "fuzzy_classifier.joblib").exists(),
        "rf_model":   (PROJECT_ROOT / "models" / "random_forest" / "random_forest.joblib").exists(),
        "results":    (PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv").exists(),
    }

def _header():
    st.markdown("""
    <div class="app-header">
      <div class="app-header-brand">
        <div class="app-header-logo">🧠</div>
        <div>
          <div class="app-header-title">Intelligence Designed To Evolve</div>
          <div class="app-header-sub">EEG Fuzzy Cognitive Load Research Platform · ds007169</div>
        </div>
      </div>
      <span class="app-header-badge">Live Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

def _disclaimer():
    st.markdown("""
    <div class="disclaimer-banner">
      <strong>⚠ Research Prototype Only.</strong> This dashboard is for academic demonstration.
      It is <strong>NOT</strong> a medical device, clinical diagnostic tool, or psychological assessment system.
      Do not use for any clinical or diagnostic purpose.
    </div>
    """, unsafe_allow_html=True)

def _section(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <div class="section-header-icon">{icon}</div>
      <div class="section-header-text">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def _plotly_apply(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:6px 0 18px; border-bottom:2px solid #e0e4ef; margin-bottom:14px;">
      <div style="font-size:16px; font-weight:900; letter-spacing:-0.02em;">
        <span style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">EEG Cognitive Load</span>
      </div>
      <div style="font-size:10px; font-weight:600; color:#94a3b8; margin-top:4px; letter-spacing:0.06em; text-transform:uppercase;">Research Dashboard &middot; v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 EEG Analysis",
            "🔬 Feature Analysis",
            "🧠 Cognitive Load Prediction",
            "📈 Model Comparison",
            "📜 Experiment History",
            "ℹ️ Methodology & Guide",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1px;background:#e0e4ef;margin:12px 0;'></div>", unsafe_allow_html=True)

    status = _data_status()
    st.markdown("<div style='font-size:10px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Pipeline Status</div>", unsafe_allow_html=True)

    status_items = [
        ("dataset", "Dataset"),
        ("preprocessed", "Preprocessed EEG"),
        ("features", "Features"),
        ("fuzzy_model", "Fuzzy Model"),
        ("rf_model", "Random Forest"),
        ("results", "Results"),
    ]
    for key, label in status_items:
        dot_cls = "status-dot-ok" if status[key] else "status-dot-fail"
        badge_color = "#dcfce7" if status[key] else "#fee2e2"
        badge_text_color = "#166534" if status[key] else "#991b1b"
        badge_text = "Ready" if status[key] else "Missing"
        st.markdown(f"""
        <div class="status-item">
          <div class="{dot_cls}"></div>
          <span style='flex:1;font-weight:500;color:#374151;'>{label}</span>
          <span style='font-size:10px;font-weight:700;background:{badge_color};color:{badge_text_color};padding:2px 7px;border-radius:999px;'>{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#e0e4ef;margin:12px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#9ca3af; line-height:1.9;'>
      <b style='color:#64748b;'>Dataset</b> &nbsp;ds007169<br>
      <b style='color:#64748b;'>Source</b> &nbsp;OpenNeuro.org<br>
      <b style='color:#64748b;'>License</b> &nbsp;CC0 Public Domain
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    _header()
    _disclaimer()

    df = _load_features()
    comp = _load_comparison()

    # ── KPI Row ──
    _section("📊", "System Overview")

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, cfg.dataset.id,                  "Dataset ID",     "OpenNeuro"),
        (k2, cfg.dataset.subjects,             "Subjects",       "Participants"),
        (k3, cfg.dataset.channels,             "EEG Channels",   "10-20 Montage"),
        (k4, f"{cfg.dataset.sampling_rate} Hz","Sampling Rate",  "250 Hz standard"),
        (k5, f"{df.shape[0]:,}" if df is not None else "—", "Windows", "4 s / 2 s overlap"),
    ]
    for col, val, label, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Dataset Overview + Model Results ──
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    col_left, col_right = st.columns([1.1, 0.9], gap="medium")

    with col_left:
        _section("📦", "Dataset Configuration")
        info_md = f"""
| Property | Value |
|---|---|
| **Dataset ID** | `{cfg.dataset.id}` |
| **Full Name** | {cfg.dataset.name} |
| **Source** | [OpenNeuro]({cfg.dataset.source}) |
| **License** | {cfg.dataset.license} |
| **EEG Format** | BrainVision (.vhdr / .eeg / .vmrk) |
| **Tasks** | N-back levels 1–4 |
| **Label Mapping** | nback_1→LOW · nback_2→MODERATE · nback_3/4→HIGH |
"""
        st.markdown(info_md)
        st.info("Label mapping is an operational research categorization based on task difficulty — not a validated cognitive scale.")

    with col_right:
        _section("⚖️", "Class Distribution")
        if df is not None:
            counts = df["label"].value_counts().reindex(["LOW", "MODERATE", "HIGH"], fill_value=0)
            if HAS_PLOTLY:
                fig_dist = go.Figure(go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker=dict(
                        color=[CLASS_COLORS[c] for c in counts.index],
                        line=dict(width=0),
                    ),
                    text=[f"{v:,} ({v/len(df)*100:.1f}%)" for v in counts.values],
                    textposition="outside",
                    textfont=dict(size=12, color="#a0a0b8"),
                ))
                fig_dist.update_layout(**PLOTLY_LAYOUT, title="Cognitive Load Windows", height=260)
                fig_dist.update_yaxes(showgrid=True)
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.bar_chart(counts)
        else:
            st.warning("Features not available. Run: `python main.py --features`")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Model Results ──
    _section("🏆", "Latest Cross-Validated Performance")
    if comp is not None:
        metric_display = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
        available = [c for c in metric_display if c in comp.columns]

        if HAS_PLOTLY:
            fig_comp = go.Figure()
            bar_colors = ["#3b82f6", "#8b5cf6"]
            for i, (_, row) in enumerate(comp.iterrows()):
                vals = [row.get(m, 0) for m in available]
                fig_comp.add_trace(go.Bar(
                    name=row["model"],
                    x=[m.replace("_", " ").title() for m in available],
                    y=vals,
                    marker_color=bar_colors[i % len(bar_colors)],
                    marker_line_width=0,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=11, color="#a0a0b8"),
                ))
            fig_comp.update_layout(**PLOTLY_LAYOUT, title="Model Performance Comparison (Subject-wise CV)", barmode="group", height=340, yaxis_range=[0, 1.1])
            st.plotly_chart(fig_comp, use_container_width=True)

        for _, row in comp.iterrows():
            c0, c1, c2, c3, c4 = st.columns(5)
            c0.metric("Model", row["model"])
            c1.metric("Accuracy", f"{row['accuracy']:.3f}")
            c2.metric("F1 (macro)", f"{row['f1_macro']:.3f}")
            c3.metric("Precision", f"{row['precision_macro']:.3f}")
            c4.metric("Recall", f"{row['recall_macro']:.3f}")
    else:
        st.info("No evaluation results yet.\n\nRun: `python main.py --all`")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    _section("🖥️", "Quick Start Commands")
    st.code("""# Complete pipeline (first run)
python main.py --all

# Individual stages
python main.py --download     # Step 1: Download dataset
python main.py --preprocess   # Step 2: Clean & filter EEG
python main.py --features     # Step 3: Extract features
python main.py --train        # Step 4: Train models
python main.py --evaluate     # Step 5: Evaluate & record

# Tests & status
pytest tests/ -v
python main.py --status""", language="bash")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: EEG ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EEG Analysis":
    _header()
    _disclaimer()
    _section("📊", "EEG Signal Visualisation")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    subject_dirs = sorted([d.name for d in processed_dir.glob("sub-*") if d.is_dir()]) if processed_dir.exists() else []

    if not subject_dirs:
        st.warning("No preprocessed EEG found.\n\nRun: `python main.py --preprocess`")
        st.stop()

    ctrl_col, viz_col = st.columns([1, 3], gap="medium")

    with ctrl_col:
        st.markdown("<p style='font-size:12px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px;'>Controls</p>", unsafe_allow_html=True)
        sub_id = st.selectbox("Subject", subject_dirs, key="eeg_sub")
        n_channels_show = st.slider("Channels to Display", 1, 19, 6, key="eeg_ch")
        time_start = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0, key="eeg_ts")
        time_end   = st.number_input("End time (s)",   min_value=1.0, value=8.0, step=1.0, key="eeg_te")

    with viz_col:
        @st.cache_resource
        def _load_preprocessed_raw(sid):
            from src.preprocessing import load_preprocessed
            return load_preprocessed(sid, config=cfg)

        with st.spinner(f"Loading {sub_id}…"):
            raw = _load_preprocessed_raw(sub_id)

        if raw is None:
            st.error(f"Could not load preprocessed data for {sub_id}.")
        else:
            sfreq     = raw.info["sfreq"]
            total_dur = raw.times[-1]
            start_s   = max(0, time_start)
            end_s     = min(total_dur, time_end)
            start_samp = int(start_s * sfreq)
            end_samp   = int(end_s * sfreq)
            data_slice = raw.get_data()[:, start_samp:end_samp]

            c1, c2, c3 = st.columns(3)
            c1.metric("EEG Channels",   len(raw.ch_names))
            c2.metric("Duration",       f"{total_dur:.1f} s")
            c3.metric("Sampling Rate",  f"{sfreq:.0f} Hz")

            try:
                import plotly.graph_objects as go
                channels = raw.ch_names[:n_channels_show]
                ch_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
                             "#14b8a6", "#f97316", "#ec4899", "#a3e635", "#06b6d4",
                             "#7c3aed", "#65a30d", "#dc2626", "#0891b2", "#c026d3",
                             "#0d9488", "#b45309", "#1d4ed8", "#166534"]

                fig_wv = go.Figure()
                n_samp = data_slice.shape[1]
                times = np.linspace(start_s, end_s, n_samp)
                offset_scale = 6.0 * np.std(data_slice[:n_channels_show]) if data_slice[:n_channels_show].std() > 0 else 1e-5

                for i, ch in enumerate(channels):
                    sig = data_slice[i]
                    offset = (len(channels) - 1 - i) * offset_scale
                    fig_wv.add_trace(go.Scatter(
                        x=times, y=sig + offset,
                        mode="lines",
                        name=ch,
                        line=dict(color=ch_colors[i % len(ch_colors)], width=1.0),
                        hovertemplate=f"<b>{ch}</b><br>Time: %{{x:.3f}} s<br>Amplitude: %{{customdata:.2e}} V<extra></extra>",
                        customdata=sig,
                    ))

                # Build layout: merge PLOTLY_LAYOUT but override yaxis separately
                wv_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                wv_layout.update({
                    "title": f"{sub_id} \u2014 EEG Waveforms ({start_s:.1f}\u2013{end_s:.1f} s)",
                    "xaxis_title": "Time (s)",
                    "yaxis": dict(
                        gridcolor="#e2e8f0",
                        linecolor="#cbd5e1",
                        tickcolor="#94a3b8",
                        zerolinecolor="#e2e8f0",
                        tickvals=[(len(channels) - 1 - i) * offset_scale for i in range(len(channels))],
                        ticktext=channels,
                        showgrid=False,
                        zeroline=False,
                    ),
                    "height": 420,
                    "showlegend": False,
                })
                fig_wv.update_layout(**wv_layout)
                st.plotly_chart(fig_wv, use_container_width=True)
            except ImportError:
                from src.visualization import plot_eeg_waveform
                fig_wv = plot_eeg_waveform(data_slice, sfreq, raw.ch_names, n_channels=n_channels_show,
                                           title=f"{sub_id} — EEG Waveform ({start_s:.1f}–{end_s:.1f} s)")
                st.pyplot(fig_wv, use_container_width=True)
                plt.close("all")

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            _section("📡", "Power Spectral Density (Welch)")

            try:
                from src.visualization import plot_power_spectrum
                fig_psd = plot_power_spectrum(data_slice, sfreq, title=f"{sub_id} — Mean PSD")
                st.pyplot(fig_psd, use_container_width=True)
                plt.close("all")
            except Exception as e:
                st.warning(f"PSD unavailable: {e}")

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            with st.expander("📋 Channel List & Types"):
                ch_df = pd.DataFrame({
                    "Channel": raw.ch_names,
                    "Type":    [raw.get_channel_types([c])[0] for c in raw.ch_names],
                })
                st.dataframe(ch_df, use_container_width=True, height=240, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: FEATURE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    _header()
    _disclaimer()
    _section("🔬", "EEG Feature Analysis")

    df = _load_features()
    if df is None:
        st.warning("Features not available. Run: `python main.py --features`")
        st.stop()

    from src.feature_extraction import get_feature_columns
    feature_cols = get_feature_columns(df)

    # ── KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, sub in [
        (k1, f"{len(df):,}",              "Total Windows",  "4 s segments"),
        (k2, df["subject_id"].nunique(),   "Subjects",       "Unique participants"),
        (k3, len(feature_cols),            "Features",       "Extracted per window"),
        (k4, df["label"].nunique(),        "Classes",        "Cognitive load levels"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Class Distribution + Subject Windows ──
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    cl, cr = st.columns(2, gap="medium")

    with cl:
        _section("📊", "Class Distribution")
        counts = df["label"].value_counts().reindex(["LOW", "MODERATE", "HIGH"], fill_value=0)
        if HAS_PLOTLY:
            fig_cls = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker=dict(color=[CLASS_COLORS[c] for c in counts.index], line=dict(width=0)),
                text=[f"{v:,} ({v/len(df)*100:.1f}%)" for v in counts.values],
                textposition="outside",
            ))
            fig_cls.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Windows", showlegend=False)
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.bar_chart(counts)

    with cr:
        _section("👤", "Windows per Subject")
        sub_counts = df.groupby("subject_id").size().sort_values(ascending=False)
        if HAS_PLOTLY:
            fig_sub = go.Figure(go.Bar(
                x=sub_counts.index.tolist(), y=sub_counts.values.tolist(),
                marker=dict(color="#3b82f6", opacity=0.8, line=dict(width=0)),
            ))
            fig_sub.update_layout(**PLOTLY_LAYOUT, height=260, xaxis_tickangle=-45, yaxis_title="Windows", showlegend=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            st.bar_chart(sub_counts)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Boxplot ──
    _section("📦", "Feature Distribution by Cognitive Load Class")
    selected_feat = st.selectbox(
        "Select feature",
        feature_cols,
        index=feature_cols.index("theta_beta_ratio") if "theta_beta_ratio" in feature_cols else 0,
    )

    present_classes = [c for c in ["LOW", "MODERATE", "HIGH"] if c in df["label"].unique()]
    data_by_class = [df[df["label"] == c][selected_feat].dropna().values for c in present_classes]

    if HAS_PLOTLY:
        from plotly.subplots import make_subplots
        fig_box = go.Figure()
        for cls, data in zip(present_classes, data_by_class):
            fig_box.add_trace(go.Box(
                y=data, name=cls,
                marker=dict(color=CLASS_COLORS[cls], opacity=0.7),
                line=dict(color=CLASS_COLORS[cls]),
                boxmean="sd",
                jitter=0.3, pointpos=-1.8,
                boxpoints="outliers",
                marker_size=4,
            ))
        fig_box.update_layout(**PLOTLY_LAYOUT, title=f"{selected_feat} by Class", yaxis_title=selected_feat, height=360)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        fig_bx, ax_bx = plt.subplots(figsize=(7, 4))
        ax_bx.boxplot(data_by_class, patch_artist=True,
                      medianprops=dict(color="#f0f0f5", linewidth=2),
                      boxprops=dict(alpha=0.7))
        ax_bx.set_xticklabels(present_classes)
        ax_bx.set_title(f"{selected_feat} by Class")
        st.pyplot(fig_bx, use_container_width=True)
        plt.close("all")

    # Stats table
    stats_data = {}
    for cls, data in zip(present_classes, data_by_class):
        if len(data) > 0:
            stats_data[cls] = {
                "Count": len(data), "Mean": np.mean(data), "Std": np.std(data),
                "Min": np.min(data), "Q25": np.percentile(data, 25),
                "Median": np.median(data), "Q75": np.percentile(data, 75),
                "Max": np.max(data),
            }
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data).T.round(4), use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Feature Importance ──
    _section("🎯", "Feature Importance — Kruskal-Wallis & Mutual Information")
    imp_df = _load_importance()
    if imp_df is not None:
        plot_df = imp_df.head(15).copy()
        if HAS_PLOTLY:
            fig_imp = go.Figure(go.Bar(
                x=plot_df.get("mutual_info", pd.Series([0]*len(plot_df))).values[::-1],
                y=plot_df["feature"].values[::-1],
                orientation="h",
                marker=dict(
                    color=plot_df.get("mutual_info", pd.Series([0]*len(plot_df))).values[::-1],
                    colorscale=[[0, "#1d4ed8"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
                    showscale=False,
                    line=dict(width=0),
                ),
                text=[f"{v:.4f}" for v in plot_df.get("mutual_info", pd.Series([0]*len(plot_df))).values[::-1]],
                textposition="outside",
            ))
            fig_imp.update_layout(**PLOTLY_LAYOUT, title="Top 15 Features — Mutual Information Score", height=420, xaxis_title="Mutual Information", showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            fig_i, ax_i = plt.subplots(figsize=(8, 5))
            ax_i.barh(plot_df["feature"].values[::-1], plot_df.get("mutual_info", 0).values[::-1], color="#3b82f6")
            ax_i.set_xlabel("Mutual Information")
            st.pyplot(fig_i, use_container_width=True)
            plt.close("all")

        with st.expander("📋 Full Importance Table"):
            st.dataframe(imp_df.style.format({"kruskal_stat": "{:.2f}", "kruskal_pvalue": "{:.4f}", "mutual_info": "{:.4f}"}), use_container_width=True)
    else:
        st.info("Feature importance not computed. Run `python main.py --features`")

    with st.expander("📋 Raw Feature Data (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

    with st.expander("📊 Feature Summary Statistics"):
        st.dataframe(df[feature_cols].describe().round(4).T, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: COGNITIVE LOAD PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive Load Prediction":
    _header()
    _disclaimer()
    _section("🧠", "Dual-Model Cognitive Load Inference")

    df         = _load_features()
    fuzzy_clf  = _load_fuzzy_model()
    rf_model   = _load_rf_model()

    if df is None:
        st.warning("Features not available. Run: `python main.py --features`")
        st.stop()
    if fuzzy_clf is None:
        st.warning("Fuzzy model not trained. Run: `python main.py --train`")
        st.stop()

    from src.feature_extraction import get_feature_columns
    from src.explainability import explain_fuzzy, explain_rf

    feature_cols = get_feature_columns(df)

    # ── Input Mode ──
    mode = st.radio("Input Mode", ["Select from Dataset", "Manual Feature Entry"], horizontal=True, key="pred_mode")

    if mode == "Select from Dataset":
        sc1, sc2 = st.columns([1, 2], gap="medium")
        with sc1:
            subjects = df["subject_id"].unique().tolist()
            sel_subject = st.selectbox("Subject", subjects, key="pred_sub")
        with sc2:
            sub_df = df[df["subject_id"] == sel_subject]
            sel_window = st.slider("Window ID", int(sub_df["window_id"].min()),
                                   int(sub_df["window_id"].max()), int(sub_df["window_id"].iloc[0]), key="pred_win")
        row = sub_df[sub_df["window_id"] == sel_window]
        if row.empty: row = sub_df.iloc[[0]]
        row = row.iloc[0]
        true_label = row.get("label", "UNKNOWN")
        x_all = row[feature_cols].values.astype(float)
        st.info(f"**True Label:** {true_label} &nbsp;·&nbsp; **Trial Type:** {row.get('trial_type', 'N/A')}")

    else:
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        means = df[feature_cols].mean()
        x_all_dict = {}
        cols_sl = st.columns(4)
        for i, feat in enumerate(feature_cols):
            mean_val = float(means.get(feat, 0.0))
            feat_min = float(df[feat].quantile(0.01))
            feat_max = float(df[feat].quantile(0.99))
            with cols_sl[i % 4]:
                x_all_dict[feat] = st.slider(feat, feat_min, feat_max, mean_val,
                                              step=(feat_max - feat_min) / 200, format="%.4f", key=f"sl_{feat}")
        x_all = np.array([x_all_dict[f] for f in feature_cols])
        true_label = "N/A (manual)"

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("⚡ Run Dual-Model Inference", type="primary", use_container_width=True):
        fuzzy_feature_idx = [feature_cols.index(f) for f in fuzzy_clf.feature_names if f in feature_cols]
        x_fuzzy = x_all[fuzzy_feature_idx]

        with st.spinner("Running Mamdani Fuzzy Inference + Random Forest…"):
            fuzzy_result = None
            try:
                fuzzy_result = fuzzy_clf.predict_single(x_fuzzy)
            except Exception as e:
                st.error(f"Fuzzy inference error: {e}")

        # ── Side-by-side results ──
        col_f, col_rf = st.columns(2, gap="medium")

        with col_f:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            _section("🔶", "Mamdani Fuzzy Classifier")
            if fuzzy_result:
                pred = fuzzy_result["predicted_class"]
                score = fuzzy_result.get("fuzzy_score", 50)
                conf = fuzzy_result.get("confidence", 0)
                st.markdown(f'<div class="pred-box pred-{pred}">{pred}</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Fuzzy Score", f"{score:.1f}/100")
                m2.metric("Confidence", f"{conf:.1%}")
                m3.metric("True Label", true_label)

                # Membership breakdown
                _section("📐", "Feature Memberships")
                mems = fuzzy_result.get("memberships", {})
                for feat_name, levels in mems.items():
                    st.markdown(f"""<div class='mem-row'>
                      <div class='mem-label'>
                        <span>{feat_name}</span>
                        <span style='color:#60607a;'>LOW={levels.get('LOW',0):.2f} · MED={levels.get('MEDIUM',0):.2f} · HIGH={levels.get('HIGH',0):.2f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    for key, color in [("LOW", "#3b82f6"), ("MEDIUM", "#f59e0b"), ("HIGH", "#ef4444")]:
                        v = levels.get(key, 0)
                        st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>
                          <span style='font-size:11px;color:#60607a;width:40px;'>{key}</span>
                          <div style='flex:1;height:5px;background:rgba(255,255,255,0.05);border-radius:3px;'>
                            <div style='width:{v*100:.1f}%;height:100%;background:{color};border-radius:3px;'></div>
                          </div>
                          <span style='font-size:11px;color:#a0a0b8;width:32px;text-align:right;'>{v:.2f}</span>
                        </div>""", unsafe_allow_html=True)

                # Activated rules
                rules = fuzzy_result.get("activated_rules", [])
                if rules:
                    _section("📜", "Activated Fuzzy Rules")
                    for r in rules[:6]:
                        consq = r.get("consequent", pred)
                        badge_cls = f"rule-badge-{consq.lower()}"
                        st.markdown(f"""<div class="rule-row">
                          <span class="rule-num">Rule {r['rule_number']}</span>
                          <span style='flex:1;'>{r['rule_text']}</span>
                          <span class="{badge_cls}">{consq}</span>
                          <span style='font-size:11px;color:#60607a;white-space:nowrap;'>str={r['strength']:.2f}</span>
                        </div>""", unsafe_allow_html=True)

                with st.expander("📄 Full Fuzzy Explanation"):
                    explanation = explain_fuzzy(fuzzy_result, x_fuzzy, fuzzy_clf.feature_names)
                    st.text(explanation)
            else:
                st.error("Fuzzy inference failed.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_rf:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            _section("🌲", "Random Forest Baseline")
            if rf_model and rf_model.is_trained:
                try:
                    proba = rf_model.predict_proba(x_all.reshape(1, -1))[0]
                    pred_int = int(np.argmax(proba))
                    class_names = ["LOW", "MODERATE", "HIGH"]
                    pred_rf = class_names[pred_int]

                    st.markdown(f'<div class="pred-box pred-{pred_rf}">{pred_rf}</div>', unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    m1.metric("Confidence", f"{proba[pred_int]:.1%}")
                    m2.metric("True Label", true_label)

                    _section("📊", "Class Probabilities")
                    prob_data = [("LOW", proba[0], "prob-fill-low"), ("MODERATE", proba[1], "prob-fill-mod"), ("HIGH", proba[2], "prob-fill-high")]
                    for cls_name, p, fill_cls in prob_data:
                        st.markdown(f"""<div style='margin-bottom:12px;'>
                          <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;'>
                            <span style='color:#a0a0b8;font-weight:500;'>{cls_name}</span>
                            <span style='color:#f0f0f5;font-weight:700;'>{p:.1%}</span>
                          </div>
                          <div class='prob-track'>
                            <div class='{fill_cls}' style='width:{p*100:.1f}%;'></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    # Feature Importance
                    fi_df = rf_model.get_feature_importance().head(8)
                    _section("🏆", "Top Feature Importances")
                    for _, fi_row in fi_df.iterrows():
                        pct = fi_row["importance"]
                        st.markdown(f"""<div style='margin-bottom:8px;'>
                          <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>
                            <span style='color:#a0a0b8;'>{fi_row['feature']}</span>
                            <span style='color:#f0f0f5;font-weight:600;'>{pct:.4f}</span>
                          </div>
                          <div style='height:4px;background:rgba(255,255,255,0.05);border-radius:2px;'>
                            <div style='width:{min(pct*10, 1)*100:.1f}%;height:100%;background:linear-gradient(90deg,#1d4ed8,#8b5cf6);border-radius:2px;'></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    with st.expander("📄 Full RF Explanation"):
                        rf_explanation = explain_rf(rf_model, x_all, feature_cols)
                        st.text(rf_explanation)

                except Exception as e:
                    st.error(f"RF prediction error: {e}")
            else:
                st.warning("Random Forest not trained. Run: `python main.py --train`")
            st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    _header()
    _disclaimer()
    _section("📈", "Cross-Validated Model Performance")

    comp = _load_comparison()
    if comp is None:
        st.warning("No evaluation results found.\n\nRun: `python main.py --evaluate`")
        st.stop()

    st.info("All metrics are from **subject-wise StratifiedGroupKFold** cross-validation. Subjects in the test fold were never seen during training — this ensures realistic generalisation estimates.")

    # ── Metrics Table ──
    display_cols = ["model", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    available = [c for c in display_cols if c in comp.columns]
    styled = comp[available].style.format({c: "{:.3f}" for c in available if c != "model"}).highlight_max(
        subset=[c for c in available if c != "model"], color="rgba(59, 82, 246, 0.25)"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Radar Chart or Bar Chart ──
    try:
        import plotly.graph_objects as go
        _section("📊", "Visual Performance Comparison")
        metric_cols = [c for c in ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"] if c in comp.columns]
        label_map = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Acc", "f1_macro": "F1 Macro", "precision_macro": "Precision", "recall_macro": "Recall"}
        bar_colors = ["#3b82f6", "#8b5cf6"]

        fig_mc = go.Figure()
        for i, (_, row) in enumerate(comp.iterrows()):
            vals = [row.get(m, 0) for m in metric_cols]
            fig_mc.add_trace(go.Bar(
                name=row["model"],
                x=[label_map.get(m, m) for m in metric_cols],
                y=vals,
                marker_color=bar_colors[i % len(bar_colors)],
                marker_line_width=0,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(size=12, color="#a0a0b8"),
            ))
        fig_mc.update_layout(**PLOTLY_LAYOUT, title="Model Comparison — Subject-wise CV", barmode="group", height=360, yaxis_range=[0, 1.15], yaxis_title="Score")
        st.plotly_chart(fig_mc, use_container_width=True)
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 5))
        n_metrics = len(metric_cols)
        x = np.arange(n_metrics)
        width = 0.35
        for i, (_, row) in enumerate(comp.iterrows()):
            vals = [row.get(m, 0) for m in metric_cols]
            ax.bar(x + (i - 0.5) * width, vals, width, label=row["model"], alpha=0.85, edgecolor="#0c0c18")
            for bar in ax.patches[-n_metrics:]:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([label_map.get(m, m) for m in metric_cols])
        ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True); plt.close("all")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    _section("🔲", "Confusion Matrices")
    figures_dir = PROJECT_ROOT / "results" / "figures"
    cm1, cm2 = st.columns(2, gap="medium")
    for col, name, fname in [
        (cm1, "Fuzzy Classifier",  "fuzzy_classifier_confusion_matrix.png"),
        (cm2, "Random Forest",     "random_forest_confusion_matrix.png"),
    ]:
        img_path = figures_dir / fname
        with col:
            st.markdown(f"<div style='font-size:13px;font-weight:600;color:#a0a0b8;margin-bottom:8px;'>{name}</div>", unsafe_allow_html=True)
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Run `python main.py --evaluate` to generate.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Fuzzy Rules ──
    _section("📜", "Mamdani Fuzzy Rules Database")
    try:
        from src.database import get_fuzzy_rules, init_db
        init_db()
        rules = get_fuzzy_rules()
        if rules:
            for r in rules:
                consq = r.consequent
                badge_cls = f"rule-badge-{consq.lower()}"
                st.markdown(f"""<div class="rule-row">
                  <span class="rule-num">Rule {r.rule_number}</span>
                  <span style='flex:1;'>{r.rule_text}</span>
                  <span class="{badge_cls}">{consq}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No fuzzy rules stored. Run `python main.py --train`")
    except Exception as e:
        st.warning(f"Could not load rules: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── RF Feature Importance ──
    _section("🌲", "Random Forest — Gini Feature Importances")
    rf_model = _load_rf_model()
    if rf_model and rf_model.is_trained:
        fi_df = rf_model.get_feature_importance()
        try:
            import plotly.graph_objects as go
            plot_df = fi_df.head(15).copy()
            fig_fi = go.Figure(go.Bar(
                x=plot_df["importance"].values[::-1],
                y=plot_df["feature"].values[::-1],
                orientation="h",
                marker=dict(color=plot_df["importance"].values[::-1], colorscale=[[0, "#1d4ed8"], [1, "#8b5cf6"]], line=dict(width=0)),
                text=[f"{v:.4f}" for v in plot_df["importance"].values[::-1]], textposition="outside",
            ))
            fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 15 — Feature Importances (Gini)", height=420, xaxis_title="Importance", showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        except ImportError:
            fig_f, ax_f = plt.subplots(figsize=(8, 5))
            ax_f.barh(fi_df["feature"].values[:15][::-1], fi_df["importance"].values[:15][::-1])
            st.pyplot(fig_f, use_container_width=True); plt.close("all")
        st.caption("Feature importance shows contribution to prediction decisions. It does NOT imply causality.")
    else:
        st.info("RF model not loaded. Run `python main.py --train`")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: EXPERIMENT HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📜 Experiment History":
    _header()
    _disclaimer()
    _section("📜", "Experiment History — SQLite Database")

    st.markdown("<div style='color:var(--text-2);font-size:14px;margin-bottom:20px;'>All model runs, evaluation metrics, and preprocessing records stored in <code>data/eeg_project.db</code>.</div>", unsafe_allow_html=True)

    try:
        from src.database import get_all_experiment_history_df, init_db
        init_db()
        history_df = get_all_experiment_history_df()

        if history_df is not None and not history_df.empty:
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            best_acc = history_df["Accuracy"].max() if "Accuracy" in history_df.columns else None
            best_f1  = history_df["F1 (Macro)"].max() if "F1 (Macro)" in history_df.columns else None
            for col, val, label, sub in [
                (k1, len(history_df),               "Total Runs",     "Recorded experiments"),
                (k2, history_df["Model"].nunique() if "Model" in history_df.columns else "—", "Models", "Unique classifiers"),
                (k3, f"{best_acc:.3f}" if best_acc else "N/A", "Peak Accuracy", "Best run"),
                (k4, f"{best_f1:.3f}"  if best_f1  else "N/A", "Peak F1 Macro", "Best run"),
            ]:
                with col:
                    st.markdown(f"""<div class="kpi-card">
                      <div class="kpi-label">{label}</div>
                      <div class="kpi-value">{val}</div>
                      <div class="kpi-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Filter controls
            fc1, fc2, fc3 = st.columns([1, 1, 2], gap="medium")
            with fc1:
                model_filter = st.selectbox("Filter by Model", ["All Models"] + sorted(history_df["Model"].unique().tolist()) if "Model" in history_df.columns else ["All Models"])
            with fc2:
                sort_col = st.selectbox("Sort by", ["Accuracy", "F1 (Macro)", "Run ID"] if all(c in history_df.columns for c in ["Accuracy", "F1 (Macro)", "Run ID"]) else history_df.columns[:3].tolist())
            with fc3:
                search_term = st.text_input("🔍  Search", placeholder="Filter by version, date, model name…")

            filtered_df = history_df.copy()
            if model_filter != "All Models" and "Model" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Model"] == model_filter]
            if search_term:
                filtered_df = filtered_df[filtered_df.astype(str).apply(lambda r: r.str.contains(search_term, case=False).any(), axis=1)]
            if sort_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(sort_col, ascending=False)

            _section("📋", f"Historical Runs ({len(filtered_df)} records)")
            fmt = {}
            for col_name in ["Accuracy", "F1 (Macro)", "Balanced Acc", "Precision", "Recall"]:
                if col_name in filtered_df.columns:
                    fmt[col_name] = "{:.3f}"
            st.dataframe(filtered_df.style.format(fmt), use_container_width=True, hide_index=True)

            # Export
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export as CSV", data=csv, file_name="eeg_experiment_history.csv", mime="text/csv")

        else:
            st.info("No experiment runs in the database yet.\n\nRun: `python main.py --evaluate`")

    except Exception as e:
        st.error(f"Could not load experiment history: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7: METHODOLOGY & GUIDE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Methodology & Guide":
    _header()
    _disclaimer()
    _section("ℹ️", "Methodology, Architecture & Viva Guide")

    st.markdown("""<div class='disclaimer-banner' style='background:rgba(239,68,68,0.06);border-color:rgba(239,68,68,0.2);border-left-color:#ef4444;'>
      <strong>⚠ Ethical & Scientific Disclaimer:</strong> This system is a <strong>cognitive-load research prototype</strong> and is NOT a medical device, clinical diagnostic tool, or psychological assessment system.
      Results must NOT be used for any medical, clinical, or diagnostic purpose.
      All EEG data is <strong>public, anonymized</strong>, and licensed under <strong>CC0</strong> (OpenNeuro ds007169).
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🏗 Architecture", "📚 Dataset & Stack", "🎓 Viva Defense"])

    with tab1:
        st.markdown("""
## Project Overview

This system demonstrates a complete end-to-end pipeline for EEG-based cognitive load classification:

1. **Signal Processing** — MNE-Python: 1–40 Hz bandpass, 50 Hz notch filter, average referencing, artifact rejection (±150 µV threshold)
2. **Feature Engineering** — 17 features: Theta/Alpha/Beta band powers, relative powers, TAR/BAR, spectral entropy, sample entropy, statistical moments
3. **Fuzzy Logic Classification** — 13-rule Mamdani inference system (scikit-fuzzy) with data-driven triangular membership functions and centroid defuzzification
4. **Machine Learning Baseline** — Random Forest (scikit-learn) with Gini feature importances
5. **Explainability** — Activated rules display, fuzzy membership breakdowns, plain-language explanations
6. **Cross-Validation** — Subject-wise StratifiedGroupKFold (subjects in test fold never seen during training)

### Cognitive Load Classes
| Class | N-back Level | Cognitive Demand |
|---|---|---|
| **LOW** | n-back level 1 | Minimal working memory load |
| **MODERATE** | n-back level 2 | Intermediate demand |
| **HIGH** | n-back levels 3 & 4 | Maximum demand |
        """)

        _section("⚠️", "Known Limitations")
        st.markdown("""
- **Small N:** Only 18 subjects — limited statistical power for generalization claims
- **Proxy Labels:** Labels derived from task difficulty, NOT validated direct brain state measures
- **No Baseline Condition:** n-back 1 used as "LOW" proxy rather than eyes-closed rest
- **EEG Variability:** Highly individual — inter-subject variability limits cross-subject generalization
- **No Clinical Validation:** Prototype for educational and research demonstration only
        """)

    with tab2:
        st.markdown("### System Architecture — Data Flow")
        st.code("""
PUBLIC EEG DATASET (OpenNeuro ds007169 · CC0)
    │  BrainVision format: .vhdr / .eeg / .vmrk
    ▼
MNE-PYTHON PREPROCESSING
    ├── 1–40 Hz FIR bandpass filter
    ├── 50 Hz notch (mains interference)
    ├── Common average referencing
    ├── Amplitude artifact rejection (±150 µV)
    └── 4s sliding windows · 2s overlap
    ▼
FEATURE EXTRACTION (17 per window)
    ├── Band powers:  Delta · Theta · Alpha · Beta
    ├── Relative powers (% total power)
    ├── Ratios: TAR (Theta/Alpha) · BAR (Beta/Alpha) · TBR
    ├── Spectral Entropy (Shannon) · Sample Entropy
    └── Statistical: Mean · Std · Skewness · Kurtosis
    ▼
FEATURE SELECTION
    ├── Kruskal-Wallis H test (inter-class separability)
    └── Mutual Information ranking
    ▼
    ┌─────────────────────────┐
    │                         │
    ▼                         ▼
MAMDANI FUZZY SYSTEM     RANDOM FOREST
 scikit-fuzzy             scikit-learn
 13 IF-THEN rules         n_estimators=200
 Triangular MFs           Gini criterion
 Centroid defuzzification Subject-wise GroupKFold
    │                         │
    ▼                         ▼
LOW / MODERATE / HIGH    LOW / MODERATE / HIGH
    │                         │
    └──────────┬──────────────┘
               ▼
    SUBJECT-WISE CV EVALUATION
    StratifiedGroupKFold (5 folds)
    Accuracy · F1 · Precision · Recall
               ▼
    EXPLAINABILITY LAYER
    Activated rule decomposition
    Membership function breakdown
    Natural language explanations
               ▼
    STREAMLIT DASHBOARD + SQLite DATABASE
""", language="text")

    with tab3:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown(f"""
### Dataset Reference
| Property | Value |
|---|---|
| **ID** | `{cfg.dataset.id}` |
| **Name** | {cfg.dataset.name} |
| **Source** | OpenNeuro.org |
| **DOI** | doi:10.18112/openneuro.ds007169.v1.0.0 |
| **License** | CC0 (Public Domain) |
| **Subjects** | 18 |
| **EEG Channels** | 19 (10-20 montage) |
| **Sampling Rate** | 250 Hz |
| **Format** | BrainVision (.vhdr/.eeg/.vmrk) |
""")
            st.code("""Booth, L., & Barras, M. (2024).
Cognitive Workload 5-level n-back [Data set].
OpenNeuro. doi:10.18112/openneuro.ds007169.v1.0.0
License: CC0 (Public Domain)""")

        with col_b:
            st.markdown("""
### Technology Stack
| Component | Library |
|---|---|
| EEG Processing | MNE-Python |
| Fuzzy Logic | scikit-fuzzy |
| Machine Learning | scikit-learn |
| Numerical | NumPy · SciPy |
| Data | Pandas |
| Visualisation | Plotly · Matplotlib |
| Dashboard | Streamlit |
| Database | SQLite · SQLAlchemy |
| Configuration | PyYAML |
""")

    with tab4:
        st.markdown("### 🎓 Viva Defense — Key Questions & Model Answers")
        st.info("Expand each question to reveal a detailed model answer for your thesis defense.")

        viva_qs = [
            ("Why use Fuzzy Logic instead of a Black-Box Neural Network?",
             "Fuzzy Mamdani inference provides 100% white-box transparency through explicit IF-THEN linguistic rules (e.g., *'IF Alpha is LOW AND Theta/Alpha is HIGH THEN Workload is HIGH'*). In neuroscience and clinical contexts, explainability is essential for trust, accountability, and human-in-the-loop decision making. Neural networks cannot provide this level of interpretability without separate post-hoc techniques.",
             "Fuzzy Logic"),
            ("Why use Subject-Wise Cross-Validation (StratifiedGroupKFold)?",
             "Random train/test splits allow individual subject EEG 'brain signatures' to leak into the test set, causing artificially inflated accuracy (often >90%). Subject-wise CV guarantees that no window from any test subject appeared during training — this is the only way to honestly estimate real-world performance across new, unseen individuals.",
             "Cross-Validation"),
            ("What is the physiological basis of Theta/Alpha Ratio (TAR)?",
             "Cognitive workload and working memory demand reliably elicit increased frontal theta oscillations (4–8 Hz) driven by hippocampal-prefrontal circuits, while simultaneously suppressing parietal alpha power (8–13 Hz) through cortical desynchronization. The TAR therefore directly reflects the balance between active working memory engagement and neural inhibition — a robust neuromarker of cognitive load.",
             "EEG Signals"),
            ("How are EEG artifacts removed during preprocessing?",
             "A multi-stage pipeline: (1) 1–40 Hz FIR bandpass filter removes DC drift and high-frequency EMG noise; (2) 50 Hz notch filter removes mains interference; (3) average reference reduces electrode-specific noise; (4) amplitude threshold rejection (±150 µV) eliminates ocular blinks, eye movements, and large myogenic artifacts. ICA could be added for further artifact separation in a production system.",
             "EEG Signals"),
            ("How are fuzzy membership functions determined?",
             "Triangular membership functions (LOW, MEDIUM, HIGH) are data-driven: the centers are set at the empirical 25th, 50th, and 75th percentiles of each feature's distribution across all training subjects. Boundaries extend to the 5th and 95th percentiles. This ensures the functions reflect the actual range and skew of the real EEG data rather than arbitrary domain assumptions.",
             "Fuzzy Logic"),
            ("What defuzzification method is used and why?",
             "Centroid (Center of Gravity) defuzzification is used, computing the weighted average of the output membership function's area. This gives a continuous numerical output (0–100) that preserves gradation between classes. The output is then thresholded: <35 → LOW, 35–65 → MODERATE, >65 → HIGH. Centroid is preferred over max methods because it produces smoother, more representative crisp values.",
             "Fuzzy Logic"),
            ("What is Spectral Entropy and why is it a useful cognitive load feature?",
             "Spectral Entropy (Shannon entropy applied to the normalised PSD) measures the uniformity of power distribution across frequencies. During low-demand states, power is broadly distributed (high entropy). During high cognitive load, power concentrates in specific bands (theta increases, alpha decreases), reducing spectral entropy. This makes it sensitive to the qualitative changes in EEG spectral structure driven by cognitive demands.",
             "EEG Signals"),
            ("What are the major ethical limitations?",
             "This is an academic research prototype only: (1) Labels are task-difficulty proxies, NOT validated cognitive state measurements; (2) Only 18 subjects — no claim of clinical generalizability can be made; (3) No comparison to clinical gold standards (NASA-TLX, pupillometry, heart rate); (4) Not validated in any real-world operational setting; (5) Cannot be used for any medical, clinical, hiring, or assessment purpose.",
             "Ethics"),
            ("Why might the Fuzzy classifier underperform Random Forest on accuracy metrics?",
             "The Fuzzy classifier applies hard-coded triangular membership functions and 13 manually defined rules — its decision boundary is constrained by the rule structure. Random Forest automatically discovers complex non-linear feature interactions across all 17 features using ensemble tree voting. However, the Fuzzy classifier's advantage is complete interpretability; every prediction can be fully explained by which rules fired and at what strength.",
             "Models"),
            ("How would you extend this system to a real-time BCI application?",
             "Key extensions: (1) Real-time artifact removal with online ICA; (2) Adaptive membership function recalibration per subject (personalization); (3) Sliding window streaming with <100 ms latency; (4) Wireless EEG hardware integration (e.g., OpenBCI, Emotiv); (5) Temporal smoothing of predictions to reduce flicker; (6) Regulatory CE/FDA approval process for any clinical use; (7) Longitudinal validation study with matched controls.",
             "Research"),
        ]

        # Category filter
        all_cats = sorted(set(q[2] for q in viva_qs))
        cat_filter = st.multiselect("Filter by topic", all_cats, default=all_cats, key="viva_cat")
        search_viva = st.text_input("🔍 Search questions", key="viva_search", placeholder="Type to filter…")

        for i, (question, answer, category) in enumerate(viva_qs):
            if category not in cat_filter:
                continue
            if search_viva and search_viva.lower() not in question.lower() and search_viva.lower() not in answer.lower():
                continue
            cat_color = {"Fuzzy Logic": "#8b5cf6", "Cross-Validation": "#3b82f6", "EEG Signals": "#14b8a6", "Ethics": "#ef4444", "Models": "#f59e0b", "Research": "#22c55e"}.get(category, "#a0a0b8")
            with st.expander(f"Q{i+1}. {question}"):
                st.markdown(f"<span style='background:{cat_color}22;border:1px solid {cat_color}44;color:{cat_color};font-size:11px;font-weight:600;padding:3px 10px;border-radius:4px;display:inline-block;margin-bottom:12px;'>{category}</span>", unsafe_allow_html=True)
                st.markdown(answer)
