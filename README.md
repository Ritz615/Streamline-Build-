# EEG Cognitive Load Classification — Research Prototype

> **⚠️ DISCLAIMER:** This is an **academic research prototype** built for a
> Bachelor of Engineering final year project. It is **NOT** a medical device,
> clinical diagnostic system, or psychological assessment tool. It uses public,
> anonymized EEG data licensed under CC0. Results should not be used for any
> medical or clinical purpose.

---

## What This Project Does

This system processes public EEG (electroencephalography) data recorded while
participants performed cognitive tasks of varying difficulty, then classifies
the cognitive workload into three research categories: **LOW**, **MODERATE**,
and **HIGH**.

The system uses:
- **Fuzzy Logic** (interpretable, rule-based) as the main classifier
- **Random Forest** as a machine-learning baseline
- Transparent explanations of every prediction

## Why It Exists

This project demonstrates that cognitive workload can be studied using
publicly available EEG data, open-source signal processing tools, and
interpretable machine learning — without requiring expensive hardware,
proprietary software, or clinical data collection.

---

## Architecture

```
PUBLIC EEG DATASET (OpenNeuro ds007169)
            ↓
    MNE-Python preprocessing
    (bandpass 1–40 Hz, notch 50 Hz, average reference)
            ↓
    4-second sliding window segmentation (2s overlap)
            ↓
    Feature extraction:
      • Theta, alpha, beta band power
      • Relative power + ratios
      • Statistical: mean, std, variance, RMS, skewness, kurtosis
      • Entropy: spectral entropy, sample entropy
            ↓
    Feature selection (Kruskal-Wallis + Mutual Information)
            ↓
      ┌──────────────────────┐
      │                      │
      ▼                      ▼
  FUZZY CLASSIFIER    RANDOM FOREST
  (scikit-fuzzy)      (scikit-learn)
      │                      │
      ▼                      ▼
  LOW/MODERATE/HIGH   LOW/MODERATE/HIGH
      │                      │
      └──────────┬───────────┘
                 ▼
  Subject-wise cross-validation evaluation
                 ▼
  Explainability output (activated rules + memberships)
                 ▼
  Streamlit Dashboard + SQLite metadata DB
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Cognitive Workload 5-level n-back |
| **ID** | ds007169 |
| **Source** | https://openneuro.org/datasets/ds007169 |
| **DOI** | doi:10.18112/openneuro.ds007169.v1.0.0 |
| **License** | CC0 (Public Domain) |
| **Subjects** | 18 |
| **Channels** | 19 (10-20 EEG montage) |
| **Sampling Rate** | 250 Hz |
| **Format** | BrainVision (.vhdr/.eeg/.vmrk) |

### Label Mapping

The dataset has n-back tasks at 4 difficulty levels. There is **no rest/baseline condition**.
We apply this operational research mapping:

| Task | Workload Class |
|------|---------------|
| nback_1 | LOW |
| nback_2 | MODERATE |
| nback_3 | HIGH |
| nback_4 | HIGH |

This mapping is **configurable** in `config.yaml`. It represents a research
categorization based on task difficulty, NOT a medically validated scale.

---

## Installation

### Prerequisites
- Python 3.11 or later
- pip
- Git

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd eeg-cognitive-load

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Complete Pipeline (recommended)

```bash
python main.py --all
```

This runs all stages automatically:
1. Downloads the dataset
2. Preprocesses EEG
3. Extracts features
4. Trains models
5. Evaluates models
6. Generates report

### Individual Stages

```bash
python main.py --download     # Download dataset
python main.py --preprocess   # Preprocess EEG
python main.py --features     # Extract features
python main.py --train        # Train models
python main.py --evaluate     # Evaluate and compare
python main.py --report       # Generate report
```

### Check Status

```bash
python main.py --status
```

### Run Tests

```bash
pytest tests/ -v
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Dataset Download

The download script tries in order:
1. **openneuro-py** (recommended): `pip install openneuro-py`
2. **AWS S3** (boto3)
3. **Manual download** — instructions printed if automated methods fail

### Manual Download

```bash
# Method 1: openneuro-py CLI
openneuro-py download --dataset=ds007169 --target=data/raw/ds007169

# Method 2: DataLad
datalad install https://github.com/OpenNeuroDatasets/ds007169
datalad get -r .
```

Expected after download:
```
data/raw/ds007169/
├── dataset_description.json
├── participants.tsv
├── sub-01/eeg/sub-01_task-nback_eeg.vhdr
├── sub-01/eeg/sub-01_task-nback_events.tsv
└── ...
```

---

## Quick Start — Run Merged Application

### 1. Launch the Complete Unified Application (Landing Page + Research Dashboard)
```bash
# Starts both the Landing Page (port 8080) and Streamlit Dashboard (port 8501)
python main.py --app
```
- **Landing Page**: [http://localhost:8080](http://localhost:8080)
- **Research Dashboard**: [http://localhost:8501](http://localhost:8501)

### 2. Launch Individual Components
```bash
# Launch Landing Page only (port 8080)
python main.py --ui

# Launch Streamlit Research Dashboard only (port 8501)
python main.py --dashboard

# Run complete end-to-end ML & EEG pipeline
python main.py --all

# Run automated tests
pytest tests/ -v
```

---

## Project Structure

```
eeg-cognitive-load/
├── index.html                   ← Root Landing Page (Project 1)
├── styles.css                   ← Exact Landing Page styling
├── main.js                      ← Count-up & mobile menu logic
├── assets/logo.webp             ← Circular brand logo
├── fonts/                       ← Geist Pixel Circle font fallback
├── web/                         ← Standalone web distribution folder
│   ├── index.html
│   ├── styles.css
│   ├── main.js
│   ├── assets/logo.webp
│   └── fonts/GeistPixel-Circle.woff2
├── app/
│   └── streamlit_app.py         ← 7-page research dashboard (Project 2)
├── src/
│   ├── config.py                ← Configuration loader
│   ├── database.py              ← SQLite ORM & run history
│   ├── data_loader.py           ← BIDS / BrainVision dataset loader
│   ├── dataset_manager.py       ← Dataset verification
│   ├── preprocessing.py         ← 1–40Hz bandpass, 50Hz notch, average ref
│   ├── segmentation.py          ← 4s sliding windows with 2s overlap
│   ├── feature_extraction.py    ← PSD band powers, ratios, entropies, stats
│   ├── feature_selection.py     ← Kruskal-Wallis + Mutual Information
│   ├── fuzzy_classifier.py      ← 13-rule Mamdani Fuzzy System (scikit-fuzzy)
│   ├── random_forest.py         ← Random Forest baseline model
│   ├── evaluation.py            ← 5-fold StratifiedGroupKFold cross-validation
│   ├── explainability.py        ← Membership values & fired rule explanations
│   ├── visualization.py         ← Signal waveforms, PSD, and EDA plots
│   └── pipeline.py              ← End-to-end pipeline orchestrator
├── scripts/                     ← Standalone stage execution scripts
├── tests/                       ← Complete pytest test suite (43 passed)
├── data/
│   ├── raw/                     ← OpenNeuro ds007169 (18 subjects)
│   ├── processed/               ← Preprocessed .fif signal files
│   ├── features/features.csv    ← Extracted feature dataset (1,046 windows)
│   └── eeg_project.db           ← SQLite experiment metadata database
├── models/
│   ├── fuzzy/                   ← Trained Fuzzy Classifier (.joblib)
│   └── random_forest/           ← Trained Random Forest model (.joblib)
├── results/
│   ├── figures/                 ← Confusion matrices, EDA plots, comparisons
│   ├── metrics/                 ← model_comparison.csv, feature_importance.csv
│   └── reports/                 ← final_experiment_report.md
├── config.yaml                  ← Global system configuration
├── requirements.txt             ← Python dependencies
├── main.py                      ← Unified CLI entry point & launcher
├── README.md                    ← Product documentation
├── PROJECT_GUIDE.md             ← Academic architectural guide
└── VIVA_GUIDE.md                ← 20 Viva defense questions & model answers
```

---

## Results

After running `python main.py --all`, find results in:

| Location | Contents |
|----------|----------|
| `results/metrics/model_comparison.csv` | Accuracy, F1, precision, recall |
| `results/figures/` | Confusion matrices, EDA plots, comparison charts |
| `results/reports/final_experiment_report.md` | Full experiment report |
| `data/eeg_project.db` | All experiment metadata |

---

## Configuration

All settings are in `config.yaml`. Important ones:

```yaml
preprocessing:
  low_frequency: 1.0
  high_frequency: 40.0
  notch_frequency: 50.0

segmentation:
  window_seconds: 4.0
  overlap_seconds: 2.0

label_mapping:
  nback_1: "LOW"
  nback_2: "MODERATE"
  nback_3: "HIGH"
  nback_4: "HIGH"

models:
  random_forest:
    n_estimators: 200
    random_state: 42
```

---

## Limitations

- Only 18 subjects — limited statistical power
- No rest/baseline condition in the dataset
- 3-class labels are derived from task difficulty, not validated measures
- EEG is noisy and highly individual — may not generalise
- Not validated for clinical use
- This is a research prototype for academic purposes only

---

## Citation

```
Booth, L., & Barras, M. (2024).
Cognitive Workload 5-level n-back [Data set].
OpenNeuro. doi:10.18112/openneuro.ds007169.v1.0.0
License: CC0 (Public Domain)
```

---

## License

This project code is released under MIT License.
The EEG dataset is CC0 (public domain).
