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

## Project Structure

```
eeg-cognitive-load/
├── app/streamlit_app.py        ← Web dashboard
├── src/
│   ├── config.py               ← Config loader
│   ├── database.py             ← SQLite ORM
│   ├── data_loader.py          ← Dataset loader
│   ├── dataset_manager.py      ← Dataset registration
│   ├── preprocessing.py        ← EEG preprocessing
│   ├── segmentation.py         ← Window segmentation
│   ├── feature_extraction.py   ← Feature computation
│   ├── feature_selection.py    ← Feature ranking
│   ├── fuzzy_classifier.py     ← Fuzzy inference system
│   ├── random_forest.py        ← RF baseline model
│   ├── evaluation.py           ← Metrics + confusion matrices
│   ├── explainability.py       ← Human-readable explanations
│   ├── visualization.py        ← EDA + plots
│   └── pipeline.py             ← Pipeline orchestrator
├── scripts/                    ← Standalone stage scripts
├── tests/                      ← pytest test suite
├── data/
│   ├── raw/                    ← Downloaded EEG (not in git)
│   ├── processed/              ← Preprocessed .fif files
│   ├── features/features.csv   ← Extracted features
│   └── eeg_project.db          ← SQLite metadata DB
├── models/                     ← Trained models
├── results/                    ← Figures, metrics, reports
├── config.yaml                 ← All configuration
├── requirements.txt
├── main.py                     ← CLI entry point
├── README.md
├── PROJECT_GUIDE.md
└── VIVA_GUIDE.md
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
