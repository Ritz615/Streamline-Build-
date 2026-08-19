# Project Guide — EEG Cognitive Load Classification

> Simple explanation of what the system does, how it works, and why
> each part was designed this way. Written for BE students.

---

## The Big Picture

```
EEG Signal (raw brain waves)
         ↓
      CLEAN IT
   (filter noise)
         ↓
  SPLIT INTO WINDOWS
  (4-second chunks)
         ↓
  EXTRACT FEATURES
  (numbers that describe
   each brain wave chunk)
         ↓
   ┌──────────────┐
   │              │
   ▼              ▼
 FUZZY         RANDOM
 LOGIC          FOREST
   │              │
   ▼              ▼
 LOW /        LOW /
 MODERATE /   MODERATE /
 HIGH          HIGH
   │              │
   └──────┬───────┘
          ▼
       COMPARE
          ↓
       EXPLAIN
       (why did the
       model say HIGH?)
```

---

## Step 1: The EEG Data

**What is EEG?**
EEG (Electroencephalography) measures electrical activity in the brain
using small sensors placed on the scalp. When neurons fire, they produce
tiny electrical signals — EEG captures thousands of these per second.

**What dataset is used?**
We use `ds007169` from OpenNeuro.org — a public, free, anonymized dataset.
18 participants performed n-back tasks of 4 difficulty levels while their
EEG was recorded at 250 samples per second using 19 electrodes.

**What is the n-back task?**
Participants see a sequence of items and must say whether the current item
matches the one shown N steps back:
- 1-back = remember the last item (easy)
- 4-back = remember 4 items back (very hard)

**Label mapping:**
```
nback_1 → LOW workload
nback_2 → MODERATE workload
nback_3 → HIGH workload
nback_4 → HIGH workload
```

> ⚠️ This is an operational research categorization based on task difficulty.
> It is NOT a medically validated measure of cognitive state.

---

## Step 2: Preprocessing

Raw EEG is very noisy. We clean it in several steps:

### Band-pass filtering (1–40 Hz)
The brain signals we care about are in the 1–40 Hz range.
Below 1 Hz = slow drifts (remove them).
Above 40 Hz = muscle noise (remove it).

### Notch filter (50 Hz)
Electrical power lines emit 50 Hz interference. We remove it.

### Average reference
Each electrode measures voltage relative to a reference. Using the average
of all electrodes as the reference reduces bias.

### Artifact rejection (±150 µV threshold)
Very large voltages are usually caused by eye blinks or movement, not brain
signals. We mark those time periods as bad and skip them.

### Why NOT ICA?
ICA (Independent Component Analysis) is a more advanced artifact removal
technique. We did not use it because:
- It requires manual inspection of components
- It adds complexity beyond BE project scope
- Amplitude thresholding is sufficient for this prototype
- ICA can be added in future work

---

## Step 3: Segmentation (windowing)

We cannot process the entire EEG recording as one unit.
Instead, we split it into small time windows:

```
RECORDING: |----nback_1----|----nback_2----|----nback_3----|
                ↓ slice
WINDOWS:   [w1][w2][w3][w4]  [w5][w6][w7]  [w8][w9]...
```

- **Window size:** 4 seconds (1000 samples @ 250 Hz)
- **Overlap:** 2 seconds (windows overlap by 50%)
- **Each window gets a label** (LOW/MODERATE/HIGH) based on what task was running

Each window becomes one training sample.

---

## Step 4: Feature Extraction

For each 4-second window, we compute numbers (features) that describe
the brain activity in that window:

### Frequency band features
The brain works in different frequency "bands":

| Band | Frequency | Meaning |
|------|-----------|---------|
| Theta | 4–8 Hz | Working memory, attention, frontal activity |
| Alpha | 8–13 Hz | Relaxation, idle brain, decreases with engagement |
| Beta | 13–30 Hz | Active thinking, alertness, increases with load |

We compute:
- **Absolute power** — how much signal is in each band
- **Relative power** — band power as fraction of total power
- **Ratios** — e.g., theta/beta (increases with cognitive load)

### Statistical features
Simple time-domain statistics:
- mean, standard deviation, variance, RMS
- skewness (asymmetry of signal), kurtosis (spikiness)

### Entropy features
- **Spectral entropy** — how spread out the frequency content is
- **Sample entropy** — how complex/unpredictable the signal is

### Result
Each window → approximately 22 numbers → one row in `data/features/features.csv`

---

## Step 5: Feature Selection

Not all features are equally useful. We rank them using:

1. **Kruskal-Wallis test** — checks if feature values differ significantly between classes
2. **Mutual Information** — measures how much a feature tells us about the class

The top 5 features are used as inputs to the fuzzy classifier.

---

## Step 6: Fuzzy Logic Classifier

### What is fuzzy logic?
Normal computers think in 0s and 1s (true/false).
Fuzzy logic thinks in degrees: "theta is 0.8 HIGH, 0.3 MEDIUM, 0.1 LOW"

### How it works here:

1. **Fuzzification** — each feature value is converted to membership degrees
   ```
   theta_beta_ratio = 2.5 → {LOW: 0.1, MEDIUM: 0.3, HIGH: 0.8}
   ```

2. **Rules** — IF-THEN rules fire based on memberships
   ```
   IF theta_beta_ratio is HIGH AND theta_power is HIGH
   THEN cognitive_load is HIGH
   ```

3. **Aggregation** — all firing rules are combined

4. **Defuzzification** — the fuzzy output is converted to a crisp class
   ```
   Fuzzy score: 73 → HIGH
   ```

### Why fuzzy logic?
- **Explainable** — you can see exactly which rules fired
- **Matches human reasoning** — "somewhat high, not exactly high"
- **No black box** — every prediction has a clear reason
- **Literature basis** — brain signals naturally fit fuzzy descriptions

### Membership functions
For each input feature, we define three membership functions:
```
LOW    ╱╲
      /  \
MEDIUM    ╱╲
         /  \
HIGH         ╱╲
            /  \
        ────────────→ feature value
```

Ranges are computed from training data percentiles (data-driven, not arbitrary).

---

## Step 7: Random Forest Baseline

Random Forest is a standard machine learning model that:
1. Creates many decision trees (200 trees)
2. Each tree votes for a class
3. Majority vote wins

We use it to compare with the fuzzy classifier.

### Why is it called a "baseline"?
A baseline is a reference point. It shows us what a standard ML approach
achieves on the same problem. If the fuzzy classifier beats or matches the
Random Forest, it proves the fuzzy approach is viable.

### Subject-wise cross-validation
We use StratifiedGroupKFold to ensure:
- Subjects in the test set are NEVER seen during training
- This tests whether the model works on NEW people

> This is scientifically critical. Random window splitting would leak information
> and give unrealistically high accuracy.

---

## Step 8: Evaluation

We measure both models using:
- **Accuracy** — fraction of correct predictions
- **F1-score (macro)** — balanced measure across all 3 classes
- **Balanced accuracy** — adjusts for class imbalance
- **Confusion matrix** — shows which classes get confused

---

## Step 9: Explainability

For every prediction, the fuzzy system explains:
```
Prediction: HIGH

Activated Rules:
  Rule 3: theta_beta_ratio is HIGH → HIGH  (strength=0.82)
  Rule 7: theta_power is HIGH      → HIGH  (strength=0.71)

Feature Memberships:
  theta_beta_ratio: LOW=0.05  MEDIUM=0.15  HIGH=0.80
  theta_power:      LOW=0.10  MEDIUM=0.25  HIGH=0.72

Summary:
  The model classified this as HIGH cognitive load because
  the theta/beta ratio and theta power had HIGH membership,
  which strongly activated the HIGH-load rules.
```

This is the main advantage of fuzzy logic over Random Forest — every decision
has a human-understandable reason.

---

## Step 10: Dashboard

The Streamlit dashboard lets you:
- View EEG waveforms and spectra
- Explore feature distributions
- Make predictions on individual windows
- See activated fuzzy rules
- Compare fuzzy vs Random Forest performance
- Read the full explanation of any prediction

Run with:
```bash
streamlit run app/streamlit_app.py
```

---

## Database (SQLite)

Every experiment is logged in `data/eeg_project.db`:
- Which dataset was used
- Which preprocessing settings were applied
- Which features were extracted
- Which model achieved what accuracy
- Which fuzzy rules were activated

This makes the research **reproducible** — you can always trace back which
settings produced which result.

---

## Important Limitations

1. **Only 18 subjects** — small sample, limited generalizability
2. **No baseline** — 1-back task is used as LOW, not true rest
3. **Individual variation** — EEG differs greatly between people
4. **Not validated** — this is a research prototype, not a clinical tool
5. **Task difficulty ≠ cognitive load** — n-back level is a proxy
6. **Single dataset** — results may differ on other datasets

---

## File Descriptions

| File | Purpose |
|------|---------|
| `config.yaml` | All project settings — edit this to change experiments |
| `main.py` | CLI entry point — run stages from here |
| `src/config.py` | Reads config.yaml |
| `src/database.py` | SQLite database setup |
| `src/data_loader.py` | Loads EEG files using MNE |
| `src/preprocessing.py` | Filters, references, artifact rejection |
| `src/segmentation.py` | Cuts EEG into labeled windows |
| `src/feature_extraction.py` | Computes all features |
| `src/feature_selection.py` | Ranks features by importance |
| `src/fuzzy_classifier.py` | Fuzzy inference system |
| `src/random_forest.py` | Random Forest model |
| `src/evaluation.py` | Metrics + confusion matrices |
| `src/explainability.py` | Human-readable explanations |
| `src/visualization.py` | EDA plots + figures |
| `src/pipeline.py` | Runs all stages in order |
| `app/streamlit_app.py` | Web dashboard |
| `tests/` | Automated tests |
