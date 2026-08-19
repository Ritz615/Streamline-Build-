# VIVA Guide — EEG Cognitive Load Classification

> Likely viva questions and model answers. Written for BE students.
> These answers reflect what was actually built — not generic theory.

---

## Q1. What is EEG?

**Answer:**
EEG stands for Electroencephalography. It measures the electrical activity
of the brain by placing small electrodes on the scalp. When neurons in the
brain fire together, they create tiny electrical signals that can be detected.
In our project, we used a 19-channel EEG system sampled at 250 Hz, which
records 250 measurements per second from each electrode.

---

## Q2. What is cognitive load?

**Answer:**
Cognitive load refers to the mental effort being used by the working memory.
When you solve a hard maths problem, your cognitive load is higher than when
you read a simple sentence. In our project, we use the term to describe the
level of mental demand during a task — classified as LOW, MODERATE, or HIGH.
We cannot directly measure cognitive load — we estimate it from EEG signals
as a research proxy.

---

## Q3. Why did you use the theta band?

**Answer:**
Theta waves (4–8 Hz) are associated with working memory and attention,
especially in the frontal regions of the brain. Research has shown that theta
power tends to increase when a person is engaged in demanding cognitive tasks.
In our system, theta power and the theta/beta ratio are among the strongest
features for distinguishing high from low workload in the n-back task.

---

## Q4. Why did you use the alpha band?

**Answer:**
Alpha waves (8–13 Hz) are associated with a relaxed, idle brain state. When
a person becomes more cognitively engaged, alpha power typically decreases —
a phenomenon called alpha event-related desynchronisation (ERD). We include
alpha power and alpha relative power as features because they provide
complementary information to theta.

---

## Q5. Why did you use the beta band?

**Answer:**
Beta waves (13–30 Hz) are associated with active thinking, alertness, and
motor activity. Some studies report beta increases during cognitive tasks.
We included beta power and the theta/beta ratio because the theta/beta ratio
is one of the most studied indices of cognitive workload in EEG literature.

---

## Q6. What is band power?

**Answer:**
Band power is the amount of signal energy in a specific frequency range.
We compute it using Welch's method — a standard algorithm that estimates
the power spectral density (how much power exists at each frequency) and
then integrates the area under the curve within the band of interest.
For example, theta band power = area under the PSD curve from 4 to 8 Hz.

---

## Q7. What is spectral entropy?

**Answer:**
Spectral entropy measures how uniformly the power is distributed across
frequencies. If one frequency dominates, entropy is LOW (organised).
If power is spread across all frequencies equally, entropy is HIGH (complex).
During demanding cognitive tasks, the EEG spectrum often becomes less
uniform, which can change spectral entropy. We calculate it using the
Shannon entropy formula applied to the normalized PSD.

---

## Q8. What is sample entropy?

**Answer:**
Sample entropy is a nonlinear measure of signal complexity and predictability.
A highly predictable signal has low sample entropy; an unpredictable, complex
signal has high entropy. It measures how often patterns of length m repeat
in the signal. We use m=2 and r=0.2×std(signal) as standard parameters.

---

## Q9. Why did you use fuzzy logic?

**Answer:**
We used fuzzy logic because:
1. It is **explainable** — every prediction has a clear rule-based reason
2. It handles **uncertainty** naturally — brain signals are not cleanly
   digital, they exist in degrees (theta is "somewhat HIGH")
3. It is **interpretable** for academic purposes — a student can read the
   rules and understand the decision
4. It does not require large amounts of labeled training data compared to
   deep learning
5. It produces a **fuzzy score** (0–100) that provides more information
   than a hard class label alone

---

## Q10. What is a membership function?

**Answer:**
A membership function defines how much a value "belongs" to a fuzzy set.
For example, for theta/beta ratio:
- A value of 1.2 might be: {LOW: 0.8, MEDIUM: 0.2, HIGH: 0.0}
- A value of 3.5 might be: {LOW: 0.0, MEDIUM: 0.1, HIGH: 0.9}

We use triangular membership functions, where:
- The center is the class mean (computed from training data)
- The width is based on the class standard deviation
This makes the membership functions data-driven rather than arbitrary.

---

## Q11. What is a fuzzy rule?

**Answer:**
A fuzzy rule is an IF-THEN statement that connects input conditions to an
output conclusion. Example:

```
IF theta_beta_ratio is HIGH AND theta_power is HIGH
THEN cognitive_load is HIGH
```

Multiple rules fire simultaneously with different strengths (0–1).
The final output is the aggregation of all fired rules, then defuzzified
to produce a crisp class label using the centroid method.

---

## Q12. Why did you use Random Forest?

**Answer:**
We used Random Forest as a **baseline model** for comparison. It is:
- Well-established and widely used
- Relatively easy to implement correctly
- Provides feature importances for analysis
- Works well on tabular feature data without deep learning complexity

The purpose of the Random Forest is to give us a reference point:
if the fuzzy classifier performs similarly to Random Forest, it proves
the fuzzy approach is viable for this problem.

---

## Q13. What is subject-wise splitting and why is it mandatory?

**Answer:**
Subject-wise splitting means that all EEG windows from a single subject
appear in EITHER the training set OR the test set — never both.

Without this safeguard, the model learns EEG patterns specific to one person,
and when tested on windows from the same person (even from a different time),
it will appear to perform well. This is called **data leakage** and produces
artificially high accuracy.

With subject-wise splitting (StratifiedGroupKFold), the model must generalise
to new people it has never seen — which is the real scientific test.

We enforced this using `StratifiedGroupKFold` from scikit-learn with subject
IDs as the grouping key.

---

## Q14. Why did you NOT use deep learning?

**Answer:**
Several reasons:
1. **Dataset size** — only 18 subjects × a few thousand windows is too small
   for CNNs or LSTMs to generalise well
2. **Explainability** — deep learning models are black boxes; our project
   requires interpretable predictions
3. **Scope** — this is a BE final year project, not a research publication
4. **Hardware** — deep learning works best on GPU; students use laptops
5. **Overengineering** — the problem does not require deep learning;
   good feature engineering + fuzzy logic achieves the research objective

---

## Q15. Why did you use a public dataset?

**Answer:**
Several important reasons:
1. **Ethics** — collecting EEG from humans requires ethics approval, informed
   consent, and proper supervision
2. **Reproducibility** — any researcher can download and verify our results
3. **Availability** — the dataset is free, CC0 licensed, and accessible globally
4. **Academic validity** — using a published dataset with known characteristics
5. **Safety** — we do not need to set up EEG hardware or risk data privacy issues

We explicitly chose NOT to collect EEG from students or patients.

---

## Q16. What are the main limitations of your system?

**Answer:**
1. Only 18 subjects — small sample, results may not generalise
2. No rest/baseline condition — 1-back used as LOW proxy, which may not
   be a perfect representation of low cognitive load
3. EEG is very noisy and varies significantly between individuals
4. The three classes (LOW/MODERATE/HIGH) are derived from task difficulty,
   not validated psychological measures
5. The fuzzy rules are data-driven heuristics — not medically validated
6. Performance may be affected by class imbalance
7. The model is NOT validated for clinical use

---

## Q17. Why LOW, MODERATE, HIGH? Why not 4 or 5 classes?

**Answer:**
We merged nback_3 and nback_4 into a single HIGH class for two reasons:
1. **Class balance** — with 4 difficulty levels and only 18 subjects,
   having 4 distinct classes would give very few samples per class
2. **Practical utility** — the practical difference between nback_3 and
   nback_4 cognitive load is smaller than between nback_1 and nback_3
3. **Literature** — most EEG workload research uses 2–3 classes

The config.yaml makes this mapping configurable.

---

## Q18. How is the dataset labeled?

**Answer:**
The dataset uses trial_type markers in events.tsv files:
- nback_1: n-back task at difficulty level 1
- nback_2: n-back task at difficulty level 2
- nback_3: n-back task at difficulty level 3
- nback_4: n-back task at difficulty level 4

There is no rest or baseline condition. Tutorial trials (istutorial=True)
are excluded from analysis. We assign workload labels based on the n-back
level active during each 4-second window.

---

## Q19. What happens during preprocessing?

**Answer:**
1. **Pick EEG channels** — remove ECG and other non-EEG channels
2. **Detect bad channels** — mark flat or excessively noisy channels
3. **Band-pass filter** (1–40 Hz) — remove low-frequency drifts and high-frequency muscle noise
4. **Notch filter** (50 Hz) — remove power line interference
5. **Average reference** — subtract mean of all electrodes from each electrode
6. **Artifact rejection** — mark time segments with amplitude > 150 µV as bad
7. **Save** — store as .fif file for reuse

---

## Q20. What is the main contribution of this project?

**Answer:**
The main contributions are:
1. **End-to-end reproducible pipeline** — from raw public EEG to classified
   cognitive load, using only open-source tools
2. **Fuzzy explainability** — every prediction explains which rules fired
   and what feature memberships activated them
3. **Subject-wise evaluation** — correct scientific validation preventing
   data leakage
4. **Transparent limitations** — honest documentation of what the system
   can and cannot do
5. **Student accessibility** — the system runs on a laptop with free data,
   demonstrating EEG analysis is accessible to BE students

---

## Quick Reference — Key Numbers

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sampling rate | 250 Hz | Dataset specification |
| EEG channels | 19 | 10-20 montage |
| Band-pass | 1–40 Hz | Preserve cognitive bands |
| Window size | 4 s | Standard in EEG research |
| Overlap | 2 s | Increase training samples |
| CV folds | 5 | Enough splits for 18 subjects |
| RF trees | 200 | Balance accuracy/speed |
| Random seed | 42 | Reproducibility |
| Fuzzy features | 5 | Top features from selection |
| Fuzzy rules | ~15 | Simple but sufficient |

---

## Common Mistakes to Avoid Saying

| ❌ Wrong | ✓ Correct |
|---------|----------|
| "EEG proves cognitive load" | "EEG provides signals associated with task difficulty" |
| "The fuzzy rules are medically validated" | "The rules are research heuristics tuned on this dataset" |
| "Our system diagnoses high cognitive load" | "Our system classifies windows as LOW/MODERATE/HIGH based on EEG features" |
| "Accuracy = X% proves it works" | "Accuracy = X% on this dataset with these conditions" |
| "This works for everyone" | "Results are based on 18 subjects and may not generalise" |
