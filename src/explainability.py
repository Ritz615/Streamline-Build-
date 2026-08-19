"""
src/explainability.py
=====================
Human-readable explanations for model predictions.

For Fuzzy Classifier:
  - Shows activated fuzzy rules with their firing strengths
  - Shows feature membership values (LOW/MEDIUM/HIGH)
  - Generates a natural-language summary

For Random Forest:
  - Shows feature importances
  - Shows class probabilities
  - Describes top contributing features

IMPORTANT LANGUAGE GUIDELINE:
  All explanations use the phrase "important for this prediction"
  rather than "causes cognitive load" or "proves high load".
  Correlation ≠ causality.

Usage:
    from src.explainability import explain_fuzzy, explain_rf
    text = explain_fuzzy(prediction_result, feature_values, feature_names)
    text = explain_rf(model, x, feature_names, class_names)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_NAMES = ["LOW", "MODERATE", "HIGH"]


# ------------------------------------------------------------------ #
# Fuzzy Explainability
# ------------------------------------------------------------------ #

def explain_fuzzy(
    prediction_result: Dict,
    feature_values: np.ndarray,
    feature_names: List[str],
) -> str:
    """
    Generate a human-readable explanation for a fuzzy prediction.

    Parameters:
        prediction_result : dict from FuzzyClassifier.predict_single()
        feature_values    : 1D array of input feature values
        feature_names     : feature names

    Returns:
        Multi-line explanation string
    """
    predicted = prediction_result.get("predicted_class", "UNKNOWN")
    fuzzy_score = prediction_result.get("fuzzy_score", 0)
    confidence = prediction_result.get("confidence", 0)
    memberships = prediction_result.get("memberships", {})
    activated_rules = prediction_result.get("activated_rules", [])

    lines = [
        "═" * 60,
        "  FUZZY CLASSIFIER — PREDICTION EXPLANATION",
        "═" * 60,
        "",
        f"  Predicted Cognitive Load: {predicted}",
        f"  Fuzzy Score:              {fuzzy_score:.1f} / 100",
        f"  Prediction Confidence:    {confidence:.1%}",
        "",
    ]

    # Score interpretation
    score_desc = _interpret_fuzzy_score(fuzzy_score)
    lines += [f"  Score Interpretation: {score_desc}", ""]

    # Feature membership summary
    lines += [
        "  ─── Feature Evidence ───────────────────────────",
        "",
    ]
    for i, feat in enumerate(feature_names):
        val = feature_values[i] if i < len(feature_values) else 0.0
        mem = memberships.get(feat, {})
        dominant = _dominant_membership(mem)
        lines.append(
            f"  {feat:<25} = {val:>10.4f}   [{dominant}]"
        )
        lines.append(
            f"    Membership:  LOW={mem.get('LOW',0):.2f}  "
            f"MEDIUM={mem.get('MEDIUM',0):.2f}  "
            f"HIGH={mem.get('HIGH',0):.2f}"
        )

    lines += [""]

    # Activated rules
    if activated_rules:
        lines += [
            "  ─── Activated Fuzzy Rules ──────────────────────",
            "",
        ]
        for r in activated_rules[:5]:  # Show top 5
            strength_bar = _bar(r["strength"])
            lines.append(
                f"  Rule {r['rule_number']:>2}: {strength_bar} (strength={r['strength']:.2f})"
            )
            lines.append(f"          {r['rule_text']}")
            lines.append("")
    else:
        lines += ["  No rules fired strongly for this input.", ""]

    # Natural language summary
    summary = _generate_fuzzy_summary(predicted, activated_rules, memberships)
    lines += [
        "  ─── Summary ────────────────────────────────────",
        "",
        summary,
        "",
        "  ⚠ RESEARCH PROTOTYPE: Not a medical assessment.",
        "═" * 60,
    ]

    return "\n".join(lines)


def _interpret_fuzzy_score(score: float) -> str:
    if score <= 33:
        return "Low cognitive demand (score in LOW range)"
    elif score <= 67:
        return "Moderate cognitive demand (score in MODERATE range)"
    else:
        return "High cognitive demand (score in HIGH range)"


def _dominant_membership(mem: Dict) -> str:
    if not mem:
        return "N/A"
    dominant = max(mem, key=lambda k: mem[k])
    return dominant


def _bar(strength: float, width: int = 10) -> str:
    filled = int(round(strength * width))
    return "█" * filled + "░" * (width - filled)


def _generate_fuzzy_summary(
    predicted: str,
    activated_rules: List[Dict],
    memberships: Dict,
) -> str:
    """
    Generate a plain-language summary of the fuzzy prediction.
    Uses careful language: 'associated with', not 'proves' or 'causes'.
    """
    if not activated_rules:
        return (
            f"  The system classified this EEG segment as {predicted} cognitive load.\n"
            "  No rules fired strongly — the input may be near a decision boundary."
        )

    top_rule = activated_rules[0]
    consequent = top_rule.get("consequent", predicted)

    # Identify strongest feature membership
    strongest_feature = ""
    strongest_level = ""
    max_val = 0.0
    for feat, levels in memberships.items():
        for level, val in levels.items():
            if val > max_val:
                max_val = val
                strongest_feature = feat
                strongest_level = level

    return (
        f"  The model classified this EEG segment as {predicted} cognitive load\n"
        f"  based on the fuzzy rules. The strongest signal came from\n"
        f"  Rule {top_rule['rule_number']} (strength={top_rule['strength']:.2f}),\n"
        f"  which suggests {consequent} load.\n\n"
        f"  The feature '{strongest_feature}' had a strong\n"
        f"  '{strongest_level}' membership (degree={max_val:.2f}),\n"
        f"  which is associated with {predicted} cognitive load in this model.\n\n"
        "  Note: EEG features are statistical signals that are\n"
        "  associated with task difficulty — not direct measures of\n"
        "  a person's cognitive state. Individual variation is high."
    )


# ------------------------------------------------------------------ #
# Random Forest Explainability
# ------------------------------------------------------------------ #

def explain_rf(
    rf_model,
    x: np.ndarray,
    feature_names: List[str],
    class_names: List[str] = CLASS_NAMES,
) -> str:
    """
    Generate a human-readable explanation for a Random Forest prediction.

    Parameters:
        rf_model      : trained RandomForestModel instance
        x             : 1D feature array
        feature_names : feature names
        class_names   : class label strings

    Returns:
        Multi-line explanation string
    """
    if not rf_model.is_trained:
        return "Random Forest model not trained. Run python main.py --train first."

    proba = rf_model.predict_proba(x.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    predicted = class_names[pred_idx]

    # Feature importances
    fi_df = rf_model.get_feature_importance()
    top_features = fi_df.head(5)

    lines = [
        "═" * 60,
        "  RANDOM FOREST — PREDICTION EXPLANATION",
        "═" * 60,
        "",
        f"  Predicted Cognitive Load: {predicted}",
        "",
        "  Class Probabilities:",
    ]
    for i, cls in enumerate(class_names):
        bar = _bar(proba[i])
        lines.append(f"    {cls:<12} {bar} {proba[i]:.1%}")

    lines += [
        "",
        "  ─── Feature Importance (model-wide) ────────────",
        "  Note: importance = how much a feature is used",
        "  across all trees. Does NOT imply causality.",
        "",
    ]
    for _, row in top_features.iterrows():
        bar = _bar(row["importance"])
        lines.append(
            f"  {row['feature']:<25} {bar} {row['importance']:.4f}"
        )

    lines += [
        "",
        "  ─── Summary ────────────────────────────────────",
        "",
        f"  The Random Forest predicted {predicted} cognitive load.",
        f"  Confidence: {proba[pred_idx]:.1%}",
        "",
        f"  The top feature for this model is '{fi_df.iloc[0]['feature']}'",
        "  which contributes most to predictions across the dataset.",
        "",
        "  ⚠ RESEARCH PROTOTYPE: Not a medical assessment.",
        "═" * 60,
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Shared utilities
# ------------------------------------------------------------------ #

def feature_value_context(
    feature_name: str,
    value: float,
    min_val: float,
    max_val: float,
) -> str:
    """
    Return a human-readable context for a feature value.
    E.g. "above average", "near the top of the observed range".
    """
    if max_val == min_val:
        return "constant across dataset"
    position = (value - min_val) / (max_val - min_val)
    if position < 0.2:
        return "very low (near minimum observed)"
    elif position < 0.4:
        return "below average"
    elif position < 0.6:
        return "average"
    elif position < 0.8:
        return "above average"
    else:
        return "very high (near maximum observed)"


def format_prediction_for_display(prediction_result: Dict) -> Dict:
    """
    Format a prediction result for Streamlit display.
    Returns a simplified dict suitable for the dashboard.
    """
    return {
        "predicted_class": prediction_result.get("predicted_class", "UNKNOWN"),
        "fuzzy_score": prediction_result.get("fuzzy_score", 0.0),
        "confidence_pct": f"{prediction_result.get('confidence', 0) * 100:.1f}%",
        "n_rules_activated": len(prediction_result.get("activated_rules", [])),
        "top_rule": (
            prediction_result.get("activated_rules", [{}])[0].get("rule_text", "N/A")
            if prediction_result.get("activated_rules")
            else "N/A"
        ),
    }
