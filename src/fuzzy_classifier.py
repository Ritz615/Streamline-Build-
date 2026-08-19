"""
src/fuzzy_classifier.py
=======================
Fuzzy inference system for EEG cognitive load classification.

Architecture:
  Inputs  : top 3–5 features (selected from feature analysis)
            Default: theta_beta_ratio, theta_power, beta_power,
                     spectral_entropy, theta_relative
  Output  : cognitive_load → LOW / MODERATE / HIGH

Membership functions: triangular / trapezoidal (data-driven ranges)
Inference:   Mamdani-style with product t-norm
Defuzz:      centroid method
Rules:       ~15 hand-crafted rules (refined with data)

IMPORTANT DISCLAIMER:
  These fuzzy rules encode engineering heuristics derived from the
  EEG literature and are tuned on this specific dataset.
  They do NOT constitute medically validated diagnostic criteria.
  The model is a RESEARCH PROTOTYPE.

Usage:
    from src.fuzzy_classifier import FuzzyClassifier
    clf = FuzzyClassifier()
    clf.train(X_train, y_train, feature_names)
    prediction = clf.predict_single(x)
    explanation = clf.explain(x)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# scikit-fuzzy imports
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    warnings.warn(
        "scikit-fuzzy not installed. FuzzyClassifier will not work. "
        "Install with: pip install scikit-fuzzy",
        ImportWarning,
    )

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

CLASS_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
INT_TO_CLASS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

# Default feature inputs for fuzzy system
# (overridden by feature selection results at training time)
DEFAULT_FUZZY_FEATURES = [
    "theta_beta_ratio",
    "theta_power",
    "beta_power",
    "spectral_entropy",
    "theta_relative",
]


# ------------------------------------------------------------------ #
# Fuzzy Classifier
# ------------------------------------------------------------------ #

class FuzzyClassifier:
    """
    Mamdani-style fuzzy inference system for cognitive load classification.

    Attributes:
        feature_names  : list of feature names used as inputs
        feature_ranges : dict of {feature: (min, max)} from training data
        membership_params : dict of per-feature LOW/MEDIUM/HIGH params
        rules          : list of rule dicts for database storage
    """

    def __init__(self, config=None):
        if not SKFUZZY_AVAILABLE:
            raise ImportError("scikit-fuzzy is required. Install: pip install scikit-fuzzy")
        self.config = config or get_config()
        self.feature_names: List[str] = []
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.membership_params: Dict[str, Dict] = {}
        self._control_system = None
        self._simulator = None
        self.is_trained = False
        self.rules: List[Dict] = []

        # Output universe: 0 (LOW) → 100 (HIGH)
        self._output_universe = np.linspace(0, 100, 200)

    # ---------------------------------------------------------------- #
    # Training / Fitting
    # ---------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """
        Configure the fuzzy system based on training data statistics.

        The membership function ranges are derived from the actual
        data distribution (percentiles) rather than arbitrary fixed values.

        Parameters:
            X            : feature matrix (n_samples × n_features)
            y            : encoded labels (0=LOW, 1=MOD, 2=HIGH)
            feature_names: names corresponding to X columns
        """
        logger.info("Training FuzzyClassifier with %d samples, %d features", len(X), len(feature_names))
        self.feature_names = feature_names

        # ---- Compute data-driven membership ranges ----
        self._compute_membership_params(X, y, feature_names)

        # ---- Build the fuzzy control system ----
        self._build_control_system()

        self.is_trained = True
        logger.info("FuzzyClassifier training complete. %d rules created.", len(self.rules))

    def _compute_membership_params(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> None:
        """
        Compute data-driven membership function parameters.

        For each feature and each class, compute:
          - class_mean: mean feature value for that class
          - class_std:  std of feature value for that class

        Membership functions:
          LOW    : triangular centered at LOW class mean
          MEDIUM : triangular centered at MODERATE class mean
          HIGH   : triangular centered at HIGH class mean
        """
        for fi, feat in enumerate(feature_names):
            feat_vals = X[:, fi]
            raw_min = float(np.min(feat_vals))
            raw_max = float(np.max(feat_vals))

            # Ensure universe covers min and max with margin
            overall_min = raw_min
            overall_max = raw_max
            if overall_max - overall_min < 1e-5:
                overall_min -= 1.0
                overall_max += 1.0
            else:
                margin = (overall_max - overall_min) * 0.1
                overall_min -= margin
                overall_max += margin

            # Per-class statistics
            class_stats = {}
            for cls_int, cls_name in INT_TO_CLASS.items():
                mask = y == cls_int
                if not np.any(mask):
                    # fallback: equal spacing
                    cls_mean = overall_min + (cls_int + 0.5) * (overall_max - overall_min) / 3
                    cls_std = (overall_max - overall_min) / 6
                else:
                    cls_vals = feat_vals[mask]
                    cls_mean = float(np.mean(cls_vals))
                    cls_std = max(float(np.std(cls_vals)), 1e-5)
                class_stats[cls_name] = {"mean": cls_mean, "std": cls_std}

            self.feature_ranges[feat] = (overall_min, overall_max)
            self.membership_params[feat] = {
                "universe_min": overall_min,
                "universe_max": overall_max,
                "class_stats": class_stats,
            }

    def _get_mf_params(self, feat: str) -> Dict:
        """
        Convert class statistics to triangular membership function points.

        Returns:
            {
                'universe': np.array,
                'low_mf':   np.array,
                'med_mf':   np.array,
                'high_mf':  np.array,
            }
        """
        params = self.membership_params[feat]
        umin = float(params["universe_min"])
        umax = float(params["universe_max"])
        urange = max(umax - umin, 1e-5)
        universe = np.linspace(umin, umax, 200)

        stats = params["class_stats"]
        low_mean = stats["LOW"]["mean"]
        med_mean = stats["MODERATE"]["mean"]
        high_mean = stats["HIGH"]["mean"]

        # Width of each MF = 2 std, but at minimum 1/4 of range
        low_width = max(stats["LOW"]["std"] * 2, urange * 0.25)
        med_width = max(stats["MODERATE"]["std"] * 2, urange * 0.25)
        high_width = max(stats["HIGH"]["std"] * 2, urange * 0.25)

        def _make_trimf(mean_val: float, width: float) -> np.ndarray:
            center = float(np.clip(mean_val, umin, umax))
            left = max(umin, center - width)
            right = min(umax, center + width)
            # Ensure strict a <= b <= c
            left = min(left, center)
            right = max(right, center)
            return fuzz.trimf(universe, [left, center, right])

        low_mf = _make_trimf(low_mean, low_width)
        med_mf = _make_trimf(med_mean, med_width)
        high_mf = _make_trimf(high_mean, high_width)

        return {
            "universe": universe,
            "low_mf": low_mf,
            "med_mf": med_mf,
            "high_mf": high_mf,
        }

    def _build_control_system(self) -> None:
        """
        Build the scikit-fuzzy Antecedents, Consequent, and Rules.
        """
        antecedents = {}
        mf_data = {}

        for feat in self.feature_names:
            mf = self._get_mf_params(feat)
            universe = mf["universe"]
            ant = ctrl.Antecedent(universe, feat)
            ant["LOW"] = mf["low_mf"]
            ant["MEDIUM"] = mf["med_mf"]
            ant["HIGH"] = mf["high_mf"]
            antecedents[feat] = ant
            mf_data[feat] = mf

        # ---- Consequent ----
        cog_load = ctrl.Consequent(self._output_universe, "cognitive_load")
        cog_load["LOW"] = fuzz.trimf(self._output_universe, [0, 0, 50])
        cog_load["MODERATE"] = fuzz.trimf(self._output_universe, [0, 50, 100])
        cog_load["HIGH"] = fuzz.trimf(self._output_universe, [50, 100, 100])

        # ---- Rules ----
        rules_obj, rules_desc = self._create_rules(antecedents, cog_load)
        self.rules = rules_desc

        # ---- Control system ----
        self._control_system = ctrl.ControlSystem(rules_obj)
        self._simulator = ctrl.ControlSystemSimulation(self._control_system)

        logger.info("Fuzzy control system built with %d rules.", len(rules_obj))

    def _create_rules(
        self, antecedents: Dict, cog_load
    ) -> Tuple[List, List[Dict]]:
        """
        Create fuzzy rules.

        Rule design rationale:
          - Theta band power increases with cognitive load
            (frontal theta associated with WM and attention)
          - Beta band power increases with cognitive load
            (associated with active processing)
          - Theta/Beta ratio increases with load
          - Spectral entropy decreases with task difficulty
            (more organised brain activity under cognitive demand)
          - Alpha power generally decreases with load
            (alpha suppression during cognitive engagement)

        These are general heuristics from the EEG cognition literature.
        The actual rule set is validated against training data.

        DISCLAIMER: These are model heuristics, NOT medical facts.
        """
        rules_obj = []
        rules_desc = []
        rule_no = 0

        # Shorthand
        A = antecedents
        feat = self.feature_names

        def get_ant(feature, level):
            if feature in A:
                return A[feature][level]
            return None

        def add_rule(antecedent_expr, consequent_level, desc):
            nonlocal rule_no
            rule_no += 1
            try:
                r = ctrl.Rule(antecedent_expr, cog_load[consequent_level])
                rules_obj.append(r)
                rules_desc.append({
                    "rule_number": rule_no,
                    "rule_text": desc,
                    "consequent": consequent_level,
                    "enabled": True,
                })
                logger.debug("Rule %d: %s → %s", rule_no, desc, consequent_level)
            except Exception as e:
                logger.warning("Rule %d creation failed: %s", rule_no, e)

        # ---------------------------------------------------------------- #
        # Rules — adapts to available features
        # ---------------------------------------------------------------- #
        has = lambda f: f in A

        # --- LOW workload rules ---
        if has("theta_beta_ratio") and has("theta_power"):
            add_rule(
                A["theta_beta_ratio"]["LOW"] & A["theta_power"]["LOW"],
                "LOW",
                "IF theta_beta_ratio is LOW AND theta_power is LOW THEN LOW"
            )
        if has("beta_power") and has("spectral_entropy"):
            add_rule(
                A["beta_power"]["LOW"] & A["spectral_entropy"]["HIGH"],
                "LOW",
                "IF beta_power is LOW AND spectral_entropy is HIGH THEN LOW"
            )
        if has("alpha_power"):
            add_rule(
                A["alpha_power"]["HIGH"],
                "LOW",
                "IF alpha_power is HIGH THEN LOW"
            )
        if has("alpha_relative"):
            add_rule(
                A["alpha_relative"]["HIGH"],
                "LOW",
                "IF alpha_relative is HIGH THEN LOW"
            )
        if has("theta_relative"):
            add_rule(
                A["theta_relative"]["LOW"],
                "LOW",
                "IF theta_relative is LOW THEN LOW"
            )
        if has("theta_power"):
            add_rule(
                A["theta_power"]["LOW"],
                "LOW",
                "IF theta_power is LOW THEN LOW"
            )

        # --- MODERATE workload rules ---
        if has("theta_beta_ratio"):
            add_rule(
                A["theta_beta_ratio"]["MEDIUM"],
                "MODERATE",
                "IF theta_beta_ratio is MEDIUM THEN MODERATE"
            )
        if has("theta_power") and has("beta_power"):
            add_rule(
                A["theta_power"]["MEDIUM"] & A["beta_power"]["MEDIUM"],
                "MODERATE",
                "IF theta_power is MEDIUM AND beta_power is MEDIUM THEN MODERATE"
            )
        if has("alpha_power"):
            add_rule(
                A["alpha_power"]["MEDIUM"],
                "MODERATE",
                "IF alpha_power is MEDIUM THEN MODERATE"
            )
        if has("spectral_entropy"):
            add_rule(
                A["spectral_entropy"]["MEDIUM"],
                "MODERATE",
                "IF spectral_entropy is MEDIUM THEN MODERATE"
            )
        if has("theta_relative") and has("theta_beta_ratio"):
            add_rule(
                A["theta_relative"]["MEDIUM"] & A["theta_beta_ratio"]["MEDIUM"],
                "MODERATE",
                "IF theta_relative is MEDIUM AND theta_beta_ratio is MEDIUM THEN MODERATE"
            )
        if has("theta_power") and has("spectral_entropy"):
            add_rule(
                A["theta_power"]["MEDIUM"] & A["spectral_entropy"]["MEDIUM"],
                "MODERATE",
                "IF theta_power is MEDIUM AND spectral_entropy is MEDIUM THEN MODERATE"
            )

        # --- HIGH workload rules ---
        if has("theta_beta_ratio") and has("theta_power"):
            add_rule(
                A["theta_beta_ratio"]["HIGH"] & A["theta_power"]["HIGH"],
                "HIGH",
                "IF theta_beta_ratio is HIGH AND theta_power is HIGH THEN HIGH"
            )
        if has("beta_power") and has("spectral_entropy"):
            add_rule(
                A["beta_power"]["HIGH"] & A["spectral_entropy"]["LOW"],
                "HIGH",
                "IF beta_power is HIGH AND spectral_entropy is LOW THEN HIGH"
            )
        if has("alpha_power"):
            add_rule(
                A["alpha_power"]["LOW"],
                "HIGH",
                "IF alpha_power is LOW THEN HIGH"
            )
        if has("alpha_relative"):
            add_rule(
                A["alpha_relative"]["LOW"],
                "HIGH",
                "IF alpha_relative is LOW THEN HIGH"
            )
        if has("theta_relative"):
            add_rule(
                A["theta_relative"]["HIGH"],
                "HIGH",
                "IF theta_relative is HIGH THEN HIGH"
            )
        if has("theta_power") and has("beta_power"):
            add_rule(
                A["theta_power"]["HIGH"] & A["beta_power"]["HIGH"],
                "HIGH",
                "IF theta_power is HIGH AND beta_power is HIGH THEN HIGH"
            )
        if has("theta_beta_ratio"):
            add_rule(
                A["theta_beta_ratio"]["HIGH"],
                "HIGH",
                "IF theta_beta_ratio is HIGH THEN HIGH"
            )

        # Catch-all: ensure every antecedent feature is used in at least one rule
        for f in feat:
            if not any(f in r["rule_text"] for r in rules_desc):
                add_rule(A[f]["LOW"], "LOW", f"IF {f} is LOW THEN LOW")
                add_rule(A[f]["MEDIUM"], "MODERATE", f"IF {f} is MEDIUM THEN MODERATE")
                add_rule(A[f]["HIGH"], "HIGH", f"IF {f} is HIGH THEN HIGH")

        logger.info("Created %d fuzzy rules.", len(rules_obj))
        return rules_obj, rules_desc

    # ---------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------- #

    def predict_single(self, x: np.ndarray) -> Dict:
        """
        Make a prediction for a single feature vector.

        Parameters:
            x : 1D array of feature values in same order as feature_names

        Returns:
            dict with:
              predicted_class : str ("LOW" / "MODERATE" / "HIGH")
              fuzzy_score     : float [0, 100]
              confidence      : float [0, 1]
              memberships     : {feature: {LOW, MEDIUM, HIGH}}
              activated_rules : list of activated rule descriptions
        """
        if not self.is_trained:
            raise RuntimeError("FuzzyClassifier is not trained. Call train() first.")

        # Reset simulator for fresh computation
        sim = ctrl.ControlSystemSimulation(self._control_system)

        # Set input values (clip to universe range)
        for i, feat in enumerate(self.feature_names):
            val = float(x[i])
            umin, umax = self.feature_ranges.get(feat, (0, 1))
            val = np.clip(val, umin, umax)
            try:
                sim.input[feat] = val
            except (KeyError, ValueError):
                pass

        try:
            sim.compute()
            fuzzy_score = float(sim.output.get("cognitive_load", 50.0))
        except Exception as e:
            logger.warning("Fuzzy computation error: %s. Using default score 50.", e)
            fuzzy_score = 50.0

        # Map fuzzy score to class
        predicted_class, confidence = self._score_to_class(fuzzy_score)

        # Compute feature memberships
        memberships = self._compute_memberships(x)

        # Determine which rules fired strongly
        activated_rules = self._get_activated_rules(memberships)

        return {
            "predicted_class": predicted_class,
            "fuzzy_score": round(fuzzy_score, 2),
            "confidence": round(confidence, 4),
            "memberships": memberships,
            "activated_rules": activated_rules,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for a feature matrix.
        Returns array of class integers (0=LOW, 1=MODERATE, 2=HIGH).
        """
        predictions = []
        for x in X:
            result = self.predict_single(x)
            predictions.append(CLASS_MAP[result["predicted_class"]])
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Returns (n_samples, 3) array.
        Probabilities are derived from fuzzy score membership.
        """
        proba = []
        for x in X:
            result = self.predict_single(x)
            p = self._fuzzy_score_to_proba(result["fuzzy_score"])
            proba.append(p)
        return np.array(proba)

    def _score_to_class(self, score: float) -> Tuple[str, float]:
        """
        Map a fuzzy score [0, 100] to a class label.
          0–33  → LOW
          34–66 → MODERATE
          67–100 → HIGH
        """
        if score <= 33.33:
            cls = "LOW"
            confidence = 1.0 - (score / 33.33)
        elif score <= 66.67:
            cls = "MODERATE"
            confidence = 1.0 - abs((score - 50.0) / 16.67)
        else:
            cls = "HIGH"
            confidence = (score - 66.67) / 33.33

        confidence = float(np.clip(confidence, 0.0, 1.0))
        return cls, confidence

    def _fuzzy_score_to_proba(self, score: float) -> np.ndarray:
        """Convert fuzzy score to approximate class probabilities."""
        low_val = float(fuzz.interp_membership(self._output_universe,
                         fuzz.trimf(self._output_universe, [0, 0, 50]), score))
        mod_val = float(fuzz.interp_membership(self._output_universe,
                         fuzz.trimf(self._output_universe, [0, 50, 100]), score))
        high_val = float(fuzz.interp_membership(self._output_universe,
                          fuzz.trimf(self._output_universe, [50, 100, 100]), score))
        total = low_val + mod_val + high_val
        if total <= 0:
            return np.array([1/3, 1/3, 1/3])
        return np.array([low_val, mod_val, high_val]) / total

    def _compute_memberships(self, x: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute membership degree for each feature and level."""
        memberships = {}
        for i, feat in enumerate(self.feature_names):
            val = float(x[i])
            mf = self._get_mf_params(feat)
            universe = mf["universe"]
            memberships[feat] = {
                "LOW": float(fuzz.interp_membership(universe, mf["low_mf"], val)),
                "MEDIUM": float(fuzz.interp_membership(universe, mf["med_mf"], val)),
                "HIGH": float(fuzz.interp_membership(universe, mf["high_mf"], val)),
            }
        return memberships

    def _get_activated_rules(self, memberships: Dict) -> List[Dict]:
        """
        Return rules that are activated (strength > 0.1) based on memberships.
        """
        activated = []
        for rule in self.rules:
            strength = self._estimate_rule_strength(rule["rule_text"], memberships)
            if strength > 0.1:
                activated.append({
                    "rule_number": rule["rule_number"],
                    "rule_text": rule["rule_text"],
                    "consequent": rule["consequent"],
                    "strength": round(strength, 3),
                })
        return sorted(activated, key=lambda r: -r["strength"])

    def _estimate_rule_strength(self, rule_text: str, memberships: Dict) -> float:
        """
        Estimate rule firing strength from membership values.
        Parses rule text to identify antecedents.
        """
        total_strength = 1.0
        n_conditions = 0

        for feat, levels in memberships.items():
            for level in ["LOW", "MEDIUM", "HIGH"]:
                # Check if this feature+level appears in rule text
                pattern = f"{feat} is {level}"
                if pattern in rule_text:
                    total_strength *= levels.get(level, 0.0)
                    n_conditions += 1

        if n_conditions == 0:
            return 0.0
        return total_strength

    # ---------------------------------------------------------------- #
    # Save / Load
    # ---------------------------------------------------------------- #

    def save(self, path: Optional[str] = None, config=None) -> str:
        """Save the trained classifier to disk."""
        cfg = config or self.config
        if path is None:
            save_dir = resolve_path(cfg.paths.models) / "fuzzy"
            save_dir.mkdir(parents=True, exist_ok=True)
            path = str(save_dir / "fuzzy_classifier.joblib")

        state = {
            "feature_names": self.feature_names,
            "feature_ranges": self.feature_ranges,
            "membership_params": self.membership_params,
            "rules": self.rules,
            "is_trained": self.is_trained,
        }
        joblib.dump(state, path)
        logger.info("FuzzyClassifier saved: %s", path)
        return path

    def load(self, path: Optional[str] = None, config=None) -> None:
        """Load a saved classifier from disk."""
        cfg = config or self.config
        if path is None:
            save_dir = resolve_path(cfg.paths.models) / "fuzzy"
            path = str(save_dir / "fuzzy_classifier.joblib")

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Fuzzy classifier not found: {path}\n"
                "Please run:  python main.py --train"
            )

        state = joblib.load(path)
        self.feature_names = state["feature_names"]
        self.feature_ranges = state["feature_ranges"]
        self.membership_params = state["membership_params"]
        self.rules = state["rules"]
        self.is_trained = state["is_trained"]

        # Rebuild control system
        self._compute_membership_params_from_state()
        self._build_control_system()
        logger.info("FuzzyClassifier loaded from: %s", path)

    def _compute_membership_params_from_state(self):
        """After loading, restore membership_params (already loaded from state)."""
        pass  # membership_params loaded directly from state dict


# ------------------------------------------------------------------ #
# Explainability
# ------------------------------------------------------------------ #

def get_fuzzy_rules_for_display(classifier: FuzzyClassifier) -> List[Dict]:
    """Return the rules list suitable for database storage and display."""
    return classifier.rules
