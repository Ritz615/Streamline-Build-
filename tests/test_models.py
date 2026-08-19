"""tests/test_models.py — Tests for fuzzy classifier and random forest."""
import numpy as np
import pytest

VALID_CLASSES = {"LOW", "MODERATE", "HIGH"}
INT_TO_CLASS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


# ── Random Forest ─────────────────────────────────────────────────────────

class TestRandomForest:

    def test_rf_trains(self, synthetic_X_y):
        """Random Forest should train without errors."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        assert rf.is_trained

    def test_rf_predicts(self, synthetic_X_y):
        """Random Forest should produce predictions of correct shape."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        preds = rf.predict(X)
        assert preds.shape == (len(X),)

    def test_rf_predictions_valid_classes(self, synthetic_X_y):
        """All RF predictions should be valid class integers (0, 1, or 2)."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        preds = rf.predict(X)
        for p in preds:
            assert p in {0, 1, 2}, f"Invalid class integer: {p}"

    def test_rf_proba_shape(self, synthetic_X_y):
        """RF probabilities should have shape (n_samples, 3)."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_rf_proba_sums_to_one(self, synthetic_X_y):
        """RF probabilities should sum to 1 per sample."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        proba = rf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_rf_feature_importance_shape(self, synthetic_X_y):
        """Feature importance DataFrame should have correct columns."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y
        rf = RandomForestModel()
        rf.train_final(X, y, feature_cols)
        fi = rf.get_feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(feature_cols)

    def test_rf_cross_validates(self, synthetic_X_y):
        """Subject-wise CV should produce valid metrics."""
        from src.random_forest import RandomForestModel
        X, y, groups, feature_cols = synthetic_X_y

        # Use 2 folds for speed in testing
        rf = RandomForestModel()
        # Only run if enough subjects for CV
        n_subjects = len(np.unique(groups))
        if n_subjects < 2:
            pytest.skip("Not enough subjects for CV test")

        result = rf.cross_validate(X, y, groups, feature_cols, n_splits=2)
        assert "accuracy_mean" in result
        assert 0.0 <= result["accuracy_mean"] <= 1.0


# ── Fuzzy Classifier ──────────────────────────────────────────────────────

class TestFuzzyClassifier:

    @pytest.fixture
    def trained_fuzzy(self, synthetic_X_y):
        """Return a trained FuzzyClassifier."""
        from src.fuzzy_classifier import FuzzyClassifier
        X, y, groups, feature_cols = synthetic_X_y
        # Use first 5 features
        try:
            clf = FuzzyClassifier()
        except ImportError:
            pytest.skip("scikit-fuzzy not installed or not compatible")
        clf.train(X[:, :5], y, feature_cols[:5])
        return clf, X[:, :5], feature_cols[:5]

    def test_fuzzy_trains(self, trained_fuzzy):
        """FuzzyClassifier should train without errors."""
        clf, X, feat = trained_fuzzy
        assert clf.is_trained

    def test_fuzzy_has_rules(self, trained_fuzzy):
        """FuzzyClassifier should have at least 5 rules."""
        clf, X, feat = trained_fuzzy
        assert len(clf.rules) >= 5

    def test_fuzzy_predict_single_returns_dict(self, trained_fuzzy):
        """predict_single should return a dict with required keys."""
        clf, X, feat = trained_fuzzy
        result = clf.predict_single(X[0])
        assert isinstance(result, dict)
        assert "predicted_class" in result
        assert "fuzzy_score" in result
        assert "confidence" in result
        assert "memberships" in result
        assert "activated_rules" in result

    def test_fuzzy_predicted_class_valid(self, trained_fuzzy):
        """Predicted class must be LOW, MODERATE, or HIGH."""
        clf, X, feat = trained_fuzzy
        result = clf.predict_single(X[0])
        assert result["predicted_class"] in VALID_CLASSES

    def test_fuzzy_score_in_range(self, trained_fuzzy):
        """Fuzzy score must be in [0, 100]."""
        clf, X, feat = trained_fuzzy
        result = clf.predict_single(X[0])
        assert 0 <= result["fuzzy_score"] <= 100

    def test_fuzzy_confidence_in_range(self, trained_fuzzy):
        """Confidence must be in [0, 1]."""
        clf, X, feat = trained_fuzzy
        result = clf.predict_single(X[0])
        assert 0 <= result["confidence"] <= 1

    def test_fuzzy_batch_predict(self, trained_fuzzy):
        """Batch predict should return integer array."""
        clf, X, feat = trained_fuzzy
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        for p in preds:
            assert p in {0, 1, 2}

    def test_fuzzy_memberships_present(self, trained_fuzzy):
        """Memberships dict should have entries for all input features."""
        clf, X, feat = trained_fuzzy
        result = clf.predict_single(X[0])
        for f in feat:
            assert f in result["memberships"]
            levels = result["memberships"][f]
            assert "LOW" in levels
            assert "MEDIUM" in levels
            assert "HIGH" in levels

    def test_fuzzy_save_load(self, trained_fuzzy, tmp_path):
        """FuzzyClassifier should save and reload correctly."""
        from src.fuzzy_classifier import FuzzyClassifier
        clf, X, feat = trained_fuzzy
        save_path = str(tmp_path / "fuzzy_test.joblib")
        clf.save(path=save_path)

        clf2 = FuzzyClassifier()
        clf2.load(path=save_path)
        assert clf2.is_trained
        result = clf2.predict_single(X[0])
        assert result["predicted_class"] in VALID_CLASSES


# ── Explainability ────────────────────────────────────────────────────────

class TestExplainability:

    def test_fuzzy_explanation_is_string(self, trained_fuzzy=None):
        """Fuzzy explanation should return a non-empty string."""
        try:
            from src.fuzzy_classifier import FuzzyClassifier
            from src.explainability import explain_fuzzy
        except ImportError:
            pytest.skip("Required modules not available")

        # Use minimal data
        prediction_result = {
            "predicted_class": "HIGH",
            "fuzzy_score": 75.0,
            "confidence": 0.8,
            "memberships": {"theta_power": {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.8}},
            "activated_rules": [{"rule_number": 1, "rule_text": "IF theta_power is HIGH THEN HIGH",
                                  "consequent": "HIGH", "strength": 0.8}],
        }
        feature_values = np.array([0.5])
        feature_names = ["theta_power"]

        text = explain_fuzzy(prediction_result, feature_values, feature_names)
        assert isinstance(text, str)
        assert len(text) > 50
        assert "HIGH" in text
