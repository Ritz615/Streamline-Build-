"""
app/server.py
=============
Unified Python backend server that serves the approved Landing Page / UI
and provides REST API endpoints for real EEG processing, feature analysis,
fuzzy inference, Random Forest predictions, and SQLite experiment history.

Usage:
    python app/server.py --port 8080
"""

import argparse
import base64
import json
import logging
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.config import get_config, PROJECT_ROOT
from src.database import (
    get_all_experiment_history_df,
    get_fuzzy_rules,
    init_db,
    record_model_run,
)
from src.evaluation import load_comparison_csv
from src.explainability import explain_fuzzy, explain_rf
from src.feature_extraction import get_feature_columns, load_features
from src.feature_selection import load_feature_importance
from src.fuzzy_classifier import FuzzyClassifier
from src.preprocessing import load_preprocessed
from src.random_forest import RandomForestModel

logger = logging.getLogger("eeg_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global caches
_CONFIG = get_config()
_FUZZY_MODEL = None
_RF_MODEL = None
_FEATURES_DF = None


def get_cached_fuzzy_model():
    global _FUZZY_MODEL
    if _FUZZY_MODEL is None:
        try:
            clf = FuzzyClassifier(config=_CONFIG)
            clf.load(config=_CONFIG)
            _FUZZY_MODEL = clf
            logger.info("Fuzzy model loaded into cache.")
        except Exception as e:
            logger.warning("Could not load fuzzy model: %s", e)
    return _FUZZY_MODEL


def get_cached_rf_model():
    global _RF_MODEL
    if _RF_MODEL is None:
        try:
            rf = RandomForestModel(config=_CONFIG)
            rf.load(config=_CONFIG)
            _RF_MODEL = rf
            logger.info("Random Forest model loaded into cache.")
        except Exception as e:
            logger.warning("Could not load RF model: %s", e)
    return _RF_MODEL


def get_cached_features():
    global _FEATURES_DF
    if _FEATURES_DF is None:
        try:
            _FEATURES_DF = load_features(config=_CONFIG)
            logger.info("Features dataset loaded into cache (%d rows).", len(_FEATURES_DF))
        except Exception as e:
            logger.warning("Could not load features: %s", e)
    return _FEATURES_DF


class EEGUnifiedRequestHandler(SimpleHTTPRequestHandler):
    """
    Handles static files from web/ and serves JSON API endpoints under /api/*
    """

    def __init__(self, *args, **kwargs):
        web_dir = str(PROJECT_ROOT / "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.startswith("/api/"):
            self._handle_api_get(path, query)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            self._handle_api_post(path, data)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---------------------------------------------------------------------- #
    # API GET Router
    # ---------------------------------------------------------------------- #
    def _handle_api_get(self, path: str, query: dict):
        try:
            if path == "/api/status":
                self._api_status()
            elif path == "/api/subjects":
                self._api_subjects()
            elif path == "/api/eeg-data":
                self._api_eeg_data(query)
            elif path == "/api/features":
                self._api_features(query)
            elif path == "/api/models/comparison":
                self._api_model_comparison()
            elif path == "/api/history":
                self._api_history()
            elif path == "/api/viva":
                self._api_viva()
            else:
                self._send_json({"error": "Unknown endpoint"}, status=404)
        except Exception as e:
            logger.error("API error on %s: %s", path, e, exc_info=True)
            self._send_json({"error": str(e)}, status=500)

    # ---------------------------------------------------------------------- #
    # API POST Router
    # ---------------------------------------------------------------------- #
    def _handle_api_post(self, path: str, data: dict):
        try:
            if path == "/api/predict":
                self._api_predict(data)
            else:
                self._send_json({"error": "Unknown POST endpoint"}, status=404)
        except Exception as e:
            logger.error("API error on %s: %s", path, e, exc_info=True)
            self._send_json({"error": str(e)}, status=500)

    # ---------------------------------------------------------------------- #
    # Endpoints
    # ---------------------------------------------------------------------- #
    def _api_status(self):
        df = get_cached_features()
        comp = load_comparison_csv(config=_CONFIG)
        comp_records = comp.to_dict(orient="records") if comp is not None else []

        data = {
            "dataset_id": _CONFIG.dataset.id,
            "dataset_name": _CONFIG.dataset.name,
            "subjects_total": _CONFIG.dataset.subjects,
            "channels": _CONFIG.dataset.channels,
            "sampling_rate": _CONFIG.dataset.sampling_rate,
            "total_windows": len(df) if df is not None else 0,
            "classes": ["LOW", "MODERATE", "HIGH"],
            "models_trained": {
                "fuzzy": get_cached_fuzzy_model() is not None,
                "random_forest": get_cached_rf_model() is not None,
            },
            "comparison": comp_records,
        }
        self._send_json(data)

    def _api_subjects(self):
        processed_dir = PROJECT_ROOT / "data" / "processed"
        subjects = sorted([d.name for d in processed_dir.glob("sub-*") if d.is_dir()]) if processed_dir.exists() else []
        self._send_json({"subjects": subjects})

    def _api_eeg_data(self, query: dict):
        subject = query.get("subject", ["sub-001"])[0]
        start_sec = float(query.get("start", [0.0])[0])
        dur_sec = float(query.get("duration", [4.0])[0])

        raw = load_preprocessed(subject, config=_CONFIG)
        if raw is None:
            self._send_json({"error": f"Subject {subject} not found"}, status=404)
            return

        sfreq = float(raw.info["sfreq"])
        total_dur = float(raw.times[-1])
        start_samp = int(max(0, start_sec) * sfreq)
        end_samp = int(min(total_dur, start_sec + dur_sec) * sfreq)

        # Pick top channels (Fz, Cz, Pz, T5, T6 or available)
        avail_chans = raw.ch_names
        pref_chans = ["Fz", "Cz", "Pz", "T5", "T6", "Oz", "Fp1", "F3", "F4", "C3", "C4", "P3", "P4"]
        chosen_chans = [c for c in pref_chans if c in avail_chans][:6] or avail_chans[:6]

        chan_indices = [avail_chans.index(c) for c in chosen_chans]
        raw_data = raw.get_data(picks=chan_indices, start=start_samp, stop=end_samp)

        # Downsample for web rendering (100 Hz display)
        step = max(1, int(sfreq / 100))
        time_points = (np.arange(raw_data.shape[1])[::step] / sfreq + start_sec).tolist()

        channels_payload = []
        for i, ch_name in enumerate(chosen_chans):
            sig = raw_data[i, ::step].astype(float)
            channels_payload.append({
                "channel": ch_name,
                "values": sig.tolist(),
            })

        # Compute Welch PSD band powers on this segment
        from scipy.signal import welch
        mean_sig = np.mean(raw_data, axis=0)
        freqs, psd = welch(mean_sig, fs=sfreq, nperseg=min(len(mean_sig), int(sfreq * 2)))

        def band_pwr(f_low, f_high):
            idx = (freqs >= f_low) & (freqs <= f_high)
            return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

        delta_p = band_pwr(1, 4)
        theta_p = band_pwr(4, 8)
        alpha_p = band_pwr(8, 13)
        beta_p = band_pwr(13, 30)
        tot = (delta_p + theta_p + alpha_p + beta_p) or 1.0

        psd_payload = {
            "freqs": freqs[(freqs >= 1) & (freqs <= 35)].tolist(),
            "psd": psd[(freqs >= 1) & (freqs <= 35)].tolist(),
            "bands": {
                "delta": round(delta_p / tot * 100, 1),
                "theta": round(theta_p / tot * 100, 1),
                "alpha": round(alpha_p / tot * 100, 1),
                "beta": round(beta_p / tot * 100, 1),
            }
        }

        self._send_json({
            "subject": subject,
            "total_duration": round(total_dur, 1),
            "sfreq": sfreq,
            "time_points": time_points,
            "channels": channels_payload,
            "psd": psd_payload,
        })

    def _api_features(self, query: dict):
        df = get_cached_features()
        if df is None:
            self._send_json({"error": "Features not extracted"}, status=404)
            return

        feature_cols = get_feature_columns(df)
        imp_df = load_feature_importance(config=_CONFIG)
        imp_records = imp_df.to_dict(orient="records") if imp_df is not None else []

        # Class distribution
        class_dist = df["label"].value_counts().to_dict()

        # Distribution stats per feature for boxplots
        selected_feat = query.get("feature", ["alpha_relative"])[0]
        if selected_feat not in feature_cols:
            selected_feat = feature_cols[0]

        box_data = {}
        for cls in ["LOW", "MODERATE", "HIGH"]:
            vals = df[df["label"] == cls][selected_feat].dropna().values
            if len(vals) > 0:
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                iqr = q75 - q25
                min_v = float(np.min(vals[vals >= q25 - 1.5 * iqr]))
                max_v = float(np.max(vals[vals <= q75 + 1.5 * iqr]))
                box_data[cls] = {
                    "min": round(min_v, 4),
                    "q25": round(float(q25), 4),
                    "median": round(float(q50), 4),
                    "q75": round(float(q75), 4),
                    "max": round(max_v, 4),
                    "mean": round(float(np.mean(vals)), 4),
                    "count": int(len(vals)),
                }

        # Sample 30 real windows for picker dropdown
        sample_windows = df.sample(min(30, len(df)), random_state=42)[
            ["subject_id", "window_id", "label"] + feature_cols[:6]
        ].to_dict(orient="records")

        self._send_json({
            "total_windows": len(df),
            "subjects_count": int(df["subject_id"].nunique()),
            "feature_names": feature_cols,
            "importance": imp_records[:15],
            "class_distribution": class_dist,
            "selected_feature": selected_feat,
            "box_plot": box_data,
            "sample_windows": sample_windows,
        })

    def _api_predict(self, data: dict):
        df = get_cached_features()
        fuzzy_clf = get_cached_fuzzy_model()
        rf_model = get_cached_rf_model()

        if fuzzy_clf is None:
            self._send_json({"error": "Fuzzy model not trained"}, status=500)
            return

        feature_cols = get_feature_columns(df) if df is not None else fuzzy_clf.feature_names

        # Case 1: Window selection from dataset
        if "subject_id" in data and "window_id" in data and df is not None:
            sub_id = data["subject_id"]
            win_id = int(data["window_id"])
            matched = df[(df["subject_id"] == sub_id) & (df["window_id"] == win_id)]
            if matched.empty:
                row = df.iloc[0]
            else:
                row = matched.iloc[0]
            true_label = row.get("label", "UNKNOWN")
            feat_values = {f: float(row.get(f, 0.0)) for f in feature_cols}
        else:
            # Case 2: Custom slider feature values
            true_label = "N/A (Manual Simulation)"
            feat_values = {}
            for f in feature_cols:
                feat_values[f] = float(data.get(f, 0.25))

        x_fuzzy = np.array([feat_values[f] for f in fuzzy_clf.feature_names if f in feat_values])
        x_all = np.array([feat_values[f] for f in feature_cols])

        # Run Fuzzy Model
        fuzzy_res = fuzzy_clf.predict_single(x_fuzzy)
        explanation = explain_fuzzy(fuzzy_res, x_fuzzy, fuzzy_clf.feature_names)

        # Run RF Model if available
        rf_res = None
        if rf_model and rf_model.is_trained:
            try:
                proba = rf_model.predict_proba(x_all.reshape(1, -1))[0]
                pred_idx = int(np.argmax(proba))
                class_names = ["LOW", "MODERATE", "HIGH"]
                rf_res = {
                    "predicted_class": class_names[pred_idx],
                    "confidence": round(float(proba[pred_idx]) * 100, 1),
                    "probabilities": {
                        "LOW": round(float(proba[0]) * 100, 1),
                        "MODERATE": round(float(proba[1]) * 100, 1),
                        "HIGH": round(float(proba[2]) * 100, 1),
                    },
                }
            except Exception as e:
                logger.error("RF predict error: %s", e)

        self._send_json({
            "true_label": true_label,
            "feature_values": feat_values,
            "fuzzy": {
                "predicted_class": fuzzy_res["predicted_class"],
                "fuzzy_score": round(float(fuzzy_res["fuzzy_score"]), 1),
                "confidence": round(float(fuzzy_res["confidence"]) * 100, 1),
                "memberships": fuzzy_res.get("memberships", {}),
                "activated_rules": fuzzy_res.get("activated_rules", []),
                "explanation": explanation,
            },
            "random_forest": rf_res,
        })

    def _api_model_comparison(self):
        comp = load_comparison_csv(config=_CONFIG)
        rules = get_fuzzy_rules()
        rf = get_cached_rf_model()

        comp_records = comp.to_dict(orient="records") if comp is not None else []
        rules_records = [
            {"rule_number": r.rule_number, "rule_text": r.rule_text, "consequent": r.consequent}
            for r in rules
        ] if rules else []

        rf_fi = []
        if rf and rf.is_trained:
            fi_df = rf.get_feature_importance()
            rf_fi = fi_df.to_dict(orient="records")

        # Encode confusion matrix images if available
        figures_dir = PROJECT_ROOT / "results" / "figures"
        cm_fuzzy = None
        cm_rf = None
        f_fuzzy_cm = figures_dir / "fuzzy_classifier_confusion_matrix.png"
        f_rf_cm = figures_dir / "random_forest_confusion_matrix.png"

        if f_fuzzy_cm.exists():
            with open(f_fuzzy_cm, "rb") as f:
                cm_fuzzy = base64.b64encode(f.read()).decode("utf-8")
        if f_rf_cm.exists():
            with open(f_rf_cm, "rb") as f:
                cm_rf = base64.b64encode(f.read()).decode("utf-8")

        self._send_json({
            "comparison": comp_records,
            "fuzzy_rules": rules_records,
            "rf_feature_importance": rf_fi[:15],
            "confusion_matrix_fuzzy": cm_fuzzy,
            "confusion_matrix_rf": cm_rf,
        })

    def _api_history(self):
        init_db()
        df = get_all_experiment_history_df()
        records = df.to_dict(orient="records") if df is not None and not df.empty else []
        self._send_json({"runs": records, "total_runs": len(records)})

    def _api_viva(self):
        viva_qna = [
            {
                "q": "Why use Fuzzy Logic instead of just a Deep Neural Network?",
                "a": "Fuzzy Mamdani systems provide 100% white-box transparency and linguistic rules (e.g. IF Alpha is LOW AND Theta/Alpha is HIGH THEN Workload is HIGH), which is critical for trustworthy human-in-the-loop neuroscience applications."
            },
            {
                "q": "Why use Subject-Wise Cross-Validation (StratifiedGroupKFold)?",
                "a": "Random train/test splits leak individual subject brain signatures, artificially inflating accuracy to >90%. Subject-wise CV tests exclusively on subjects never seen during training, giving true generalized performance."
            },
            {
                "q": "What is the physiological basis of Theta/Alpha Ratio (TAR)?",
                "a": "Increased cognitive workload increases frontal theta power (working memory load) and suppresses parietal alpha power (cortical activation/desynchronization). Higher TAR reflects higher mental workload."
            },
            {
                "q": "How are artifacts handled during preprocessing?",
                "a": "Raw signals are bandpass-filtered (1–40 Hz), notch-filtered at 50 Hz (mains interference), re-referenced to average, and windowed with amplitude thresholding (±150 µV) to reject blinks and muscle spikes."
            },
            {
                "q": "What is the ethical limitation of this prototype?",
                "a": "This is an educational research prototype using task difficulty as an operational proxy for cognitive load. It is NOT a medical device."
            }
        ]
        self._send_json({"viva_questions": viva_qna})


def run_server(port: int = 8080):
    """Start the unified threading HTTP server."""
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, EEGUnifiedRequestHandler)
    print("=" * 65)
    print("  EEG COGNITIVE LOAD -- UNIFIED RESEARCH PLATFORM")
    print("=" * 65)
    print(f"  * Server running at: http://localhost:{port}")
    print(f"  * Serving Web UI from: {PROJECT_ROOT / 'web'}")
    print(f"  * Real Python API endpoints available under /api/*")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Unified Platform Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    args = parser.parse_args()
    run_server(args.port)
