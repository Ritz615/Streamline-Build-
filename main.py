#!/usr/bin/env python
"""
main.py
=======
Command-line entry point for the EEG Cognitive Load Classification Pipeline.

Usage:
    python main.py --all          # Run complete pipeline
    python main.py --download     # Download dataset only
    python main.py --preprocess   # Preprocess EEG only
    python main.py --features     # Extract features only
    python main.py --train        # Train models only
    python main.py --evaluate     # Evaluate models only
    python main.py --report       # Generate report only
    python main.py --status       # Show project status

After running --all, launch the dashboard:
    streamlit run app/streamlit_app.py
"""

import argparse
import io
import sys
import logging
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Cognitive Load Classification Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all             Run the complete pipeline
  python main.py --download        Download ds007169 dataset
  python main.py --preprocess      Preprocess EEG signals
  python main.py --features        Extract EEG features
  python main.py --train           Train fuzzy + RF models
  python main.py --evaluate        Evaluate models
  python main.py --report          Generate experiment report
  python main.py --status          Show project status

Disclaimer:
  This is a RESEARCH PROTOTYPE using public anonymized EEG data.
  It is NOT a medical device or diagnostic system.
        """,
    )
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess EEG")
    parser.add_argument("--features", action="store_true", help="Extract features")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--status", action="store_true", help="Show project status")
    parser.add_argument("--ui", action="store_true", help="Launch the Web Landing Page (port 8080)")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit Research Dashboard (port 8501)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser.parse_args()


def show_status(config):
    """Display current project status."""
    from src.dataset_manager import DatasetManager
    from src.evaluation import load_comparison_csv

    print("\n" + "=" * 60)
    print("  EEG COGNITIVE LOAD — PROJECT STATUS")
    print("=" * 60)

    manager = DatasetManager(config=config)
    data_ok = manager.is_data_available()
    ok = lambda b: "[OK]" if b else "[--]"

    print(f"\n  Dataset available:    {ok(data_ok)}")

    processed_dir = Path("data/processed")
    preproc_ok = processed_dir.exists() and any(processed_dir.rglob("*.fif"))
    print(f"  Preprocessed EEG:     {ok(preproc_ok)}")

    features_file = Path("data/features/features.csv")
    features_ok = features_file.exists()
    if features_ok:
        import pandas as pd
        try:
            df = pd.read_csv(features_file)
            print(f"  Features extracted:   [OK] ({len(df)} windows)")
        except Exception:
            print(f"  Features extracted:   [OK]")
    else:
        print(f"  Features extracted:   [--]")

    fuzzy_path = Path("models/fuzzy/fuzzy_classifier.joblib")
    rf_path = Path("models/random_forest/random_forest.joblib")
    print(f"  Fuzzy model trained:  {ok(fuzzy_path.exists())}")
    print(f"  RF model trained:     {ok(rf_path.exists())}")

    comp_df = load_comparison_csv(config)
    if comp_df is not None:
        print("\n  Latest Results:")
        for _, row in comp_df.iterrows():
            print(f"    {row['model']:<20} Acc={row['accuracy']:.3f}  F1={row['f1_macro']:.3f}")
    else:
        print("  Results:              [--] (run --evaluate)")

    print("\n" + "=" * 60 + "\n")


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = get_config(config_path)
    setup_logging(config)

    logger = logging.getLogger("main")
    logger.info("EEG Cognitive Load Classification Research Prototype v1.0")
    logger.info("DISCLAIMER: Research prototype only. NOT a medical system.")

    if args.status:
        show_status(config)
        return

    if args.ui:
        print("\n" + "=" * 60)
        print("  LAUNCHING WEB LANDING PAGE ON http://localhost:8080")
        print("=" * 60 + "\n")
        import subprocess
        subprocess.run([sys.executable, "-m", "http.server", "8080", "--directory", "web"])
        return

    if args.dashboard:
        print("\n" + "=" * 60)
        print("  LAUNCHING STREAMLIT DASHBOARD ON http://localhost:8501")
        print("=" * 60 + "\n")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
        return

    # Check at least one action specified
    if not any([args.all, args.download, args.preprocess,
                args.features, args.train, args.evaluate, args.report]):
        print("\nNo action specified. Use --all to run the complete pipeline.")
        print("Options:")
        print("  python main.py --ui          Launch Web Landing Page (http://localhost:8080)")
        print("  python main.py --dashboard   Launch Streamlit Dashboard (http://localhost:8501)")
        print("  python main.py --all         Run full end-to-end ML pipeline\n")
        show_status(config)
        return

    from src.pipeline import Pipeline
    pipeline = Pipeline(config=config)

    try:
        if args.all:
            pipeline.run_all()
            print("\n[OK] Pipeline complete! Launch dashboard with:")
            print("    streamlit run app/streamlit_app.py\n")

        else:
            if args.download:
                ok = pipeline.run_download()
                if not ok:
                    sys.exit(1)

            if args.preprocess:
                pipeline.run_preprocess()

            if args.features:
                pipeline.run_features()

            if args.train:
                pipeline.run_train()

            if args.evaluate:
                pipeline.run_evaluate()

            if args.report:
                path = pipeline.run_report()
                print(f"\n✓ Report generated: {path}\n")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
