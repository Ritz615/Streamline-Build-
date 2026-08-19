#!/usr/bin/env python
"""scripts/evaluate_models.py — Evaluate both models with subject-wise CV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config, setup_logging
from src.pipeline import Pipeline

config = get_config()
setup_logging(config)
p = Pipeline(config=config)
p.run_evaluate()
