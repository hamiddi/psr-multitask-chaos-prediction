"""
Run the complete PSR + hierarchical multitask chaos-prediction experiment.

Usage from the repository root:
    PYTHONPATH=src python scripts/run_experiment.py

Optional environment variable to reduce TensorFlow oneDNN messages:
    TF_ENABLE_ONEDNN_OPTS=0 PYTHONPATH=src python scripts/run_experiment.py
"""

import os
import warnings

# Reduce TensorFlow logging before TensorFlow is imported inside full_pipeline.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from psrchaos.full_pipeline import (
    Config,
    run_full_workflow,
    plot_workflow_figures,
    print_summary,
    run_advanced_early_warning_evaluation,
)


def main() -> None:
    """Execute the full workflow, generate plots, print summary, and run evaluation."""
    warnings.filterwarnings("ignore")
    tf.random.set_seed(42)
    np.random.seed(42)

    cfg = Config()

    results = run_full_workflow(cfg)

    plot_workflow_figures(
        results,
        cfg,
        example_b=28.0,
        example_ic_id=0,
        show=False,
    )

    print_summary(results)

    run_advanced_early_warning_evaluation(results)


if __name__ == "__main__":
    main()
