"""
Fast smoke test for checking that the complete pipeline imports and runs.

This uses fewer b values, fewer time points, fewer folds, and fewer epochs.
It is intended only for debugging the installation, not for reproducing
paper-quality results.

Usage:
    PYTHONPATH=src python scripts/run_fast_smoke_test.py
"""

import os
import warnings
import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from psrchaos.full_pipeline import Config, run_full_workflow, print_summary


def main() -> None:
    warnings.filterwarnings("ignore")
    tf.random.set_seed(42)
    np.random.seed(42)

    cfg = Config(
        # Small settings for quick execution only.
        t_end=8.0,
        num_points=800,
        b_values=tuple(np.linspace(20.0, 30.0, 6)),
        initial_conditions=((0.1, 0.3, 1.0), (0.10001, 0.3, 1.0)),
        window_size=40,
        future_horizon=8,
        stride=20,
        n_splits=2,
        epochs=2,
        batch_size=16,
        output_dir="psr_ssm_outputs_smoke_test",
    )

    results = run_full_workflow(cfg)
    print_summary(results)


if __name__ == "__main__":
    main()
