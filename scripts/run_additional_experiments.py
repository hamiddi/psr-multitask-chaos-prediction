"""
Run additional experiments.

This script adds experiments requested by reviewers without changing the original
paper pipeline. It saves tables and plots under:

    psr_ssm_outputs/additional_experiments/

Usage:
    PYTHONPATH=src python scripts/run_additional_experiments.py

Fast test mode:
    PYTHONPATH=src python scripts/run_additional_experiments.py --fast
"""

import argparse
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from psrchaos.full_pipeline import Config
from psrchaos.extensions import run_additional_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a quick reduced experiment to verify the code.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    tf.random.set_seed(42)
    np.random.seed(42)

    cfg = Config()
    run_additional_experiments(cfg=cfg, fast=args.fast)


if __name__ == "__main__":
    main()
