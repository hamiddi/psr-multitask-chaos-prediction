"""
========================================================================
PSR + HIERARCHICAL MULTITASK DEEP LEARNING FOR 3-CLASS REGIME PREDICTION
FULLY MERGED COMPLETE SCRIPT WITH COMMENTS
========================================================================

3-Class regime definition
-------------------------
Class 0: stable
Class 1: transitional_oscillatory   (near_instability + oscillatory merged)
Class 2: chaotic

Hierarchical classification
---------------------------
Level 1:
    nonchaotic  vs  chaotic

Level 2 inside nonchaotic:
    stable  vs  transitional_oscillatory

Representation design
---------------------
- Classification branch uses:
      [last hidden state ; attention context]
- Forecasting branch uses:
      last hidden state only
- Future PSR branch uses:
      last hidden state only

Main workflow
-------------
Step 1:  Simulate Lorenz system across a parameter sweep
Step 2:  Estimate instability region from eigenvalue analysis
Step 3:  Build phase-space reconstruction (PSR)
Step 4:  Assign 3 regime labels
Step 5:  Build multitask dataset
Step 6:  Train hierarchical multitask model with grouped CV
Step 7:  Evaluate classification + forecasting
Step 8:  Save dataset, metrics, and plots
Step 9:  Run advanced early-warning evaluation

Outputs
-------
- psr_ssm_outputs/psr_multitask_dataset.csv
- psr_ssm_outputs/metrics_summary.json
- psr_ssm_outputs/plots/*.png
- psr_ssm_outputs/evaluation/*.csv
- psr_ssm_outputs/evaluation/*.json
- psr_ssm_outputs/evaluation/*.png
========================================================================
"""

from __future__ import annotations

# ======================================================================
# STEP 0: IMPORTS
# ======================================================================

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional
import math
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.stats import entropy

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping


# ======================================================================
# STEP 0: GLOBAL CONSTANTS
# ======================================================================

# 3-class regime setup
CLASS_NAMES = ["stable", "transitional_oscillatory", "chaotic"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Hierarchy:
# coarse:
#   0 = nonchaotic
#   1 = chaotic
#
# fine inside nonchaotic:
#   0 = stable
#   1 = transitional_oscillatory

COARSE_CLASS_NAMES = ["nonchaotic", "chaotic"]
NONCHAOTIC_FINE_CLASS_NAMES = ["stable", "transitional_oscillatory"]

ORIG_TO_COARSE = {
    0: 0,  # stable -> nonchaotic
    1: 0,  # transitional_oscillatory -> nonchaotic
    2: 1,  # chaotic -> chaotic
}

ORIG_TO_NONCHAOTIC_FINE = {
    0: 0,  # stable
    1: 1,  # transitional_oscillatory
}


# ======================================================================
# STEP 0: CONFIGURATION
# ======================================================================

@dataclass
class Config:
    # Lorenz parameters
    a: float = 10.0
    c: float = 8.0 / 3.0

    # Simulation settings
    t_start: float = 0.0
    t_end: float = 60.0
    num_points: int = 6000
    transient_fraction: float = 0.20

    # Parameter sweep for b
    b_values: Tuple[float, ...] = tuple(np.concatenate([
        np.linspace(1.0, 20.0, 8),
        np.linspace(20.0, 30.0, 21),
        np.linspace(30.0, 35.0, 6),
    ]))

    # Initial conditions
    initial_conditions: Tuple[Tuple[float, float, float], ...] = (
        (0.1, 0.3, 1.0),
        (0.10001, 0.3, 1.0),
        (0.15, 0.3, 1.0),
        (-1.0, 3.0, 4.0),
    )

    # PSR settings
    psr_variable: str = "x"
    embedding_dim: int = 3
    delay_tau: int = 15

    # Window settings
    window_size: int = 150
    future_horizon: int = 20
    stride: int = 10

    # Regime labeling heuristics
    near_margin_b: float = 1.0
    chaos_anchor_b: float = 28.0
    stable_hi_k: float = 1.5

    # Model settings
    state_dim: int = 64
    hidden_dim: int = 64
    dropout_rate: float = 0.20
    attention_hidden_dim: int = 32

    # Training settings
    n_splits: int = 5
    batch_size: int = 32
    epochs: int = 25
    validation_split: float = 0.20
    early_stopping_patience: int = 5
    learning_rate: float = 1e-3
    random_seed: int = 42

    # Multitask loss weights
    lambda_class: float = 1.0
    lambda_forecast: float = 1.0
    lambda_phase: float = 0.2

    # Chaotic threshold selection
    chaos_prob_target_recall: float = 0.90

    # Output settings
    output_dir: str = "psr_ssm_outputs"
    dataset_csv: str = "psr_multitask_dataset.csv"
    metrics_json: str = "metrics_summary.json"

    # PCA for hybrid-ready features
    n_reduced_features: int = 6


# ======================================================================
# STEP 1: LORENZ SYSTEM
# ======================================================================

def lorenz_rhs(t: float, state: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Lorenz equations:
        dx/dt = a(y - x)
        dy/dt = x(b - z) - y
        dz/dt = x*y - c*z
    """
    x, y, z = state
    dx = a * (y - x)
    dy = x * (b - z) - y
    dz = x * y - c * z
    return np.array([dx, dy, dz], dtype=float)


def simulate_lorenz(
    a: float,
    b: float,
    c: float,
    initial_state: Tuple[float, float, float],
    t_start: float,
    t_end: float,
    num_points: int,
) -> Dict[str, np.ndarray]:
    """
    Numerically integrate the Lorenz system.
    """
    t_eval = np.linspace(t_start, t_end, num_points)

    sol = solve_ivp(
        fun=lambda t, y: lorenz_rhs(t, y, a=a, b=b, c=c),
        t_span=(t_start, t_end),
        y0=np.array(initial_state, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed for b={b}, ic={initial_state}: {sol.message}")

    return {
        "t": sol.t,
        "x": sol.y[0],
        "y": sol.y[1],
        "z": sol.y[2],
    }


def run_parameter_sweep(cfg: Config) -> List[Dict[str, object]]:
    """
    Run simulations for all b values and all initial conditions.
    """
    all_runs: List[Dict[str, object]] = []

    for b in cfg.b_values:
        for ic_id, ic in enumerate(cfg.initial_conditions):
            sim = simulate_lorenz(
                a=cfg.a,
                b=b,
                c=cfg.c,
                initial_state=ic,
                t_start=cfg.t_start,
                t_end=cfg.t_end,
                num_points=cfg.num_points,
            )
            all_runs.append({
                "b": float(b),
                "ic_id": ic_id,
                "initial_condition": ic,
                "simulation": sim,
            })

    return all_runs


# ======================================================================
# STEP 2: EQUILIBRIA AND EIGENVALUE ANALYSIS
# ======================================================================

def lorenz_equilibria(a: float, b: float, c: float) -> List[np.ndarray]:
    """
    Lorenz equilibrium points.
    """
    eqs = [np.array([0.0, 0.0, 0.0], dtype=float)]
    if b > 1.0:
        s = math.sqrt(c * (b - 1.0))
        eqs.append(np.array([s, s, b - 1.0], dtype=float))
        eqs.append(np.array([-s, -s, b - 1.0], dtype=float))
    return eqs


def lorenz_jacobian_at_point(point: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Jacobian of Lorenz system at point (x, y, z).
    """
    x, y, z = point
    return np.array([
        [-a,   a,   0.0],
        [b-z, -1.0, -x ],
        [y,    x,   -c ],
    ], dtype=float)


def equilibrium_eigenvalues(a: float, b: float, c: float) -> Dict[str, np.ndarray]:
    """
    Compute eigenvalues at all equilibrium points.
    """
    eqs = lorenz_equilibria(a, b, c)
    names = ["E0", "E_plus", "E_minus"]
    out: Dict[str, np.ndarray] = {}

    for name, point in zip(names, eqs):
        J = lorenz_jacobian_at_point(point, a, b, c)
        out[name] = np.linalg.eigvals(J)

    return out


def critical_b_hopf_formula(a: float, c: float) -> Optional[float]:
    """
    Classical Lorenz Hopf-like critical estimate:
        b = a(a + c + 3) / (a - c - 1)
    """
    denom = a - c - 1.0
    if denom <= 0:
        return None
    return a * (a + c + 3.0) / denom


def scan_eigenvalues_over_b(a: float, c: float, b_values: Iterable[float]) -> pd.DataFrame:
    """
    Summarize eigenvalues over b using maximum real part.
    """
    rows = []
    for b in b_values:
        eigs = equilibrium_eigenvalues(a, b, c)
        for eq_name, vals in eigs.items():
            rows.append({
                "b": float(b),
                "equilibrium": eq_name,
                "max_real_part": float(np.max(np.real(vals))),
                "eigvals": vals,
            })
    return pd.DataFrame(rows)


# ======================================================================
# STEP 3: PREPROCESSING AND PSR
# ======================================================================

def remove_transient(simulation: Dict[str, np.ndarray], transient_fraction: float) -> Dict[str, np.ndarray]:
    """
    Remove the initial transient region from a trajectory.
    """
    n = len(simulation["t"])
    cut = int(n * transient_fraction)

    return {
        "t": simulation["t"][cut:],
        "x": simulation["x"][cut:],
        "y": simulation["y"][cut:],
        "z": simulation["z"][cut:],
    }


def phase_space_reconstruct(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Phase-space reconstruction:
        Y(t) = [x(t), x(t-tau), x(t-2tau), ..., x(t-(m-1)tau)]
    """
    max_lag = (m - 1) * tau
    if len(series) <= max_lag:
        raise ValueError("Series too short for requested PSR parameters.")

    vectors = []
    for t in range(max_lag, len(series)):
        vec = [series[t - k * tau] for k in range(m)]
        vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


def align_series_after_psr(sim: Dict[str, np.ndarray], m: int, tau: int) -> Dict[str, np.ndarray]:
    """
    Align original time series to match the PSR sequence length.
    """
    max_lag = (m - 1) * tau
    return {
        "t": sim["t"][max_lag:],
        "x": sim["x"][max_lag:],
        "y": sim["y"][max_lag:],
        "z": sim["z"][max_lag:],
    }


def build_psr_run(cfg: Config, sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Build PSR from the chosen scalar variable.
    """
    if cfg.psr_variable not in sim:
        raise ValueError(f"psr_variable must be one of x,y,z; got {cfg.psr_variable}")

    scalar_series = sim[cfg.psr_variable]
    psr = phase_space_reconstruct(scalar_series, cfg.embedding_dim, cfg.delay_tau)
    aligned = align_series_after_psr(sim, cfg.embedding_dim, cfg.delay_tau)

    return {
        "psr": psr,
        "t": aligned["t"],
        "x": aligned["x"],
        "y": aligned["y"],
        "z": aligned["z"],
    }


def build_psr_windows(
    psr_run: Dict[str, np.ndarray],
    window_size: int,
    future_horizon: int,
    stride: int,
) -> List[Dict[str, np.ndarray]]:
    """
    Build input windows and future targets on the PSR trajectory.
    """
    psr = psr_run["psr"]
    t = psr_run["t"]
    x = psr_run["x"]

    n = len(t)
    windows = []

    end_limit = n - window_size - future_horizon
    for i in range(0, max(end_limit, 0), stride):
        in_slice = slice(i, i + window_size)
        fut_slice = slice(i + window_size, i + window_size + future_horizon)

        windows.append({
            "psr_input": psr[in_slice],
            "x_future": x[fut_slice],
            "psr_future": psr[fut_slice],
            "t_input": t[in_slice],
            "t_future": t[fut_slice],
            "window_end_time": float(t[i + window_size - 1]),
            "future_start_time": float(t[i + window_size]),
        })

    return windows


# ======================================================================
# STEP 4: 3-CLASS LABELING
# ======================================================================

def zero_crossing_rate(signal: np.ndarray) -> float:
    """
    Fraction of sign changes in a 1D signal.
    """
    s = np.sign(signal)
    changes = np.sum(s[1:] * s[:-1] < 0)
    return float(changes / max(len(signal) - 1, 1))


def spectral_entropy_1d(signal: np.ndarray) -> float:
    """
    Normalized spectral entropy of a 1D signal.
    """
    sig = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(sig))
    total = np.sum(fft_vals)
    if total == 0:
        return 0.0
    p = fft_vals / total
    h = entropy(p, base=2)
    max_h = np.log2(len(p)) if len(p) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def future_window_stats(window: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute future-window statistics used for regime labeling.
    """
    xf = window["x_future"]
    psrf = window["psr_future"]

    variability = float(np.std(xf))
    amplitude = float(np.max(xf) - np.min(xf))
    switch_rate = float(zero_crossing_rate(xf))
    irregularity = float(spectral_entropy_1d(xf))
    psr_variation = float(np.mean(np.std(psrf, axis=0)))

    combined_metric = variability + 0.5 * switch_rate + 0.5 * irregularity + 0.25 * psr_variation

    return {
        "variability": variability,
        "amplitude": amplitude,
        "switch_rate": switch_rate,
        "irregularity": irregularity,
        "psr_variation": psr_variation,
        "combined_metric": combined_metric,
    }


def estimate_stable_metric_stats(cfg: Config, sweep_results: List[Dict[str, object]], b_hopf_est: float) -> Dict[str, float]:
    """
    Estimate future metric statistics from clearly stable runs.
    """
    stable_metrics = []
    stable_b_cutoff = b_hopf_est - cfg.near_margin_b

    for run in sweep_results:
        if float(run["b"]) > stable_b_cutoff:
            continue

        sim = remove_transient(run["simulation"], cfg.transient_fraction)
        psr_run = build_psr_run(cfg, sim)
        windows = build_psr_windows(psr_run, cfg.window_size, cfg.future_horizon, cfg.stride)

        for w in windows:
            stats = future_window_stats(w)
            stable_metrics.append(stats["combined_metric"])

    if len(stable_metrics) == 0:
        raise ValueError("No stable metrics available. Adjust sweep or margin settings.")

    stable_metrics = np.array(stable_metrics, dtype=float)
    return {
        "mean": float(np.mean(stable_metrics)),
        "std": float(np.std(stable_metrics)),
        "p95": float(np.percentile(stable_metrics, 95)),
        "p99": float(np.percentile(stable_metrics, 99)),
    }


def assign_regime_label(
    b: float,
    future_stats: Dict[str, float],
    b_hopf_est: float,
    stable_metric_stats: Dict[str, float],
    cfg: Config,
) -> int:
    """
    Assign one of the 3 classes:
      0 = stable
      1 = transitional_oscillatory
      2 = chaotic
    """
    M = future_stats["combined_metric"]
    irregularity = future_stats["irregularity"]
    switch_rate = future_stats["switch_rate"]

    stable_hi = stable_metric_stats["mean"] + cfg.stable_hi_k * stable_metric_stats["std"]

    # Stable
    if (b <= b_hopf_est - cfg.near_margin_b) and (M <= stable_hi):
        return CLASS_TO_INDEX["stable"]

    # Chaotic
    if (b >= cfg.chaos_anchor_b) or (irregularity >= 0.75 and switch_rate >= 0.05):
        return CLASS_TO_INDEX["chaotic"]

    # Everything between stable and chaotic becomes transitional_oscillatory
    return CLASS_TO_INDEX["transitional_oscillatory"]


# ======================================================================
# STEP 5: DATASET BUILDING
# ======================================================================

def extract_psr_summary_features(psr_window: np.ndarray, b_value: float) -> Dict[str, float]:
    """
    Extract handcrafted summary features from a PSR input window.
    """
    feats = {"b": float(b_value)}
    m = psr_window.shape[1]

    for j in range(m):
        col = psr_window[:, j]
        feats[f"psr{j+1}_mean"] = float(np.mean(col))
        feats[f"psr{j+1}_std"] = float(np.std(col))
        feats[f"psr{j+1}_amp"] = float(np.max(col) - np.min(col))
        feats[f"psr{j+1}_rms"] = float(np.sqrt(np.mean(col ** 2)))
        feats[f"psr{j+1}_zcr"] = float(zero_crossing_rate(col))
        feats[f"psr{j+1}_entropy"] = float(spectral_entropy_1d(col))

    return feats


def build_multitask_dataset(
    cfg: Config,
    sweep_results: List[Dict[str, object]],
    b_hopf_est: float,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build:
      dataset_df  : tabular summary dataset
      X_seq       : PSR input sequences
      y_class     : 3-class labels
      y_forecast  : future x targets
      y_psr       : future PSR targets
      groups      : run_id groups for GroupKFold
    """
    stable_metric_stats = estimate_stable_metric_stats(cfg, sweep_results, b_hopf_est)

    rows = []
    X_seq = []
    y_class = []
    y_forecast = []
    y_psr = []
    groups = []

    for run_id, run in enumerate(sweep_results):
        b = float(run["b"])
        sim = remove_transient(run["simulation"], cfg.transient_fraction)
        psr_run = build_psr_run(cfg, sim)
        windows = build_psr_windows(psr_run, cfg.window_size, cfg.future_horizon, cfg.stride)

        for w_id, w in enumerate(windows):
            stats = future_window_stats(w)
            label = assign_regime_label(
                b=b,
                future_stats=stats,
                b_hopf_est=b_hopf_est,
                stable_metric_stats=stable_metric_stats,
                cfg=cfg,
            )

            X_seq.append(w["psr_input"])
            y_class.append(label)
            y_forecast.append(w["x_future"])
            y_psr.append(w["psr_future"])
            groups.append(run_id)

            row = {
                "run_id": run_id,
                "window_id": w_id,
                "b": b,
                "ic_id": int(run["ic_id"]),
                "window_end_time": w["window_end_time"],
                "future_start_time": w["future_start_time"],
                "future_variability": stats["variability"],
                "future_amplitude": stats["amplitude"],
                "future_switch_rate": stats["switch_rate"],
                "future_irregularity": stats["irregularity"],
                "future_psr_variation": stats["psr_variation"],
                "future_metric": stats["combined_metric"],
                "label": label,
                "label_name": CLASS_NAMES[label],
            }
            row.update(extract_psr_summary_features(w["psr_input"], b))
            rows.append(row)

    dataset_df = pd.DataFrame(rows)
    X_seq = np.array(X_seq, dtype=np.float32)
    y_class = np.array(y_class, dtype=np.int32)
    y_forecast = np.array(y_forecast, dtype=np.float32)
    y_psr = np.array(y_psr, dtype=np.float32)
    groups = np.array(groups, dtype=np.int32)

    return dataset_df, X_seq, y_class, y_forecast, y_psr, groups, stable_metric_stats


def save_dataset_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the summary dataset to CSV.
    """
    df.to_csv(filepath, index=False)
    print(f"Saved dataset to: {filepath}")


def get_summary_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all handcrafted feature columns used for PCA reduction.
    """
    exclude = {
        "run_id", "window_id", "ic_id",
        "window_end_time", "future_start_time",
        "label", "label_name"
    }
    return [c for c in df.columns if c not in exclude]


# ======================================================================
# STEP 6: HIERARCHICAL TARGETS AND SAMPLE WEIGHTS
# ======================================================================

def make_hierarchical_targets(y_class: np.ndarray):
    """
    Convert original 3-class labels into:
      - coarse labels: 0=nonchaotic, 1=chaotic
      - fine labels inside nonchaotic: 0=stable, 1=transitional_oscillatory
    """
    y_coarse = np.array([ORIG_TO_COARSE[int(y)] for y in y_class], dtype=np.int32)

    y_nonchaotic_fine = np.zeros_like(y_class, dtype=np.int32)
    for i, y in enumerate(y_class):
        y = int(y)
        if y in ORIG_TO_NONCHAOTIC_FINE:
            y_nonchaotic_fine[i] = ORIG_TO_NONCHAOTIC_FINE[y]

    return y_coarse, y_nonchaotic_fine


def inverse_frequency_class_weights(y: np.ndarray, n_classes: int) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights.
    """
    counts = np.bincount(y, minlength=n_classes)
    total = counts.sum()

    weights = {}
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = float(total / (n_classes * counts[c]))
        else:
            weights[c] = 0.0
    return weights


def build_sample_weights_for_hierarchy(y_class: np.ndarray):
    """
    Build sample weights for:
      - coarse classifier: nonchaotic vs chaotic
      - nonchaotic fine classifier: stable vs transitional_oscillatory
    """
    y_coarse, y_nonchaotic_fine = make_hierarchical_targets(y_class)

    coarse_weights = inverse_frequency_class_weights(y_coarse, n_classes=2)

    nonchaotic_mask = np.isin(
        y_class,
        [CLASS_TO_INDEX["stable"], CLASS_TO_INDEX["transitional_oscillatory"]]
    ).astype(np.float32)

    if np.sum(nonchaotic_mask) > 0:
        nonchaotic_weights = inverse_frequency_class_weights(
            y_nonchaotic_fine[nonchaotic_mask == 1],
            n_classes=2,
        )
    else:
        nonchaotic_weights = {0: 0.0, 1: 0.0}

    sw_coarse = np.array([coarse_weights[int(c)] for c in y_coarse], dtype=np.float32)

    sw_nonchaotic_fine = np.zeros_like(y_class, dtype=np.float32)
    for i, y in enumerate(y_class):
        if int(y) in ORIG_TO_NONCHAOTIC_FINE:
            sw_nonchaotic_fine[i] = nonchaotic_weights[int(y_nonchaotic_fine[i])]

    return {
        "y_coarse": y_coarse,
        "y_nonchaotic_fine": y_nonchaotic_fine,
        "sw_coarse": sw_coarse,
        "sw_nonchaotic_fine": sw_nonchaotic_fine,
        "coarse_class_weights": coarse_weights,
        "nonchaotic_class_weights": nonchaotic_weights,
    }


def combine_hierarchical_probabilities(
    p_coarse: np.ndarray,
    p_nonchaotic: np.ndarray,
) -> np.ndarray:
    """
    Combine hierarchical probabilities into final 3-class probabilities.

    Output ordering:
      [stable, transitional_oscillatory, chaotic]
    """
    p_final = np.zeros((p_coarse.shape[0], 3), dtype=np.float32)

    p_final[:, CLASS_TO_INDEX["stable"]] = p_coarse[:, 0] * p_nonchaotic[:, 0]
    p_final[:, CLASS_TO_INDEX["transitional_oscillatory"]] = p_coarse[:, 0] * p_nonchaotic[:, 1]
    p_final[:, CLASS_TO_INDEX["chaotic"]] = p_coarse[:, 1]

    return p_final


# ======================================================================
# STEP 6: ATTENTION POOLING LAYER
# ======================================================================

class AttentionPooling1D(layers.Layer):
    """
    Attention pooling over time steps.

    Input:
        sequence tensor of shape (batch, time, features)

    Output:
        context vector of shape (batch, features)
    """

    def __init__(self, hidden_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.W = self.add_weight(
            shape=(feature_dim, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_W",
        )
        self.b = self.add_weight(
            shape=(self.hidden_dim,),
            initializer="zeros",
            trainable=True,
            name="attn_b",
        )
        self.v = self.add_weight(
            shape=(self.hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_v",
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (B, T, F)
        score_hidden = tf.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        scores = tf.tensordot(score_hidden, self.v, axes=[[2], [0]])   # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)                        # (B, T, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)              # (B, F)
        return context


# ======================================================================
# STEP 6: MODEL COMPONENTS
# ======================================================================

class SimpleStateSpaceCell(layers.Layer):
    """
    Lightweight trainable state-space-style recurrent cell:
        h_t = tanh(x_t W_x + h_{t-1} W_h + b)
    """

    def __init__(self, state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.state_size = state_dim
        self.output_size = state_dim

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.W_x = self.add_weight(
            shape=(input_dim, self.state_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W_x",
        )
        self.W_h = self.add_weight(
            shape=(self.state_dim, self.state_dim),
            initializer="orthogonal",
            trainable=True,
            name="W_h",
        )
        self.b = self.add_weight(
            shape=(self.state_dim,),
            initializer="zeros",
            trainable=True,
            name="b",
        )
        super().build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]
        h = tf.nn.tanh(tf.matmul(inputs, self.W_x) + tf.matmul(h_prev, self.W_h) + self.b)
        return h, [h]


def build_hierarchical_multitask_state_space_model_last_plus_attention(cfg: Config, input_dim: int) -> Model:
    """
    3-class hierarchical multitask model.

    Classification:
      - coarse head: nonchaotic vs chaotic
      - fine nonchaotic head: stable vs transitional_oscillatory

    Representation:
      - classification uses: last_state + attention_context
      - forecasting uses: last_state only
      - PSR future uses: last_state only
    """
    seq_in = layers.Input(shape=(cfg.window_size, input_dim), name="psr_input")

    # Shared encoder outputs full hidden-state sequence
    encoder_states = layers.RNN(
        SimpleStateSpaceCell(cfg.state_dim),
        return_sequences=True,
        name="ssm_encoder"
    )(seq_in)
    encoder_states = layers.Dropout(cfg.dropout_rate)(encoder_states)

    # Recent dynamics summary for forecasting
    last_state = layers.Lambda(lambda t: t[:, -1, :], name="last_state")(encoder_states)

    # Broader window summary for classification
    attention_context = AttentionPooling1D(
        hidden_dim=cfg.attention_hidden_dim,
        name="attention_pool"
    )(encoder_states)

    classification_context = layers.Concatenate(name="classification_context")(
        [last_state, attention_context]
    )

    # Coarse head: nonchaotic vs chaotic
    hc = layers.Dense(cfg.hidden_dim, activation="relu")(classification_context)
    hc = layers.Dropout(cfg.dropout_rate)(hc)
    coarse_out = layers.Dense(2, activation="softmax", name="coarse_output")(hc)

    # Fine nonchaotic head: stable vs transitional_oscillatory
    hn = layers.Dense(cfg.hidden_dim, activation="relu")(classification_context)
    hn = layers.Dropout(cfg.dropout_rate)(hn)
    nonchaotic_out = layers.Dense(2, activation="softmax", name="nonchaotic_fine_output")(hn)

    # Forecast branch uses last hidden state only
    hf = layers.Dense(cfg.hidden_dim, activation="relu")(last_state)
    hf = layers.Dropout(cfg.dropout_rate)(hf)
    forecast_out = layers.Dense(cfg.future_horizon, activation="linear", name="forecast_output")(hf)

    # Future PSR branch also uses last hidden state only
    hp = layers.Dense(cfg.hidden_dim, activation="relu")(last_state)
    hp = layers.Dropout(cfg.dropout_rate)(hp)
    psr_flat_dim = cfg.future_horizon * input_dim
    psr_flat = layers.Dense(psr_flat_dim, activation="linear")(hp)
    psr_out = layers.Reshape((cfg.future_horizon, input_dim), name="psr_output")(psr_flat)

    model = Model(
        inputs=seq_in,
        outputs=[coarse_out, nonchaotic_out, forecast_out, psr_out],
        name="hierarchical_psr_state_space_3class_last_plus_attention"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=[
            "sparse_categorical_crossentropy",  # coarse_output
            "sparse_categorical_crossentropy",  # nonchaotic_fine_output
            "mse",                              # forecast_output
            "mse",                              # psr_output
        ],
        loss_weights=[
            cfg.lambda_class,
            cfg.lambda_class,
            cfg.lambda_forecast,
            cfg.lambda_phase,
        ],
        metrics=[
            ["accuracy"],
            ["accuracy"],
            ["mae"],
            ["mae"],
        ],
    )

    return model


def normalize_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize input sequences using train-set statistics only.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def make_explicit_validation_split(
    X: np.ndarray,
    yc: np.ndarray,
    yf: np.ndarray,
    yp: np.ndarray,
    train_h: Dict[str, np.ndarray],
    val_fraction: float,
    seed: int,
):
    """
    Create an explicit train/validation split.

    This avoids Keras internal split issues with multi-output sample weights.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = max(1, int(n * val_fraction))
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]

    X_tr = X[tr_idx]
    X_val = X[val_idx]

    yc_tr = train_h["y_coarse"][tr_idx]
    yc_val = train_h["y_coarse"][val_idx]

    yn_tr = train_h["y_nonchaotic_fine"][tr_idx]
    yn_val = train_h["y_nonchaotic_fine"][val_idx]

    yf_tr = yf[tr_idx]
    yf_val = yf[val_idx]

    yp_tr = yp[tr_idx]
    yp_val = yp[val_idx]

    swc_tr = train_h["sw_coarse"][tr_idx]
    swc_val = train_h["sw_coarse"][val_idx]

    swn_tr = train_h["sw_nonchaotic_fine"][tr_idx]
    swn_val = train_h["sw_nonchaotic_fine"][val_idx]

    return {
        "X_tr": X_tr,
        "X_val": X_val,
        "y_tr_list": [yc_tr, yn_tr, yf_tr, yp_tr],
        "y_val_list": [yc_val, yn_val, yf_val, yp_val],
        "sw_tr_list": [
            swc_tr,
            swn_tr,
            np.ones((len(X_tr), 1), dtype=np.float32),
            np.ones((len(X_tr), 1, 1), dtype=np.float32),
        ],
        "sw_val_list": [
            swc_val,
            swn_val,
            np.ones((len(X_val), 1), dtype=np.float32),
            np.ones((len(X_val), 1, 1), dtype=np.float32),
        ],
    }


# ======================================================================
# STEP 7: METRICS AND THRESHOLDS
# ======================================================================

def multiclass_false_alarm_rate(cm: np.ndarray, positive_class: int) -> float:
    """
    One-vs-rest false alarm rate for a chosen class.
    """
    tp = cm[positive_class, positive_class]
    fp = cm[:, positive_class].sum() - tp
    tn = cm.sum() - cm[positive_class, :].sum() - fp
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean absolute percentage error.
    """
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100.0)


def choose_chaotic_threshold_by_recall_constraint(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.90,
) -> float:
    """
    Choose a chaotic-class probability threshold that achieves at least a
    target recall in a one-vs-rest setting.
    """
    chaos_idx = CLASS_TO_INDEX["chaotic"]
    y_bin = (y_true == chaos_idx).astype(int)
    p = y_prob[:, chaos_idx]

    precision, recall, thresholds = precision_recall_curve(y_bin, p)

    valid_thresholds = []
    for i, thr in enumerate(thresholds):
        if recall[i] >= min_recall:
            valid_thresholds.append(float(thr))

    if not valid_thresholds:
        return 0.50

    return min(valid_thresholds)


# ======================================================================
# STEP 6+7: GROUPED CV TRAINING
# ======================================================================

def train_hierarchical_multitask_model_grouped_cv_last_plus_attention(
    cfg: Config,
    X_seq: np.ndarray,
    y_class: np.ndarray,
    y_forecast: np.ndarray,
    y_psr: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    """
    Train the 3-class hierarchical multitask model with:
      - grouped cross-validation
      - class-weighted classification
      - classification using last state + attention
      - forecasting using last state only
    """
    gkf = GroupKFold(n_splits=cfg.n_splits)

    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    all_forecast_true = []
    all_forecast_pred = []
    all_test_indices = []

    n_classes = len(CLASS_NAMES)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_seq, y_class, groups=groups), start=1):
        print(f"  -> Training fold {fold_idx}/{cfg.n_splits}")

        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        yc_train, yc_test = y_class[train_idx], y_class[test_idx]
        yf_train, yf_test = y_forecast[train_idx], y_forecast[test_idx]
        yp_train, yp_test = y_psr[train_idx], y_psr[test_idx]

        # Normalize input only
        X_train, X_test, _, _ = normalize_sequences(X_train, X_test)

        # Build hierarchical targets and weights
        train_h = build_sample_weights_for_hierarchy(yc_train)

        # Explicit train/validation split
        split = make_explicit_validation_split(
            X=X_train,
            yc=yc_train,
            yf=yf_train,
            yp=yp_train,
            train_h=train_h,
            val_fraction=cfg.validation_split,
            seed=cfg.random_seed + fold_idx,
        )

        model = build_hierarchical_multitask_state_space_model_last_plus_attention(
            cfg, input_dim=X_train.shape[2]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            split["X_tr"],
            split["y_tr_list"],
            sample_weight=split["sw_tr_list"],
            validation_data=(split["X_val"], split["y_val_list"], split["sw_val_list"]),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=[early_stop],
        )

        # Predict
        pred_outputs = model.predict(X_test, verbose=0)
        p_coarse, p_nonchaotic, pred_forecast, pred_psr = pred_outputs

        pred_class_prob = combine_hierarchical_probabilities(p_coarse, p_nonchaotic)
        pred_class = np.argmax(pred_class_prob, axis=1)

        cm = confusion_matrix(yc_test, pred_class, labels=np.arange(n_classes))

        report = classification_report(
            yc_test,
            pred_class,
            labels=np.arange(n_classes),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        yc_test_bin = label_binarize(yc_test, classes=np.arange(n_classes))
        try:
            roc_auc_macro = roc_auc_score(yc_test_bin, pred_class_prob, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc_macro = np.nan

        forecast_rmse = rmse(yf_test.ravel(), pred_forecast.ravel())
        forecast_mae = float(mean_absolute_error(yf_test.ravel(), pred_forecast.ravel()))
        forecast_mape = mape(yf_test.ravel(), pred_forecast.ravel())

        chaotic_far = multiclass_false_alarm_rate(cm, positive_class=CLASS_TO_INDEX["chaotic"])

        fold_result = {
            "fold": fold_idx,
            "model": model,
            "y_test": yc_test,
            "y_prob": pred_class_prob,
            "y_pred": pred_class,
            "forecast_true": yf_test,
            "forecast_pred": pred_forecast,
            "psr_true": yp_test,
            "psr_pred": pred_psr,
            "test_indices": test_idx,
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": accuracy_score(yc_test, pred_class),
            "precision_macro": precision_score(yc_test, pred_class, average="macro", zero_division=0),
            "recall_macro": recall_score(yc_test, pred_class, average="macro", zero_division=0),
            "f1_macro": f1_score(yc_test, pred_class, average="macro", zero_division=0),
            "precision_weighted": precision_score(yc_test, pred_class, average="weighted", zero_division=0),
            "recall_weighted": recall_score(yc_test, pred_class, average="weighted", zero_division=0),
            "f1_weighted": f1_score(yc_test, pred_class, average="weighted", zero_division=0),
            "roc_auc_macro_ovr": roc_auc_macro,
            "chaotic_false_alarm_rate": chaotic_far,
            "forecast_rmse": forecast_rmse,
            "forecast_mae": forecast_mae,
            "forecast_mape": forecast_mape,
        }
        fold_results.append(fold_result)

        all_y_true.append(yc_test)
        all_y_prob.append(pred_class_prob)
        all_y_pred.append(pred_class)
        all_forecast_true.append(yf_test)
        all_forecast_pred.append(pred_forecast)
        all_test_indices.append(test_idx)

    all_y_true = np.concatenate(all_y_true)
    all_y_prob = np.concatenate(all_y_prob)
    all_y_pred = np.concatenate(all_y_pred)
    all_forecast_true = np.concatenate(all_forecast_true)
    all_forecast_pred = np.concatenate(all_forecast_pred)
    all_test_indices = np.concatenate(all_test_indices)

    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(n_classes))
    overall_report = classification_report(
        all_y_true,
        all_y_pred,
        labels=np.arange(n_classes),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    all_y_true_bin = label_binarize(all_y_true, classes=np.arange(n_classes))
    try:
        overall_roc_auc_macro = roc_auc_score(all_y_true_bin, all_y_prob, multi_class="ovr", average="macro")
    except ValueError:
        overall_roc_auc_macro = np.nan

    summary_metrics = {
        "accuracy_mean": float(np.mean([fr["accuracy"] for fr in fold_results])),
        "accuracy_std": float(np.std([fr["accuracy"] for fr in fold_results])),
        "precision_macro_mean": float(np.mean([fr["precision_macro"] for fr in fold_results])),
        "precision_macro_std": float(np.std([fr["precision_macro"] for fr in fold_results])),
        "recall_macro_mean": float(np.mean([fr["recall_macro"] for fr in fold_results])),
        "recall_macro_std": float(np.std([fr["recall_macro"] for fr in fold_results])),
        "f1_macro_mean": float(np.mean([fr["f1_macro"] for fr in fold_results])),
        "f1_macro_std": float(np.std([fr["f1_macro"] for fr in fold_results])),
        "precision_weighted_mean": float(np.mean([fr["precision_weighted"] for fr in fold_results])),
        "precision_weighted_std": float(np.std([fr["precision_weighted"] for fr in fold_results])),
        "recall_weighted_mean": float(np.mean([fr["recall_weighted"] for fr in fold_results])),
        "recall_weighted_std": float(np.std([fr["recall_weighted"] for fr in fold_results])),
        "f1_weighted_mean": float(np.mean([fr["f1_weighted"] for fr in fold_results])),
        "f1_weighted_std": float(np.std([fr["f1_weighted"] for fr in fold_results])),
        "roc_auc_macro_ovr_mean": float(np.nanmean([fr["roc_auc_macro_ovr"] for fr in fold_results])),
        "roc_auc_macro_ovr_std": float(np.nanstd([fr["roc_auc_macro_ovr"] for fr in fold_results])),
        "chaotic_false_alarm_rate_mean": float(np.mean([fr["chaotic_false_alarm_rate"] for fr in fold_results])),
        "chaotic_false_alarm_rate_std": float(np.std([fr["chaotic_false_alarm_rate"] for fr in fold_results])),
        "forecast_rmse_mean": float(np.mean([fr["forecast_rmse"] for fr in fold_results])),
        "forecast_rmse_std": float(np.std([fr["forecast_rmse"] for fr in fold_results])),
        "forecast_mae_mean": float(np.mean([fr["forecast_mae"] for fr in fold_results])),
        "forecast_mae_std": float(np.std([fr["forecast_mae"] for fr in fold_results])),
        "forecast_mape_mean": float(np.mean([fr["forecast_mape"] for fr in fold_results])),
        "forecast_mape_std": float(np.std([fr["forecast_mape"] for fr in fold_results])),
    }

    return {
        "fold_results": fold_results,
        "all_y_true": all_y_true,
        "all_y_prob": all_y_prob,
        "all_y_pred": all_y_pred,
        "all_forecast_true": all_forecast_true,
        "all_forecast_pred": all_forecast_pred,
        "all_test_indices": all_test_indices,
        "overall_confusion_matrix": overall_cm,
        "overall_classification_report": overall_report,
        "overall_roc_auc_macro_ovr": overall_roc_auc_macro,
        "summary_metrics": summary_metrics,
    }


# ======================================================================
# STEP 8: PCA REDUCTION FOR HYBRID-READY FEATURES
# ======================================================================

def reduce_features_for_hybrid(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_components: int = 6,
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce handcrafted summary features using PCA.
    """
    X = df[feature_columns].values
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_scaled = (X - mean) / std

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca


def build_hybrid_model_placeholder(input_dim: int) -> None:
    """
    Placeholder for future hybrid quantum-classical model input.
    """
    print(f"[Placeholder] Hybrid quantum-classical model would take input_dim={input_dim}")


# ======================================================================
# STEP 8: OUTPUT DIRECTORY HELPERS
# ======================================================================

def ensure_output_dir(cfg: Config) -> Path:
    """
    Ensure main output directory exists.
    """
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def ensure_plot_dir(output_dir: Path) -> Path:
    """
    Ensure plot directory exists.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def save_metrics_json(metrics: Dict[str, object], filepath: Path) -> None:
    """
    Save metrics summary to JSON.
    """
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {filepath}")


def finalize_plot(save_path: Optional[Path] = None, show: bool = False) -> None:
    """
    Save and/or show current plot, then close it.
    """
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


# ======================================================================
# STEP 9: PLOT FUNCTIONS - PHYSICS AND MODEL
# ======================================================================

def plot_time_series(sim, title="Lorenz time series", save_path: Optional[Path] = None, show: bool = False):
    plt.figure(figsize=(10, 5))
    plt.plot(sim["t"], sim["x"], label="x(t)")
    plt.plot(sim["t"], sim["y"], label="y(t)")
    plt.plot(sim["t"], sim["z"], label="z(t)")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_lorenz_3d(sim, title="Lorenz attractor", save_path: Optional[Path] = None, show: bool = False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sim["x"], sim["y"], sim["z"], linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    finalize_plot(save_path, show)


def plot_phase_plane(sim, x_key="y", y_key="z", title="Phase plane", save_path: Optional[Path] = None, show: bool = False):
    plt.figure(figsize=(6, 6))
    plt.plot(sim[x_key], sim[y_key], linewidth=0.8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_amplitude_vs_b(sweep_results, transient_fraction=0.2, state_key="x", save_path: Optional[Path] = None, show: bool = False):
    b_vals = []
    amps = []

    for run in sweep_results:
        sim = remove_transient(run["simulation"], transient_fraction)
        amp = np.max(sim[state_key]) - np.min(sim[state_key])
        b_vals.append(run["b"])
        amps.append(amp)

    plt.figure(figsize=(8, 5))
    plt.scatter(b_vals, amps, s=16)
    plt.xlabel("b")
    plt.ylabel(f"Amplitude of {state_key}")
    plt.title(f"Oscillation amplitude of {state_key} vs b")
    plt.grid(True)
    finalize_plot(save_path, show)


def local_maxima(signal):
    """
    Simple local maxima finder.
    """
    idx = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            idx.append(i)
    return np.array(idx, dtype=int)


def plot_bifurcation_diagram(sweep_results, transient_fraction=0.2, state_key="x", save_path: Optional[Path] = None, show: bool = False):
    b_plot = []
    ymax_plot = []

    for run in sweep_results:
        sim = remove_transient(run["simulation"], transient_fraction)
        idx = local_maxima(sim[state_key])
        for i in idx:
            b_plot.append(run["b"])
            ymax_plot.append(sim[state_key][i])

    plt.figure(figsize=(9, 5))
    plt.scatter(b_plot, ymax_plot, s=4)
    plt.xlabel("b")
    plt.ylabel(f"Local maxima of {state_key}")
    plt.title(f"Bifurcation diagram using local maxima of {state_key}")
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_max_real_eigenvalue(eig_df, save_path: Optional[Path] = None, show: bool = False):
    grouped = eig_df.groupby(["b", "equilibrium"])["max_real_part"].max().reset_index()

    plt.figure(figsize=(8, 5))
    for eq_name in grouped["equilibrium"].unique():
        sub = grouped[grouped["equilibrium"] == eq_name]
        plt.plot(sub["b"], sub["max_real_part"], label=eq_name)

    plt.axhline(0, linestyle="--")
    plt.xlabel("b")
    plt.ylabel("Max real part of eigenvalues")
    plt.title("Stability indicator vs b")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_trajectory_separation(sim1, sim2, title="Sensitivity to initial conditions", save_path: Optional[Path] = None, show: bool = False):
    d = np.sqrt(
        (sim1["x"] - sim2["x"])**2 +
        (sim1["y"] - sim2["y"])**2 +
        (sim1["z"] - sim2["z"])**2
    )

    plt.figure(figsize=(8, 5))
    plt.plot(sim1["t"], d)
    plt.xlabel("Time")
    plt.ylabel("Trajectory separation")
    plt.title(title)
    plt.grid(True)
    finalize_plot(save_path, show)


# ------------------------------------------------------------
# True vs predicted trajectory separation versus time
# ------------------------------------------------------------
def plot_true_vs_predicted_separation_vs_time(
    true_future_1: np.ndarray,
    true_future_2: np.ndarray,
    pred_future_1: np.ndarray,
    pred_future_2: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
    title: str = "True vs predicted trajectory separation vs time",
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot trajectory separation versus time for:
      - true future trajectories
      - predicted future trajectories

    This is used to visualize sensitivity to initial conditions
    in the chaotic regime.
    """
    true_sep = np.abs(true_future_1 - true_future_2)
    pred_sep = np.abs(pred_future_1 - pred_future_2)

    n = min(len(true_sep), len(pred_sep))

    if time_axis is None:
        t = np.arange(n)
    else:
        t = np.asarray(time_axis[:n])

    plt.figure(figsize=(9, 5))
    plt.plot(t, true_sep[:n], label="True separation", linewidth=2)
    plt.plot(t, pred_sep[:n], label="Predicted separation", linewidth=2, linestyle="--")
    plt.xlabel("Time / forecast step")
    plt.ylabel("Trajectory separation")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    finalize_plot(save_path, show)


# ------------------------------------------------------------
# True vs predicted reconstructed 3D attractor
# ------------------------------------------------------------
def plot_true_vs_predicted_reconstructed_attractor_3d(
    true_psr: np.ndarray,
    pred_psr: np.ndarray,
    title: str = "True vs predicted reconstructed attractor",
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot side-by-side 3D reconstructed attractors using:

      - true future x-window -> delay embedding
      - predicted future x-window -> delay embedding

    Important:
    This is NOT a true predicted (x, y, z) Lorenz attractor.
    It is a reconstructed attractor from forecasted x(t), which is
    appropriate for the current model because the model predicts future x.
    """
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(true_psr[:, 0], true_psr[:, 1], true_psr[:, 2], linewidth=1.0)
    ax1.set_title("True reconstructed attractor")
    ax1.set_xlabel("x(t)")
    ax1.set_ylabel("x(t+1)")
    ax1.set_zlabel("x(t+2)")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(pred_psr[:, 0], pred_psr[:, 1], pred_psr[:, 2], linewidth=1.0)
    ax2.set_title("Predicted reconstructed attractor")
    ax2.set_xlabel("x(t)")
    ax2.set_ylabel("x(t+1)")
    ax2.set_zlabel("x(t+2)")

    plt.suptitle(title)
    finalize_plot(save_path, show)


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[Path] = None, show: bool = False):
    counts = df["label_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Count")
    plt.title("Class distribution")
    plt.grid(True, axis="y")
    finalize_plot(save_path, show)


def plot_class_distribution_vs_b(df: pd.DataFrame, save_path: Optional[Path] = None, show: bool = False):
    ct = pd.crosstab(df["b"], df["label_name"]).reindex(columns=CLASS_NAMES, fill_value=0)

    plt.figure(figsize=(10, 5))
    for cname in CLASS_NAMES:
        plt.plot(ct.index, ct[cname], label=cname)
    plt.xlabel("b")
    plt.ylabel("Window count")
    plt.title("Class distribution across b")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_future_metric_histogram_by_class(df: pd.DataFrame, save_path: Optional[Path] = None, show: bool = False):
    plt.figure(figsize=(9, 5))
    for cname in CLASS_NAMES:
        vals = df[df["label_name"] == cname]["future_metric"].values
        if len(vals) > 0:
            plt.hist(vals, bins=40, alpha=0.5, label=cname)
    plt.xlabel("Future metric")
    plt.ylabel("Count")
    plt.title("Future metric distribution by class")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_multiclass_confusion_matrix(cm: np.ndarray, class_names=CLASS_NAMES, save_path: Optional[Path] = None, show: bool = False):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=30)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    finalize_plot(save_path, show)


def plot_multiclass_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None, show: bool = False):
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))

    plt.figure(figsize=(7, 6))
    for i, cname in enumerate(CLASS_NAMES):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{cname} (AUC={auc_i:.3f})")
        except ValueError:
            pass

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_multiclass_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None, show: bool = False):
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))

    plt.figure(figsize=(7, 6))
    for i, cname in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=cname)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_probability_distribution_by_true_class(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None, show: bool = False):
    plt.figure(figsize=(10, 6))
    for i, cname in enumerate(CLASS_NAMES):
        vals = y_prob[y_true == i, i]
        if len(vals) > 0:
            plt.hist(vals, bins=30, alpha=0.5, label=cname)
    plt.xlabel("Predicted probability assigned to the true class")
    plt.ylabel("Count")
    plt.title("Probability distribution by true class")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_chaotic_probability_over_windows(y_prob: np.ndarray, threshold: float, save_path: Optional[Path] = None, show: bool = False):
    chaos_idx = CLASS_TO_INDEX["chaotic"]
    p = y_prob[:, chaos_idx]

    plt.figure(figsize=(10, 4))
    plt.plot(p, label="P(chaotic)")
    plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    plt.xlabel("Window index")
    plt.ylabel("Probability")
    plt.title("Chaotic-class probability over all CV test windows")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_cv_metrics_by_fold_multitask(baseline_results, save_path: Optional[Path] = None, show: bool = False):
    fr = baseline_results["fold_results"]
    folds = [x["fold"] for x in fr]

    acc = [x["accuracy"] for x in fr]
    f1m = [x["f1_macro"] for x in fr]
    aucm = [x["roc_auc_macro_ovr"] for x in fr]
    cfar = [x["chaotic_false_alarm_rate"] for x in fr]
    frmse = [x["forecast_rmse"] for x in fr]

    x = np.arange(len(folds))
    width = 0.16

    plt.figure(figsize=(12, 5))
    plt.bar(x - 2*width, acc, width, label="Accuracy")
    plt.bar(x - 1*width, f1m, width, label="Macro F1")
    plt.bar(x + 0*width, aucm, width, label="Macro ROC-AUC")
    plt.bar(x + 1*width, cfar, width, label="Chaotic FAR")
    plt.bar(x + 2*width, frmse, width, label="Forecast RMSE")

    plt.xticks(x, [f"Fold {f}" for f in folds])
    plt.ylabel("Score")
    plt.title("Cross-validation metrics by fold")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_cv_metric_boxplot_multitask(baseline_results, save_path: Optional[Path] = None, show: bool = False):
    fr = baseline_results["fold_results"]

    data = [
        [x["accuracy"] for x in fr],
        [x["f1_macro"] for x in fr],
        [x["roc_auc_macro_ovr"] for x in fr],
        [x["chaotic_false_alarm_rate"] for x in fr],
        [x["forecast_rmse"] for x in fr],
        [x["forecast_mae"] for x in fr],
    ]
    labels = ["Accuracy", "Macro F1", "Macro ROC-AUC", "Chaotic FAR", "Forecast RMSE", "Forecast MAE"]

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Score")
    plt.title("Distribution of CV metrics across folds")
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_forecast_examples(y_true: np.ndarray, y_pred: np.ndarray, n_examples: int = 5, save_path: Optional[Path] = None, show: bool = False):
    n_examples = min(n_examples, len(y_true))
    idxs = np.linspace(0, len(y_true) - 1, n_examples, dtype=int)

    plt.figure(figsize=(10, 2.5 * n_examples))
    for k, idx in enumerate(idxs, start=1):
        plt.subplot(n_examples, 1, k)
        plt.plot(y_true[idx], label="True")
        plt.plot(y_pred[idx], label="Predicted")
        plt.title(f"Forecast example #{idx}")
        plt.xlabel("Future step")
        plt.ylabel("x")
        plt.grid(True)
        plt.legend()

    finalize_plot(save_path, show)


def plot_forecast_error_by_class(y_true_class: np.ndarray, y_true_forecast: np.ndarray, y_pred_forecast: np.ndarray, save_path: Optional[Path] = None, show: bool = False):
    rmses = []
    for i in range(len(CLASS_NAMES)):
        mask = (y_true_class == i)
        if np.sum(mask) == 0:
            rmses.append(np.nan)
        else:
            rmses.append(rmse(y_true_forecast[mask].ravel(), y_pred_forecast[mask].ravel()))

    plt.figure(figsize=(7, 4))
    plt.bar(CLASS_NAMES, rmses)
    plt.ylabel("RMSE")
    plt.title("Forecast RMSE by true class")
    plt.grid(True, axis="y")
    finalize_plot(save_path, show)


# ------------------------------------------------------------
# Helper: find two nearby forecast windows for same b
# ------------------------------------------------------------
def find_two_nearby_windows_with_time_for_same_b(
    dataset_df: pd.DataFrame,
    all_test_indices: np.ndarray,
    all_forecast_true: np.ndarray,
    all_forecast_pred: np.ndarray,
    target_b: float,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Find two forecasted test windows for the same b value, preferably from
    different runs, and return:

      - true future trajectories
      - predicted future trajectories
      - time axis
    """
    df_test = dataset_df.iloc[all_test_indices].copy()
    df_test = df_test.reset_index(drop=True)

    mask = np.isclose(df_test["b"].values.astype(float), float(target_b))
    sub = df_test.loc[mask].copy()

    if len(sub) < 2:
        return None

    unique_runs = sub["run_id"].unique()

    if len(unique_runs) >= 2:
        row1 = sub[sub["run_id"] == unique_runs[0]].iloc[0]
        row2 = sub[sub["run_id"] == unique_runs[1]].iloc[0]
    else:
        row1 = sub.iloc[0]
        row2 = sub.iloc[1]

    i1 = int(row1.name)
    i2 = int(row2.name)

    horizon = len(all_forecast_true[i1])
    time_axis = np.arange(horizon)

    return {
        "true_future_1": all_forecast_true[i1],
        "true_future_2": all_forecast_true[i2],
        "pred_future_1": all_forecast_pred[i1],
        "pred_future_2": all_forecast_pred[i2],
        "time_axis": time_axis,
    }


# ------------------------------------------------------------
# Helper: find one representative forecast example for same b
# ------------------------------------------------------------
def find_one_forecast_example_for_same_b(
    dataset_df: pd.DataFrame,
    all_test_indices: np.ndarray,
    target_b: float,
    preferred_class: Optional[int] = None,
) -> Optional[int]:
    """
    Find one representative forecast example for a given b value.
    Optionally prefer a specific true class.
    """
    df_test = dataset_df.iloc[all_test_indices].copy()
    df_test = df_test.reset_index(drop=True)

    mask = np.isclose(df_test["b"].values.astype(float), float(target_b))

    if preferred_class is not None:
        mask = mask & (df_test["label"].values.astype(int) == int(preferred_class))

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    return int(idx[0])


# ------------------------------------------------------------
# Helper: build simple 3D delay embedding from forecast x
# ------------------------------------------------------------
def build_simple_delay_embedding_from_forecast(x_forecast: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a simple 3D delay embedding from a 1D forecast window.

    Input:
        x_forecast = [x(t+1), x(t+2), ..., x(t+H)]

    Output:
        array of shape (H-2, 3):
            [x(t), x(t+1), x(t+2)]-style local triples
    """
    if len(x_forecast) < 3:
        return None

    return np.column_stack([
        x_forecast[:-2],
        x_forecast[1:-1],
        x_forecast[2:],
    ])


# ======================================================================
# STEP 9: GENERATE AND SAVE MAIN PLOTS
# ======================================================================

def plot_workflow_figures(results, cfg: Config, example_b: float = 28.0, example_ic_id: int = 0, show: bool = False):
    """
    Generate and save all main plots automatically.
    """
    sweep_results = results["sweep_results"]
    eig_df = results["eigenvalue_summary"]
    dataset_df = results["dataset"]
    baseline = results["baseline_results"]

    output_dir = Path(results["output_dir"])
    plot_dir = ensure_plot_dir(output_dir)

    # ------------------------------------------------------------
    # Find an example run for the system-level plots
    # ------------------------------------------------------------
    example_run = None
    for run in sweep_results:
        if np.isclose(run["b"], example_b) and run["ic_id"] == example_ic_id:
            example_run = run
            break

    if example_run is None:
        example_run = sweep_results[0]

    sim_trim = remove_transient(example_run["simulation"], cfg.transient_fraction)

    # ------------------------------------------------------------
    # 1. Example time series
    # ------------------------------------------------------------
    plot_time_series(
        sim_trim,
        title=f"Time series after transient removal (b={example_run['b']}, ic_id={example_run['ic_id']})",
        save_path=plot_dir / "01_time_series.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 2. 3D Lorenz attractor
    # ------------------------------------------------------------
    plot_lorenz_3d(
        sim_trim,
        title=f"3D Lorenz attractor (b={example_run['b']}, ic_id={example_run['ic_id']})",
        save_path=plot_dir / "02_lorenz_3d.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 3. Phase plane
    # ------------------------------------------------------------
    plot_phase_plane(
        sim_trim,
        x_key="y",
        y_key="z",
        title=f"Phase plane y-z (b={example_run['b']}, ic_id={example_run['ic_id']})",
        save_path=plot_dir / "03_phase_plane_yz.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 4. Amplitude vs b
    # ------------------------------------------------------------
    plot_amplitude_vs_b(
        sweep_results,
        transient_fraction=cfg.transient_fraction,
        state_key="x",
        save_path=plot_dir / "04_amplitude_vs_b.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 5. Bifurcation diagram
    # ------------------------------------------------------------
    plot_bifurcation_diagram(
        sweep_results,
        transient_fraction=cfg.transient_fraction,
        state_key="x",
        save_path=plot_dir / "05_bifurcation_diagram.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 6. Maximum real eigenvalue
    # ------------------------------------------------------------
    plot_max_real_eigenvalue(
        eig_df,
        save_path=plot_dir / "06_max_real_eigenvalue.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 7. True sensitivity to initial conditions
    # ------------------------------------------------------------
    same_b_runs = [r for r in sweep_results if np.isclose(r["b"], example_run["b"])]
    if len(same_b_runs) >= 2:
        sim1 = same_b_runs[0]["simulation"]
        sim2 = same_b_runs[1]["simulation"]

        plot_trajectory_separation(
            sim1,
            sim2,
            title=f"Sensitivity to initial conditions at b={example_run['b']}",
            save_path=plot_dir / "07_trajectory_separation.png",
            show=show,
        )

    # ------------------------------------------------------------
    # 8. True vs predicted trajectory separation versus time
    # ------------------------------------------------------------
    pair = find_two_nearby_windows_with_time_for_same_b(
        dataset_df=dataset_df,
        all_test_indices=baseline["all_test_indices"],
        all_forecast_true=baseline["all_forecast_true"],
        all_forecast_pred=baseline["all_forecast_pred"],
        target_b=example_run["b"],
    )

    if pair is not None:
        plot_true_vs_predicted_separation_vs_time(
            true_future_1=pair["true_future_1"],
            true_future_2=pair["true_future_2"],
            pred_future_1=pair["pred_future_1"],
            pred_future_2=pair["pred_future_2"],
            time_axis=pair["time_axis"],
            title=f"True vs predicted trajectory separation at b={example_run['b']}",
            save_path=plot_dir / "08_true_vs_predicted_trajectory_separation.png",
            show=show,
        )
    else:
        print(
            "Skipping 08_true_vs_predicted_trajectory_separation.png "
            "because no suitable forecast window pair was found."
        )

    # ------------------------------------------------------------
    # 9. True vs predicted reconstructed 3D attractor
    # ------------------------------------------------------------
    example_idx = find_one_forecast_example_for_same_b(
        dataset_df=dataset_df,
        all_test_indices=baseline["all_test_indices"],
        target_b=example_run["b"],
        preferred_class=CLASS_TO_INDEX["chaotic"],
    )

    if example_idx is not None:
        true_x = baseline["all_forecast_true"][example_idx]
        pred_x = baseline["all_forecast_pred"][example_idx]

        true_psr_vis = build_simple_delay_embedding_from_forecast(true_x)
        pred_psr_vis = build_simple_delay_embedding_from_forecast(pred_x)

        if true_psr_vis is not None and pred_psr_vis is not None:
            plot_true_vs_predicted_reconstructed_attractor_3d(
                true_psr=true_psr_vis,
                pred_psr=pred_psr_vis,
                title=f"True vs predicted reconstructed attractor at b={example_run['b']}",
                save_path=plot_dir / "09_true_vs_predicted_reconstructed_attractor_3d.png",
                show=show,
            )
        else:
            print(
                "Skipping 09_true_vs_predicted_reconstructed_attractor_3d.png "
                "because forecast horizon is too short."
            )
    else:
        print(
            "Skipping 09_true_vs_predicted_reconstructed_attractor_3d.png "
            "because no suitable example was found."
        )

    # ------------------------------------------------------------
    # 10. Class distribution
    # ------------------------------------------------------------
    plot_class_distribution(
        dataset_df,
        save_path=plot_dir / "10_class_distribution.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 11. Class distribution vs b
    # ------------------------------------------------------------
    plot_class_distribution_vs_b(
        dataset_df,
        save_path=plot_dir / "11_class_distribution_vs_b.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 12. Future metric histogram by class
    # ------------------------------------------------------------
    plot_future_metric_histogram_by_class(
        dataset_df,
        save_path=plot_dir / "12_future_metric_histogram_by_class.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 13. Multiclass ROC curves
    # ------------------------------------------------------------
    plot_multiclass_roc_curves(
        baseline["all_y_true"],
        baseline["all_y_prob"],
        save_path=plot_dir / "13_multiclass_roc.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 14. Multiclass precision-recall curves
    # ------------------------------------------------------------
    plot_multiclass_pr_curves(
        baseline["all_y_true"],
        baseline["all_y_prob"],
        save_path=plot_dir / "14_multiclass_pr.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 15. Confusion matrix
    # ------------------------------------------------------------
    plot_multiclass_confusion_matrix(
        baseline["overall_confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=plot_dir / "15_confusion_matrix.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 16. Probability distribution by true class
    # ------------------------------------------------------------
    plot_probability_distribution_by_true_class(
        baseline["all_y_true"],
        baseline["all_y_prob"],
        save_path=plot_dir / "16_probability_distribution_by_true_class.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 17. Chaotic probability over windows
    # ------------------------------------------------------------
    plot_chaotic_probability_over_windows(
        baseline["all_y_prob"],
        results["suggested_chaotic_threshold"],
        save_path=plot_dir / "17_chaotic_probability_over_windows.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 18. Cross-validation metrics by fold
    # ------------------------------------------------------------
    plot_cv_metrics_by_fold_multitask(
        baseline,
        save_path=plot_dir / "18_cv_metrics_by_fold.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 19. Cross-validation metric boxplot
    # ------------------------------------------------------------
    plot_cv_metric_boxplot_multitask(
        baseline,
        save_path=plot_dir / "19_cv_metric_boxplot.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 20. Forecast examples
    # ------------------------------------------------------------
    plot_forecast_examples(
        baseline["all_forecast_true"],
        baseline["all_forecast_pred"],
        n_examples=5,
        save_path=plot_dir / "20_forecast_examples.png",
        show=show,
    )

    # ------------------------------------------------------------
    # 21. Forecast error by class
    # ------------------------------------------------------------
    plot_forecast_error_by_class(
        baseline["all_y_true"],
        baseline["all_forecast_true"],
        baseline["all_forecast_pred"],
        save_path=plot_dir / "21_forecast_error_by_class.png",
        show=show,
    )

    print(f"All plots saved in: {plot_dir}")


# ======================================================================
# STEP 10: ADVANCED EARLY-WARNING EVALUATION
# ======================================================================

def ensure_eval_dir(output_dir: Path) -> Path:
    """
    Create directory for advanced evaluation outputs.
    """
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def compute_per_class_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """
    Compute precision, recall, F1, and support for each class.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for cname in class_names:
        rows.append({
            "class_name": cname,
            "precision": report[cname]["precision"],
            "recall": report[cname]["recall"],
            "f1_score": report[cname]["f1-score"],
            "support": int(report[cname]["support"]),
        })

    return pd.DataFrame(rows)


def merge_predictions_with_dataset(
    dataset_df: pd.DataFrame,
    baseline_results: Dict[str, object],
) -> pd.DataFrame:
    """
    Merge out-of-fold predictions back into the dataset rows.
    """
    df = dataset_df.copy().reset_index(drop=True)

    df["pred_label"] = np.nan
    df["pred_label_name"] = None

    for cname in CLASS_NAMES:
        df[f"prob_{cname}"] = np.nan

    all_test_indices = baseline_results["all_test_indices"]
    all_y_pred = baseline_results["all_y_pred"]
    all_y_prob = baseline_results["all_y_prob"]

    df.loc[all_test_indices, "pred_label"] = all_y_pred
    df.loc[all_test_indices, "pred_label_name"] = [CLASS_NAMES[int(y)] for y in all_y_pred]

    for j, cname in enumerate(CLASS_NAMES):
        df.loc[all_test_indices, f"prob_{cname}"] = all_y_prob[:, j]

    df["pred_label"] = df["pred_label"].astype("Int64")
    return df


def extract_run_event_summary(
    df_pred: pd.DataFrame,
    event_class: str = "chaotic",
    alarm_prob_col: str = "prob_chaotic",
    alarm_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build run-level event/alarm summary.
    """
    event_idx = CLASS_TO_INDEX[event_class]
    rows = []

    for run_id, sub in df_pred.groupby("run_id"):
        sub = sub.sort_values("future_start_time").reset_index(drop=True)

        true_event_mask = (sub["label"].astype(int) == event_idx)
        alarm_mask = (sub[alarm_prob_col].astype(float) >= alarm_threshold)

        has_event = bool(true_event_mask.any())
        has_alarm = bool(alarm_mask.any())

        event_time = float(sub.loc[true_event_mask, "future_start_time"].iloc[0]) if has_event else np.nan
        alarm_time = float(sub.loc[alarm_mask, "future_start_time"].iloc[0]) if has_alarm else np.nan

        if has_event and has_alarm:
            lead_time = event_time - alarm_time
        else:
            lead_time = np.nan

        rows.append({
            "run_id": int(run_id),
            "has_event": has_event,
            "has_alarm": has_alarm,
            "event_time": event_time,
            "alarm_time": alarm_time,
            "lead_time": lead_time,
            "detected_early": bool(has_event and has_alarm and (alarm_time < event_time)),
            "detected_on_time_or_late": bool(has_event and has_alarm and (alarm_time >= event_time)),
            "missed_event": bool(has_event and not has_alarm),
            "false_alarm_run": bool((not has_event) and has_alarm),
            "alarms_in_run": int(alarm_mask.sum()),
            "windows_in_run": int(len(sub)),
            "alarm_fraction_in_run": float(alarm_mask.mean()),
        })

    return pd.DataFrame(rows)


def summarize_run_event_metrics(run_event_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize run-level early-warning performance.
    """
    n_runs = len(run_event_df)
    n_event_runs = int(run_event_df["has_event"].sum())
    n_non_event_runs = n_runs - n_event_runs

    early_mask = run_event_df["detected_early"]
    late_mask = run_event_df["detected_on_time_or_late"]
    missed_mask = run_event_df["missed_event"]
    false_alarm_mask = run_event_df["false_alarm_run"]

    positive_leads = run_event_df.loc[early_mask, "lead_time"].dropna()
    all_detected_leads = run_event_df.loc[run_event_df["has_event"] & run_event_df["has_alarm"], "lead_time"].dropna()

    return {
        "n_runs": n_runs,
        "n_event_runs": n_event_runs,
        "n_non_event_runs": n_non_event_runs,
        "early_detection_rate_over_event_runs": float(early_mask.sum() / n_event_runs) if n_event_runs > 0 else np.nan,
        "late_detection_rate_over_event_runs": float(late_mask.sum() / n_event_runs) if n_event_runs > 0 else np.nan,
        "missed_event_rate_over_event_runs": float(missed_mask.sum() / n_event_runs) if n_event_runs > 0 else np.nan,
        "false_alarm_run_rate_over_non_event_runs": float(false_alarm_mask.sum() / n_non_event_runs) if n_non_event_runs > 0 else np.nan,
        "mean_alarms_per_run": float(run_event_df["alarms_in_run"].mean()),
        "mean_alarm_fraction_per_run": float(run_event_df["alarm_fraction_in_run"].mean()),
        "mean_positive_lead_time": float(positive_leads.mean()) if len(positive_leads) > 0 else np.nan,
        "median_positive_lead_time": float(positive_leads.median()) if len(positive_leads) > 0 else np.nan,
        "mean_detected_lead_time": float(all_detected_leads.mean()) if len(all_detected_leads) > 0 else np.nan,
        "median_detected_lead_time": float(all_detected_leads.median()) if len(all_detected_leads) > 0 else np.nan,
    }


def compute_forecast_error_by_true_class(
    y_true_class: np.ndarray,
    y_true_forecast: np.ndarray,
    y_pred_forecast: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """
    Compute forecast RMSE, MAE, and MAPE by true class.
    """
    rows = []

    for i, cname in enumerate(class_names):
        mask = (y_true_class == i)

        if np.sum(mask) == 0:
            rows.append({
                "class_name": cname,
                "n_samples": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
            })
            continue

        yt = y_true_forecast[mask].ravel()
        yp = y_pred_forecast[mask].ravel()

        rows.append({
            "class_name": cname,
            "n_samples": int(np.sum(mask)),
            "rmse": rmse(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
            "mape": mape(yt, yp),
        })

    return pd.DataFrame(rows)


def compute_brier_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """
    Compute one-vs-rest Brier score for each class.
    """
    rows = []
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        score = brier_score_loss(y_bin, y_prob[:, i])
        rows.append({
            "class_name": cname,
            "brier_score": float(score),
        })
    return pd.DataFrame(rows)


def expected_calibration_error_binary(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for a binary problem.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if np.sum(mask) == 0:
            continue

        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true_binary[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def compute_ece_by_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute one-vs-rest ECE for each class.
    """
    rows = []
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        ece = expected_calibration_error_binary(y_bin, y_prob[:, i], n_bins=n_bins)
        rows.append({
            "class_name": cname,
            "ece": ece,
        })
    return pd.DataFrame(rows)


def sweep_chaotic_thresholds(
    df_pred: pd.DataFrame,
    thresholds: np.ndarray,
    event_class: str = "chaotic",
    prob_col: str = "prob_chaotic",
) -> pd.DataFrame:
    """
    Sweep thresholds for chaotic warning and compute run-level metrics.
    """
    rows = []

    for thr in thresholds:
        run_event_df = extract_run_event_summary(
            df_pred=df_pred,
            event_class=event_class,
            alarm_prob_col=prob_col,
            alarm_threshold=float(thr),
        )
        summary = summarize_run_event_metrics(run_event_df)
        summary["threshold"] = float(thr)
        rows.append(summary)

    return pd.DataFrame(rows)


def plot_lead_time_histogram(
    run_event_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot histogram of lead times for detected event runs.
    """
    vals = run_event_df.loc[
        run_event_df["has_event"] & run_event_df["has_alarm"],
        "lead_time"
    ].dropna().values

    plt.figure(figsize=(8, 5))
    if len(vals) > 0:
        plt.hist(vals, bins=20)
    plt.xlabel("Lead time (event_time - alarm_time)")
    plt.ylabel("Count")
    plt.title("Lead Time Distribution")
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_threshold_sweep(
    threshold_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot threshold vs early-warning metrics.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["threshold"], threshold_df["early_detection_rate_over_event_runs"], label="Early detection rate")
    plt.plot(threshold_df["threshold"], threshold_df["missed_event_rate_over_event_runs"], label="Missed event rate")
    plt.plot(threshold_df["threshold"], threshold_df["false_alarm_run_rate_over_non_event_runs"], label="False alarm run rate")
    plt.plot(threshold_df["threshold"], threshold_df["mean_alarm_fraction_per_run"], label="Mean alarm fraction/run")
    plt.xlabel("Chaotic probability threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold Sweep for Chaotic Warning")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def plot_reliability_curve_one_vs_rest(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_index: int,
    class_name: str,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Reliability diagram for one-vs-rest probability calibration.
    """
    y_bin = (y_true == class_index).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf = []
    frac_pos = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob[:, class_index] >= lo) & (y_prob[:, class_index] < hi)
        else:
            mask = (y_prob[:, class_index] >= lo) & (y_prob[:, class_index] <= hi)

        if np.sum(mask) == 0:
            continue

        mean_conf.append(np.mean(y_prob[mask, class_index]))
        frac_pos.append(np.mean(y_bin[mask]))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if len(mean_conf) > 0:
        plt.plot(mean_conf, frac_pos, marker="o", label=class_name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability Curve: {class_name}")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save dataframe to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_json(data: Dict[str, object], path: Path) -> None:
    """
    Save dictionary to JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {path}")


def run_advanced_early_warning_evaluation(results: Dict[str, object], chaotic_threshold: Optional[float] = None) -> Dict[str, object]:
    """
    Run advanced evaluation for early-warning usefulness.
    """
    output_dir = Path(results["output_dir"])
    eval_dir = ensure_eval_dir(output_dir)

    dataset_df = results["dataset"]
    baseline = results["baseline_results"]

    y_true = baseline["all_y_true"]
    y_pred = baseline["all_y_pred"]
    y_prob = baseline["all_y_prob"]
    y_true_forecast = baseline["all_forecast_true"]
    y_pred_forecast = baseline["all_forecast_pred"]

    if chaotic_threshold is None:
        chaotic_threshold = float(results["suggested_chaotic_threshold"])

    # 1. Per-class metrics
    per_class_df = compute_per_class_metrics_from_predictions(y_true, y_pred)
    save_dataframe(per_class_df, eval_dir / "per_class_metrics.csv")

    # 2. Merge predictions back to dataset
    df_pred = merge_predictions_with_dataset(dataset_df, baseline)
    save_dataframe(df_pred, eval_dir / "dataset_with_oof_predictions.csv")

    # 3. Run-level event summary
    run_event_df = extract_run_event_summary(
        df_pred=df_pred,
        event_class="chaotic",
        alarm_prob_col="prob_chaotic",
        alarm_threshold=chaotic_threshold,
    )
    save_dataframe(run_event_df, eval_dir / "run_level_event_summary.csv")

    run_event_summary = summarize_run_event_metrics(run_event_df)
    save_json(run_event_summary, eval_dir / "run_level_event_summary.json")

    # 4. Forecast error by class
    forecast_by_class_df = compute_forecast_error_by_true_class(
        y_true_class=y_true,
        y_true_forecast=y_true_forecast,
        y_pred_forecast=y_pred_forecast,
    )
    save_dataframe(forecast_by_class_df, eval_dir / "forecast_error_by_class.csv")

    # 5. Calibration metrics
    brier_df = compute_brier_scores(y_true, y_prob)
    ece_df = compute_ece_by_class(y_true, y_prob, n_bins=10)
    save_dataframe(brier_df, eval_dir / "brier_scores.csv")
    save_dataframe(ece_df, eval_dir / "ece_by_class.csv")

    # 6. Threshold sweep
    thresholds = np.linspace(0.01, 0.99, 50)
    threshold_df = sweep_chaotic_thresholds(
        df_pred=df_pred,
        thresholds=thresholds,
        event_class="chaotic",
        prob_col="prob_chaotic",
    )
    save_dataframe(threshold_df, eval_dir / "chaotic_threshold_sweep.csv")

    # 7. Evaluation plots
    plot_lead_time_histogram(
        run_event_df,
        save_path=eval_dir / "lead_time_histogram.png",
        show=False,
    )

    plot_threshold_sweep(
        threshold_df,
        save_path=eval_dir / "chaotic_threshold_sweep.png",
        show=False,
    )

    plot_reliability_curve_one_vs_rest(
        y_true=y_true,
        y_prob=y_prob,
        class_index=CLASS_TO_INDEX["transitional_oscillatory"],
        class_name="transitional_oscillatory",
        save_path=eval_dir / "reliability_transitional_oscillatory.png",
        show=False,
    )

    plot_reliability_curve_one_vs_rest(
        y_true=y_true,
        y_prob=y_prob,
        class_index=CLASS_TO_INDEX["chaotic"],
        class_name="chaotic",
        save_path=eval_dir / "reliability_chaotic.png",
        show=False,
    )

    print("\n========== ADVANCED EARLY-WARNING EVALUATION ==========")
    print(f"Chaotic threshold used: {chaotic_threshold:.6f}")
    print("\nPer-class metrics:")
    print(per_class_df.to_string(index=False))

    print("\nRun-level event summary:")
    for k, v in run_event_summary.items():
        print(f"{k}: {v}")

    print("\nForecast error by class:")
    print(forecast_by_class_df.to_string(index=False))

    print("\nBrier scores:")
    print(brier_df.to_string(index=False))

    print("\nECE by class:")
    print(ece_df.to_string(index=False))

    return {
        "per_class_metrics": per_class_df,
        "dataset_with_predictions": df_pred,
        "run_level_event_summary_table": run_event_df,
        "run_level_event_summary": run_event_summary,
        "forecast_error_by_class": forecast_by_class_df,
        "brier_scores": brier_df,
        "ece_by_class": ece_df,
        "chaotic_threshold_sweep": threshold_df,
        "evaluation_dir": str(eval_dir),
    }


# ======================================================================
# STEP 11: MAIN WORKFLOW
# ======================================================================

def run_full_workflow(cfg: Config) -> Dict[str, object]:
    """
    Run the full 3-class workflow end-to-end.
    """
    outdir = ensure_output_dir(cfg)

    # Step 1
    print("=== Step 1: Run parameter sweep simulations ===")
    sweep_results = run_parameter_sweep(cfg)

    # Step 2
    print("=== Step 2: Scan equilibrium eigenvalues and estimate instability region ===")
    eig_df = scan_eigenvalues_over_b(cfg.a, cfg.c, cfg.b_values)
    b_hopf_est = critical_b_hopf_formula(cfg.a, cfg.c)

    # Step 3
    print("=== Step 3: Build PSR-based multitask dataset ===")
    dataset_df, X_seq, y_class, y_forecast, y_psr, groups, stable_metric_stats = build_multitask_dataset(
        cfg=cfg,
        sweep_results=sweep_results,
        b_hopf_est=b_hopf_est,
    )

    # Step 4
    print("=== Step 4: Save extracted PSR summary features and labels to CSV ===")
    save_dataset_to_csv(dataset_df, outdir / cfg.dataset_csv)

    # Step 5
    print("=== Step 5: Build raw PSR sequence dataset for multitask model ===")
    print(f"PSR sequence dataset shape: {X_seq.shape}")

    # Step 6
    print("=== Step 6: Train hierarchical 3-class multitask model with grouped CV ===")
    baseline_results = train_hierarchical_multitask_model_grouped_cv_last_plus_attention(
        cfg=cfg,
        X_seq=X_seq,
        y_class=y_class,
        y_forecast=y_forecast,
        y_psr=y_psr,
        groups=groups,
    )

    # Step 7
    print("=== Step 7: Choose practical threshold for chaotic warning ===")
    suggested_chaotic_threshold = choose_chaotic_threshold_by_recall_constraint(
        y_true=baseline_results["all_y_true"],
        y_prob=baseline_results["all_y_prob"],
        min_recall=cfg.chaos_prob_target_recall,
    )

    # Step 8
    print("=== Step 8: Reduce PSR summary features for hybrid-ready input ===")
    feature_columns = get_summary_feature_columns(dataset_df)
    X_reduced, pca = reduce_features_for_hybrid(
        dataset_df,
        feature_columns=feature_columns,
        n_components=cfg.n_reduced_features,
    )
    build_hybrid_model_placeholder(input_dim=X_reduced.shape[1])

    metrics_payload = {
        "summary_metrics": baseline_results["summary_metrics"],
        "overall_roc_auc_macro_ovr": baseline_results["overall_roc_auc_macro_ovr"],
        "suggested_chaotic_threshold": suggested_chaotic_threshold,
        "b_hopf_estimate": b_hopf_est,
        "stable_metric_stats": stable_metric_stats,
        "model_type": "3-class hierarchical multitask classifier; classification uses last hidden state + attention context; forecasting uses last hidden state only",
    }
    save_metrics_json(metrics_payload, outdir / cfg.metrics_json)

    return {
        "config": asdict(cfg),
        "sweep_results": sweep_results,
        "eigenvalue_summary": eig_df,
        "hopf_estimate_b_formula": b_hopf_est,
        "stable_metric_stats": stable_metric_stats,
        "dataset": dataset_df,
        "feature_columns": feature_columns,
        "baseline_results": baseline_results,
        "suggested_chaotic_threshold": suggested_chaotic_threshold,
        "hybrid_reduced_features": X_reduced,
        "pca_model": pca,
        "X_seq_shape": X_seq.shape,
        "output_dir": str(outdir),
    }


# ======================================================================
# STEP 12: SUMMARY REPORT
# ======================================================================

def print_summary(results: Dict[str, object]) -> None:
    """
    Print a concise summary of workflow outputs and CV metrics.
    """
    print("\n========== WORKFLOW SUMMARY ==========")
    print("Model type: 3-class hierarchical multitask classifier")
    print("Classification representation: last hidden state + attention context")
    print("Forecast representation: last hidden state only")
    print(f"Estimated Hopf-like instability formula value for b: {results['hopf_estimate_b_formula']}")
    print(f"Suggested chaotic-class warning threshold: {results['suggested_chaotic_threshold']:.4f}")
    print(f"PSR sequence dataset shape: {results['X_seq_shape']}")
    print(f"Output directory: {results['output_dir']}")

    dataset_df = results["dataset"]
    print(f"\nDataset shape: {dataset_df.shape}")

    print("\nClass counts:")
    print(dataset_df["label_name"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    sm = results["baseline_results"]["summary_metrics"]

    print("\nCross-validation metrics (mean ± std):")
    print(f"Accuracy:              {sm['accuracy_mean']:.4f} ± {sm['accuracy_std']:.4f}")
    print(f"Macro Precision:       {sm['precision_macro_mean']:.4f} ± {sm['precision_macro_std']:.4f}")
    print(f"Macro Recall:          {sm['recall_macro_mean']:.4f} ± {sm['recall_macro_std']:.4f}")
    print(f"Macro F1:              {sm['f1_macro_mean']:.4f} ± {sm['f1_macro_std']:.4f}")
    print(f"Weighted Precision:    {sm['precision_weighted_mean']:.4f} ± {sm['precision_weighted_std']:.4f}")
    print(f"Weighted Recall:       {sm['recall_weighted_mean']:.4f} ± {sm['recall_weighted_std']:.4f}")
    print(f"Weighted F1:           {sm['f1_weighted_mean']:.4f} ± {sm['f1_weighted_std']:.4f}")
    print(f"Macro ROC-AUC (OVR):   {sm['roc_auc_macro_ovr_mean']:.4f} ± {sm['roc_auc_macro_ovr_std']:.4f}")
    print(f"Chaotic False Alarm:   {sm['chaotic_false_alarm_rate_mean']:.4f} ± {sm['chaotic_false_alarm_rate_std']:.4f}")
    print(f"Forecast RMSE:         {sm['forecast_rmse_mean']:.4f} ± {sm['forecast_rmse_std']:.4f}")
    print(f"Forecast MAE:          {sm['forecast_mae_mean']:.4f} ± {sm['forecast_mae_std']:.4f}")
    print(f"Forecast MAPE:         {sm['forecast_mape_mean']:.4f} ± {sm['forecast_mape_std']:.4f}")

    cm = results["baseline_results"]["overall_confusion_matrix"]
    print("\nOverall confusion matrix:")
    print(cm)

    report = results["baseline_results"]["overall_classification_report"]
    print("\nOverall classification report:")
    for cname in CLASS_NAMES:
        print(
            f"{cname:28s} "
            f"precision={report[cname]['precision']:.4f}, "
            f"recall={report[cname]['recall']:.4f}, "
            f"f1={report[cname]['f1-score']:.4f}, "
            f"support={int(report[cname]['support'])}"
        )


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.random.set_seed(42)
    np.random.seed(42)

    cfg = Config()

    # Main workflow
    results = run_full_workflow(cfg)

    # Main plots
    plot_workflow_figures(results, cfg, example_b=28.0, example_ic_id=0, show=False)

    # Summary
    print_summary(results)

    # Advanced early-warning evaluation
    advanced_eval = run_advanced_early_warning_evaluation(results)
