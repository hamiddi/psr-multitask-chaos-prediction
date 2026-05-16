"""
extensions.py
####################################################################################
Paper title: Attention-Based Multitask Learning with Phase-Space Reconstruction for 
             Early Detection of Instability and Chaos in the Lorenz System
Authors: Hamid D. Ismail, Ahmad Harb, Marwan Bikdash
May 16, 2026
####################################################################################


Additional experiments added to address reviewer comments for the paper:
"Attention-Based Multitask Learning with Phase-Space Reconstruction for Early Detection
of Instability and Chaos in the Lorenz System".

This module is intentionally separate from full_pipeline.py so the original paper code
remains stable. The functions here add:

1. Independent held-out parameter-region testing
2. Baseline model comparison: LSTM, CNN-LSTM, Transformer, and Echo-State Network
3. Loss-weight sensitivity and ablation experiments
4. Multitask gradient-alignment diagnostics
5. Noise robustness experiments
6. Latent-space PCA visualization
7. Computational complexity and inference-time reporting
8. Cross-system validation using Lorenz, Rossler, and Mackey-Glass

Run from repository root:
    PYTHONPATH=src python scripts/run_additional_experiments.py
"""

from __future__ import annotations

import json
import time
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

from .full_pipeline import (
    Config,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    run_parameter_sweep,
    scan_eigenvalues_over_b,
    critical_b_hopf_formula,
    build_multitask_dataset,
    build_hierarchical_multitask_state_space_model_last_plus_attention,
    build_sample_weights_for_hierarchy,
    combine_hierarchical_probabilities,
    normalize_sequences,
    make_explicit_validation_split,
    rmse,
    mape,
    multiclass_false_alarm_rate,
    choose_chaotic_threshold_by_recall_constraint,
    ensure_output_dir,
    finalize_plot,
    zero_crossing_rate,
    spectral_entropy_1d,
)


# ======================================================================
# GENERAL UTILITIES
# ======================================================================

def ensure_reviewer_dir(cfg: Config) -> Path:
    """Create a directory for additional experiments."""
    out = Path(cfg.output_dir) / "additional_experiments"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a pandas table and print the path for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_json(obj: Dict, path: Path) -> None:
    """Save a JSON file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON: {path}")


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, and MAPE for any forecast tensor shape."""
    return {
        "forecast_rmse": rmse(y_true.ravel(), y_pred.ravel()),
        "forecast_mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "forecast_mape": mape(y_true.ravel(), y_pred.ravel()),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return standard multiclass classification metrics."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
            out["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        except Exception:
            out["roc_auc_macro_ovr"] = np.nan

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
        out["chaotic_false_alarm_rate"] = multiclass_false_alarm_rate(cm, positive_class=CLASS_TO_INDEX["chaotic"])

    return out


def prepare_lorenz_dataset(cfg: Config):
    """
    Recreate the Lorenz dataset without running the full plotting pipeline.

    This is used by all experiments to ensure that baseline and
    ablation tests use the same data generation procedure as the paper.
    """
    sweep_results = run_parameter_sweep(cfg)
    eig_df = scan_eigenvalues_over_b(cfg.a, cfg.c, cfg.b_values)
    b_hopf_est = critical_b_hopf_formula(cfg.a, cfg.c)
    dataset_df, X_seq, y_class, y_forecast, y_psr, groups, stable_stats = build_multitask_dataset(
        cfg=cfg,
        sweep_results=sweep_results,
        b_hopf_est=b_hopf_est,
    )
    return {
        "sweep_results": sweep_results,
        "eigenvalue_summary": eig_df,
        "hopf_estimate": b_hopf_est,
        "stable_metric_stats": stable_stats,
        "dataset": dataset_df,
        "X_seq": X_seq,
        "y_class": y_class,
        "y_forecast": y_forecast,
        "y_psr": y_psr,
        "groups": groups,
    }


# ======================================================================
# PROPOSED MODEL: SINGLE TRAIN/TEST SPLIT
# ======================================================================

def train_proposed_on_split(
    cfg: Config,
    X_seq: np.ndarray,
    y_class: np.ndarray,
    y_forecast: np.ndarray,
    y_psr: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: Optional[int] = None,
) -> Dict[str, object]:
    """
    Train the proposed hierarchical multitask model on one explicit split.

    This function is used for independent parameter-region testing, noise
    robustness, ablation comparison, and complexity timing.
    """
    local_cfg = replace(cfg)
    if epochs is not None:
        local_cfg.epochs = int(epochs)

    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    yf_train, yf_test = y_forecast[train_idx], y_forecast[test_idx]
    yp_train, yp_test = y_psr[train_idx], y_psr[test_idx]

    X_train_norm, X_test_norm, norm_mean, norm_std = normalize_sequences(X_train, X_test)

    train_h = build_sample_weights_for_hierarchy(yc_train)
    split = make_explicit_validation_split(
        X=X_train_norm,
        yc=yc_train,
        yf=yf_train,
        yp=yp_train,
        train_h=train_h,
        val_fraction=local_cfg.validation_split,
        seed=local_cfg.random_seed,
    )

    model = build_hierarchical_multitask_state_space_model_last_plus_attention(
        local_cfg,
        input_dim=X_train.shape[2],
    )

    start_train = time.perf_counter()
    model.fit(
        split["X_tr"],
        split["y_tr_list"],
        sample_weight=split["sw_tr_list"],
        validation_data=(split["X_val"], split["y_val_list"], split["sw_val_list"]),
        epochs=local_cfg.epochs,
        batch_size=local_cfg.batch_size,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=local_cfg.early_stopping_patience, restore_best_weights=True)],
    )
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    p_coarse, p_nonchaotic, pred_forecast, pred_psr = model.predict(X_test_norm, verbose=0)
    infer_seconds = time.perf_counter() - start_pred

    y_prob = combine_hierarchical_probabilities(p_coarse, p_nonchaotic)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {}
    metrics.update(classification_metrics(yc_test, y_pred, y_prob))
    metrics.update(forecast_metrics(yf_test, pred_forecast))
    metrics["training_time_seconds"] = float(train_seconds)
    metrics["inference_time_seconds_total"] = float(infer_seconds)
    metrics["inference_time_ms_per_window"] = float(1000.0 * infer_seconds / max(len(test_idx), 1))
    metrics["model_parameters"] = int(model.count_params())

    return {
        "model": model,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "X_test_norm": X_test_norm,
        "y_test": yc_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "forecast_true": yf_test,
        "forecast_pred": pred_forecast,
        "psr_pred": pred_psr,
        "metrics": metrics,
    }


# ======================================================================
# 1. INDEPENDENT PARAMETER-REGION TEST
# ======================================================================

def run_independent_parameter_region_test(
    cfg: Config,
    data: Dict[str, object],
    heldout_b_min: float = 24.0,
    heldout_b_max: float = 30.0,
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Train on parameter regions outside [heldout_b_min, heldout_b_max]
    and test only on that independent unseen parameter interval.

    This directly addresses concerns about validation beyond
    grouped cross-validation neighborhoods.
    """
    df = data["dataset"]
    b = df["b"].values.astype(float)
    test_mask = (b >= heldout_b_min) & (b <= heldout_b_max)
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    result = train_proposed_on_split(
        cfg,
        data["X_seq"],
        data["y_class"],
        data["y_forecast"],
        data["y_psr"],
        train_idx,
        test_idx,
        epochs=epochs,
    )

    row = {
        "experiment": "independent_parameter_region",
        "heldout_b_min": heldout_b_min,
        "heldout_b_max": heldout_b_max,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **result["metrics"],
    }
    out = pd.DataFrame([row])
    if outdir is not None:
        save_table(out, outdir / "independent_parameter_region_test.csv")
    return out


# ======================================================================
# 2. BASELINE MODELS
# ======================================================================

def build_lstm_baseline(cfg: Config, forecast_shape: Tuple[int, ...]) -> Model:
    """Standard LSTM baseline with one classification head and one forecast head."""
    inp = layers.Input(shape=(cfg.window_size, cfg.embedding_dim))
    x = layers.LSTM(cfg.state_dim)(inp)
    x = layers.Dropout(cfg.dropout_rate)(x)
    h = layers.Dense(cfg.hidden_dim, activation="relu")(x)
    class_out = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_output")(h)
    flat_dim = int(np.prod(forecast_shape))
    forecast = layers.Dense(flat_dim, name="forecast_flat")(h)
    forecast_out = layers.Reshape(forecast_shape, name="forecast_output")(forecast)
    model = Model(inp, [class_out, forecast_out], name="baseline_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[1.0, 1.0])
    return model


def build_cnn_lstm_baseline(cfg: Config, forecast_shape: Tuple[int, ...]) -> Model:
    """CNN-LSTM baseline that extracts local temporal patterns before recurrence."""
    inp = layers.Input(shape=(cfg.window_size, cfg.embedding_dim))
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(cfg.state_dim)(x)
    h = layers.Dense(cfg.hidden_dim, activation="relu")(x)
    class_out = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_output")(h)
    flat_dim = int(np.prod(forecast_shape))
    forecast = layers.Dense(flat_dim, name="forecast_flat")(h)
    forecast_out = layers.Reshape(forecast_shape, name="forecast_output")(forecast)
    model = Model(inp, [class_out, forecast_out], name="baseline_cnn_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[1.0, 1.0])
    return model


def build_transformer_baseline(cfg: Config, forecast_shape: Tuple[int, ...]) -> Model:
    """Small Transformer encoder baseline for nonlinear time-series windows."""
    inp = layers.Input(shape=(cfg.window_size, cfg.embedding_dim))
    x = layers.Dense(cfg.state_dim)(inp)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(cfg.state_dim // 4, 8))(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(cfg.state_dim * 2, activation="relu")(x)
    ff = layers.Dense(cfg.state_dim)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    h = layers.Dense(cfg.hidden_dim, activation="relu")(x)
    class_out = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_output")(h)
    flat_dim = int(np.prod(forecast_shape))
    forecast = layers.Dense(flat_dim, name="forecast_flat")(h)
    forecast_out = layers.Reshape(forecast_shape, name="forecast_output")(forecast)
    model = Model(inp, [class_out, forecast_out], name="baseline_transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[1.0, 1.0])
    return model


def train_keras_baseline(
    cfg: Config,
    builder,
    name: str,
    X_seq: np.ndarray,
    y_class: np.ndarray,
    y_forecast: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: Optional[int] = None,
) -> Dict[str, float]:
    """Train and evaluate a Keras baseline model on one split."""
    local_cfg = replace(cfg)
    if epochs is not None:
        local_cfg.epochs = int(epochs)

    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    X_train_norm, X_test_norm, _, _ = normalize_sequences(X_train, X_test)
    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    yf_train, yf_test = y_forecast[train_idx], y_forecast[test_idx]

    model = builder(local_cfg, y_forecast.shape[1:])
    start_train = time.perf_counter()
    model.fit(
        X_train_norm,
        [yc_train, yf_train],
        validation_split=local_cfg.validation_split,
        epochs=local_cfg.epochs,
        batch_size=local_cfg.batch_size,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=local_cfg.early_stopping_patience, restore_best_weights=True)],
    )
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_prob, y_forecast_pred = model.predict(X_test_norm, verbose=0)
    infer_seconds = time.perf_counter() - start_pred
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {"model": name, "n_train": len(train_idx), "n_test": len(test_idx)}
    metrics.update(classification_metrics(yc_test, y_pred, y_prob))
    metrics.update(forecast_metrics(yf_test, y_forecast_pred))
    metrics["training_time_seconds"] = float(train_seconds)
    metrics["inference_time_ms_per_window"] = float(1000.0 * infer_seconds / max(len(test_idx), 1))
    metrics["model_parameters"] = int(model.count_params())
    return metrics


def esn_features(X: np.ndarray, reservoir_size: int = 200, spectral_radius: float = 0.9, seed: int = 42) -> np.ndarray:
    """
    Lightweight Echo-State Network feature extractor.

    The reservoir is fixed randomly; only downstream linear models are trained.
    This is useful as a reservoir-computing baseline.
    """
    rng = np.random.default_rng(seed)
    input_dim = X.shape[2]
    Win = rng.normal(scale=0.2, size=(input_dim, reservoir_size))
    W = rng.normal(scale=0.1, size=(reservoir_size, reservoir_size))
    eig_max = np.max(np.abs(np.linalg.eigvals(W)))
    if eig_max > 0:
        W = W * (spectral_radius / eig_max)
    H = np.zeros((X.shape[0], reservoir_size), dtype=np.float32)
    for i in range(X.shape[0]):
        h = np.zeros(reservoir_size)
        for t in range(X.shape[1]):
            h = np.tanh(X[i, t] @ Win + h @ W)
        H[i] = h
    return H


def train_esn_baseline(
    cfg: Config,
    X_seq: np.ndarray,
    y_class: np.ndarray,
    y_forecast: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float]:
    """Train an ESN/reservoir baseline with logistic and ridge readouts."""
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    X_train_norm, X_test_norm, _, _ = normalize_sequences(X_train, X_test)
    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    yf_train, yf_test = y_forecast[train_idx], y_forecast[test_idx]

    start_train = time.perf_counter()
    H_train = esn_features(X_train_norm, seed=cfg.random_seed)
    H_test = esn_features(X_test_norm, seed=cfg.random_seed)
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(H_train, yc_train)
    reg = Ridge(alpha=1.0)
    reg.fit(H_train, yf_train.reshape(len(yf_train), -1))
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = clf.predict(H_test)
    y_prob = clf.predict_proba(H_test)
    yf_pred = reg.predict(H_test).reshape(yf_test.shape)
    infer_seconds = time.perf_counter() - start_pred

    metrics = {"model": "Echo State Network", "n_train": len(train_idx), "n_test": len(test_idx)}
    metrics.update(classification_metrics(yc_test, y_pred, y_prob))
    metrics.update(forecast_metrics(yf_test, yf_pred))
    metrics["training_time_seconds"] = float(train_seconds)
    metrics["inference_time_ms_per_window"] = float(1000.0 * infer_seconds / max(len(test_idx), 1))
    metrics["model_parameters"] = int(H_train.shape[1])
    return metrics


def run_baseline_model_comparison(
    cfg: Config,
    data: Dict[str, object],
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compare the proposed model with LSTM, CNN-LSTM, Transformer, and ESN baselines."""
    X_seq = data["X_seq"]
    y_class = data["y_class"]
    y_forecast = data["y_forecast"]
    y_psr = data["y_psr"]
    groups = data["groups"]

    train_idx, test_idx = next(GroupKFold(n_splits=cfg.n_splits).split(X_seq, y_class, groups=groups))

    rows = []

    proposed = train_proposed_on_split(cfg, X_seq, y_class, y_forecast, y_psr, train_idx, test_idx, epochs=epochs)
    rows.append({"model": "Proposed PSR-Attention Multitask", "n_train": len(train_idx), "n_test": len(test_idx), **proposed["metrics"]})

    rows.append(train_keras_baseline(cfg, build_lstm_baseline, "LSTM", X_seq, y_class, y_forecast, train_idx, test_idx, epochs=epochs))
    rows.append(train_keras_baseline(cfg, build_cnn_lstm_baseline, "CNN-LSTM", X_seq, y_class, y_forecast, train_idx, test_idx, epochs=epochs))
    rows.append(train_keras_baseline(cfg, build_transformer_baseline, "Transformer", X_seq, y_class, y_forecast, train_idx, test_idx, epochs=epochs))
    rows.append(train_esn_baseline(cfg, X_seq, y_class, y_forecast, train_idx, test_idx))

    df = pd.DataFrame(rows)
    if outdir is not None:
        save_table(df, outdir / "baseline_model_comparison.csv")
        plot_baseline_comparison(df, outdir / "baseline_model_comparison.png")
    return df


def plot_baseline_comparison(df: pd.DataFrame, save_path: Path) -> None:
    """Create a compact bar plot comparing baseline macro F1 and forecast RMSE."""
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, df["f1_macro"], width, label="Macro F1")
    plt.bar(x + width / 2, df["forecast_rmse"], width, label="Forecast RMSE")
    plt.xticks(x, df["model"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Baseline comparison")
    plt.legend()
    plt.grid(True, axis="y")
    finalize_plot(save_path, show=False)


# ======================================================================
# 3. ABLATION AND LOSS-WEIGHT SENSITIVITY
# ======================================================================

def run_loss_weight_sensitivity(
    cfg: Config,
    data: Dict[str, object],
    weight_grid: Optional[List[Tuple[float, float, float]]] = None,
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate sensitivity to multitask loss weights.

    Each tuple is: (lambda_class, lambda_forecast, lambda_phase).
    """
    if weight_grid is None:
        weight_grid = [
            (1.0, 0.0, 0.0),  # classification-dominant
            (0.0, 1.0, 0.0),  # forecasting-dominant
            (1.0, 1.0, 0.0),  # classification + forecasting
            (1.0, 1.0, 0.2),  # full paper model
            (2.0, 1.0, 0.2),
            (1.0, 2.0, 0.2),
        ]

    X_seq = data["X_seq"]
    y_class = data["y_class"]
    y_forecast = data["y_forecast"]
    y_psr = data["y_psr"]
    groups = data["groups"]
    train_idx, test_idx = next(GroupKFold(n_splits=cfg.n_splits).split(X_seq, y_class, groups=groups))

    rows = []
    for lc, lf, lp in weight_grid:
        local_cfg = replace(cfg, lambda_class=lc, lambda_forecast=lf, lambda_phase=lp)
        result = train_proposed_on_split(local_cfg, X_seq, y_class, y_forecast, y_psr, train_idx, test_idx, epochs=epochs)
        rows.append({
            "lambda_class": lc,
            "lambda_forecast": lf,
            "lambda_phase": lp,
            **result["metrics"],
        })

    df = pd.DataFrame(rows)
    if outdir is not None:
        save_table(df, outdir / "loss_weight_sensitivity.csv")
        plot_loss_weight_sensitivity(df, outdir / "loss_weight_sensitivity.png")
    return df


def plot_loss_weight_sensitivity(df: pd.DataFrame, save_path: Path) -> None:
    """Plot macro F1 and RMSE across loss-weight settings."""
    labels = [f"{r.lambda_class:g},{r.lambda_forecast:g},{r.lambda_phase:g}" for r in df.itertuples()]
    x = np.arange(len(df))
    plt.figure(figsize=(10, 5))
    plt.plot(x, df["f1_macro"], marker="o", label="Macro F1")
    plt.plot(x, df["forecast_rmse"], marker="s", label="Forecast RMSE")
    plt.xticks(x, labels, rotation=30)
    plt.xlabel("Loss weights: lambda_class, lambda_forecast, lambda_phase")
    plt.ylabel("Score")
    plt.title("Loss-weight sensitivity")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show=False)


def compute_gradient_cosine_similarity(
    cfg: Config,
    data: Dict[str, object],
    n_samples: int = 128,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute cosine similarity between classification and forecasting gradients.

    Negative values indicate potential gradient conflict between tasks.
    This addresses concern about multitask optimization conflict.
    """
    X = data["X_seq"][:n_samples]
    y = data["y_class"][:n_samples]
    yf = data["y_forecast"][:n_samples]
    yp = data["y_psr"][:n_samples]

    X_norm, _, _, _ = normalize_sequences(X, X)
    h = build_sample_weights_for_hierarchy(y)
    model = build_hierarchical_multitask_state_space_model_last_plus_attention(cfg, input_dim=X.shape[2])

    x_tensor = tf.convert_to_tensor(X_norm, dtype=tf.float32)
    y_coarse = tf.convert_to_tensor(h["y_coarse"], dtype=tf.int32)
    y_fine = tf.convert_to_tensor(h["y_nonchaotic_fine"], dtype=tf.int32)
    yf_tensor = tf.convert_to_tensor(yf, dtype=tf.float32)

    with tf.GradientTape() as tape1:
        p_coarse, p_fine, pred_forecast, pred_psr = model(x_tensor, training=True)
        loss_class = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_coarse, p_coarse))
        loss_class += tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_fine, p_fine))
    grads_class = tape1.gradient(loss_class, model.trainable_variables)

    with tf.GradientTape() as tape2:
        p_coarse, p_fine, pred_forecast, pred_psr = model(x_tensor, training=True)
        #loss_forecast = tf.reduce_mean(tf.keras.losses.mean_squared_error(yf_tensor, pred_forecast))
        loss_forecast = tf.reduce_mean(tf.square(yf_tensor - pred_forecast))
    grads_forecast = tape2.gradient(loss_forecast, model.trainable_variables)

    flat_class = []
    flat_forecast = []
    for gc, gf in zip(grads_class, grads_forecast):
        if gc is not None and gf is not None:
            flat_class.append(tf.reshape(gc, [-1]))
            flat_forecast.append(tf.reshape(gf, [-1]))
    vc = tf.concat(flat_class, axis=0)
    vf = tf.concat(flat_forecast, axis=0)
    cosine = tf.reduce_sum(vc * vf) / (tf.norm(vc) * tf.norm(vf) + 1e-12)

    df = pd.DataFrame([{
        "n_samples": int(n_samples),
        "classification_loss": float(loss_class.numpy()),
        "forecast_loss": float(loss_forecast.numpy()),
        "gradient_cosine_similarity": float(cosine.numpy()),
    }])
    if outdir is not None:
        save_table(df, outdir / "multitask_gradient_cosine_similarity.csv")
    return df


# ======================================================================
# 4. NOISE ROBUSTNESS
# ======================================================================

def add_gaussian_noise(X: np.ndarray, noise_level: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise proportional to the standard deviation of each feature.

    noise_level=0.05 means 5% of feature standard deviation.
    """
    rng = np.random.default_rng(seed)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return X + rng.normal(scale=noise_level * std, size=X.shape)


def run_noise_robustness(
    cfg: Config,
    data: Dict[str, object],
    noise_levels: Optional[List[float]] = None,
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Train on clean data and test on noisy windows at multiple noise levels."""
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.10]

    X_seq = data["X_seq"]
    y_class = data["y_class"]
    y_forecast = data["y_forecast"]
    y_psr = data["y_psr"]
    groups = data["groups"]
    train_idx, test_idx = next(GroupKFold(n_splits=cfg.n_splits).split(X_seq, y_class, groups=groups))

    trained = train_proposed_on_split(cfg, X_seq, y_class, y_forecast, y_psr, train_idx, test_idx, epochs=epochs)
    model = trained["model"]
    norm_mean = trained["norm_mean"]
    norm_std = trained["norm_std"]

    rows = []
    for nl in noise_levels:
        X_noisy = add_gaussian_noise(X_seq[test_idx], nl, seed=cfg.random_seed)
        X_noisy_norm = (X_noisy - norm_mean) / norm_std
        p_coarse, p_fine, pred_forecast, _ = model.predict(X_noisy_norm, verbose=0)
        y_prob = combine_hierarchical_probabilities(p_coarse, p_fine)
        y_pred = np.argmax(y_prob, axis=1)
        row = {"noise_level": nl}
        row.update(classification_metrics(y_class[test_idx], y_pred, y_prob))
        row.update(forecast_metrics(y_forecast[test_idx], pred_forecast))
        rows.append(row)

    df = pd.DataFrame(rows)
    if outdir is not None:
        save_table(df, outdir / "noise_robustness.csv")
        plot_noise_robustness(df, outdir / "noise_robustness.png")
    return df


def plot_noise_robustness(df: pd.DataFrame, save_path: Path) -> None:
    """Plot degradation under increasing noise."""
    plt.figure(figsize=(8, 5))
    plt.plot(df["noise_level"], df["f1_macro"], marker="o", label="Macro F1")
    plt.plot(df["noise_level"], df["forecast_rmse"], marker="s", label="Forecast RMSE")
    plt.xlabel("Gaussian noise level")
    plt.ylabel("Score")
    plt.title("Noise robustness")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show=False)


# ======================================================================
# 5. LATENT SPACE ANALYSIS
# ======================================================================

def run_latent_space_pca(
    cfg: Config,
    data: Dict[str, object],
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Extract encoder latent vectors and project them using PCA.

    This supports discussion of why the transitional oscillatory class is
    intrinsically more overlapping/ambiguous.
    """
    X_seq = data["X_seq"]
    y_class = data["y_class"]
    y_forecast = data["y_forecast"]
    y_psr = data["y_psr"]
    groups = data["groups"]
    train_idx, test_idx = next(GroupKFold(n_splits=cfg.n_splits).split(X_seq, y_class, groups=groups))

    result = train_proposed_on_split(cfg, X_seq, y_class, y_forecast, y_psr, train_idx, test_idx, epochs=epochs)
    model = result["model"]
    latent_model = Model(inputs=model.input, outputs=model.get_layer("classification_context").output)
    latent = latent_model.predict(result["X_test_norm"], verbose=0)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(latent)

    df = pd.DataFrame({
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
        "label": result["y_test"],
        "label_name": [CLASS_NAMES[int(i)] for i in result["y_test"]],
    })
    if outdir is not None:
        save_table(df, outdir / "latent_space_pca.csv")
        plot_latent_space_pca(df, outdir / "latent_space_pca.png")
    return df


def plot_latent_space_pca(df: pd.DataFrame, save_path: Path) -> None:
    """Plot PCA latent space colored by true regime."""
    plt.figure(figsize=(7, 6))
    for cname in CLASS_NAMES:
        sub = df[df["label_name"] == cname]
        plt.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.6, label=cname)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent-space PCA of classification representation")
    plt.legend()
    plt.grid(True)
    finalize_plot(save_path, show=False)


# ======================================================================
# 6. COMPUTATIONAL COMPLEXITY
# ======================================================================

def run_complexity_report(
    cfg: Config,
    data: Dict[str, object],
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Report parameter count, training time, inference time, and environment."""
    X_seq = data["X_seq"]
    y_class = data["y_class"]
    y_forecast = data["y_forecast"]
    y_psr = data["y_psr"]
    groups = data["groups"]
    train_idx, test_idx = next(GroupKFold(n_splits=cfg.n_splits).split(X_seq, y_class, groups=groups))
    result = train_proposed_on_split(cfg, X_seq, y_class, y_forecast, y_psr, train_idx, test_idx, epochs=epochs)
    row = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "window_size": cfg.window_size,
        "embedding_dim": cfg.embedding_dim,
        "future_horizon": cfg.future_horizon,
        **result["metrics"],
    }
    df = pd.DataFrame([row])
    if outdir is not None:
        save_table(df, outdir / "computational_complexity.csv")
    return df


# ======================================================================
# 7. ADDITIONAL NONLINEAR SYSTEMS
# ======================================================================

def rossler_rhs(t, state, a=0.2, b=0.2, c=5.7):
    """Rossler system."""
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]

# we will not use this because it has many parameters
def duffing_rhs(t, state, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2):
    """Forced Duffing oscillator converted to first-order form."""
    x, v = state
    dx = v
    dv = -delta * v - alpha * x - beta * x ** 3 + gamma * np.cos(omega * t)
    return [dx, dv]

def simulate_mackey_glass(beta=0.2, gamma=0.1, n=10, tau=17, dt=0.1, steps=6000, x0=1.2):
    """Simple Euler simulation of Mackey-Glass delay equation."""
    delay_steps = int(tau / dt)
    x = np.ones(steps + delay_steps + 1) * x0
    for i in range(delay_steps, steps + delay_steps):
        x_tau = x[i - delay_steps]
        dx = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[i]
        x[i + 1] = x[i] + dt * dx
    t = np.arange(steps) * dt
    series = x[delay_steps:delay_steps + steps]
    return {"t": t, "x": series, "y": np.gradient(series), "z": np.gradient(np.gradient(series))}


def simulate_additional_system(system_name: str, steps: int = 6000, t_end: float = 120.0) -> Dict[str, np.ndarray]:
    """Simulate one additional nonlinear system and return t, x, y, z-like arrays."""
    if system_name == "rossler":
        t_eval = np.linspace(0, t_end, steps)
        sol = solve_ivp(lambda t, y: rossler_rhs(t, y), (0, t_end), [1.0, 1.0, 1.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
        return {"t": sol.t, "x": sol.y[0], "y": sol.y[1], "z": sol.y[2]}
    if system_name == "duffing":
        t_eval = np.linspace(0, t_end, steps)
        sol = solve_ivp(lambda t, y: duffing_rhs(t, y), (0, t_end), [0.1, 0.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
        x = sol.y[0]
        v = sol.y[1]
        return {"t": sol.t, "x": x, "y": v, "z": np.gradient(v)}
    if system_name == "mackey_glass":
        return simulate_mackey_glass(steps=steps)
    raise ValueError(f"Unknown system: {system_name}")



def build_single_system_multiregime_dataset(
    cfg: Config,
    system_name: str,
    noise_level: float = 0.0,
) -> Dict[str, object]:
    """
    Build a three-regime dataset for an additional nonlinear system.

    Because Rossler, Duffing, and Mackey-Glass do not use the same Lorenz rho
    bifurcation parameter, labels are assigned from the distribution of a
    future-window dynamical metric:

        lower third  -> stable-like
        middle third -> transitional-like
        upper third  -> chaotic-like

    This provides cross-system validation of the workflow rather than claiming
    identical physical bifurcation structure across systems.
    """
    sim = simulate_additional_system(
        system_name,
        steps=cfg.num_points,
        t_end=cfg.t_end,
    )

    if noise_level > 0:
        rng = np.random.default_rng(cfg.random_seed)
        for key in ["x", "y", "z"]:
            std = np.std(sim[key]) + 1e-8
            sim[key] = sim[key] + rng.normal(
                scale=noise_level * std,
                size=sim[key].shape,
            )

    n = len(sim["t"])
    cut = int(n * cfg.transient_fraction)

    t = sim["t"][cut:]
    x = sim["x"][cut:]
    y = sim["y"][cut:]
    z = sim["z"][cut:]

    max_lag = (cfg.embedding_dim - 1) * cfg.delay_tau

    psr = []
    for i in range(max_lag, len(x)):
        psr.append([x[i - j * cfg.delay_tau] for j in range(cfg.embedding_dim)])
    psr = np.array(psr, dtype=np.float32)

    t = t[max_lag:]
    x = x[max_lag:]
    y = y[max_lag:]
    z = z[max_lag:]

    X_seq = []
    y_forecast = []
    y_psr = []
    metrics = []

    end_limit = len(t) - cfg.window_size - cfg.future_horizon

    for i in range(0, max(end_limit, 0), cfg.stride):
        in_slice = slice(i, i + cfg.window_size)
        fut_slice = slice(i + cfg.window_size, i + cfg.window_size + cfg.future_horizon)

        psr_input = psr[in_slice]
        psr_future = psr[fut_slice]

        # Keep the cross-system forecast target compatible with the
        # original proposed model, which predicts a scalar future signal.
        state_future = x[fut_slice].astype(np.float32)

        xf = x[fut_slice]

        future_metric = (
            np.std(xf)
            + 0.5 * zero_crossing_rate(xf)
            + 0.5 * spectral_entropy_1d(xf)
            + 0.25 * np.mean(np.std(psr_future, axis=0))
        )

        X_seq.append(psr_input)
        y_forecast.append(state_future)
        y_psr.append(psr_future)
        metrics.append(future_metric)

    X_seq = np.array(X_seq, dtype=np.float32)
    y_forecast = np.array(y_forecast, dtype=np.float32)
    y_psr = np.array(y_psr, dtype=np.float32)
    metrics = np.array(metrics, dtype=float)

    q1, q2 = np.quantile(metrics, [1 / 3, 2 / 3])

    y_class = np.zeros(len(metrics), dtype=np.int32)
    y_class[(metrics > q1) & (metrics <= q2)] = CLASS_TO_INDEX["transitional_oscillatory"]
    y_class[metrics > q2] = CLASS_TO_INDEX["chaotic"]

    groups = np.arange(len(y_class)) // 50

    dataset_df = pd.DataFrame({
        "system": system_name,
        "window_id": np.arange(len(y_class)),
        "future_metric": metrics,
        "label": y_class,
        "label_name": [CLASS_NAMES[int(i)] for i in y_class],
    })

    return {
        "dataset": dataset_df,
        "X_seq": X_seq,
        "y_class": y_class,
        "y_forecast": y_forecast,
        "y_psr": y_psr,
        "groups": groups,
    }


def run_cross_system_validation(
    cfg: Config,
    data: Optional[Dict[str, object]] = None,
    systems: Optional[List[str]] = None,
    epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Train and evaluate the proposed model across Lorenz, Rossler,
    and Mackey-Glass systems.

    Important implementation detail:
    - Lorenz is evaluated using the original dataset generated by the
      main paper pipeline through prepare_lorenz_dataset().
    - Rossler and Mackey-Glass are generated separately using the same
      phase-space reconstruction and windowing workflow.
    - The additional systems use quantile-based regime labels derived
      from future-window dynamical metrics. These labels are intended
      for cross-system workflow validation, not for claiming that all
      systems share the same physical bifurcation parameter.
    """
    if systems is None:
        systems = ["lorenz", "rossler", "mackey_glass"]

    rows = []

    for system in systems:
        print(f"Running cross-system validation for: {system}")

        # ------------------------------------------------------------
        # Lorenz benchmark system
        # ------------------------------------------------------------
        # Lorenz must NOT be sent to simulate_additional_system(), because
        # this file only defines additional-system simulators for Rossler
        # and Mackey-Glass. Instead, we reuse the original Lorenz dataset
        # already prepared above in run_all_additional_experiments().
        if system == "lorenz":

            if data is None:
                data = prepare_lorenz_dataset(cfg)

            X_seq = data["X_seq"]
            y_class = data["y_class"]
            y_forecast = data["y_forecast"]
            y_psr = data["y_psr"]
            groups = data["groups"]

        # ------------------------------------------------------------
        # Additional nonlinear systems
        # ------------------------------------------------------------
        # Rossler and Mackey-Glass are generated here using the same
        # input representation: scalar x(t), PSR embedding, overlapping
        # windows, and future forecasting target.
        else:

            sys_data = build_single_system_multiregime_dataset(
                cfg=cfg,
                system_name=system,
                noise_level=0.0,
            )

            X_seq = sys_data["X_seq"]
            y_class = sys_data["y_class"]
            y_forecast = sys_data["y_forecast"]
            y_psr = sys_data["y_psr"]
            groups = sys_data["groups"]

        # Guard against invalid or very small datasets.
        if len(y_class) < 100 or len(np.unique(groups)) < 2:
            rows.append({
                "system": system,
                "status": "skipped_too_few_windows",
                "n_samples": int(len(y_class)),
            })
            continue

        # Grouped split avoids placing highly correlated neighboring
        # windows from the same local region into both train and test sets.
        gkf = GroupKFold(
            n_splits=min(cfg.n_splits, len(np.unique(groups)))
        )

        train_idx, test_idx = next(
            gkf.split(X_seq, y_class, groups=groups)
        )

        result = train_proposed_on_split(
            cfg=cfg,
            X_seq=X_seq,
            y_class=y_class,
            y_forecast=y_forecast,
            y_psr=y_psr,
            train_idx=train_idx,
            test_idx=test_idx,
            epochs=epochs,
        )

        rows.append({
            "system": system,
            "status": "completed",
            "n_samples": int(len(y_class)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            **result["metrics"],
        })

    df = pd.DataFrame(rows)

    if outdir is not None:

        save_table(df, outdir / "cross_system_validation.csv")

        completed = df[df["status"] == "completed"].copy()

        if len(completed) > 0:

            x = np.arange(len(completed))
            width = 0.35

            plt.figure(figsize=(9, 5))

            plt.bar(
                x - width / 2,
                completed["f1_macro"],
                width,
                label="Macro F1",
            )

            plt.bar(
                x + width / 2,
                completed["forecast_rmse"],
                width,
                label="Forecast RMSE",
            )

            plt.xticks(
                x,
                completed["system"],
                rotation=20,
            )

            plt.ylabel("Score")
            plt.title("Cross-system validation")
            plt.legend()
            plt.grid(True, axis="y")

            finalize_plot(
                outdir / "cross_system_validation.png",
                show=False,
            )

    return df



# ======================================================================
# MASTER RUNNER
# ======================================================================

def run_additional_experiments(cfg: Optional[Config] = None, fast: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Run all experiments and save outputs.

    Parameters
    ----------
    cfg:
        Main configuration. If None, default Config() is used.
    fast:
        If True, reduces epochs and dataset size for quick verification.
        Use fast=False for paper-quality reruns.
    """
    if cfg is None:
        cfg = Config()

    if fast:
        cfg = replace(
            cfg,
            epochs=3,
            num_points=min(cfg.num_points, 2500),
            b_values=tuple(np.linspace(10.0, 32.0, 10)),
        )

    outdir = ensure_reviewer_dir(cfg)
    save_json({"config": asdict(cfg), "fast_mode": fast}, outdir / "experiment_config.json")

    print("Preparing Lorenz dataset for experiments...")
    data = prepare_lorenz_dataset(cfg)

    outputs = {}
    outputs["independent_parameter_region"] = run_independent_parameter_region_test(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["baseline_model_comparison"] = run_baseline_model_comparison(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["loss_weight_sensitivity"] = run_loss_weight_sensitivity(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["gradient_cosine"] = compute_gradient_cosine_similarity(cfg, data, outdir=outdir)
    outputs["noise_robustness"] = run_noise_robustness(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["latent_space_pca"] = run_latent_space_pca(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["complexity"] = run_complexity_report(cfg, data, epochs=cfg.epochs if not fast else 3, outdir=outdir)
    outputs["cross_system_validation"] = run_cross_system_validation(
        cfg,
        data=data,
        epochs=cfg.epochs if not fast else 3,
        outdir=outdir,
    )

    print(f"Additional experiment outputs saved in: {outdir}")
    return outputs

