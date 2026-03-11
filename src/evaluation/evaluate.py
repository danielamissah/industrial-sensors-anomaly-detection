"""
Model evaluation module.

Computes:
  - Reconstruction error distribution
  - Precision, Recall, F1, AUC-ROC at the calibrated threshold
  - SHAP values to explain which sensor channels drive anomaly scores
  - Visualisations saved to outputs/figures/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from loguru import logger

from lstm_autoencoder import LSTMAutoencoder


plt.style.use("dark_background")
PALETTE = ["#00e5ff", "#ff4060", "#00ff88", "#ffb800"]


def load_model_and_threshold(model_dir: str, device):
    model_dir = Path(model_dir)

    with open(model_dir / "model_config.json") as f:
        cfg = json.load(f)

    with open(model_dir / "threshold.json") as f:
        threshold_data = json.load(f)
        threshold = threshold_data["threshold"]

    model = LSTMAutoencoder(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["sequence_length"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device).eval()

    return model, threshold, cfg


def compute_metrics(y_true: np.ndarray, errors: np.ndarray, threshold: float) -> dict:
    y_pred = (errors > threshold).astype(int)
    metrics = {
        "precision":      precision_score(y_true, y_pred, zero_division=0),
        "recall":         recall_score(y_true, y_pred, zero_division=0),
        "f1":             f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":        roc_auc_score(y_true, errors),
        "threshold":      threshold,
        "anomaly_rate_pred": y_pred.mean(),
        "anomaly_rate_true": y_true.mean(),
    }
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    return metrics


def plot_reconstruction_error_distribution(errors, y_true, threshold, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors[y_true == 0], bins=60, alpha=0.7, label="Normal",  color=PALETTE[0], density=True)
    ax.hist(errors[y_true == 1], bins=60, alpha=0.7, label="Anomaly", color=PALETTE[1], density=True)
    ax.axvline(threshold, color=PALETTE[3], linestyle="--", linewidth=2, label=f"Threshold: {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Reconstruction Error Distribution: Normal vs Anomalous Sequences", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_roc_curve(y_true, errors, auc, save_path):
    fpr, tpr, _ = roc_curve(y_true, errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE[2], linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Anomaly Detection", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, errors, threshold, save_path):
    y_pred = (errors > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_shap_values(model, X_background: np.ndarray, X_explain: np.ndarray,
                        feature_names: list, save_path: str):
    """
    Use SHAP DeepExplainer to attribute anomaly scores to individual sensor channels.
    Background: random sample of normal sequences.
    Explain: anomalous sequences.
    """
    logger.info("Computing SHAP values (this may take a few minutes)...")

    # SHAP works on the reconstruction error output
    # We wrap the model to return scalar anomaly score per sample
    class AnomalyScorer(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, x):
            x_hat = self.base(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2), keepdim=True)

    scorer = AnomalyScorer(model)

    bg_tensor = torch.FloatTensor(X_background[:100])
    ex_tensor = torch.FloatTensor(X_explain[:50])

    explainer   = shap.DeepExplainer(scorer, bg_tensor)
    shap_values = explainer.shap_values(ex_tensor)  # (n, seq_len, n_features)

    # Aggregate across time dimension → (n, n_features)
    shap_agg = np.abs(shap_values).mean(axis=1)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_shap = shap_agg.mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]

    ax.barh(
        [feature_names[i] for i in sorted_idx[:15]][::-1],
        mean_shap[sorted_idx[:15]][::-1],
        color=PALETTE[0]
    )
    ax.set_xlabel("Mean |SHAP value| (contribution to anomaly score)", fontsize=11)
    ax.set_title("Top 15 Sensor Channels Driving Anomaly Detections\n(SHAP Feature Importance)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP plot: {save_path}")

    return shap_agg, feature_names


def run_evaluation(model_dir: str = "outputs/models",
                   data_dir:  str = "data/processed",
                   fig_dir:   str = "outputs/figures"):
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load ──────────────────────────────────────────────────────
    model, threshold, cfg = load_model_and_threshold(model_dir, device)
    X_test = np.load(Path(data_dir) / "X_test.npy")
    y_test = np.load(Path(data_dir) / "y_test.npy")

    # ── Reconstruction errors ─────────────────────────────────────
    test_tensor = torch.FloatTensor(X_test).to(device)
    errors = model.reconstruction_error(test_tensor)

    # ── Metrics ───────────────────────────────────────────────────
    logger.info("=== Evaluation Metrics ===")
    metrics = compute_metrics(y_test, errors, threshold)

    # ── Plots ─────────────────────────────────────────────────────
    plot_reconstruction_error_distribution(
        errors, y_test, threshold, fig_path / "reconstruction_error_dist.png"
    )
    plot_roc_curve(y_test, errors, metrics["auc_roc"], fig_path / "roc_curve.png")
    plot_confusion_matrix(y_test, errors, threshold, fig_path / "confusion_matrix.png")

    # ── SHAP ──────────────────────────────────────────────────────
    n_features   = X_test.shape[2]
    feature_names = [f"sensor_{i}" for i in range(n_features)]  # replace with actual names

    X_normal   = X_test[y_test == 0]
    X_anomalous = X_test[y_test == 1]

    if len(X_anomalous) > 0:
        compute_shap_values(
            model, X_normal, X_anomalous,
            feature_names, str(fig_path / "shap_importance.png")
        )

    # ── Save metrics ──────────────────────────────────────────────
    metrics_path = Path(model_dir) / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    logger.success(f"Evaluation complete. Figures → {fig_dir}")
    return metrics


if __name__ == "__main__":
    run_evaluation()