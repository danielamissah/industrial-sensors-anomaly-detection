"""
Evaluation module — runs across all 4 CMAPSS subsets and produces:
  - Per-subset metrics (Precision, Recall, F1, AUC-ROC)
  - Cross-subset comparison table
  - Reconstruction error distribution plots
  - ROC curves
  - SHAP sensor importance plots
  - Outputs saved to outputs/figures/<subset>/ and outputs/reports/
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
    roc_auc_score, confusion_matrix, roc_curve,
)
from loguru import logger
import yaml
import argparse

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from lstm_autoencoder import LSTMAutoencoder


plt.style.use("dark_background")
COLORS = ["#00e5ff", "#ff4060", "#00ff88", "#ffb800", "#b388ff"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_artifacts(model_dir: str, subset: str, device):
    mdir = Path(model_dir) / subset
    with open(mdir / "model_config.json") as f: cfg = json.load(f)
    with open(mdir / "threshold.json")    as f: thr = json.load(f)

    model = LSTMAutoencoder(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"], dropout=0.0,
    )
    model.load_state_dict(torch.load(mdir / "best_model.pt", map_location=device))
    model.to(device).eval()
    return model, thr["threshold"], cfg


def get_feature_names(data_dir: str, subset: str) -> list:
    feat_path = Path(data_dir) / subset / "feature_cols.json"
    if feat_path.exists():
        with open(feat_path) as f:
            return json.load(f)
    return [f"feature_{i}" for i in range(100)]


def compute_metrics(y_true, errors, threshold) -> dict:
    y_pred = (errors > threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, errors)),
        "threshold": float(threshold),
        "anomaly_rate_pred": float(y_pred.mean()),
        "anomaly_rate_true": float(y_true.mean()),
    }


def plot_error_distribution(errors, y_true, threshold, subset, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors[y_true==0], bins=60, alpha=0.7, label="Normal",  color=COLORS[0], density=True)
    ax.hist(errors[y_true==1], bins=60, alpha=0.7, label="Anomaly", color=COLORS[1], density=True)
    ax.axvline(threshold, color=COLORS[3], lw=2, linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_title(f"[{subset}] Reconstruction Error Distribution", fontsize=13)
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    path = save_dir / f"{subset}_error_dist.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, errors, auc, subset, save_dir):
    fpr, tpr, _ = roc_curve(y_true, errors)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=COLORS[0], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], color=COLORS[2], lw=1, linestyle="--")
    ax.set_title(f"[{subset}] ROC Curve", fontsize=12)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{subset}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_subset_comparison(all_metrics: dict, save_dir: Path):
    """Bar chart comparing F1 and AUC-ROC across all 4 subsets."""
    subsets = list(all_metrics.keys())
    f1s     = [all_metrics[s]["f1"]      for s in subsets]
    aucs    = [all_metrics[s]["auc_roc"] for s in subsets]

    x = np.arange(len(subsets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, f1s,  w, label="F1 Score",  color=COLORS[0], alpha=0.85)
    ax.bar(x + w/2, aucs, w, label="AUC-ROC",   color=COLORS[2], alpha=0.85)
    ax.set_xticks(x)
    subset_labels = {
        "FD001": "1 cond / 1 fault",
        "FD002": "6 cond / 1 fault",
        "FD003": "1 cond / 2 faults",
        "FD004": "6 cond / 2 faults",
    }
    ax.set_xticklabels([f"{s}\n{subset_labels.get(s, '')}" for s in subsets], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Subset Performance Comparison\n(Harder subsets have more operating conditions and fault modes)", fontsize=11)
    ax.legend()
    ax.axhline(0.75, color=COLORS[3], lw=1, linestyle=":", label="Minimum threshold")
    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i - w/2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9, color=COLORS[0])
        ax.text(i + w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9, color=COLORS[2])
    plt.tight_layout()
    plt.savefig(save_dir / "cross_subset_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cross-subset comparison saved.")


def plot_shap(model, X_normal, X_anomalous, feature_names, subset, save_dir):
    logger.info(f"  [{subset}] Computing SHAP values...")

    class AnomalyScorer(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x):
            x_hat = self.base(x)
            return ((x - x_hat) ** 2).mean(dim=(1,2), keepdim=True)

    scorer  = AnomalyScorer(model)
    bg      = torch.FloatTensor(X_normal[:100])
    explain = torch.FloatTensor(X_anomalous[:50])

    try:
        explainer   = shap.DeepExplainer(scorer, bg)
        shap_vals   = explainer.shap_values(explain)
        shap_agg    = np.abs(shap_vals).mean(axis=1)   # (n, n_features)
        mean_shap   = shap_agg.mean(axis=0)
        sorted_idx  = np.argsort(mean_shap)[::-1][:15]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in sorted_idx][::-1],
                mean_shap[sorted_idx][::-1], color=COLORS[0])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"[{subset}] Top 15 Sensor Channels — Anomaly Attribution\n(SHAP DeepExplainer)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / f"{subset}_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  [{subset}] SHAP plot saved.")
    except Exception as e:
        logger.warning(f"  [{subset}] SHAP skipped: {e}")


def evaluate_subset(subset: str, model_dir: str, data_dir: str,
                    fig_dir: Path, device) -> dict:
    logger.info(f"\n  Evaluating: {subset}")

    model, threshold, cfg = load_model_and_artifacts(model_dir, subset, device)
    X_test = np.load(Path(data_dir) / subset / "X_test.npy")
    y_test = np.load(Path(data_dir) / subset / "y_test.npy")
    X_train = np.load(Path(data_dir) / subset / "X_train.npy")
    y_train = np.load(Path(data_dir) / subset / "y_train.npy")

    errors  = model.reconstruction_error(torch.FloatTensor(X_test).to(device))
    metrics = compute_metrics(y_test, errors, threshold)

    logger.info(f"  F1={metrics['f1']:.3f}  AUC={metrics['auc_roc']:.3f}  "
                f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")

    subset_fig_dir = fig_dir / subset
    subset_fig_dir.mkdir(parents=True, exist_ok=True)

    plot_error_distribution(errors, y_test, threshold, subset, subset_fig_dir)
    plot_roc(y_test, errors, metrics["auc_roc"], subset, subset_fig_dir)

    feature_names = get_feature_names(data_dir, subset)
    X_normal    = X_train[y_train == 0]
    X_anomalous = X_test[y_test   == 1]
    if len(X_anomalous) > 0:
        plot_shap(model, X_normal, X_anomalous, feature_names, subset, subset_fig_dir)

    return metrics


def run_evaluation(model_dir="outputs/models", data_dir="data/processed",
                   fig_dir="outputs/figures", report_dir="outputs/reports",
                   subsets=None, config_path="configs/config.yaml"):

    config   = load_config(config_path)
    subsets  = subsets or list(config["data"]["subsets"].keys())
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = {}
    for subset in subsets:
        all_metrics[subset] = evaluate_subset(
            subset, model_dir, data_dir, fig_path, device)

    # Cross-subset comparison plot
    if len(all_metrics) > 1:
        plot_cross_subset_comparison(all_metrics, fig_path)

    # Save consolidated report
    report = {"subsets": all_metrics}
    with open(Path(report_dir) / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    logger.info("\n" + "="*60)
    logger.info(f"{'Subset':<8} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Recall':>6}")
    logger.info("="*60)
    for s, m in all_metrics.items():
        logger.info(f"{s:<8} {m['f1']:>6.3f} {m['auc_roc']:>6.3f} "
                    f"{m['precision']:>6.3f} {m['recall']:>6.3f}")

    logger.success(f"\nEvaluation complete. Figures → {fig_dir}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["FD001","FD002","FD003","FD004"])
    parser.add_argument("--model-dir",  default="outputs/models")
    parser.add_argument("--data-dir",   default="data/processed")
    parser.add_argument("--fig-dir",    default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.data_dir, args.fig_dir,
                   args.report_dir, args.subsets, args.config)