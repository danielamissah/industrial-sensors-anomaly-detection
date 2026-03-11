"""
Training script for LSTM Autoencoder.
Compatible with both local execution and SageMaker Training Jobs.

SageMaker passes hyperparameters as CLI args and expects:
  - Input data in /opt/ml/input/data/
  - Model artifacts saved to /opt/ml/model/
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from loguru import logger
import yaml

from lstm_autoencoder import build_model, compute_threshold


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker passes these automatically
    parser.add_argument("--model-dir",   default=os.environ.get("SM_MODEL_DIR",   "outputs/models"))
    parser.add_argument("--data-dir",    default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
    parser.add_argument("--output-dir",  default=os.environ.get("SM_OUTPUT_DATA_DIR", "outputs"))

    # Hyperparameters (override config.yaml or pass via SageMaker)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--hidden-size",   type=int,   default=64)
    parser.add_argument("--num-layers",    type=int,   default=2)
    parser.add_argument("--latent-dim",    type=int,   default=32)
    parser.add_argument("--dropout",       type=float, default=0.2)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--threshold-pct", type=int,   default=95)

    return parser.parse_args()


def load_data(data_dir: str):
    data_path = Path(data_dir)
    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")

    # For autoencoder: train ONLY on normal sequences (no anomalies)
    normal_idx = np.where(y_train == 0)[0]
    X_normal   = X_train[normal_idx]
    logger.info(f"Training on {len(X_normal)} normal sequences "
                f"({len(X_normal)/len(X_train):.1%} of total)")

    n_val  = int(len(X_normal) * 0.1)
    X_val  = X_normal[:n_val]
    X_tr   = X_normal[n_val:]

    return X_tr, X_val, X_train, y_train


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.FloatTensor(X)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss   = criterion(output, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            total_loss += criterion(output, batch).item()
    return total_loss / len(loader)


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────
    X_tr, X_val, X_train_full, y_train_full = load_data(args.data_dir)
    input_size = X_tr.shape[2]

    train_loader = make_loader(X_tr,  args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val, args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────
    config_override = {
        "model": {
            "hidden_size": args.hidden_size,
            "num_layers":  args.num_layers,
            "latent_dim":  args.latent_dim,
            "dropout":     args.dropout,
        },
        "data": {"sequence_length": X_tr.shape[1]}
    }
    model = build_model(config_override, input_size).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # ── MLflow tracking ───────────────────────────────────────────
    mlflow.set_experiment("sensor-anomaly-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "hidden_size": args.hidden_size,
            "num_layers":  args.num_layers,
            "latent_dim":  args.latent_dim,
            "dropout":     args.dropout,
            "input_size":  input_size,
            "n_train_seq": len(X_tr),
        })

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = val_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Path(args.model_dir) / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # ── Threshold calibration on full training data ────────────
        model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt"))
        all_tensor = torch.FloatTensor(X_train_full).to(device)
        errors = model.reconstruction_error(all_tensor)
        normal_errors = errors[y_train_full == 0]
        threshold = compute_threshold(normal_errors, percentile=args.threshold_pct)

        mlflow.log_metrics({
            "best_val_loss":        best_val_loss,
            "anomaly_threshold":    threshold,
            "threshold_percentile": args.threshold_pct,
        })

        # ── Save artefacts ─────────────────────────────────────────
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save threshold alongside model
        threshold_path = model_dir / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "percentile": args.threshold_pct}, f)

        # Save model config for inference container
        model_config = {
            "input_size":   input_size,
            "hidden_size":  args.hidden_size,
            "num_layers":   args.num_layers,
            "latent_dim":   args.latent_dim,
            "dropout":      args.dropout,
            "sequence_length": X_tr.shape[1],
        }
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f)

        mlflow.pytorch.log_model(model, "model")
        logger.success(f"Training complete. Best val loss: {best_val_loss:.6f}")
        logger.success(f"Artefacts saved to: {args.model_dir}")


if __name__ == "__main__":
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    run_training(args)