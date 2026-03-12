"""
Training script for LSTM Autoencoder.
Supports all 4 CMAPSS subsets — trains one model per subset.
Compatible with local execution and SageMaker Training Jobs.

Usage (local):
    python src/models/train.py --subset FD001
    python src/models/train.py --subset all
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pytorch
from loguru import logger
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from lstm_autoencoder import build_model, compute_threshold


# ── Default config used inside SageMaker container (no config.yaml available) ──
SAGEMAKER_DEFAULT_CONFIG = {
    "data": {
        "subsets": {
            "FD001": {}, "FD002": {}, "FD003": {}, "FD004": {}
        },
        "sequence_length": 50,
        "anomaly_threshold_rul": 30,
        "drop_sensors": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    },
    "model": {
        "hidden_size": 64, "num_layers": 2, "latent_dim": 32, "dropout": 0.2,
    },
    "training": {"seed": 42},
    "evaluation": {"reconstruction_threshold_percentile": 95},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",      default="all",
                        choices=["FD001","FD002","FD003","FD004","all"])
    parser.add_argument("--model-dir",   default=os.environ.get("SM_MODEL_DIR",   "outputs/models"))
    parser.add_argument("--data-dir",    default=os.environ.get("SM_CHANNEL_TRAIN","data/processed"))
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--latent-dim",  type=int,   default=32)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--threshold-pct", type=int, default=95)
    parser.add_argument("--config",      default="configs/config.yaml")
    return parser.parse_args()


def load_config(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found at '{path}' — using SageMaker defaults.")
        return SAGEMAKER_DEFAULT_CONFIG


def load_subset_data(data_dir: str, subset: str):
    path = Path(data_dir) / subset
    X_train = np.load(path / "X_train.npy")
    y_train = np.load(path / "y_train.npy")
    X_test  = np.load(path / "X_test.npy")
    y_test  = np.load(path / "y_test.npy")

    normal_idx = np.where(y_train == 0)[0]
    X_normal   = X_train[normal_idx]
    n_val      = max(1, int(len(X_normal) * 0.1))
    X_val      = X_normal[:n_val]
    X_tr       = X_normal[n_val:]

    logger.info(f"  [{subset}] train={len(X_tr):,}  val={len(X_val):,}  "
                f"test={len(X_test):,}  n_features={X_tr.shape[2]}")
    return X_tr, X_val, X_train, y_train, X_test, y_test


def make_loader(X, batch_size, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X)),
                      batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            total += criterion(model(batch), batch).item()
    return total / len(loader)


def train_one_subset(subset: str, args, config: dict, device,
                     data_dir: str = None) -> dict:
    logger.info(f"\n{'='*55}")
    logger.info(f"Training: {subset}")
    logger.info(f"{'='*55}")

    _data_dir = data_dir or args.data_dir
    X_tr, X_val, X_train_full, y_train_full, X_test, y_test = \
        load_subset_data(_data_dir, subset)

    input_size = X_tr.shape[2]
    seq_len    = X_tr.shape[1]

    model_cfg = {
        "model": {"hidden_size": args.hidden_size, "num_layers": args.num_layers,
                  "latent_dim": args.latent_dim, "dropout": args.dropout},
        "data":  {"sequence_length": seq_len}
    }
    model     = build_model(model_cfg, input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_loader = make_loader(X_tr,  args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val, args.batch_size, shuffle=False)

    model_out = Path(args.model_dir) / subset
    model_out.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("sensor-anomaly-detection")
    with mlflow.start_run(run_name=subset):
        mlflow.log_params({
            "subset": subset, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "hidden_size": args.hidden_size,
            "num_layers": args.num_layers, "latent_dim": args.latent_dim,
            "dropout": args.dropout, "input_size": input_size,
        })

        best_val, patience_ctr = float("inf"), 0

        for epoch in range(1, args.epochs + 1):
            tr_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = val_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch:03d} | train={tr_loss:.6f} | val={val_loss:.6f}")

            if val_loss < best_val:
                best_val, patience_ctr = val_loss, 0
                torch.save(model.state_dict(), model_out / "best_model.pt")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(model_out / "best_model.pt"))
        all_tensor    = torch.FloatTensor(X_train_full).to(device)
        errors        = model.reconstruction_error(all_tensor)
        normal_errors = errors[y_train_full == 0]
        threshold     = compute_threshold(normal_errors, args.threshold_pct)

        mlflow.log_metrics({"best_val_loss": best_val, "anomaly_threshold": threshold})

        model_config = {
            "subset": subset, "input_size": input_size, "seq_len": seq_len,
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "latent_dim": args.latent_dim, "dropout": args.dropout,
        }
        with open(model_out / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        with open(model_out / "threshold.json", "w") as f:
            json.dump({"threshold": threshold, "percentile": args.threshold_pct}, f, indent=2)

        mlflow.pytorch.log_model(model, f"model_{subset}")
        logger.success(f"  [{subset}] Done. best_val={best_val:.6f}  threshold={threshold:.6f}")

    return {"subset": subset, "best_val_loss": best_val, "threshold": threshold}


# ── Feature engineering inline (used in SageMaker where only raw data is available) ──
def engineer_features(raw_dir: Path, subset: str, config: dict) -> Path:
    COLS      = ["unit","cycle"] + [f"os{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]
    DROP      = config["data"]["drop_sensors"]
    SEQ       = config["data"]["sequence_length"]
    RUL_THRESH= config["data"]["anomaly_threshold_rul"]

    proc_dir  = Path("/tmp/processed") / subset
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(raw_dir / f"train_{subset}.txt",
                           sep=r"\s+", header=None, names=COLS).dropna(axis=1)
    test_df  = pd.read_csv(raw_dir / f"test_{subset}.txt",
                           sep=r"\s+", header=None, names=COLS).dropna(axis=1)
    rul_df   = pd.read_csv(raw_dir / f"RUL_{subset}.txt",
                           sep=r"\s+", header=None, names=["rul_at_end"])

    # RUL labels — train
    max_cy = train_df.groupby("unit")["cycle"].max().reset_index()
    max_cy.columns = ["unit","max_cycle"]
    train_df = train_df.merge(max_cy, on="unit")
    train_df["rul"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop(columns=["max_cycle"], inplace=True)
    train_df["anomaly"] = (train_df["rul"] <= RUL_THRESH).astype(int)

    # RUL labels — test
    units   = sorted(test_df["unit"].unique())
    rul_map = {u: rul_df.iloc[i]["rul_at_end"] for i, u in enumerate(units)}
    max_cy_te = test_df.groupby("unit")["cycle"].max().reset_index()
    max_cy_te.columns = ["unit","max_cycle"]
    test_df = test_df.merge(max_cy_te, on="unit")
    test_df["rul_at_end"] = test_df["unit"].map(rul_map)
    test_df["rul"] = test_df["rul_at_end"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df.drop(columns=["rul_at_end","max_cycle"], inplace=True)
    test_df["anomaly"] = (test_df["rul"] <= RUL_THRESH).astype(int)

    # Drop low-variance sensors
    for df in [train_df, test_df]:
        df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)

    sensor_cols = [c for c in train_df.columns if c.startswith("s")]

    # Rolling features
    for col in sensor_cols:
        for df in [train_df, test_df]:
            grp = df.groupby("unit")[col]
            df[f"{col}_rm"] = grp.transform(lambda x: x.rolling(10, min_periods=1).mean())
            df[f"{col}_rs"] = grp.transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))

    feat_cols = [c for c in train_df.columns
                 if c not in ["unit","cycle","rul","anomaly"] and not c.startswith("os")]

    # Normalise
    scaler = MinMaxScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols]  = scaler.transform(test_df[feat_cols])

    # Build sequences
    def make_seqs(df):
        X_l, y_l = [], []
        for _, grp in df.groupby("unit"):
            grp = grp.sort_values("cycle").reset_index(drop=True)
            F   = grp[feat_cols].values.astype(np.float32)
            L   = grp["anomaly"].values
            for i in range(len(grp) - SEQ + 1):
                X_l.append(F[i:i+SEQ])
                y_l.append(L[i+SEQ-1])
        return np.array(X_l, dtype=np.float32), np.array(y_l, dtype=np.int64)

    X_train, y_train = make_seqs(train_df)
    X_test,  y_test  = make_seqs(test_df)

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_test.npy",  X_test)
    np.save(proc_dir / "y_test.npy",  y_test)

    logger.info(f"  [{subset}] X_train={X_train.shape}  X_test={X_test.shape}  "
                f"anomaly_rate={y_train.mean():.2%}")
    return proc_dir.parent  # return /tmp/processed


# ── SageMaker entry point ─────────────────────────────────────────────────────
def run_sagemaker(args):
    raw_dir   = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    config    = load_config(args.config)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"SageMaker mode | raw_dir={raw_dir} | device={device}")
    logger.info(f"Files: {sorted([f.name for f in raw_dir.glob('*.txt')])}")

    subsets = ["FD001","FD002","FD003","FD004"] if args.subset == "all" else [args.subset]
    summary = []

    for subset in subsets:
        if not (raw_dir / f"train_{subset}.txt").exists():
            logger.warning(f"Skipping {subset} — files not found in {raw_dir}")
            continue

        proc_parent = engineer_features(raw_dir, subset, config)
        args.model_dir = str(model_dir)
        result = train_one_subset(subset, args, config, device,
                                  data_dir=str(proc_parent))
        summary.append(result)

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.success("SageMaker training complete.")
    for r in summary:
        logger.info(f"  {r['subset']}  best_val={r['best_val_loss']:.6f}  "
                    f"threshold={r['threshold']:.6f}")


# ── Local entry point ─────────────────────────────────────────────────────────
def run_training(args):
    config  = load_config(args.config)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    subsets = (list(config["data"]["subsets"].keys())
               if args.subset == "all" else [args.subset])

    summary = []
    for subset in subsets:
        result = train_one_subset(subset, args, config, device)
        summary.append(result)

    logger.info("\n" + "="*55)
    logger.info("TRAINING SUMMARY")
    logger.info("="*55)
    for r in summary:
        logger.info(f"  {r['subset']}  best_val={r['best_val_loss']:.6f}  "
                    f"threshold={r['threshold']:.6f}")

    summary_path = Path(args.model_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if os.environ.get("SM_MODEL_DIR"):
        run_sagemaker(args)
    else:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        run_training(args)