"""
Feature engineering pipeline for NASA CMAPSS sensor data.
Supports all 4 subsets: FD001, FD002, FD003, FD004.

Key differences across subsets:
  FD001/FD003 — single operating condition  (no clustering needed)
  FD002/FD004 — 6 operating conditions      (normalise per-condition cluster)

Steps:
  1. Parse raw text files
  2. For multi-condition subsets: cluster operating points, normalise per cluster
  3. Drop near-zero variance sensors
  4. Rolling statistics (mean, std) per engine
  5. Min-max normalise (fit on train, apply to test)
  6. Sliding window sequences for LSTM input
  7. Save as .npy + .parquet per subset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from loguru import logger
import yaml
import joblib
import argparse


COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"os{i}" for i in range(1, 4)]
    + [f"s{i}"  for i in range(1, 22)]
)

MULTI_CONDITION_SUBSETS = {"FD002", "FD004"}
N_OPERATING_CONDITIONS  = 6


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_raw_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df.dropna(axis=1, inplace=True)
    logger.info(f"Loaded {Path(filepath).name}: {len(df):,} rows")
    return df


def add_rul_labels(df: pd.DataFrame, rul_file: str = None, mode: str = "train",
                   anomaly_rul: int = 30) -> pd.DataFrame:
    if mode == "train":
        max_cycles = df.groupby("unit")["cycle"].max().reset_index()
        max_cycles.columns = ["unit", "max_cycle"]
        df = df.merge(max_cycles, on="unit")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop(columns=["max_cycle"], inplace=True)
    elif mode == "test" and rul_file:
        rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul_at_end"])
        # Map each engine unit to its ground-truth RUL at last observed cycle
        units = sorted(df["unit"].unique())
        rul_map = {unit: rul_df.iloc[i]["rul_at_end"] for i, unit in enumerate(units)}
        max_cycles = df.groupby("unit")["cycle"].max().reset_index()
        max_cycles.columns = ["unit", "max_cycle"]
        df = df.merge(max_cycles, on="unit")
        df["rul_at_end"] = df["unit"].map(rul_map)
        df["rul"] = df["rul_at_end"] + (df["max_cycle"] - df["cycle"])
        df.drop(columns=["rul_at_end", "max_cycle"], inplace=True)

    df["anomaly"] = (df["rul"] <= anomaly_rul).astype(int)
    logger.info(f"  Anomaly rate: {df['anomaly'].mean():.2%}")
    return df


def assign_operating_condition(df: pd.DataFrame, kmeans: KMeans = None):
    """
    For FD002/FD004: cluster the 3 operating settings into 6 discrete conditions.
    Fit on train, apply same cluster labels to test.
    """
    os_cols = ["os1", "os2", "os3"]
    if kmeans is None:
        kmeans = KMeans(n_clusters=N_OPERATING_CONDITIONS, random_state=42, n_init=10)
        kmeans.fit(df[os_cols])
    df["op_condition"] = kmeans.predict(df[os_cols])
    return df, kmeans


def normalise_per_condition(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             feature_cols: list, subset: str):
    """
    For multi-condition subsets: fit a separate scaler per operating condition.
    This removes condition-driven variance so the model learns degradation patterns.
    """
    scalers = {}
    for cond in train_df["op_condition"].unique():
        scaler = MinMaxScaler()
        mask_tr = train_df["op_condition"] == cond
        mask_te = test_df["op_condition"] == cond
        train_df.loc[mask_tr, feature_cols] = scaler.fit_transform(
            train_df.loc[mask_tr, feature_cols])
        if mask_te.any():
            test_df.loc[mask_te, feature_cols] = scaler.transform(
                test_df.loc[mask_te, feature_cols])
        scalers[cond] = scaler
    logger.info(f"  Per-condition normalisation applied ({len(scalers)} conditions)")
    return train_df, test_df, scalers


def drop_low_variance_sensors(df: pd.DataFrame, drop_list: list) -> pd.DataFrame:
    cols_to_drop = [c for c in drop_list if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def add_rolling_features(df: pd.DataFrame, sensor_cols: list, window: int = 10) -> pd.DataFrame:
    new_cols = {}
    for col in sensor_cols:
        grp = df.groupby("unit")[col]
        new_cols[f"{col}_rmean"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        new_cols[f"{col}_rstd"]  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def create_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    X_list, y_list, meta_list = [], [], []
    for unit_id, group in df.groupby("unit"):
        group    = group.sort_values("cycle").reset_index(drop=True)
        features = group[feature_cols].values.astype(np.float32)
        labels   = group["anomaly"].values
        ruls     = group["rul"].values
        cycles   = group["cycle"].values
        for i in range(len(group) - seq_len + 1):
            X_list.append(features[i : i + seq_len])
            y_list.append(labels[i + seq_len - 1])
            meta_list.append({"unit": unit_id,
                              "cycle": cycles[i + seq_len - 1],
                              "rul":   ruls[i + seq_len - 1]})
    X    = np.array(X_list, dtype=np.float32)
    y    = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_list)
    logger.info(f"  Sequences: {X.shape}  |  anomaly rate: {y.mean():.2%}")
    return X, y, meta


def process_subset(subset_name: str, config: dict) -> dict:
    """Process one CMAPSS subset end-to-end. Returns feature column names."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing subset: {subset_name}")
    logger.info(f"{'='*50}")

    raw_dir  = Path(config["data"]["raw_path"])
    proc_dir = Path(config["data"]["processed_path"]) / subset_name
    proc_dir.mkdir(parents=True, exist_ok=True)

    subset_cfg  = config["data"]["subsets"][subset_name]
    drop_sens   = config["data"]["drop_sensors"]
    seq_len     = config["data"]["sequence_length"]
    anomaly_rul = config["data"]["anomaly_threshold_rul"]

    # ── Load ──────────────────────────────────────────────────────
    train_df = parse_raw_file(raw_dir / subset_cfg["train_file"])
    test_df  = parse_raw_file(raw_dir / subset_cfg["test_file"])

    # ── RUL + Labels ──────────────────────────────────────────────
    train_df = add_rul_labels(train_df, mode="train", anomaly_rul=anomaly_rul)
    test_df  = add_rul_labels(test_df,
                              rul_file=str(raw_dir / subset_cfg["rul_file"]),
                              mode="test", anomaly_rul=anomaly_rul)

    # ── Operating condition handling ──────────────────────────────
    kmeans = None
    if subset_name in MULTI_CONDITION_SUBSETS:
        logger.info(f"  Multi-condition subset — clustering operating points")
        train_df, kmeans = assign_operating_condition(train_df)
        test_df,  _      = assign_operating_condition(test_df, kmeans=kmeans)
        joblib.dump(kmeans, proc_dir / "kmeans_conditions.pkl")

    # ── Drop low-variance sensors ─────────────────────────────────
    train_df = drop_low_variance_sensors(train_df, drop_sens)
    test_df  = drop_low_variance_sensors(test_df,  drop_sens)

    sensor_cols = [c for c in train_df.columns if c.startswith("s")]

    # ── Rolling features ──────────────────────────────────────────
    train_df = add_rolling_features(train_df, sensor_cols)
    test_df  = add_rolling_features(test_df,  sensor_cols)

    feature_cols = [c for c in train_df.columns
                    if c not in ["unit", "cycle", "rul", "anomaly", "op_condition"]
                    and not c.startswith("os")]

    # ── Normalise ─────────────────────────────────────────────────
    if subset_name in MULTI_CONDITION_SUBSETS:
        train_df, test_df, scalers = normalise_per_condition(
            train_df, test_df, feature_cols, subset_name)
        joblib.dump(scalers, proc_dir / "scalers_per_condition.pkl")
    else:
        scaler = MinMaxScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
        joblib.dump(scaler, proc_dir / "scaler.pkl")

    # ── Sequences ─────────────────────────────────────────────────
    X_train, y_train, meta_train = create_sequences(train_df, feature_cols, seq_len)
    X_test,  y_test,  meta_test  = create_sequences(test_df,  feature_cols, seq_len)

    # ── Save ──────────────────────────────────────────────────────
    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_test.npy",  X_test)
    np.save(proc_dir / "y_test.npy",  y_test)
    meta_train.to_parquet(proc_dir / "meta_train.parquet", index=False)
    meta_test.to_parquet(proc_dir  / "meta_test.parquet",  index=False)

    # Save feature column names for downstream use
    import json
    with open(proc_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    logger.success(f"Saved to {proc_dir}  |  n_features={len(feature_cols)}")
    return {"subset": subset_name, "n_features": len(feature_cols),
            "feature_cols": feature_cols}


def run_pipeline(config_path: str = "configs/config.yaml",
                 subsets: list = None) -> None:
    config  = load_config(config_path)
    subsets = subsets or list(config["data"]["subsets"].keys())
    logger.info(f"Processing subsets: {subsets}")

    results = {}
    for subset in subsets:
        results[subset] = process_subset(subset, config)

    logger.success(f"\nAll subsets processed: {list(results.keys())}")
    for name, r in results.items():
        logger.info(f"  {name}: {r['n_features']} features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+",
                        default=["FD001", "FD002", "FD003", "FD004"],
                        help="Which subsets to process")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, subsets=args.subsets)