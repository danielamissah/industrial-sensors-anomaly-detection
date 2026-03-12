"""
SageMaker Inference Handler for LSTM Autoencoder anomaly detection.

SageMaker calls these four functions in order:
  model_fn   → load model once on startup
  input_fn   → deserialise each request
  predict_fn → run inference
  output_fn  → serialise response

Payload format (JSON):
  Input:  {"sequence": [[[f1, f2, ...], ...], ...]}   shape: (batch, seq_len, n_features)
  Output: {"anomaly": bool, "reconstruction_error": float, "threshold": float,
           "anomaly_flags": [...], "reconstruction_errors": [...]}
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make lstm_autoencoder importable — it lives alongside this file in the container
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_autoencoder import LSTMAutoencoder


def _find_model_artifacts(model_dir: Path):
    """
    Locate model_config.json and best_model.pt.
    They may be directly in model_dir or inside a subset subfolder (e.g. FD001/).
    """
    # Direct path
    if (model_dir / "model_config.json").exists():
        return model_dir

    # Search one level deep for subset subfolders
    for sub in sorted(model_dir.iterdir()):
        if sub.is_dir() and (sub / "model_config.json").exists():
            return sub

    raise FileNotFoundError(
        f"Could not find model_config.json in {model_dir} or its subdirectories. "
        f"Contents: {list(model_dir.iterdir())}"
    )


def model_fn(model_dir: str):
    """Load model artifacts. Called once when the endpoint container starts."""
    model_dir  = Path(model_dir)
    artifact_dir = _find_model_artifacts(model_dir)

    with open(artifact_dir / "model_config.json") as f:
        cfg = json.load(f)

    with open(artifact_dir / "threshold.json") as f:
        threshold_data = json.load(f)

    # config uses "seq_len" key
    seq_len = cfg.get("seq_len") or cfg.get("sequence_length", 50)

    device = torch.device("cpu")
    model  = LSTMAutoencoder(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        latent_dim  = cfg["latent_dim"],
        seq_len     = seq_len,
        dropout     = 0.0,
    )
    model.load_state_dict(
        torch.load(artifact_dir / "best_model.pt", map_location=device)
    )
    model.eval()

    print(f"[model_fn] Loaded model for subset={cfg.get('subset','?')} "
          f"input_size={cfg['input_size']} seq_len={seq_len} "
          f"threshold={threshold_data['threshold']:.6f}")

    return {
        "model":     model,
        "threshold": threshold_data["threshold"],
        "config":    cfg,
        "device":    device,
    }


def input_fn(request_body, request_content_type):
    """Deserialise incoming request body."""
    # Decode bytes if needed
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if request_content_type in ("application/json", "application/octet-stream", ""):
        try:
            data = json.loads(request_body)
        except (json.JSONDecodeError, TypeError):
            data = request_body

        # If it's already a list (bare array), use directly
        if isinstance(data, list):
            return np.array(data, dtype=np.float32)

        if isinstance(data, dict):
            # Accept "sequences", "sequence", "inputs", "data", or "instances"
            for key in ("sequences", "sequence", "inputs", "data", "instances"):
                if key in data:
                    return np.array(data[key], dtype=np.float32)
            # Last resort: take the first value in the dict
            first_val = next(iter(data.values()))
            return np.array(first_val, dtype=np.float32)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: np.ndarray, model_artifacts: dict):
    """Run anomaly detection on input sequences."""
    model     = model_artifacts["model"]
    threshold = model_artifacts["threshold"]
    device    = model_artifacts["device"]

    # Ensure 3D: (batch, seq_len, n_features)
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, ...]

    tensor = torch.FloatTensor(input_data).to(device)
    with torch.no_grad():
        errors = model.reconstruction_error(tensor)

    errors_list = errors.tolist() if hasattr(errors, "tolist") else list(errors)
    flags       = [float(e) > threshold for e in errors_list]

    return {
        "anomaly":               flags[0] if len(flags) == 1 else flags,
        "reconstruction_error":  errors_list[0] if len(errors_list) == 1 else errors_list,
        "reconstruction_errors": errors_list,
        "anomaly_flags":         flags,
        "threshold":             threshold,
        "n_anomalies":           sum(flags),
        "anomaly_rate":          sum(flags) / max(len(flags), 1),
    }


def output_fn(prediction: dict, accept: str):
    """Serialise prediction to JSON."""
    return json.dumps(prediction), "application/json"