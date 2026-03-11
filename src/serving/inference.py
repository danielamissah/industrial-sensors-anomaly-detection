"""
SageMaker Inference Handler.

SageMaker calls these four functions to load and serve the model.
The endpoint accepts JSON payloads and returns anomaly scores + decisions.
"""

import json
import os
import tarfile
from pathlib import Path
import numpy as np
import torch

# These imports are available inside the SageMaker container
from lstm_autoencoder import LSTMAutoencoder


def model_fn(model_dir: str):
    """Load model artifacts from model_dir. Called once on container start."""
    model_dir = Path(model_dir)

    with open(model_dir / "model_config.json") as f:
        cfg = json.load(f)

    with open(model_dir / "threshold.json") as f:
        threshold_data = json.load(f)

    device = torch.device("cpu")  # inference endpoints typically CPU
    model = LSTMAutoencoder(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["sequence_length"],
        dropout=0.0,  # disable dropout at inference
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    return {
        "model":     model,
        "threshold": threshold_data["threshold"],
        "config":    cfg,
        "device":    device,
    }


def input_fn(request_body: str, request_content_type: str):
    """Deserialise incoming request."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Expects: {"sequences": [[[s1, s2, ...], ...], ...]}
        # Shape: (batch, seq_len, n_features)
        return np.array(data["sequences"], dtype=np.float32)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: np.ndarray, model_artifacts: dict):
    """Run inference. Returns dict with scores and anomaly flags."""
    model     = model_artifacts["model"]
    threshold = model_artifacts["threshold"]
    device    = model_artifacts["device"]

    tensor = torch.FloatTensor(input_data).to(device)
    errors = model.reconstruction_error(tensor)
    flags  = (errors > threshold).tolist()

    return {
        "reconstruction_errors": errors.tolist(),
        "anomaly_flags":         flags,
        "threshold":             threshold,
        "n_anomalies":           sum(flags),
        "anomaly_rate":          sum(flags) / len(flags),
    }


def output_fn(prediction: dict, accept: str):
    """Serialise prediction output."""
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")