"""
LSTM Autoencoder for unsupervised anomaly detection on multivariate sensor data.

Architecture:
  Encoder: LSTM → compressed latent representation
  Decoder: LSTM → reconstructed input sequence

Anomaly scoring:
  Reconstruction error (MSE per timestep) — normal sequences reconstruct
  well; anomalous sequences produce high reconstruction error.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 latent_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        z = self.fc(h_n[-1])   # (batch, latent_dim)
        return z


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, num_layers: int,
                 output_size: int, seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, latent_dim)
        h0 = self.fc(z).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        # Repeat latent vector across sequence length
        z_seq = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_seq, (h0, torch.zeros_like(h0)))
        reconstruction = self.output_layer(out)  # (batch, seq_len, output_size)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 latent_dim: int = 32, seq_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, latent_dim, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, num_layers, input_size, seq_len, dropout)
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """
        Per-sample mean squared reconstruction error.
        Used as the anomaly score — higher = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))  # (batch,)
        return mse.cpu().numpy()

    def get_latent(self, x: torch.Tensor) -> np.ndarray:
        """Extract latent representations for SHAP analysis."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z.cpu().numpy()


def build_model(config: dict, input_size: int) -> LSTMAutoencoder:
    model_cfg = config["model"]
    return LSTMAutoencoder(
        input_size=input_size,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        latent_dim=model_cfg["latent_dim"],
        seq_len=config["data"]["sequence_length"],
        dropout=model_cfg["dropout"],
    )


def compute_threshold(reconstruction_errors: np.ndarray, percentile: int = 95) -> float:
    """
    Set anomaly threshold at the given percentile of training reconstruction errors.
    Sequences above this threshold are flagged as anomalous.
    """
    threshold = float(np.percentile(reconstruction_errors, percentile))
    logger.info(f"Anomaly threshold ({percentile}th percentile): {threshold:.6f}")
    return threshold