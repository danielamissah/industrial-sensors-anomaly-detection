"""
Prometheus metrics exporter for sensor anomaly detection.

Calls the live SageMaker endpoint with real CMAPSS sensor sequences
and exports metrics for Grafana dashboards.

Metrics:
  sad_inference_requests_total    - Counter: total requests sent
  sad_anomalies_total             - Counter: total anomalies detected
  sad_anomaly_rate                - Gauge: current anomaly rate per subset
  sad_reconstruction_error        - Gauge: current mean reconstruction error
  sad_anomaly_threshold           - Gauge: model threshold per subset
  sad_inference_latency_seconds   - Histogram: endpoint latency
  sad_endpoint_healthy            - Gauge: 1 if endpoint responding, 0 if not

Usage:
    python src/monitoring/metrics_exporter.py
    # or via Docker: make grafana-up
"""

import os
import re
import json
import time
import logging
from pathlib import Path

import numpy as np
import boto3
import yaml
from prometheus_client import (
    start_http_server, Gauge, Counter, Histogram
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path="configs/config.yaml"):
    env_path = Path(path).parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    with open(path) as f:
        raw = f.read()
    raw = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(raw)


# Prometheus metrics
INFERENCE_REQUESTS = Counter(
    "sad_inference_requests_total",
    "Total inference requests sent to SageMaker endpoint",
    ["subset"]
)
ANOMALIES_TOTAL = Counter(
    "sad_anomalies_total",
    "Total anomalies detected",
    ["subset"]
)
ANOMALY_RATE = Gauge(
    "sad_anomaly_rate",
    "Current batch anomaly rate (0-1)",
    ["subset"]
)
RECON_ERROR = Gauge(
    "sad_reconstruction_error",
    "Mean reconstruction error for current batch",
    ["subset"]
)
THRESHOLD = Gauge(
    "sad_anomaly_threshold",
    "Anomaly detection threshold",
    ["subset"]
)
INFERENCE_LATENCY = Histogram(
    "sad_inference_latency_seconds",
    "SageMaker endpoint inference latency",
    ["subset"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
ENDPOINT_HEALTHY = Gauge(
    "sad_endpoint_healthy",
    "1 if endpoint is responding, 0 otherwise",
    ["subset"]
)


class AnomalyMetricsExporter:
    def __init__(self, config_path="configs/config.yaml"):
        self.cfg             = load_config(config_path)
        self.endpoint_name   = self.cfg["aws"]["endpoint_name"]
        self.region          = self.cfg["aws"]["region"]
        self.subsets         = self.cfg["data"]["subsets"]
        self.seq_len         = self.cfg["data"]["sequence_length"]
        self.n_features      = 14
        self.push_interval   = 30  # seconds between scrapes
        self.batch_size      = 10  # sequences per scrape

        self.runtime = boto3.client(
            "sagemaker-runtime",
            region_name=self.region
        )

        # Load real test data if available
        self.test_data = self._load_test_data()
        logger.info(f"Exporter ready — endpoint={self.endpoint_name}")

    def _load_test_data(self) -> dict:
        """Load preprocessed test sequences from data/processed/."""
        processed_dir = Path(self.cfg["data"]["processed_path"])
        data = {}
        for subset in self.subsets:
            x_path = processed_dir / subset / "X_test.npy"
            if x_path.exists():
                data[subset] = np.load(x_path)
                logger.info(f"Loaded test data: {subset} shape={data[subset].shape}")
            else:
                logger.warning(f"No test data found for {subset}: {x_path}")
        return data

    def _get_batch(self, subset: str) -> np.ndarray:
        """Return a random batch of sequences for the given subset."""
        if subset in self.test_data and len(self.test_data[subset]) > 0:
            data = self.test_data[subset]
            idx  = np.random.choice(len(data), size=min(self.batch_size, len(data)),
                                    replace=False)
            return data[idx]
        else:
            # Fallback: synthetic noise if no real data available
            return np.random.randn(self.batch_size, self.seq_len, self.n_features)

    def _call_endpoint(self, sequences: np.ndarray, subset: str) -> dict:
        """Call SageMaker endpoint and return parsed response."""
        payload = json.dumps({"sequences": sequences.tolist()})
        t0      = time.time()

        response = self.runtime.invoke_endpoint(
            EndpointName  = self.endpoint_name,
            ContentType   = "application/json",
            Body          = payload,
        )
        latency = time.time() - t0
        result  = json.loads(response["Body"].read())
        return result, latency

    def scrape_and_export(self, subset: str):
        """Run one scrape cycle for a subset."""
        sequences = self._get_batch(subset)

        try:
            result, latency = self._call_endpoint(sequences, subset)

            anomaly_flags = result.get("anomaly_flags", result.get("anomaly", []))
            recon_errors  = result.get("reconstruction_error", [])
            threshold     = result.get("threshold", 0.0)

            n_requests = len(sequences)
            n_anomalies = sum(anomaly_flags) if anomaly_flags else 0
            anomaly_rate = n_anomalies / n_requests if n_requests > 0 else 0
            mean_error   = float(np.mean(recon_errors)) if recon_errors else 0.0

            # Update metrics
            INFERENCE_REQUESTS.labels(subset=subset).inc(n_requests)
            ANOMALIES_TOTAL.labels(subset=subset).inc(n_anomalies)
            ANOMALY_RATE.labels(subset=subset).set(anomaly_rate)
            RECON_ERROR.labels(subset=subset).set(mean_error)
            THRESHOLD.labels(subset=subset).set(float(threshold))
            INFERENCE_LATENCY.labels(subset=subset).observe(latency)
            ENDPOINT_HEALTHY.labels(subset=subset).set(1)

            logger.info(
                f"{subset} | requests={n_requests} anomalies={n_anomalies} "
                f"rate={anomaly_rate:.1%} recon_error={mean_error:.4f} "
                f"threshold={threshold:.4f} latency={latency*1000:.0f}ms"
            )

            if anomaly_rate > 0.4:
                logger.warning(f"HIGH ANOMALY RATE: {subset} = {anomaly_rate:.1%}")

        except Exception as e:
            ENDPOINT_HEALTHY.labels(subset=subset).set(0)
            logger.error(f"Endpoint call failed for {subset}: {e}")

    def run(self):
        port = self.cfg.get("monitoring", {}).get("prometheus_port", 8000)
        start_http_server(port)
        logger.info(f"Metrics server started on :{port}/metrics")
        logger.info(f"Scraping endpoint: {self.endpoint_name}")

        while True:
            for subset in self.subsets:
                self.scrape_and_export(subset)
            time.sleep(self.push_interval)


if __name__ == "__main__":
    exporter = AnomalyMetricsExporter()
    exporter.run()