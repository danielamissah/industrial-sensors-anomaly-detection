"""
Test the live SageMaker endpoint with both dummy and real data.

Usage:
    python scripts/test_endpoint.py              # dummy random sequences
    python scripts/test_endpoint.py --real       # real sequences from data/processed/FD001/
    python scripts/test_endpoint.py --subset FD002 --real
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def test_endpoint(use_real=False, subset="FD001", config_path="configs/config.yaml"):
    cfg           = load_config(config_path)
    region        = cfg["aws"]["region"]
    endpoint_name = cfg["aws"]["endpoint_name"]
    seq_len       = cfg["data"]["sequence_length"]

    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=120, connect_timeout=10,
                      retries={"max_attempts": 3, "mode": "adaptive"}),
    )

    if use_real:
        data_path = Path("data/processed") / subset
        X_test = np.load(data_path / "X_test.npy")
        y_test = np.load(data_path / "y_test.npy")

        normal_idx  = np.where(y_test == 0)[0]
        anomaly_idx = np.where(y_test == 1)[0]

        if len(normal_idx) == 0 or len(anomaly_idx) == 0:
            logger.warning("Could not find both normal and anomaly samples — sending first 2")
            sequences = X_test[:2].tolist()
            labels    = ["sample_0", "sample_1"]
        else:
            sequences = X_test[[normal_idx[0], anomaly_idx[0]]].tolist()
            labels    = ["normal", "anomaly"]

        logger.info(f"Sending 2 real sequences from {subset}: 1 normal + 1 anomalous")
        logger.info(f"  shape={np.array(sequences).shape}")
    else:
        n_features = 14
        sequences  = np.random.randn(2, seq_len, n_features).tolist()
        labels     = ["dummy_1", "dummy_2"]
        logger.info(f"Sending 2 dummy sequences  shape=(2, {seq_len}, {n_features})")

    payload = json.dumps({"sequences": sequences}).encode("utf-8")

    logger.info(f"Endpoint: {endpoint_name}  |  region: {region}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            response = runtime.invoke_endpoint(
                EndpointName = endpoint_name,
                ContentType  = "application/json",
                Accept       = "application/json",
                Body         = payload,
            )
            result = json.loads(response["Body"].read().decode("utf-8"))
            logger.info(f"SUCCESS on attempt {attempt + 1}")
            logger.info(f"Response:\n{json.dumps(result, indent=2)}")

            # Per-sequence summary
            errors = result.get("reconstruction_errors", [])
            flags  = result.get("anomaly_flags", [])
            logger.info("\n── Per-sequence results ──")
            for label, err, flag in zip(labels, errors, flags):
                status = "ANOMALY" if flag else "NORMAL"
                logger.info(f"  [{label}]  error={err:.6f}  {status}")
            logger.info(f"  threshold = {result['threshold']:.6f}")
            return result

        except (ReadTimeoutError, ClientError) as e:
            if attempt == max_retries - 1:
                logger.error(f"FINAL FAILURE after {max_retries} attempts")
                logger.error(f"Error: {e}")
                raise
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:120]}...")
            logger.warning(f"   Waiting {wait}s before retry...")
            time.sleep(wait)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",   action="store_true",
                        help="Use real test data from data/processed/<subset>/")
    parser.add_argument("--subset", default="FD001",
                        choices=["FD001","FD002","FD003","FD004"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    result = test_endpoint(use_real=args.real, subset=args.subset,
                           config_path=args.config)
    if result:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed")
        exit(1)