"""
Data ingestion module.
Downloads the NASA CMAPSS Turbofan Engine Degradation dataset and uploads to S3.

Dataset: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
"""

import os
import boto3
import zipfile
import urllib.request
from pathlib import Path
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_cmapss(raw_dir: str) -> None:
    """
    Downloads NASA CMAPSS FD001 dataset.
    Manual download required from:
    https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

    Place the following files in data/raw/:
      - train_FD001.txt
      - test_FD001.txt
      - RUL_FD001.txt
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    expected_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    missing = [f for f in expected_files if not (raw_path / f).exists()]

    if missing:
        logger.warning(f"Missing dataset files: {missing}")
        logger.info("Please download from: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        logger.info(f"Place files in: {raw_dir}")
    else:
        logger.info(f"All dataset files present in {raw_dir}")


def upload_to_s3(local_dir: str, bucket: str, prefix: str = "data/raw/") -> None:
    """Upload raw data files to S3 for SageMaker access."""
    s3 = boto3.client("s3")
    raw_path = Path(local_dir)

    for file_path in raw_path.glob("*.txt"):
        s3_key = f"{prefix}{file_path.name}"
        logger.info(f"Uploading {file_path.name} → s3://{bucket}/{s3_key}")
        s3.upload_file(str(file_path), bucket, s3_key)
        logger.success(f"Uploaded {file_path.name}")


def validate_data(raw_dir: str) -> bool:
    """Basic data validation — checks files exist and are non-empty."""
    raw_path = Path(raw_dir)
    files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    valid = True

    for f in files:
        path = raw_path / f
        if not path.exists():
            logger.error(f"Missing: {f}")
            valid = False
        elif path.stat().st_size == 0:
            logger.error(f"Empty file: {f}")
            valid = False
        else:
            logger.info(f"✓ {f} ({path.stat().st_size / 1024:.1f} KB)")

    return valid


if __name__ == "__main__":
    config = load_config()
    raw_dir = config["data"]["raw_path"]

    download_cmapss(raw_dir)

    if validate_data(raw_dir):
        logger.info("Data validation passed. Uploading to S3...")
        upload_to_s3(raw_dir, config["aws"]["s3_bucket"])
    else:
        logger.error("Data validation failed. Fix issues before uploading.")