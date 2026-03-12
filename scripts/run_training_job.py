"""
Launches a SageMaker Training Job for the LSTM Autoencoder.
Runs all 4 subsets as a single job using the local model code.

Usage:
    python scripts/run_training_job.py
    python scripts/run_training_job.py --subset FD001
"""

import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from loguru import logger
import yaml
from datetime import datetime


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_training_job(subset="all", config_path="configs/config.yaml"):
    config  = load_config(config_path)
    region  = config["aws"]["region"]
    bucket  = config["aws"]["s3_bucket"]
    role    = config["aws"]["sagemaker_role"]

    session   = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name  = f"sensor-anomaly-{subset}-{timestamp}"

    logger.info(f"Launching SageMaker Training Job: {job_name}")
    logger.info(f"  Region : {region}")
    logger.info(f"  Bucket : s3://{bucket}")
    logger.info(f"  Subset : {subset}")

    hyperparameters = {
        "subset":      subset,
        "epochs":      50,
        "batch-size":  64,
        "lr":          0.001,
        "hidden-size": 64,
        "num-layers":  2,
        "latent-dim":  32,
        "dropout":     0.2,
        "patience":    10,
    }

    data_input = TrainingInput(
        s3_data      = f"s3://{bucket}/data/raw/",
        content_type = "text/plain"
    )

    estimator = PyTorch(
        entry_point       = "train.py",
        source_dir        = "src/models",
        role              = role,
        framework_version = "2.1",
        py_version        = "py310",
        instance_type     = config["aws"]["instance_type_training"],
        instance_count    = 1,
        hyperparameters   = hyperparameters,
        output_path       = f"s3://{bucket}/outputs/models/",
        code_location     = f"s3://{bucket}/code/",
        base_job_name     = "sensor-anomaly",
        sagemaker_session = session,
        environment       = {"SUBSET": subset},
        max_run           = 3600,
    )

    logger.info("Starting training — this takes 10-15 min. Live logs below:")
    logger.info(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs")

    estimator.fit(
        inputs   = {"training": data_input},
        job_name = job_name,
        wait     = True,
        logs     = True,
    )

    logger.success(f"Job complete: {job_name}")
    logger.info(f"Artifacts: s3://{bucket}/outputs/models/{job_name}/output/model.tar.gz")

    return estimator, job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all",
                        choices=["FD001","FD002","FD003","FD004","all"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_training_job(args.subset, args.config)