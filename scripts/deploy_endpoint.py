"""
Deploys the trained LSTM Autoencoder to a SageMaker real-time endpoint.

Usage:
    python scripts/deploy_endpoint.py --job-name sensor-anomaly-all-20260312-123456
"""

import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from loguru import logger
import yaml
from datetime import datetime


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def deploy_endpoint(job_name: str, config_path: str = "configs/config.yaml"):
    config        = load_config(config_path)
    region        = config["aws"]["region"]
    bucket        = config["aws"]["s3_bucket"]
    role          = config["aws"]["sagemaker_role"]
    endpoint_name = config["aws"]["endpoint_name"]

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    # Get model artifacts from the completed training job
    sm_client     = boto3.client("sagemaker", region_name=region)
    job_desc      = sm_client.describe_training_job(TrainingJobName=job_name)
    model_s3_uri  = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    logger.info(f"Model artifacts: {model_s3_uri}")
    logger.info(f"Deploying to endpoint: {endpoint_name}")

    model = PyTorchModel(
        model_data        = model_s3_uri,
        role              = role,
        entry_point       = "inference.py",
        source_dir        = "src/serving",
        framework_version = "2.1",
        py_version        = "py310",
        sagemaker_session = session,
    )

    predictor = model.deploy(
        initial_instance_count = 1,
        instance_type          = config["aws"]["instance_type_inference"],
        endpoint_name          = endpoint_name,
        wait                   = True,
        update_endpoint        = False,  # Update if endpoint already exists
    )

    logger.success(f"Endpoint live: {endpoint_name}")
    logger.info(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}")

    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True, help="Completed training job name")
    parser.add_argument("--config",   default="configs/config.yaml")
    args = parser.parse_args()
    deploy_endpoint(args.job_name, args.config)