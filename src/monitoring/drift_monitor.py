"""
SageMaker Model Monitor setup for data quality and drift detection.
"""

import logging
import boto3
import sagemaker
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_session(region: str):
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def upload_baseline_data(bucket: str, region: str) -> str:
    """Upload local processed data to S3 as baseline for Model Monitor."""
    import numpy as np
    import json
    from pathlib import Path

    s3 = boto3.client("s3", region_name=region)
    baseline_s3_prefix = "monitoring/baseline-data"

    # Use FD001 training data as baseline
    data_dir = Path("data/processed/FD001")
    X_train  = np.load(data_dir / "X_train.npy")
    y_train  = np.load(data_dir / "y_train.npy")

    # Sample 500 sequences max to keep baseline job fast
    idx = list(range(min(500, len(X_train))))
    records = []
    for i in idx:
        records.append({"features": X_train[i].tolist(), "label": int(y_train[i])})

    # Write as JSON lines
    tmp_path = "/tmp/baseline_data.jsonl"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    s3_key = f"{baseline_s3_prefix}/baseline_data.jsonl"
    s3.upload_file(tmp_path, bucket, s3_key)
    uri = f"s3://{bucket}/{baseline_s3_prefix}"
    logger.info(f"Baseline data uploaded: {uri}  ({len(records)} records)")
    return uri


def enable_data_capture(endpoint_name: str, bucket: str, region: str, capture_pct: int = 100):
    """Enable data capture on the endpoint."""
    sm = boto3.client("sagemaker", region_name=region)

    capture_config = {
        "EnableCapture": True,
        "InitialSamplingPercentage": capture_pct,
        "DestinationS3Uri": f"s3://{bucket}/monitoring/captured-data",
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        "CaptureContentTypeHeader": {
            "JsonContentTypes": ["application/json"]
        },
    }

    # Get current endpoint config name
    endpoint_desc    = sm.describe_endpoint(EndpointName=endpoint_name)
    current_cfg_name = endpoint_desc["EndpointConfigName"]
    endpoint_cfg     = sm.describe_endpoint_config(EndpointConfigName=current_cfg_name)

    new_cfg_name = f"{current_cfg_name}-capture"
    sm.create_endpoint_config(
        EndpointConfigName  = new_cfg_name,
        ProductionVariants  = endpoint_cfg["ProductionVariants"],
        DataCaptureConfig   = capture_config,
    )
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=new_cfg_name)
    logger.info(f"Data capture enabled — waiting for endpoint update...")

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name,
                WaiterConfig={"Delay": 30, "MaxAttempts": 20})
    logger.info(f"Endpoint updated with data capture: {endpoint_name}")


def create_baseline(baseline_data_uri: str, bucket: str, role: str, session):
    """Run baseline job to compute statistics on training data."""
    monitor = DefaultModelMonitor(
        role             = role,
        instance_count   = 1,
        instance_type    = "ml.t3.xlarge",
        volume_size_in_gb= 20,
        max_runtime_in_seconds = 3600,
        sagemaker_session= session,
    )

    monitor.suggest_baseline(
        baseline_dataset = baseline_data_uri,
        dataset_format   = DatasetFormat.json(lines=True),
        output_s3_uri    = f"s3://{bucket}/monitoring/baseline-results",
        wait             = True,
        logs             = True,
    )

    logger.info("Baseline job complete.")
    return monitor


def schedule_monitoring(monitor, endpoint_name: str, bucket: str):
    """Schedule hourly monitoring checks."""
    monitor.create_monitoring_schedule(
        monitor_schedule_name    = f"{endpoint_name}-monitor",
        endpoint_input           = endpoint_name,
        output_s3_uri            = f"s3://{bucket}/monitoring/reports",
        statistics               = monitor.baseline_statistics(),
        constraints              = monitor.suggested_constraints(),
        schedule_cron_expression = CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics= True,
    )
    logger.info(f"Monitoring schedule created: {endpoint_name}-monitor")


def setup_cloudwatch_alarm(endpoint_name: str, region: str, drift_threshold: float = 0.15):
    """CloudWatch alarm when drift exceeds threshold."""
    cw = boto3.client("cloudwatch", region_name=region)
    cw.put_metric_alarm(
        AlarmName           = f"{endpoint_name}-drift-alarm",
        AlarmDescription    = "Fires when sensor data distribution drifts from training baseline",
        MetricName          = "feature_drift",
        Namespace           = f"SageMaker/{endpoint_name}",
        Statistic           = "Average",
        Period              = 3600,
        EvaluationPeriods   = 3,
        Threshold           = drift_threshold,
        ComparisonOperator  = "GreaterThanThreshold",
        TreatMissingData    = "notBreaching",
    )
    logger.info(f"CloudWatch alarm created: {endpoint_name}-drift-alarm")


if __name__ == "__main__":
    config        = load_config()
    region        = config["aws"]["region"]
    endpoint_name = config["aws"]["endpoint_name"]
    bucket        = config["aws"]["s3_bucket"]
    role          = config["aws"]["sagemaker_role"]
    session       = get_session(region)

    # 1. Upload baseline data to S3
    baseline_uri = upload_baseline_data(bucket, region)

    # 2. Enable data capture on endpoint
    enable_data_capture(endpoint_name, bucket, region)

    # 3. Create baseline statistics
    monitor = create_baseline(baseline_uri, bucket, role, session)

    # 4. Schedule hourly monitoring
    schedule_monitoring(monitor, endpoint_name, bucket)

    # 5. CloudWatch alarm
    drift_threshold = config.get("monitoring", {}).get("drift_threshold", 0.15)
    setup_cloudwatch_alarm(endpoint_name, region, drift_threshold)

    logger.info("Monitoring fully configured.")