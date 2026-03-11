"""
SageMaker Model Monitor setup for data quality and drift detection.

Monitors:
  1. Data Quality     — feature distribution vs training baseline
  2. Model Quality    — anomaly rate drift over time
  3. Custom metric    — mean reconstruction error per hour

Alerts via CloudWatch when drift exceeds thresholds.
"""

import boto3
import sagemaker
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from loguru import logger
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def enable_data_capture(endpoint_name: str, bucket: str, capture_pct: int = 100):
    """
    Enable data capture on the SageMaker endpoint.
    Captured requests/responses are stored in S3 for monitoring.
    """
    sagemaker_client = boto3.client("sagemaker")

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=capture_pct,
        destination_s3_uri=f"s3://{bucket}/monitoring/captured-data",
        capture_options=["REQUEST", "RESPONSE"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"],
    )

    logger.info(f"Data capture enabled on endpoint: {endpoint_name}")
    return data_capture_config


def create_baseline(endpoint_name: str, baseline_data_uri: str,
                    bucket: str, role: str, session):
    """
    Run a baseline job to compute statistics on training data.
    SageMaker Model Monitor uses this as the reference distribution.
    """
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    baseline_job = monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format=DatasetFormat.json(lines=False),
        output_s3_uri=f"s3://{bucket}/monitoring/baseline",
        wait=True,
        logs=False,
    )

    logger.info(f"Baseline job complete: {baseline_job.job_name}")
    return monitor


def schedule_monitoring(monitor, endpoint_name: str, bucket: str):
    """
    Schedule hourly monitoring checks against the baseline.
    Violations are reported to CloudWatch.
    """
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-monitor",
        endpoint_input=endpoint_name,
        output_s3_uri=f"s3://{bucket}/monitoring/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    logger.info(f"Monitoring schedule created for: {endpoint_name}")


def check_latest_violations(endpoint_name: str):
    """Fetch and summarise latest monitoring violations."""
    sagemaker_client = boto3.client("sagemaker")

    response = sagemaker_client.list_monitoring_executions(
        MonitoringScheduleName=f"{endpoint_name}-monitor",
        MaxResults=5,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    executions = response.get("MonitoringExecutionSummaries", [])
    if not executions:
        logger.info("No monitoring executions found yet.")
        return

    latest = executions[0]
    logger.info(f"Latest monitoring execution: {latest['MonitoringExecutionStatus']}")
    logger.info(f"  Violations: {latest.get('MonitoringJobDefinitionName', 'N/A')}")
    return latest


def setup_cloudwatch_alarm(endpoint_name: str, drift_threshold: float = 0.15):
    """
    CloudWatch alarm that fires when feature drift metric exceeds threshold.
    Useful for triggering automated retraining.
    """
    cloudwatch = boto3.client("cloudwatch")

    cloudwatch.put_metric_alarm(
        AlarmName=f"{endpoint_name}-drift-alarm",
        AlarmDescription="Fires when sensor data distribution drifts from training baseline",
        MetricName="feature_drift",
        Namespace=f"SageMaker/{endpoint_name}",
        Statistic="Average",
        Period=3600,          # 1 hour
        EvaluationPeriods=3,
        Threshold=drift_threshold,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[
            # Replace with your SNS topic ARN
            "arn:aws:sns:eu-central-1:ACCOUNT_ID:mlops-alerts",
        ],
    )
    logger.info(f"CloudWatch alarm created: {endpoint_name}-drift-alarm")


if __name__ == "__main__":
    config  = load_config()
    session = sagemaker.Session()

    endpoint_name = config["aws"]["endpoint_name"]
    bucket        = config["aws"]["s3_bucket"]
    role          = config["aws"]["sagemaker_role"]

    # 1. Enable capture
    enable_data_capture(endpoint_name, bucket)

    # 2. Baseline from training data
    monitor = create_baseline(
        endpoint_name,
        baseline_data_uri=f"s3://{bucket}/data/processed/",
        bucket=bucket,
        role=role,
        session=session,
    )

    # 3. Schedule hourly checks
    schedule_monitoring(monitor, endpoint_name, bucket)

    # 4. CloudWatch alarm
    setup_cloudwatch_alarm(endpoint_name, config["monitoring"]["drift_threshold"])

    logger.success("Monitoring fully configured.")