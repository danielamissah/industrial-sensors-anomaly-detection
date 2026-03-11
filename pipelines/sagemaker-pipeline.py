"""
SageMaker Pipeline — Sensor Anomaly Detection

DAG:
  1. ProcessingStep  — feature engineering (SKLearnProcessor)
  2. TrainingStep    — LSTM autoencoder training
  3. EvaluationStep  — compute metrics + SHAP
  4. ConditionStep   — gate on F1 >= 0.75
  5. RegisterStep    — register model in Model Registry
  6. DeployStep      — create/update real-time endpoint

Run with:
    python pipelines/sagemaker_pipeline.py --run
"""

import argparse
import json
from pathlib import Path

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, TransformStep
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterInteger, ParameterFloat, ParameterString
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.step_collections import RegisterModel
import yaml
from loguru import logger


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict, sagemaker_session) -> Pipeline:
    aws_cfg  = config["aws"]
    role     = aws_cfg["sagemaker_role"]
    bucket   = aws_cfg["s3_bucket"]
    region   = aws_cfg["region"]

    # ── Pipeline Parameters (can override at run time) ────────────
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type    = ParameterString( name="TrainingInstanceType",    default_value=aws_cfg["instance_type_training"])
    model_approval_status     = ParameterString( name="ModelApprovalStatus",     default_value="PendingManualApproval")
    f1_threshold              = ParameterFloat(  name="F1Threshold",             default_value=0.75)

    # ── S3 URIs ───────────────────────────────────────────────────
    s3_raw       = f"s3://{bucket}/data/raw"
    s3_processed = f"s3://{bucket}/data/processed"
    s3_model     = f"s3://{bucket}/models"
    s3_eval      = f"s3://{bucket}/evaluation"

    # ────────────────────────────────────────────────────────────
    # STEP 1: Feature Engineering (Processing Job)
    # ────────────────────────────────────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,
        role=role,
        sagemaker_session=sagemaker_session,
    )

    feature_engineering_step = ProcessingStep(
        name="FeatureEngineering",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=s3_raw, destination="/opt/ml/processing/input/raw"),
        ],
        outputs=[
            ProcessingOutput(output_name="processed", source="/opt/ml/processing/output",
                             destination=s3_processed),
        ],
        code="src/features/engineer.py",
        job_arguments=[
            "--raw-dir",       "/opt/ml/processing/input/raw",
            "--processed-dir", "/opt/ml/processing/output",
        ],
    )

    # ────────────────────────────────────────────────────────────
    # STEP 2: Model Training
    # ────────────────────────────────────────────────────────────
    pytorch_estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            "pytorch", region, version="2.1", py_version="py310",
            image_scope="training", instance_type=aws_cfg["instance_type_training"]
        ),
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=s3_model,
        hyperparameters={
            "epochs":        50,
            "batch-size":    64,
            "lr":            0.001,
            "hidden-size":   64,
            "num-layers":    2,
            "latent-dim":    32,
            "dropout":       0.2,
            "patience":      10,
            "threshold-pct": 95,
        },
        sagemaker_session=sagemaker_session,
    )

    training_step = TrainingStep(
        name="TrainLSTMAutoencoder",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=feature_engineering_step.properties.ProcessingOutputConfig.Outputs[
                    "processed"
                ].S3Output.S3Uri,
                content_type="application/x-npy",
            )
        },
    )

    # ────────────────────────────────────────────────────────────
    # STEP 3: Evaluation
    # ────────────────────────────────────────────────────────────
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(source=s3_processed, destination="/opt/ml/processing/data"),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",
                             destination=s3_eval),
        ],
        code="src/evaluation/evaluate.py",
        property_files=[
            sagemaker.workflow.properties.PropertyFile(
                name="EvaluationReport",
                output_name="evaluation",
                path="evaluation_metrics.json",
            )
        ],
    )

    # ────────────────────────────────────────────────────────────
    # STEP 4: Quality Gate — only proceed if F1 >= threshold
    # ────────────────────────────────────────────────────────────

    # ── If model passes: register in Model Registry ───────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{s3_eval}/evaluation_metrics.json",
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=pytorch_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[aws_cfg["instance_type_inference"]],
        transform_instances=["ml.m5.large"],
        model_package_group_name=aws_cfg["model_package_group"],
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    condition_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file="EvaluationReport",
                    json_path="f1",
                ),
                right=f1_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[],  # could add SNS alert here
    )

    # ────────────────────────────────────────────────────────────
    # ASSEMBLE PIPELINE
    # ────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        name="SensorAnomalyDetectionPipeline",
        parameters=[
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            f1_threshold,
        ],
        steps=[
            feature_engineering_step,
            training_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",    action="store_true", help="Execute the pipeline")
    parser.add_argument("--upsert", action="store_true", help="Create or update pipeline definition")
    args = parser.parse_args()

    config  = load_config()
    session = sagemaker.Session()

    pipeline = build_pipeline(config, session)

    if args.upsert or args.run:
        pipeline.upsert(role_arn=config["aws"]["sagemaker_role"])
        logger.info("Pipeline upserted to SageMaker.")

    if args.run:
        execution = pipeline.start()
        logger.info(f"Pipeline execution started: {execution.arn}")
        logger.info("Monitor at: https://console.aws.amazon.com/sagemaker/home#/pipelines")


if __name__ == "__main__":
    main()