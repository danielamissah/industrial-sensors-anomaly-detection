.PHONY: help setup data features train evaluate deploy monitor test clean

help:
	@echo "Sensor Anomaly Detection — Available Commands"
	@echo "─────────────────────────────────────────────"
	@echo "  make setup       Install dependencies"
	@echo "  make data        Download and upload dataset to S3"
	@echo "  make features    Run feature engineering pipeline"
	@echo "  make train       Train LSTM autoencoder (local)"
	@echo "  make evaluate    Evaluate model + generate SHAP plots"
	@echo "  make pipeline    Run full SageMaker pipeline"
	@echo "  make deploy      Deploy model to SageMaker endpoint"
	@echo "  make monitor     Set up Model Monitor + drift alarms"
	@echo "  make test        Run unit and integration tests"
	@echo "  make mlflow      Start local MLflow tracking server"
	@echo "  make clean       Remove cached files"

setup:
	pip install -r requirements.txt

data:
	python src/ingestion/download_data.py

features:
	python src/features/engineer.py

train:
	python src/models/train.py \
		--model-dir outputs/models \
		--data-dir  data/processed \
		--epochs 50 \
		--batch-size 64 \
		--lr 0.001

evaluate:
	python src/evaluation/evaluate.py

pipeline:
	python pipelines/sagemaker_pipeline.py --upsert --run

monitor:
	python src/monitoring/drift_monitor.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage