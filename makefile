.PHONY: help setup data features train evaluate pipeline monitor test mlflow clean

help:
	@echo ""
	@echo "  Sensor Anomaly Detection — Commands"
	@echo "  ──────────────────────────────────────────────────────"
	@echo "  make setup              Install all Python dependencies"
	@echo "  make data               Validate dataset files + upload to S3"
	@echo "  make features           Feature engineering (all 4 subsets)"
	@echo "  make features SUBSET=FD001   Single subset"
	@echo "  make train              Train models (all 4 subsets)"
	@echo "  make train SUBSET=FD001      Single subset"
	@echo "  make evaluate           Evaluate + SHAP plots (all 4 subsets)"
	@echo "  make pipeline           Run full SageMaker Pipeline"
	@echo "  make monitor            Configure Model Monitor + alerts"
	@echo "  make mlflow             Start MLflow UI at localhost:5001"
	@echo "  make test               Run pytest suite"
	@echo "  make clean              Remove cached files"
	@echo ""

SUBSET  ?= all
CONFIG   = configs/config.yaml
PYPATH   = PYTHONPATH=src/models:src/features:src/evaluation:src/ingestion:src/serving:src/monitoring:.

setup:
	pip install -r requirements.txt

data:
	$(PYPATH) python src/ingestion/download_data.py --config $(CONFIG)

features:
	$(PYPATH) python src/features/engineer.py \
		--subsets $(if $(filter all,$(SUBSET)),FD001 FD002 FD003 FD004,$(SUBSET)) \
		--config $(CONFIG)

train:
	$(PYPATH) python src/models/train.py \
		--subset $(SUBSET) \
		--model-dir outputs/models \
		--data-dir  data/processed \
		--epochs    50 \
		--batch-size 64 \
		--lr        0.001 \
		--config    $(CONFIG)

evaluate:
	$(PYPATH) python src/evaluation/evaluate.py \
		--subsets $(if $(filter all,$(SUBSET)),FD001 FD002 FD003 FD004,$(SUBSET)) \
		--model-dir  outputs/models \
		--data-dir   data/processed \
		--fig-dir    outputs/figures \
		--report-dir outputs/reports \
		--config     $(CONFIG)

pipeline:
	$(PYPATH) python pipelines/sagemaker_pipeline.py --upsert --run

monitor:
	$(PYPATH) python src/monitoring/drift_monitor.py --config $(CONFIG)

mlflow:
	mlflow ui --host 0.0.0.0 --port 5001

test:
	$(PYPATH) pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage 2>/dev/null; true