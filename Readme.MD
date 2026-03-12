
# Sensor Anomaly Detection Pipeline

**End-to-end MLOps system for industrial predictive maintenance on AWS SageMaker**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-orange)](https://pytorch.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-yellow)](https://aws.amazon.com/sagemaker/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://claude.ai/chat/LICENSE)

---

## Overview

This project builds a production-ready machine learning pipeline that monitors multivariate sensor data from industrial turbofan engines and automatically detects early signs of mechanical degradation — hours or cycles before failure occurs.

The system uses an **LSTM Autoencoder** trained exclusively on normal operating sequences. Anomalies are flagged when the model cannot accurately reconstruct an incoming sensor sequence — high reconstruction error signals a deviation from the learned normal operating pattern.

The full pipeline runs on **AWS SageMaker** with automated retraining, model versioning, real-time inference, and statistical drift monitoring.

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation**
[Download from NASA](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

The dataset contains run-to-failure sensor readings from simulated turbofan engines across 4 subsets of increasing complexity:

| Subset | Operating Conditions | Fault Modes | Train Engines | Description                             |
| ------ | -------------------- | ----------- | ------------- | --------------------------------------- |
| FD001  | 1                    | 1 (HPC)     | 100           | Baseline — clean, single condition     |
| FD002  | 6                    | 1 (HPC)     | 260           | Multi-condition, same fault             |
| FD003  | 1                    | 2 (HPC+Fan) | 100           | Single condition, two fault types       |
| FD004  | 6                    | 2 (HPC+Fan) | 249           | Hardest — multi-condition, multi-fault |

Each subset has 3 files: `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt` — 12 files total.

Each engine is described by **21 sensor channels** (temperature, pressure, fan speed, fuel flow, etc.) sampled at each operational cycle. Ground-truth Remaining Useful Life (RUL) labels are provided for the test set.

**Anomaly definition:** sequences where RUL ≤ 30 cycles are labelled as anomalous (pre-failure zone).

---

## Architecture

```
Raw Sensor Data (S3)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  Parse → drop low-variance sensors → rolling stats
│  (Processing Job)    │  → per-condition normalisation → sliding windows
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   LSTM Autoencoder   │  Encoder: LSTM → latent vector (dim=32)
│   Training Job       │  Decoder: LSTM → reconstructed sequence
│   (one per subset)   │  Threshold: 95th percentile of normal errors
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Evaluation        │  Precision / Recall / F1 / AUC-ROC
│    + SHAP            │  SHAP DeepExplainer → per-sensor attribution
└──────────┬──────────┘
           │  (F1 quality gate)
           ▼
┌─────────────────────┐
│   Model Registry     │  Versioned, with metrics + lineage
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SageMaker Endpoint  │  Real-time inference < 50ms
│  + Model Monitor     │  Hourly drift checks → CloudWatch alarms
└─────────────────────┘
```

---

## Results (Baseline — v1.0)

Models trained and evaluated independently on each CMAPSS subset. These are real results from the current baseline implementation.

| Subset | F1 Score | AUC-ROC | Precision | Recall | Anomaly Rate (True) | Anomaly Rate (Pred) |
| ------ | -------- | ------- | --------- | ------ | ------------------- | ------------------- |
| FD001  | 0.134    | 0.655   | 0.094     | 0.235  | 4.0%                | 10.0%               |
| FD002  | 0.251    | 0.692   | 0.223     | 0.285  | 5.0%                | 6.4%                |
| FD003  | 0.035    | 0.713   | 0.022     | 0.086  | 2.5%                | 9.8%                |
| FD004  | 0.063    | 0.584   | 0.044     | 0.110  | 2.9%                | 7.3%                |

### Honest Interpretation

The AUC-ROC scores (0.58–0.71) confirm the model is learning a real signal — it is not random (AUC = 0.5). However, F1 and precision are low across all subsets, indicating the threshold calibration is generating too many false positives relative to the true anomaly rate.

**Root cause analysis:**

The true anomaly rate in the test sets is very low (2.5–5.0%), but the model flags 6–10% of sequences as anomalous. This mismatch between predicted and true anomaly rates drives down precision. The 95th percentile threshold, while standard, is too permissive for datasets where anomalies represent fewer than 5% of sequences.

**Planned improvements (v1.1):**

* Raise threshold to 99th percentile to reduce false positive rate
* Experiment with sequence length (currently 50 cycles) — longer windows may capture degradation trends more cleanly
* Add a supervised fine-tuning stage using the RUL labels to sharpen the decision boundary
* Implement class-weighted reconstruction loss to penalise false positives more heavily

> These results represent the honest baseline of the current implementation. The pipeline infrastructure — SageMaker Pipelines, Model Monitor, MLflow tracking, SHAP explainability — is fully operational. Model performance is an active area of iteration, which is standard practice in production ML systems.

---

## Project Structure

```
sensor-anomaly-detection/
│
├── configs/
│   └── config.yaml                  # All settings — models, AWS, thresholds
│
├── data/
│   ├── raw/                         # Place the 12 NASA CMAPSS files here
│   └── processed/                   # Auto-generated after make features
│       ├── FD001/
│       ├── FD002/
│       ├── FD003/
│       └── FD004/
│
├── src/
│   ├── ingestion/
│   │   └── download_data.py         # Data validation + S3 upload
│   ├── features/
│   │   └── engineer.py              # Full feature pipeline (all 4 subsets)
│   ├── models/
│   │   ├── lstm_autoencoder.py      # Model architecture
│   │   └── train.py                 # Training loop + MLflow logging
│   ├── evaluation/
│   │   └── evaluate.py              # Metrics + SHAP + cross-subset plots
│   ├── serving/
│   │   └── inference.py             # SageMaker inference handler
│   └── monitoring/
│       └── drift_monitor.py         # Model Monitor + CloudWatch setup
│
├── pipelines/
│   └── sagemaker_pipeline.py        # Full SageMaker Pipeline DAG
│
├── infrastructure/
│   └── docker/
│       └── Dockerfile               # Custom SageMaker training container
│
├── notebooks/                       # EDA and experiment notebooks
├── tests/
│   ├── unit/
│   └── integration/
│
├── outputs/                         # Git-ignored — generated by the pipeline
│   ├── models/                      # Saved model weights + thresholds
│   ├── figures/                     # Plots (error distributions, ROC, SHAP)
│   └── reports/                     # evaluation_report.json
│
├── Makefile                         # Single-command entry points
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/sensor-anomaly-detection.git
cd sensor-anomaly-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
make setup
# or: pip install -r requirements.txt
```

### 4. Download the dataset

1. Go to: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
2. Download the ZIP file
3. Extract all 12 `.txt` files into `data/raw/`

Your `data/raw/` folder should contain:

```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

### 5. (Optional) Configure AWS

Only needed for the SageMaker pipeline. Skip for local execution.

```bash
aws configure
# Enter: Access Key ID, Secret Key, region (eu-central-1), output format (json)
```

Then update `configs/config.yaml`:

```yaml
aws:
  s3_bucket: "your-bucket-name"
  sagemaker_role: "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
```

---

## Running the Pipeline

### Local (recommended for development)

```bash
# Step 1 — Feature engineering for all 4 subsets
make features

# Step 2 — Train models (~15 min on CPU, ~3 min on GPU)
make train

# Step 3 — Evaluate + generate all plots including SHAP
make evaluate

# Step 4 — Inspect training runs in MLflow
make mlflow
# Open: http://localhost:5001
```

Run a single subset:

```bash
make features SUBSET=FD001
make train    SUBSET=FD001
make evaluate SUBSET=FD001
```

### AWS SageMaker (cloud)

```bash
make data      # Upload data to S3
make pipeline  # Run full automated pipeline
make monitor   # Configure drift monitoring
```

---

## Output Files

```
outputs/models/FD001/
    ├── best_model.pt          # PyTorch model weights
    ├── model_config.json      # Architecture config for inference
    └── threshold.json         # Calibrated anomaly threshold

outputs/figures/
    ├── cross_subset_comparison.png
    └── FD001/
        ├── FD001_error_dist.png
        ├── FD001_roc.png
        └── FD001_shap_importance.png

outputs/reports/
    └── evaluation_report.json
```

---

## MLOps Concepts Demonstrated

| Concept                   | Implementation                               |
| ------------------------- | -------------------------------------------- |
| Automated ML pipeline     | SageMaker Pipelines (5-step DAG)             |
| Multi-stage Docker builds | `infrastructure/docker/Dockerfile`         |
| Model quality gate        | ConditionStep — configurable F1 threshold   |
| Model versioning          | SageMaker Model Registry with lineage        |
| Real-time inference       | SageMaker Endpoint with custom handler       |
| Data drift detection      | SageMaker Model Monitor (hourly)             |
| Alerting                  | CloudWatch Alarms → SNS                     |
| Experiment tracking       | MLflow — params, metrics, artefacts per run |
| Explainability            | SHAP DeepExplainer — per-sensor attribution |
| Config management         | Single `config.yaml`— no hardcoded values |
| Reproducibility           | Fixed seeds, saved scalers and thresholds    |

---

## Key Design Decisions

**Why an LSTM Autoencoder?**
Industrial anomaly detection is fundamentally unsupervised — in real deployments, labelled failure data is scarce. An autoencoder trained only on normal data generalises better than supervised classifiers when anomaly types are unknown in advance.

**Why train one model per subset?**
FD002 and FD004 have 6 distinct operating conditions. A single model trained across all conditions conflates operating-condition variance with degradation signal. Training independently ensures each model learns the correct normal distribution for its environment.

**Why per-condition normalisation for FD002/FD004?**
Operating conditions cause large, systematic shifts in sensor readings. Normalising within each condition cluster removes this confound before sequences are passed to the LSTM.

**Why SHAP over gradient-based attribution?**
Operations teams need to know which specific sensor channels are behaving anomalously. SHAP values are additive and model-agnostic — they map directly to physical sensor labels that maintenance engineers understand.

**Why publish low baseline results?**
Honest reporting is a core principle of responsible ML engineering. The pipeline is production-grade; model performance is an iterative process. Documenting the baseline and its known failure modes demonstrates engineering maturity, not weakness.

---

## Technologies

* **PyTorch 2.6+** — LSTM Autoencoder implementation
* **scikit-learn** — Preprocessing, KMeans condition clustering, metrics
* **SHAP** — Sensor attribution via DeepExplainer
* **AWS SageMaker** — Training jobs, Processing jobs, Endpoints, Pipelines, Model Registry, Model Monitor
* **MLflow** — Experiment tracking and model logging
* **boto3** — AWS SDK for S3, CloudWatch, SageMaker APIs
* **Docker** — Custom training container
* **pytest + moto** — Unit testing with AWS mocking

---

## References

* Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* NASA Prognostics Center of Excellence.
* Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
* AWS Documentation: [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html), [Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
* Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).

---

## License

MIT License — see [LICENSE](https://claude.ai/chat/LICENSE) for details.
