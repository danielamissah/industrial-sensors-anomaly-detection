# Custom SageMaker Training Container
# Extends the official PyTorch training image with additional dependencies

FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker

WORKDIR /opt/ml/code

# Install additional dependencies not in base image
COPY requirements.txt .
RUN pip install --no-cache-dir \
    shap==0.44.1 \
    mlflow==2.10.0 \
    great-expectations==0.18.8 \
    loguru==0.7.2

# Copy source code
COPY src/models/lstm_autoencoder.py .
COPY src/models/train.py .

# SageMaker expects the entry point at /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py