ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/huggingface-pytorch-trcomp-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04

# Disable prompts during package installation
ENV DEBIAN_FRONTEND="noninteractive"

COPY test/container/trcomp_pytorch/sagemaker_training.tar.gz /sagemaker_training.tar.gz

RUN pip install --no-cache-dir \
   /sagemaker_training.tar.gz

RUN rm /sagemaker_training.tar.gz

COPY test/container/trcomp_pytorch/train.py /opt/ml/code/train.py
COPY test/container/trcomp_pytorch/requirements.txt /opt/ml/code/requirements.txt

ENV SAGEMAKER_PROGRAM train.py

RUN pip install tox