FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker

COPY dummy/sagemaker_training.tar.gz /sagemaker_training.tar.gz

RUN ${PIP} install --no-cache-dir \
   /sagemaker_training.tar.gz

RUN rm /sagemaker_training.tar.gz

COPY tensorflow/train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
	