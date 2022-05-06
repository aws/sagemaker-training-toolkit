FROM public.ecr.aws/lts/ubuntu:20.04

# Disable prompts during package installation
ENV DEBIAN_FRONTEND="noninteractive"

ARG PYTHON=python3
ARG PIP=pip3
ARG PYTHON_VERSION=3.7.10

RUN apt-get update \
   && apt-get install -y --no-install-recommends \
   build-essential \
   ca-certificates \
   wget \
   zlib1g-dev \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
   && apt-get install -y --no-install-recommends \
   libbz2-dev \
   libc6-dev \
   libffi-dev \
   libgdbm-dev \
   liblzma-dev \
   libncursesw5-dev \
   libreadline-gplv2-dev \
   libsqlite3-dev \
   libssl-dev \
   tk-dev \
   ffmpeg \
   libsm6 \
   libxext6 \
   && rm -rf /var/lib/apt/lists/* \
   && apt-get clean

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
   && tar -xvf Python-$PYTHON_VERSION.tgz \
   && cd Python-$PYTHON_VERSION \
   && ./configure && make && make install \
   && rm -rf ../Python-$PYTHON_VERSION*

RUN ${PIP} --no-cache-dir install --upgrade pip

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python \
   && ln -s $(which ${PIP}) /usr/bin/pip

COPY dummy/sagemaker_training.tar.gz /sagemaker_training.tar.gz

RUN ${PIP} install --no-cache-dir \
   /sagemaker_training.tar.gz

RUN rm /sagemaker_training.tar.gz

COPY dummy/train.py /opt/ml/code/train.py
COPY dummy/requirements.txt /opt/ml/code/requirements.txt

ENV SAGEMAKER_PROGRAM train.py
