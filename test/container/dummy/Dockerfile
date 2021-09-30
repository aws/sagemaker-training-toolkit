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

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
   && tar -xvf Python-$PYTHON_VERSION.tgz \
   && cd Python-$PYTHON_VERSION \
   && ./configure \
   && make \
   && make install \
   && apt-get update \
   && apt-get install -y --no-install-recommends \
   libreadline-gplv2-dev \
   libncursesw5-dev \
   libssl1.1 \
   openssl \
   libssl-dev \
   libsqlite3-dev \
   libgdbm-dev \
   libc6-dev \
   libbz2-dev \
   tk-dev \
   && make \
   && make install \
   && rm -rf ../Python-$PYTHON_VERSION* \
   && ln -s /usr/local/bin/pip3 /usr/bin/pip \
   && ln -s $(which ${PYTHON}) /usr/local/bin/python \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

COPY dummy/sagemaker_training.tar.gz /sagemaker_training.tar.gz

RUN ${PIP} install --upgrade pip==21.0.1

RUN ${PIP} install --no-cache-dir \
   /sagemaker_training.tar.gz

RUN rm /sagemaker_training.tar.gz

COPY dummy/train.py /opt/ml/code/train.py
COPY dummy/requirements.txt /opt/ml/code/requirements.txt

ENV SAGEMAKER_PROGRAM train.py
