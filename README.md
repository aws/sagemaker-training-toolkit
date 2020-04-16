# SageMaker Training Toolkit

[![Code style:
black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)

SageMaker Training Toolkit gives you tools to create
SageMaker-compatible Docker containers, and has additional tools for
letting you create Frameworks (SageMaker-compatible Docker containers
that can run arbitrary Python or shell scripts).

Currently, this library is used by the following containers: [TensorFlow
Script
Mode](https://github.com/aws/sagemaker-tensorflow-container/tree/script-mode),
[MXNet](https://github.com/aws/sagemaker-mxnet-container),
[PyTorch](https://github.com/aws/sagemaker-pytorch-container),
[Chainer](https://github.com/aws/sagemaker-chainer-container), and
[Scikit-learn](https://github.com/aws/sagemaker-scikit-learn-container).

# Getting Started

## Creating a container using SageMaker Training Toolkit

Here we'll demonstrate how to create a Docker image using SageMaker
Training Toolkit in order to show the simplicity of using this library.

Let's suppose we need to train a model with the following training
script `train.py` using TF 2.0 in SageMaker:

``` python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test, y_test)
```

### The Dockerfile

We then create a Dockerfile with our dependencies and define the program
that will be executed in SageMaker:

``` docker
FROM tensorflow/tensorflow:2.0.0a0

RUN pip install sagemaker-training-toolkit

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py
```

More documentation on how to build a Docker container can be found
[here](https://docs.docker.com/get-started/part2/#define-a-container-with-dockerfile)

### Building the container

We then build the Docker image using `docker build`:

``` shell
docker build -t tf-2.0 .
```

### Training with Local Mode

We can use [Local
Mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode)
to test the container locally:

``` python
from sagemaker.estimator import Estimator

estimator = Estimator(image_name='tf-2.0',
                      role='SageMakerRole',
                      train_instance_count=1,
                      train_instance_type='local')

estimator.fit()
```

After using Local Mode, we can push the image to ECR and run a SageMaker
training job. To see a complete example on how to create a container
using SageMaker Container, including pushing it to ECR, see the example
notebook
[tensorflow\_bring\_your\_own.ipynb](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb).

## How a script is executed inside the container

The training script must be located under the folder `/opt/ml/code` and
its relative path is defined in the environment variable
`SAGEMAKER_PROGRAM`. The following scripts are supported:

  - **Python scripts**: uses the Python interpreter for any script with
    .py suffix
  - **Shell scripts**: uses the Shell interpreter to execute any other
    script

When training starts, the interpreter executes the entry point, from the
example above:

``` python
python train.py
```

### Mapping hyperparameters to script arguments

Any hyperparameters provided by the training job will be passed by the
interpreter to the entry point as script arguments. For example the
training job hyperparameters:

``` python
{"HyperParameters": {"batch-size": 256, "learning-rate": 0.0001, "communicator": "pure_nccl"}}
```

Will be executed as:

``` shell
./user_script.sh --batch-size 256 --learning_rate 0.0001 --communicator pure_nccl
```

The entry point is responsible for parsing these script arguments. For
example, in a Python script:

``` python
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning-rate', type=int, default=1)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--communicator', type=str)
  parser.add_argument('--frequency', type=int, default=20)

  args = parser.parse_args()
  ...
```

### Reading additional information from the container

Very often, an entry point needs additional information from the
container that is not available in `hyperparameters`. SageMaker
Containers writes this information as **environment variables** that are
available inside the script. For example, the training job below
includes the channels **training** and **testing**:

``` python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point='train.py', ...)

estimator.fit({'training': 's3://bucket/path/to/training/data', 
               'testing': 's3://bucket/path/to/testing/data'})
```

The environment variable `SM_CHANNEL_{channel_name}` provides the path
were the channel is located:

``` python
import argparse
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  ...

  # reads input channels training and testing from the environment variables
  parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
  parser.add_argument('--testing', type=str, default=os.environ['SM_CHANNEL_TESTING'])

  args = parser.parse_args()
  ...
```

When training starts, SageMaker Training Toolkit will print all
available environment variables.

