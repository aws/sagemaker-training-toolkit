![SageMaker](https://github.com/aws/sagemaker-training-toolkit/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Training Toolkit

[![Latest Version](https://img.shields.io/pypi/v/sagemaker-training.svg)](https://pypi.python.org/pypi/sagemaker-training) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker-training.svg)](https://pypi.python.org/pypi/sagemaker-training) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)

Train machine learning models within a Docker container using Amazon SageMaker.


## :books: Background

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

To train a model, you can include your training script and dependencies in a [Docker container](https://www.docker.com/resources/what-container) that runs your training code.
A container provides an effectively isolated environment, ensuring a consistent runtime and reliable training process. 

The **SageMaker Training Toolkit** can be easily added to any Docker container, making it compatible with SageMaker for [training models](https://aws.amazon.com/sagemaker/train/).
If you use a [prebuilt SageMaker Docker image for training](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), this library may already be included.

For more information, see the Amazon SageMaker Developer Guide sections on [using Docker containers for training](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html).

## :hammer_and_wrench: Installation

To install this library in your Docker image, add the following line to your [Dockerfile](https://docs.docker.com/engine/reference/builder/):

``` dockerfile
RUN pip3 install sagemaker-training-toolkit
```

## :computer: Usage

### Create a Docker image and train a model 

1. Write a training script. (For example, this script named `train.py` uses Tensorflow.)

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

2. [Define a container with a Dockerfile](https://docs.docker.com/get-started/part2/#define-a-container-with-dockerfile) that includes the training script and any dependencies.

    The training script must be located in the `/opt/ml/code` directory.
    The environment variable `SAGEMAKER_PROGRAM` defines which file inside the `/opt/ml/code` directory to use as the training entry point.
    When training starts, the interpreter executes the entry point defined by `SAGEMAKER_PROGRAM`.
    Python and shell scripts are both supported.
    
    ``` docker
    FROM tensorflow/tensorflow:2.0.0a0

    RUN pip install sagemaker-training-toolkit

    # Copies the training code inside the container
    COPY train.py /opt/ml/code/train.py

    # Defines train.py as script entry point
    ENV SAGEMAKER_PROGRAM train.py
    ```

3. Build and tag the Docker image.

    ``` shell
    docker build -t tf-2.0 .
    ```

4. Use the Docker image to start a training job using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

    This example uses [Local Mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode) to test the container locally:

    ``` python
    from sagemaker.estimator import Estimator

    estimator = Estimator(image_name='tf-2.0',
                          role='SageMakerRole',
                          train_instance_count=1,
                          train_instance_type='local')

    estimator.fit()
    ```
    
    To train a model using the image on SageMaker, [push the image to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html) and start a SageMaker training job with the image URI.
    

### Mapping hyperparameters to script arguments

Any hyperparameters provided by the training job will be passed by the interpreter to the entry point as script arguments.
For example the training job hyperparameters:

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

When the SageMaker Python SDK is used to create a training job with a framework containers, it passes special hyperparameters to the training job, which are parsed by SageMaker Container and the framework containers.
For example:

``` python
from sagemaker.tensorflow import TensorFlow

model_dir = 's3://SAGEMAKER-BUCKET/hvd-job-377/model'

mpi_distribution = {
  'mpi': {
    'enabled': True, 
    'custom_mpi_options': '-x HOROVOD_HIERARCHICAL_ALLREDUCE=1', 
    'processes_per_host': 8}}

estimator = TensorFlow(entry_point='train_horovod_imagenet.sh',
                       model_dir=model_dir,
                       hyperparameters={'lr': 0.3},
                       distributions=mpi_distribution,
                       ...)
```

When a training job is created using the estimator above, i.e.
`estimator.fit()` is called, the Python SDK will create additional
hyperparameters and invoke the training job as follow:

``` python
import boto3

job_hyperparameters = {
  # user provided hyperparameters
  'lr': '0.3',

  # hyperparameters created by the Python SDK and used by SageMaker Training Toolkit
  'sagemaker_job_name': 'JOB_NAME',
  'sagemaker_program': 'train_horovod_imagenet.sh',
  'sagemaker_region': 'us-west-2',
  'sagemaker_submit_directory': 's3://SAGEMAKER-BUCKET/JOB_NAME/source.tar.gz'
  'sagemaker_container_log_level': '20',
  'sagemaker_mpi_enabled': 'true',
  'sagemaker_mpi_num_of_processes_per_host': '8',

  # hyperparameters created by the Python SDK and used by the TF container
  'model_dir': 's3://SAGEMAKER-BUCKET/hvd-job-377/model'
}

boto3.client('sagemaker').create_training_job(HyperParameters=job_hyperparameters, ...)
```

As you can see in the example, in addition to user-provided
hyperparameters, the SageMaker Python SDK includes hyperparameters that
will be used by SageMaker Training Toolkit and or the framework
container. The most important SageMaker hyperparameters for training
are:

  - `sagemaker_program`: name of the user-provided entry point, it is
    **mandatory** unless environment variable `SAGEMAKER_PROGRAM` is
    provided.
  - `sagemaker_submit_directory`: local or S3 URI location of the
    source.tar.gz file containing the entry point code. It is
    **mandatory** unless the code is already located under the
    `/opt/ml/code` folder.

The complete list of hyperparameters is available
[here](https://github.com/aws/sagemaker-training-toolkit/blob/v2.4.4/src/sagemaker_training/_params.py).

### Reading additional information from the container

Very often, an entry point needs additional information from the container that is not available in `hyperparameters`.
SageMaker Containers writes this information as **environment variables** that are available inside the script.
For example, the training job below includes the channels **training** and **testing**:

``` python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point='train.py', ...)

estimator.fit({'training': 's3://bucket/path/to/training/data', 
               'testing': 's3://bucket/path/to/testing/data'})
```

The environment variable `SM_CHANNEL_{channel_name}` provides the path where the channel is located:

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

When training starts, SageMaker Training Toolkit will print all available environment variables.

### Get information from the container environment

**Training environment** creation is initialized by [environment.Environment()](https://github.com/aws/sagemaker-training-toolkit/blob/v2.4.4/src/sagemaker_training/__init__.py#L16) function call, this function returns an [Environment](https://github.com/aws/sagemaker-training-toolkit/blob/v2.4.4/src/sagemaker_training/_env.py#L413) object.
The `Environment` provides access to aspects of the training environment relevant to training jobs, including hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.
It is a read-only snapshot of the container environment during training and it doesn't contain any form of state.
Example on how a script can use Environment:

``` python
import sagemaker_training

env = sagemaker_training.environment.Environment()

# get the path of the channel 'training' from the ``inputdataconfig.json`` file
training_dir = env.channel_input_dirs['training']

# get a the hyperparameter 'training_data_file' from ``hyperparameters.json`` file
file_name = env.hyperparameters['training_data_file']

# get the folder where the model should be saved
model_dir = env.model_dir
data = np.load(os.path.join(training_dir, file_name))
x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])
model = ResNet50(weights='imagenet')
...
model.fit(x_train, y_train)

#save the model in the end of training
model.save(os.path.join(model_dir, 'saved_model'))
```

### Entry point execution

**Entry point execution** is encapsulted by [entry\_point.run(uri, user\_entry\_point, args, env\_vars)](https://github.com/aws/sagemaker-training-toolkit/blob/v2.4.4/src/sagemaker_training/entry_point.py#L22).
It prepares and executes the user entry point, passing `env_vars` as environment variables and `args` as command arguments. If the entry point is:

  - **A Python script:** executes the script as `ENV_VARS python entry
    point_name ARGS`
  - **Any other script:** executes the command as `ENV_VARS /bin/sh -c
    ./module_name ARGS`

Usage example:

``` python
import sagemaker_training
from sagemaker_training.beta.framework import entry_point

env = sagemaker_training.environment.Environment()
# {'channel-input-dirs': {'training': '/opt/ml/input/training'}, 'model_dir': '/opt/ml/model', ...}

# reading hyperparameters as a dictionary
hyperparameters = env.hyperparameters
# {'batch-size': 128, 'model_dir': '/opt/ml/model'}

# reading hyperparameters as script arguments
args = env.to_cmd_args(hyperparameters)
# ['--batch-size', '128', '--model_dir', '/opt/ml/model']

# reading the training environment as env vars
env_vars = env.to_env_vars()
# {'SAGEMAKER_CHANNELS':'training', 
#  'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
#  'MODEL_DIR':'/opt/ml/model', ...}

# executes user entry point named user_script.py as follow:
#
# SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
# SAGEMAKER_MODEL_DIR=/opt/ml/model python user_script.py --batch-size 128 --model_dir /opt/ml/model
entry_point.run('user_script.py', args, env_vars)
```

If the entry point execution fails, `trainer.train()` will write the error message to `/opt/ml/output/failure`.

The entry point touches the sucess file under `/opt/ml/success` otherwise.

### Create a framework container for training

A framework container is composed by a Dockerfile and framework-specific
logic. Let's see the MXNet container as an example:

#### Creating the Dockerfile

**Dockerfile**

``` docker
FROM mxnet/python

# install SageMaker Training Toolkit and SageMaker MXNet Container
RUN pip install sagemaker-training-toolkit sagemaker_mxnet_container

# set sagemaker_mxnet_container.training.main as framework entry point
ENV SAGEMAKER_TRAINING_MODULE sagemaker_mxnet_container.training:train
```

In the example above, MXNet and Python libraries are already installed
in the base container. The framework container only needs to install
SageMaker Training Toolkit and the SageMaker MXNet container package.
The environment variable `SAGEMAKER_TRAINING_MODULE` determines that the
function `train` under the module `training` of the container package is
going to be invoked when the container starts.

**The training package**

``` python
from sagemaker_training.beta import framework

# name of the user entry point from sagemaker hyperparameter
user_entry_point = env.module_name

# local or S3 URI location of the source.tar.gz file
module_dir = env.module_dir


def train(env):
  env = framework.environment.Environment()
  framework.entry point.run(module_dir,
                           user_entry_point,
                           env.to_cmd_args(),
                           env.to_env_vars())
```

The code above covers everything necessary for single training using
MXNet. The following example includes framework specific logic required
for distributed training.

``` python
def train(env):
  env = framework.environment.Environment()

  ps_port = '8000'

  # starts the MXNet scheduler only in the first instance
  if env.current_host == 'algo-1':
      _run_mxnet_process('scheduler', env.hosts, ps_port)

  # starts MXNet parameter server in all instances
  _run_mxnet_process('server', env.hosts, ps_port)

  framework.entry point.run(module_dir,
                           user_entry_point,
                           env.to_cmd_args(),
                           env.to_env_vars()) 
```

The implementation of `run_mxnet_process` can be found
[here](https://github.com/aws/sagemaker-mxnet-container/blob/64c5c8ed68e34fae50b6ac9521a0a28156fa8cff/src/sagemaker_mxnet_container/training.py#L45).
The example above starts the mxnet **scheduler** in the first instance
and starts the mxnet **server** in all instances.
## :scroll: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws/sagemaker-training-toolkit/blob/master/LICENSE) file.

## :handshake: Contributing

Contributions are welcome!
Please read our [contributing guidelines](https://github.com/aws/sagemaker-training-toolkit/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.
