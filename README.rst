.. _header-n957:

SageMaker Training Toolkit
==========================

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

SageMaker Training Toolkit gives you tools to create SageMaker-compatible Docker containers, and has additional tools for letting you create Frameworks
(SageMaker-compatible Docker containers that can run arbitrary Python or shell scripts).

Currently, this library is used by the following containers: `TensorFlow
Script Mode <https://github.com/aws/sagemaker-tensorflow-container/tree/script-mode>`__,
`MXNet <https://github.com/aws/sagemaker-mxnet-container>`__,
`PyTorch <https://github.com/aws/sagemaker-pytorch-container>`__,
`Chainer <https://github.com/aws/sagemaker-chainer-container>`__, and
`Scikit-learn <https://github.com/aws/sagemaker-scikit-learn-container>`__.

.. contents::

.. _header-n1174:

Getting Started
===============

.. _header-n962:

Creating a container using SageMaker Training Toolkit
-----------------------------------------------------

Here we'll demonstrate how to create a Docker image using SageMaker Training Toolkit in order to show the simplicity of using this library.

Let's suppose we need to train a model with the following training script ``train.py`` using TF 2.0 in SageMaker:

.. code:: python

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

.. _header-n965:

The Dockerfile
~~~~~~~~~~~~~~

We then create a Dockerfile with our dependencies and define the
program that will be executed in SageMaker:

.. code:: docker

   FROM tensorflow/tensorflow:2.0.0a0

   RUN pip install sagemaker-training-toolkit

   # Copies the training code inside the container
   COPY train.py /opt/ml/code/train.py

   # Defines train.py as script entry point
   ENV SAGEMAKER_PROGRAM train.py

More documentation on how to build a Docker container can be found `here <https://docs.docker.com/get-started/part2/#define-a-container-with-dockerfile>`__

.. _header-n968:

Building the container
~~~~~~~~~~~~~~~~~~~~~~

We then build the Docker image using ``docker build``:

.. code:: shell

   docker build -t tf-2.0 .

.. _header-n971:

Training with Local Mode
~~~~~~~~~~~~~~~~~~~~~~~~

We can use `Local
Mode <https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode>`__
to test the container locally:

.. code:: python

   from sagemaker.estimator import Estimator

   estimator = Estimator(image_name='tf-2.0',
                         role='SageMakerRole',
                         train_instance_count=1,
                         train_instance_type='local')

   estimator.fit()

After using Local Mode, we can push the image to ECR and run a SageMaker training job. To see a complete example on how to create a container using SageMaker
Container, including pushing it to ECR, see the example notebook `tensorflow_bring_your_own.ipynb  <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb>`__.

.. _header-n975:

How a script is executed inside the container
---------------------------------------------

The training script must be located under the folder ``/opt/ml/code`` and its relative path is defined in the environment variable ``SAGEMAKER_PROGRAM``. The following scripts are supported:

-  **Python scripts**: uses the Python interpreter for any script with
   .py suffix

-  **Shell scripts**: uses the Shell interpreter to execute any other
   script

When training starts, the interpreter executes the entry point, from the
example above:

.. code:: python

   python train.py

.. _header-n984:

Mapping hyperparameters to script arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any hyperparameters provided by the training job will be passed by the
interpreter to the entry point as script arguments. For example the
training job hyperparameters:

.. code:: python

   {"HyperParameters": {"batch-size": 256, "learning-rate": 0.0001, "communicator": "pure_nccl"}}

Will be executed as:

.. code:: shell

   ./user_script.sh --batch-size 256 --learning_rate 0.0001 --communicator pure_nccl

The entry point is responsible for parsing these script arguments. For
example, in a Python script:

.. code:: python

   import argparse
   
   if __name__ == '__main__':
     parser = argparse.ArgumentParser()

     parser.add_argument('--learning-rate', type=int, default=1)
     parser.add_argument('--batch-size', type=int, default=64)
     parser.add_argument('--communicator', type=str)
     parser.add_argument('--frequency', type=int, default=20)

     args = parser.parse_args()
     ...

.. _header-n991:

Reading additional information from the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very often, an entry point needs additional information from the
container that is not available in ``hyperparameters``. SageMaker
Containers writes this information as **environment variables** that are
available inside the script. For example, the training job below
includes the channels **training** and **testing**:

.. code:: python

   from sagemaker.pytorch import PyTorch

   estimator = PyTorch(entry_point='train.py', ...)

   estimator.fit({'training': 's3://bucket/path/to/training/data', 
                  'testing': 's3://bucket/path/to/testing/data'})

The environment variable ``SM_CHANNEL_{channel_name}`` provides the
path were the channel is located:

.. code:: python

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

When training starts, SageMaker Training Toolkit will print all available
environment variables.

.. _header-n997:

IMPORTANT ENVIRONMENT VARIABLES
-------------------------------

These environment variables are those that you're likely to use when
writing a user script. A full list of environment variables is given
below.

.. _header-n999:

SM_MODEL_DIR
~~~~~~~~~~~~

.. code:: shell

   SM_MODEL_DIR=/opt/ml/model

When the training job finishes, the container will be **deleted**
including its file system with **exception** of the ``/opt/ml/model`` and
``/opt/ml/output`` folders. Use ``/opt/ml/model`` to save the model
checkpoints. These checkpoints will be uploaded to the default S3
bucket. Usage example:

.. code:: python

   import os

   # using it in argparse
   parser.add_argument('model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

   # using it as variable
   model_dir = os.environ['SM_MODEL_DIR']

   # saving checkpoints to model dir in chainer
   serializers.save_npz(os.path.join(os.environ['SM_MODEL_DIR'], 'model.npz'), model)

For more information, see: `How Amazon SageMaker Processes Training
Output <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-envvariables>`__.

.. _header-n1004:

SM_CHANNELS
~~~~~~~~~~~

.. code:: shell

   SM_CHANNELS='["testing","training"]'

Contains the list of input data channels in the container.

When you run training, you can partition your training data into
different logical "channels". Depending on your problem, some common
channel ideas are: "training", "testing", "evaluation" or "images" and
"labels".

``SM_CHANNELS`` includes the name of the available channels in the
container as a JSON encoded list. Usage example:

.. code:: python

   import os
   import json

   # using it in argparse
   parser.add_argument('channel_names', default=json.loads(os.environ['SM_CHANNELS'])))

   # using it as variable
   channel_names = json.loads(os.environ['SM_CHANNELS']))

.. _header-n1010:

SM_CHANNEL_{channel_name}
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_CHANNEL_TRAINING='/opt/ml/input/data/training'
   SM_CHANNEL_TESTING='/opt/ml/input/data/testing'

Contains the directory where the channel named ``channel_name`` is
located in the container. Usage examples:

.. code:: python

   import os
   import json

   parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
   parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])


   args = parser.parse_args()

   train_file = np.load(os.path.join(args.train, 'train.npz'))
   test_file = np.load(os.path.join(args.test, 'test.npz'))

.. _header-n1014:

SM_HPS
~~~~~~

.. code:: shell

   SM_HPS='{"batch-size": "256", "learning-rate": "0.0001","communicator": "pure_nccl"}'

Contains a JSON encoded dictionary with the user provided
hyperparameters. Example usage:

.. code:: python

   import os
   import json

   hyperparameters = json.loads(os.environ['SM_HPS']))
   # {"batch-size": 256, "learning-rate": 0.0001, "communicator": "pure_nccl"}

.. _header-n1020:

SM_HP_{hyperparameter_name}
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_HP_LEARNING-RATE=0.0001
   SM_HP_BATCH-SIZE=10000
   SM_HP_COMMUNICATOR=pure_nccl

Contains value of the hyperparameter named ``hyperparameter_name``.
Usage examples:

.. code:: python

   learning_rate = float(os.environ['SM_HP_LEARNING-RATE'])
   batch_size = int(os.environ['SM_HP_BATCH-SIZE'])
   comminicator = os.environ['SM_HP_COMMUNICATOR']

.. _header-n1026:

SM_CURRENT_HOST
~~~~~~~~~~~~~~~

.. code:: shell

   SM_CURRENT_HOST=algo-1

The name of the current container on the container network. Usage
example:

.. code:: python

   import os

   # using it in argparse
   parser.add_argument('current_host', type=str, default=os.environ['SM_CURRENT_HOST'])

   # using it as variable
   current_host = os.environ['SM_CURRENT_HOST']

.. _header-n1032:

SM_HOSTS
~~~~~~~~

.. code:: shell

   SM_HOSTS='["algo-1","algo-2"]'

JSON encoded list containing all the hosts . Usage example:

.. code:: python

   import os
   import json

   # using it in argparse
   parser.add_argument('hosts', type=str, default=json.loads(os.environ['SM_HOSTS']))

   # using it as variable
   hosts = json.loads(os.environ['SM_HOSTS'])

.. _header-n1038:

SM_NUM_GPUS
~~~~~~~~~~~

.. code:: shell

   SM_NUM_GPUS=1

The number of gpus available in the current container. Usage example:

.. code:: python

   import os
   
   # using it in argparse
   parser.add_argument('num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])

   # using it as variable
   num_gpus = int(os.environ['SM_NUM_GPUS'])

.. _header-n1042:

List of provided environment variables by SageMaker Training Toolkit
--------------------------------------------------------------------

.. _header-n1043:

SM_NUM_CPUS
~~~~~~~~~~~

.. code:: shell

   SM_NUM_CPUS=32

The number of cpus available in the current container. Usage example:

.. code:: python

   # using it in argparse
   parser.add_argument('num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])

   # using it as variable
   num_cpus = int(os.environ['SM_NUM_CPUS'])

.. _header-n1047:

SM_LOG_LEVEL
~~~~~~~~~~~~

.. code:: shell

   SM_LOG_LEVEL=20

The current log level in the container. Usage example:

.. code:: python

   import os
   import logging

   logger = logging.getLogger(__name__)

   logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))

.. _header-n1053:

SM_NETWORK_INTERFACE_NAME
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_NETWORK_INTERFACE_NAME=ethwe

Name of the network interface, useful for distributed training. Usage
example:

.. code:: python

   # using it in argparse
   parser.add_argument('network_interface', type=str, default=os.environ['SM_NETWORK_INTERFACE_NAME'])

   # using it as variable
   network_interface = os.environ['SM_NETWORK_INTERFACE_NAME']

.. _header-n1057:

SM_USER_ARGS
~~~~~~~~~~~~

.. code:: shell

   SM_USER_ARGS='["--batch-size","256","--learning_rate","0.0001","--communicator","pure_nccl"]'

JSON encoded list with the script arguments provided for training.

.. _header-n1060:

SM_INPUT_DIR
~~~~~~~~~~~~

.. code:: shell

   SM_INPUT_DIR=/opt/ml/input/

The path of the input directory, e.g. ``/opt/ml/input/`` The input_dir,
e.g. ``/opt/ml/input/``, is the directory where SageMaker saves input
data and configuration files before and during training.

.. _header-n1063:

SM_INPUT_CONFIG_DIR
~~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_INPUT_CONFIG_DIR=/opt/ml/input/config

The path of the input configuration directory, e.g. ``/opt/ml/input/config/``. The
directory where standard SageMaker configuration files are located, e.g.
``/opt/ml/input/config/``.

SageMaker training creates the following files in this folder when
training starts: 

- ``hyperparameters.json``: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob request available in this file. 

- ``inputdataconfig.json``: You specify data channel information in the InputDataConfig parameter in a CreateTrainingJob request. Amazon SageMaker makes this information available in this file. 

- ``resourceconfig.json``: name of the current host and all host containers in the training.

More information about this files can be find here:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

.. _header-n1068:

SM_OUTPUT_DATA_DIR
~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_OUTPUT_DATA_DIR=/opt/ml/output/data/algo-1

The dir to write non-model training artifacts (e.g. evaluation results)
which will be retained by SageMaker, e.g. ``/opt/ml/output/data``.

As your algorithm runs in a container, it generates output including the
status of the training job and model and output artifacts. Your
algorithm should write this information to the this directory.

.. _header-n1072:

SM_RESOURCE_CONFIG
~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_RESOURCE_CONFIG='{"current_host":"algo-1","hosts":["algo-1","algo-2"]}'

The contents from ``/opt/ml/input/config/resourceconfig.json``. It has
the following keys:

-  current_host: The name of the current container on the container
   network. For example, ``'algo-1'``.

-  hosts: The list of names of all containers on the container network,
   sorted lexicographically. For example,
   ``['algo-1', 'algo-2', 'algo-3']`` for a three-node cluster.

For more information about ``resourceconfig.json``:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

.. _header-n1081:

SM_INPUT_DATA_CONFIG
~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   SM_INPUT_DATA_CONFIG='{
       "testing": {
           "RecordWrapperType": "None",
           "S3DistributionType": "FullyReplicated",
           "TrainingInputMode": "File"
       },
       "training": {
           "RecordWrapperType": "None",
           "S3DistributionType": "FullyReplicated",
           "TrainingInputMode": "File"
       }
   }'

Input data configuration from
``/opt/ml/input/config/inputdataconfig.json``.

For more information about ``inpudataconfig.json``:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

.. _header-n1085:

SM_TRAINING_ENV
~~~~~~~~~~~~~~~

.. code:: shell

   SM_TRAINING_ENV='
   {
       "channel_input_dirs": {
           "test": "/opt/ml/input/data/testing",
           "train": "/opt/ml/input/data/training"
       },
       "current_host": "algo-1",
       "framework_module": "sagemaker_chainer_container.training:main",
       "hosts": [
           "algo-1",
           "algo-2"
       ],
       "hyperparameters": {
           "batch-size": 10000,
           "epochs": 1
       },
       "input_config_dir": "/opt/ml/input/config",
       "input_data_config": {
           "test": {
               "RecordWrapperType": "None",
               "S3DistributionType": "FullyReplicated",
               "TrainingInputMode": "File"
           },
           "train": {
               "RecordWrapperType": "None",
               "S3DistributionType": "FullyReplicated",
               "TrainingInputMode": "File"
           }
       },
       "input_dir": "/opt/ml/input",
       "job_name": "preprod-chainer-2018-05-31-06-27-15-511",
       "log_level": 20,
       "model_dir": "/opt/ml/model",
       "module_dir": "s3://sagemaker-{aws-region}-{aws-id}/{training-job-name}/source/sourcedir.tar.gz",
       "module_name": "user_script",
       "network_interface_name": "ethwe",
       "num_cpus": 4,
       "num_gpus": 1,
       "output_data_dir": "/opt/ml/output/data/algo-1",
       "output_dir": "/opt/ml/output",
       "resource_config": {
           "current_host": "algo-1",
           "hosts": [
               "algo-1",
               "algo-2"
           ]
       }
   }'

Provides the entire training information as a JSON-encoded dictionary.
