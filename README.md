# SageMaker Containers

SageMaker Containers contains common functionality necessary to create a container compatible with SageMaker. It can be simply used by any container by just installing the module:

```bash
pip install sagemaker-containers
```

SageMaker Containers gives you tools to create SageMaker-compatible containers, and has additional tools for letting you create Frameworks (SageMaker-compatible containers that can run arbitrary python scripts).

[SageMaker Chainer Container](https://github.com/aws/sagemaker-chainer-container) uses sagemaker-containers.

[SageMaker TensorFlow Container](https://github.com/aws/sagemaker-tensorflow-container) and [SageMaker MXNet Container](https://github.com/aws/sagemaker-mxnet-container) will be ported to use it as well in the future. 

## Getting Started -  Executing User Scripts on Amazon SageMaker

The objective of this tutorial is to explain how a script is executed inside any SageMaker-compatible container using **SageMaker Containers**.

### Creating the training job

A SageMaker training job created using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk#sagemaker-python-sdk-overview) [```Chainer```](https://github.com/aws/sagemaker-python-sdk#chainer-sagemaker-estimators), [```TensorFlow```](https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators) and [```MXNet```](https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators) takes an user script containing the model to be trained, the Hyperparameters required by the script, and information about the input data. For example:

```python
from sagemaker.chainer import Chainer

# for complete list of parameters, see 
# https://github.com/aws/sagemaker-python-sdk#sagemaker-python-sdk-overview
estimator = Chainer(entry_point='user-script.py', 
                    hyperparameters={'batch-size':256, 
                                     'learning-rate':0.0001, 
                                     'communicator':'pure_nccl'},
                    ...) 

# starts the training job with an input data channel named training pointing to 
# s3://bucket/path/to/training/data
# for more information about data channels, see
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig
chainer_estimator.fit({'training': 's3://bucket/path/to/training/data', 'testing': 's3://bucket/path/to/testing/data')
```

### How a script is executed inside the container

When the container starts for training, **SageMaker Containers** installs the user script as a Python module. The module name matches the script name. In the case above, **user-script.py** is transformed in a Python module named **user-script**.

After that, the Python interpreter executes the user module, passing ```hyperparameters``` as script arguments. The example above will be executed by **SageMaker Containers** as follow:

```bash
python -m user-script --batch-size 256 --learning_rate 0.0001 --communicator pure_nccl
```

A user provide script consumes the hyperparameters using any argument parsing library, for example:

```python
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

Very often, a user script needs additional information from the container that is not available in ```hyperparameters```.
SageMaker Containers writes this information as **environment variables** that are available inside the script.

For example, the example above can read information about the **training** channel provided in the training job request:

```python
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  ...
  
  # reads input channels training and testing from the environment variables
  parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
  parser.add_argument('--testing', type=str, default=os.environ['SM_CHANNEL_TESTING'])

  args = parser.parse_args()
  ...
```
### List of provided environment variables by SageMaker Containers

The list of the environment variables is logged and available in cloudwatch logs. From the example above:
```bash
SM_NUM_GPUS=1
SM_NUM_CPUS=4
SM_NETWORK_INTERFACE_NAME=ethwe

SM_CURRENT_HOST=algo-1
SM_HOSTS=["algo-1","algo-2"]
SM_LOG_LEVEL=20

SM_USER_ARGS=["--batch-size","256","--learning-rate","0.0001","--communicator","pure_nccl"]

SM_HP_LEARNING_RATE=0.0001
SM_HP_BATCH-SIZE=10000

SM_HPS={"batch-size": '256', "learning-rate": "0.0001","communicator": "pure_nccl"}

SM_CHANNELS=["testing","training"]
SM_CHANNEL_TRAINING=/opt/ml/input/data/training
SM_CHANNEL_TESTING=/opt/ml/input/data/test

SM_MODULE_NAME=user_script
SM_MODULE_DIR=s3://sagemaker-{aws-region}-{aws-id}/{training-job-name}/source/sourcedir.tar.gz

SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_OUTPUT_DATA_DIR=/opt/ml/output/data/algo-1
SM_MODEL_DIR=/opt/ml/model

SM_RESOURCE_CONFIG=
{
    "current_host": "algo-1",
    "hosts": [
        "algo-1",
        "algo-2"
    ]
}

SM_INPUT_DATA_CONFIG=
{
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
}


SM_FRAMEWORK_MODULE=sagemaker_chainer_container.training:main

SM_TRAINING_ENV=
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
}
```
## IMPORTANT ENVIRONMENT VARIABLES
These environment variables are those that you're likely to use when writing a user script. A full list of environment variables is given below.
### SM_MODEL_DIR
```json
SM_MODEL_DIR=/opt/ml/model
```
When the training job finishes, the container will be **deleted** including its file system expect for **/opt/ml/model** and **/opt/ml/output**. Use **/opt/ml/model** to save the model checkpoints. These checkpoints will be uploaded to the default S3 bucket. Usage example:
```python
# using it in argparse
parser.add_argument('model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

# using it as variable
model_dir = os.environ['SM_MODEL_DIR']

# saving checkpoints to model dir in chainer
serializers.save_npz(os.path.join(os.environ['SM_MODEL_DIR'], 'model.npz'), model)
```

For more information, see: [How Amazon SageMaker Processes Training Output](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-envvariables).

### SM_CHANNELS
```bash
SM_CHANNELS='["testing","training"]'
```
Contains the list of input data channels in the container.

When you run training, you can partition your training data into different logical "channels".
Depending on your problem, some common channel ideas are: "training", "testing", "evaluation" or "images'and "labels".

```SM_CHANNELS``` includes the name of the available channels in the container as a JSON encoded list. Usage example:

```python
import json

# using it in argparse
parser.add_argument('channel_names', type=int, default=json.loads(os.environ['SM_CHANNELS'])))

# using it as variable
channel_names = json.loads(os.environ['SM_CHANNELS']))
```

### SM_CHANNEL_```{channel_name}```
```bash
SM_CHANNEL_TRAINING='/opt/ml/input/data/training'
SM_CHANNEL_TESTING='/opt/ml/input/data/testing'
```
Contains the directory where the channel named ```channel_name``` is located in the container. Usage examples:

```python
import json

parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])

    
args = parser.parse_args()

train_file = np.load(os.path.join(args.train, 'train.npz'))
test_file = np.load(os.path.join(args.test, 'test.npz'))
```

### SM_HPS
```bash
SM_HPS='{"batch-size": "256", "learning-rate": "0.0001","communicator": "pure_nccl"}'
```
Contains a JSON encoded dictionary with the user provided hyperparameters. Example usage:

```python
import json

hyperparameters = json.loads(os.environ['SM_HPS']))
# {"batch-size": 256, "learning-rate": 0.0001, "communicator": "pure_nccl"}
```
### SM_HP_```{hyperparameter_name}```
```bash
SM_HP_LEARNING-RATE=0.0001
SM_HP_BATCH-SIZE=10000
SM_HP_COMMUNICATOR=pure_nccl
```
Contains value of the hyperparameter named ```hyperparameter_name```. Usage examples:

```python
learning_rate = float(os.environ['SM_HP_LEARNING-RATE'])
batch_size = int(os.environ['SM_HP_BATCH-SIZE'])
comminicator = os.environ['SM_HP_COMMUNICATOR']
```
#### SM_CURRENT_HOST
```json
SM_CURRENT_HOST=algo-1
```
The name of the current container on the container network. Usage example:

```python
# using it in argparse
parser.add_argument('current_host', type=str, default=os.environ['SM_CURRENT_HOST'])

# using it as variable
current_host = os.environ['SM_CURRENT_HOST']
```

#### SM_HOSTS
```json
SM_HOSTS='["algo-1","algo-2"]'
```
JSON encoded list containing all the hosts . Usage example:

```python
import json

# using it in argparse
parser.add_argument('hosts', type=nargs, default=json.loads(os.environ['SM_HOSTS']))

# using it as variable
hosts = json.loads(os.environ['SM_HOSTS'])
```

#### SM_NUM_GPUS
```json
SM_NUM_GPUS=1
```
The number of gpus available in the current container. Usage example:

```python
# using it in argparse
parser.add_argument('num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])

# using it as variable
num_gpus = int(os.environ['SM_NUM_GPUS'])
```
## Environment Variables full specification:
#### SM_NUM_CPUS
```json
SM_NUM_CPUS=32
```
The number of cpus available in the current container. Usage example:
```python
# using it in argparse
parser.add_argument('num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])

# using it as variable
num_cpus = int(os.environ['SM_NUM_CPUS'])
```


#### SM_LOG_LEVEL
```json
SM_LOG_LEVEL=20
```

The current log level in the container. Usage example:
```python
import logging

logger = logging.getLogger(__name__)

logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))
```

### SM_NETWORK_INTERFACE_NAME
```json
SM_NETWORK_INTERFACE_NAME=ethwe
```
Name of the network interface, useful for distributed training. Usage example:
```python
# using it in argparse
parser.add_argument('network_interface', type=str, default=os.environ['SM_NETWORK_INTERFACE_NAME'])

# using it as variable
network_interface = os.environ['SM_NETWORK_INTERFACE_NAME']
```

### SM_USER_ARGS
```json
SM_USER_ARGS='["--batch-size","256","--learning_rate","0.0001","--communicator","pure_nccl"]'
```

JSON encoded list with the script arguments provided for training.

### SM_INPUT_DIR
```json
SM_INPUT_DIR=/opt/ml/input/
```
The path of the input directory, e.g. ```/opt/ml/input/```
The input_dir, e.g. ```/opt/ml/input/```, is the directory where SageMaker saves input data and configuration files before and during training.

### SM_INPUT_CONFIG_DIR
```json
SM_INPUT_DIR=/opt/ml/input/config
```
The path of the input directory, e.g. ```/opt/ml/input/config/```. The directory where standard SageMaker configuration files are located, e.g. ```/opt/ml/input/config/```.

SageMaker training creates the following files in this folder when training starts:
- `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob request available in this file.
- `inputdataconfig.json`: You specify data channel information in the InputDataConfig parameter in a CreateTrainingJob request. Amazon SageMaker makes this information available in this file.
- `resourceconfig.json`: name of the current host and all host containers in the training.

More information about this files can be find here:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

### SM_OUTPUT_DATA_DIR
```json
SM_OUTPUT_DATA_DIR=/opt/ml/output/data/algo-1
```
The dir to write non-model training artifacts (e.g. evaluation results) which will be retained by SageMaker, e.g. ```/opt/ml/output/data```. 

As your algorithm runs in a container, it generates output including the status of the training job and model and output artifacts. Your algorithm should write this information to the this directory.

### SM_RESOURCE_CONFIG
```json
SM_RESOURCE_CONFIG='{"current_host":"algo-1","hosts":["algo-1","algo-2"]}'
```
The contents from ```/opt/ml/input/config/resourceconfig.json```. It has the following keys:
- current_host: The name of the current container on the container network.
    For example, ```'algo-1'```.
-  hosts: The list of names of all containers on the container network,
    sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
    for a three-node cluster.

For more information about resourceconfig.json:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

### SM_INPUT_DATA_CONFIG
```json
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
```
Input data configuration from ```/opt/ml/input/config/inputdataconfig.json```.

For more information about inpudataconfig.json:
  https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

### SM_TRAINING_ENV
```python
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
```
Provides the entire training information as a JSON encoded dictionary.
## License

This library is licensed under the Apache 2.0 License.
