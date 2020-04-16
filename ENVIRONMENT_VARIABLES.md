## IMPORTANT ENVIRONMENT VARIABLES

These environment variables are those that you're likely to use when
writing a user script. A full list of environment variables is given
below.

### SM\_MODEL\_DIR

``` shell
SM_MODEL_DIR=/opt/ml/model
```

When the training job finishes, the container will be **deleted**
including its file system with **exception** of the `/opt/ml/model` and
`/opt/ml/output` folders. Use `/opt/ml/model` to save the model
checkpoints. These checkpoints will be uploaded to the default S3
bucket. Usage example:

``` python
import os

# using it in argparse
parser.add_argument('model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

# using it as variable
model_dir = os.environ['SM_MODEL_DIR']

# saving checkpoints to model dir in chainer
serializers.save_npz(os.path.join(os.environ['SM_MODEL_DIR'], 'model.npz'), model)
```

For more information, see: [How Amazon SageMaker Processes Training
Output](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-envvariables).

### SM\_CHANNELS

``` shell
SM_CHANNELS='["testing","training"]'
```

Contains the list of input data channels in the container.

When you run training, you can partition your training data into
different logical "channels". Depending on your problem, some common
channel ideas are: "training", "testing", "evaluation" or "images" and
"labels".

`SM_CHANNELS` includes the name of the available channels in the
container as a JSON encoded list. Usage example:

``` python
import os
import json

# using it in argparse
parser.add_argument('channel_names', default=json.loads(os.environ['SM_CHANNELS'])))

# using it as variable
channel_names = json.loads(os.environ['SM_CHANNELS']))
```

### [SM\_CHANNEL](){channel\_name}

``` shell
SM_CHANNEL_TRAINING='/opt/ml/input/data/training'
SM_CHANNEL_TESTING='/opt/ml/input/data/testing'
```

Contains the directory where the channel named `channel_name` is located
in the container. Usage examples:

``` python
import os
import json

parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])


args = parser.parse_args()

train_file = np.load(os.path.join(args.train, 'train.npz'))
test_file = np.load(os.path.join(args.test, 'test.npz'))
```

### SM\_HPS

``` shell
SM_HPS='{"batch-size": "256", "learning-rate": "0.0001","communicator": "pure_nccl"}'
```

Contains a JSON encoded dictionary with the user provided
hyperparameters. Example usage:

``` python
import os
import json

hyperparameters = json.loads(os.environ['SM_HPS']))
# {"batch-size": 256, "learning-rate": 0.0001, "communicator": "pure_nccl"}
```

### [SM\_HP](){hyperparameter\_name}

``` shell
SM_HP_LEARNING-RATE=0.0001
SM_HP_BATCH-SIZE=10000
SM_HP_COMMUNICATOR=pure_nccl
```

Contains value of the hyperparameter named `hyperparameter_name`. Usage
examples:

``` python
learning_rate = float(os.environ['SM_HP_LEARNING-RATE'])
batch_size = int(os.environ['SM_HP_BATCH-SIZE'])
comminicator = os.environ['SM_HP_COMMUNICATOR']
```

### SM\_CURRENT\_HOST

``` shell
SM_CURRENT_HOST=algo-1
```

The name of the current container on the container network. Usage
example:

``` python
import os

# using it in argparse
parser.add_argument('current_host', type=str, default=os.environ['SM_CURRENT_HOST'])

# using it as variable
current_host = os.environ['SM_CURRENT_HOST']
```

### SM\_HOSTS

``` shell
SM_HOSTS='["algo-1","algo-2"]'
```

JSON encoded list containing all the hosts . Usage example:

``` python
import os
import json

# using it in argparse
parser.add_argument('hosts', type=str, default=json.loads(os.environ['SM_HOSTS']))

# using it as variable
hosts = json.loads(os.environ['SM_HOSTS'])
```

### SM\_NUM\_GPUS

``` shell
SM_NUM_GPUS=1
```

The number of gpus available in the current container. Usage example:

``` python
import os

# using it in argparse
parser.add_argument('num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])

# using it as variable
num_gpus = int(os.environ['SM_NUM_GPUS'])
```

## List of provided environment variables by SageMaker Training Toolkit

### SM\_NUM\_CPUS

``` shell
SM_NUM_CPUS=32
```

The number of cpus available in the current container. Usage example:

``` python
# using it in argparse
parser.add_argument('num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])

# using it as variable
num_cpus = int(os.environ['SM_NUM_CPUS'])
```

### SM\_LOG\_LEVEL

``` shell
SM_LOG_LEVEL=20
```

The current log level in the container. Usage example:

``` python
import os
import logging

logger = logging.getLogger(__name__)

logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))
```

### SM\_NETWORK\_INTERFACE\_NAME

``` shell
SM_NETWORK_INTERFACE_NAME=ethwe
```

Name of the network interface, useful for distributed training. Usage
example:

``` python
# using it in argparse
parser.add_argument('network_interface', type=str, default=os.environ['SM_NETWORK_INTERFACE_NAME'])

# using it as variable
network_interface = os.environ['SM_NETWORK_INTERFACE_NAME']
```

### SM\_USER\_ARGS

``` shell
SM_USER_ARGS='["--batch-size","256","--learning_rate","0.0001","--communicator","pure_nccl"]'
```

JSON encoded list with the script arguments provided for training.

### SM\_INPUT\_DIR

``` shell
SM_INPUT_DIR=/opt/ml/input/
```

The path of the input directory, e.g. `/opt/ml/input/` The input\_dir,
e.g. `/opt/ml/input/`, is the directory where SageMaker saves input data
and configuration files before and during training.

### SM\_INPUT\_CONFIG\_DIR

``` shell
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
```

The path of the input configuration directory, e.g.
`/opt/ml/input/config/`. The directory where standard SageMaker
configuration files are located, e.g. `/opt/ml/input/config/`.

SageMaker training creates the following files in this folder when
training starts:

  - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters
    in a CreateTrainingJob request available in this file.
  - `inputdataconfig.json`: You specify data channel information in the
    InputDataConfig parameter in a CreateTrainingJob request. Amazon
    SageMaker makes this information available in this file.
  - `resourceconfig.json`: name of the current host and all host
    containers in the training.

More information about this files can be find here:
<https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>

### SM\_OUTPUT\_DATA\_DIR

``` shell
SM_OUTPUT_DATA_DIR=/opt/ml/output/data/algo-1
```

The dir to write non-model training artifacts (e.g. evaluation results)
which will be retained by SageMaker, e.g. `/opt/ml/output/data`.

As your algorithm runs in a container, it generates output including the
status of the training job and model and output artifacts. Your
algorithm should write this information to the this directory.

### SM\_RESOURCE\_CONFIG

``` shell
SM_RESOURCE_CONFIG='{"current_host":"algo-1","hosts":["algo-1","algo-2"]}'
```

The contents from `/opt/ml/input/config/resourceconfig.json`. It has the
following keys:

  - current\_host: The name of the current container on the container
    network. For example, `'algo-1'`.
  - hosts: The list of names of all containers on the container network,
    sorted lexicographically. For example, `['algo-1', 'algo-2',
    'algo-3']` for a three-node cluster.

For more information about `resourceconfig.json`:
<https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training>

### SM\_INPUT\_DATA\_CONFIG

``` shell
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

Input data configuration from
`/opt/ml/input/config/inputdataconfig.json`.

For more information about `inpudataconfig.json`:
<https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training>

### SM\_TRAINING\_ENV

``` shell
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

Provides the entire training information as a JSON-encoded dictionary.
