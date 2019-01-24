# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from distutils import util
import json
import logging
import multiprocessing
import os
import shlex
import socket
import subprocess
import sys
import time

import boto3

from sagemaker_containers import _content_types, _logging, _mapping, _params

logger = _logging.get_logger()

SAGEMAKER_BASE_PATH = os.path.join('/opt', 'ml')  # type: str
BASE_PATH_ENV = 'SAGEMAKER_BASE_DIR'  # type: str


def _write_json(obj, path):  # type: (object, str) -> None
    """Writes a serializeable object as a JSON file"""
    with open(path, 'w') as f:
        json.dump(obj, f)


def _is_training_path_configured():  # type: () -> bool
    """Check if the tree structure with data and configuration files used for training
    exists.

    When a SageMaker Training Job is created, the Docker container that will be used for
    training is executed with the folder /opt/ml attached. The /opt/ml folder contains
    data and configurations files necessary for training.

    Outside SageMaker, the environment variable SAGEMAKER_BASE_DIR defines the location
    of the base folder.

    This function checks wheter /opt/ml exists or if the base folder variable exists

    Returns:
        (bool): indicating whether the training path is configured or not.

    """
    return os.path.exists(SAGEMAKER_BASE_PATH) or BASE_PATH_ENV in os.environ


def _set_base_path_env():  # type: () -> None
    """Sets the environment variable SAGEMAKER_BASE_DIR as
    ~/sagemaker_local/{timestamp}/opt/ml

    Returns:
        (bool): indicating whe
    """

    local_config_dir = os.path.join(os.path.expanduser('~'), 'sagemaker_local', 'jobs',
                                    str(time.time()), 'opt', 'ml')

    logger.info('Setting environment variable SAGEMAKER_BASE_DIR as %s .' % local_config_dir)
    os.environ[BASE_PATH_ENV] = local_config_dir


_is_path_configured = _is_training_path_configured()

if not _is_path_configured:
    logger.info('Directory /opt/ml does not exist.')
    _set_base_path_env()

base_dir = os.environ.get(BASE_PATH_ENV, SAGEMAKER_BASE_PATH)  # type: str

code_dir = os.path.join(base_dir, 'code')
"""str: the path of the user's code directory, e.g., /opt/ml/code/"""

model_dir = os.path.join(base_dir, 'model')  # type: str
"""str: the directory where models should be saved, e.g., /opt/ml/model/"""

input_dir = os.path.join(base_dir, 'input')  # type: str
"""str: the path of the input directory, e.g. /opt/ml/input/

The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
        and configuration files before and during training.
        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`_input_data_dir`)
        Returns:
            str: the path of the input directory, e.g. /opt/ml/input/
"""

_input_data_dir = os.path.join(input_dir, 'data')  # type: str

input_config_dir = os.path.join(input_dir, 'config')  # type: str
"""str: the path of the input directory, e.g. /opt/ml/input/config/

The directory where standard SageMaker configuration files are located, e.g. /opt/ml/input/config/.
SageMaker training creates the following files in this folder when training starts:
    - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
        request available in this file.
    - `inputdataconfig.json`: You specify data channel information in the InputDataConfig parameter
        in a CreateTrainingJob request. Amazon SageMaker makes this information available
        in this file.
    - `resourceconfig.json`: name of the current host and all host containers in the training
More information about this files can be find here:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
Returns:
    str: the path of the input directory, e.g. /opt/ml/input/config/
"""

output_dir = os.path.join(base_dir, 'output')  # type: str
"""str: the path to the output directory, e.g. /opt/ml/output/.

The directory where training success/failure indications will be written, e.g. /opt/ml/output.
To save non-model artifacts check `output_data_dir`.
Returns:
    str: the path to the output directory, e.g. /opt/ml/output/.
"""

output_data_dir = os.path.join(output_dir, 'data')  # type: str

output_intermediate_dir = os.path.join(output_dir, 'intermediate')  # type: str
"""str: the path to the intermediate output directory, e.g. /opt/ml/output/intermediate.

The directory special behavior is to move artifacts from the training instance to
s3 directory during training.
Returns:
    str: the path to the intermediate output directory, e.g. /opt/ml/output/intermediate.
"""

HYPERPARAMETERS_FILE = 'hyperparameters.json'  # type: str
RESOURCE_CONFIG_FILE = 'resourceconfig.json'  # type: str
INPUT_DATA_CONFIG_FILE = 'inputdataconfig.json'  # type: str

hyperparameters_file_dir = os.path.join(input_config_dir, HYPERPARAMETERS_FILE)  # type: str
input_data_config_file_dir = os.path.join(input_config_dir, INPUT_DATA_CONFIG_FILE)  # type: str
resource_config_file_dir = os.path.join(input_config_dir, RESOURCE_CONFIG_FILE)  # type: str


def _create_training_directories():
    """Creates the directory structure and files necessary for training under the base path
    """
    logger.info('Creating a new training folder under %s .' % base_dir)

    os.makedirs(model_dir)
    os.makedirs(input_config_dir)
    os.makedirs(output_data_dir)

    _write_json({}, hyperparameters_file_dir)
    _write_json({}, input_data_config_file_dir)

    host_name = socket.gethostname()

    resources_dict = {
        "current_host": host_name,
        "hosts": [host_name]
    }
    _write_json(resources_dict, resource_config_file_dir)


if not _is_path_configured:
    _create_training_directories()


def _create_code_dir():  # type: () -> None
    """Creates /opt/ml/code when the module is imported."""
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)


_create_code_dir()


def _read_json(path):  # type: (str) -> dict
    """Read a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        (dict[object, object]): A dictionary representation of the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def read_hyperparameters():  # type: () -> dict
    """Read the hyperparameters from /opt/ml/input/config/hyperparameters.json.

    For more information about hyperparameters.json:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-hyperparameters

    Returns:
         (dict[string, object]): a dictionary containing the hyperparameters.
    """
    hyperparameters = _read_json(hyperparameters_file_dir)

    deserialized_hps = {}

    for k, v in hyperparameters.items():
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            logger.info("Failed to parse hyperparameter %s value %s to Json.\n"
                        "Returning the value itself", k, v)

        deserialized_hps[k] = v

    return deserialized_hps


def read_resource_config():  # type: () -> dict
    """Read the resource configuration from /opt/ml/input/config/resourceconfig.json.

    For more information about resourceconfig.json:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

    Returns:
        resource_config (dict[string, object]): the contents from /opt/ml/input/config/resourceconfig.json.
                        It has the following keys:
                            - current_host: The name of the current container on the container network.
                                For example, 'algo-1'.
                            -  hosts: The list of names of all containers on the container network,
                                sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
                                for a three-node cluster.
    """
    return _read_json(resource_config_file_dir)


def read_input_data_config():  # type: () -> dict
    """Read the input data configuration from /opt/ml/input/config/inputdataconfig.json.

        For example, suppose that you specify three data channels (train, evaluation, and
        validation) in your request. This dictionary will contain:

        {'train': {
            'ContentType':  'trainingContentType',
            'TrainingInputMode': 'File',
            'S3DistributionType': 'FullyReplicated',
            'RecordWrapperType': 'None'
        },
        'evaluation' : {
            'ContentType': 'evalContentType',
            'TrainingInputMode': 'File',
            'S3DistributionType': 'FullyReplicated',
            'RecordWrapperType': 'None'
        },
        'validation': {
            'TrainingInputMode': 'File',
            'S3DistributionType': 'FullyReplicated',
            'RecordWrapperType': 'None'
        }}

        For more information about inpudataconfig.json:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-dist-training

    Returns:
            input_data_config (dict[string, object]): contents from /opt/ml/input/config/inputdataconfig.json.
    """
    return _read_json(input_data_config_file_dir)


def channel_path(channel):  # type: (str) -> str
    """ Returns the directory containing the channel data file(s) which is:
    - <self.base_dir>/input/data/<channel>

    For more information about channels: https://docs.aws.amazon.com/sagemaker/latest/dg/API_Channel.html

    Returns:
        str: The input data directory for the specified channel.
    """
    return os.path.join(_input_data_dir, channel)


def num_gpus():  # type: () -> int
    """The number of gpus available in the current container.

    Returns:
        int: number of gpus available in the current container.
    """
    try:
        cmd = shlex.split('nvidia-smi --list-gpus')
        output = subprocess.check_output(cmd).decode('utf-8')
        return sum([1 for x in output.split('\n') if x.startswith('GPU ')])
    except (OSError, subprocess.CalledProcessError):
        logger.info('No GPUs detected (normal if no gpus installed)')
        return 0


def num_cpus():  # type: () -> int
    """The number of cpus available in the current container.

    Returns:
        int: number of cpus available in the current container.
    """
    return multiprocessing.cpu_count()


class _Env(_mapping.MappingMixin):
    """Base Class which provides access to aspects of the environment including
    system characteristics, filesystem locations, environment variables and configuration settings.

    The Env is a read-only snapshot of the container environment. It does not contain any form of state.
    It is a dictionary like object, allowing any builtin function that works with dictionary.

    Attributes:
            current_host (str): The name of the current container on the container network. For example, 'algo-1'.
            module_name (str): The name of the user provided module.
            module_dir (str): The full path location of the user provided module.
    """

    def __init__(self):
        current_host = os.environ.get(_params.CURRENT_HOST_ENV)
        module_name = os.environ.get(_params.USER_PROGRAM_ENV, None)
        module_dir = os.environ.get(_params.SUBMIT_DIR_ENV, code_dir)
        log_level = int(os.environ.get(_params.LOG_LEVEL_ENV, logging.INFO))

        self._current_host = current_host
        self._num_gpus = num_gpus()
        self._num_cpus = num_cpus()
        self._module_name = module_name
        self._user_entry_point = module_name
        self._module_dir = module_dir
        self._log_level = log_level
        self._model_dir = model_dir

    @property
    def model_dir(self):  # type: () -> str
        """Returns:
            str: the directory where models should be saved, e.g., /opt/ml/model/"""
        return self._model_dir

    @property
    def current_host(self):  # type: () -> str
        """The name of the current container on the container network. For example, 'algo-1'.
        Returns:
            str: current host.
        """
        return self._current_host

    @property
    def num_gpus(self):  # type: () -> int
        """The number of gpus available in the current container.
        Returns:
            int: number of gpus available in the current container.
        """
        return self._num_gpus

    @property
    def num_cpus(self):  # type: () -> int
        """The number of cpus available in the current container.
        Returns:
            int: number of cpus available in the current container.
        """
        return self._num_cpus

    @property
    def module_name(self):  # type: () -> str
        """The name of the user provided module.
        Returns:
            str: name of the user provided module
        """
        return self._parse_module_name(self._module_name)

    @property
    def module_dir(self):  # type: () -> str
        """The full path location of the user provided module.
        Returns:
            str: full path location of the user provided module.
        """
        return self._module_dir

    @property
    def log_level(self):  # type: () -> int
        """Environment logging level.
        Returns:
            int: environment logging level.
        """
        return self._log_level

    @property
    def user_entry_point(self):  # type: () -> str
        """The name of provided user entry point.
        Returns:
            str: The name of provided user entry point
        """
        return self._user_entry_point

    @staticmethod
    def _parse_module_name(program_param):
        """Given a module name or a script name, Returns the module name.
        This function is used for backwards compatibility.
        Args:
            program_param (str): Module or script name.
        Returns:
            str: Module name
        """
        if program_param and program_param.endswith('.py'):
            return program_param[:-3]
        return program_param


class TrainingEnv(_Env):
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.

    The TrainingEnv is a read-only snapshot of the container environment during training. It does not contain any form
    of state.

    It is a dictionary like object, allowing any builtin function that works with dictionary.

    Example on how a script can use training environment:
            >>>import sagemaker_containers

            >>>env = sagemaker_containers.training_env()

            get the path of the channel 'training' from the inputdataconfig.json file
            >>>training_dir = env.channel_input_dirs['training']

            get a the hyperparameter 'training_data_file' from hyperparameters.json file
            >>>file_name = env.hyperparameters['training_data_file']

            get the folder where the model should be saved
            >>>model_dir = env.model_dir

            >>>data = np.load(os.path.join(training_dir, file_name))
            >>>x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])
            >>>model = ResNet50(weights='imagenet')
            ...
            >>>model.fit(x_train, y_train)

            save the model in the end of training
            >>>model.save(os.path.join(model_dir, 'saved_model'))

    Attributes:
        input_dir (str): The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
            and configuration files before and during training. The input data directory has the
            following subdirectories: config (`input_config_dir`) and data (`_input_data_dir`)

        input_config_dir (str): The directory where standard SageMaker configuration files are located,
            e.g. /opt/ml/input/config/.

            SageMaker training creates the following files in this folder when training starts:
                - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
                        request available in this file.
                - `inputdataconfig.json`: You specify data channel information in the InputDataConfig
                        parameter in a CreateTrainingJob request. Amazon SageMaker makes this information
                        available in this file.
                - `resourceconfig.json`: name of the current host and all host containers in the training

            More information about these files can be find here:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

        model_dir (str):  the directory where models should be saved, e.g., /opt/ml/model/

        output_dir (str): The directory where training success/failure indications will be written, e.g.
            /opt/ml/output. To save non-model artifacts check `output_data_dir`.

        hyperparameters (dict[string, object]): An instance of `HyperParameters` containing the training job
                                                hyperparameters.

        resource_config (dict[string, object]): the contents from /opt/ml/input/config/resourceconfig.json.
            It has the following keys:
                - current_host: The name of the current container on the container network.
                    For example, 'algo-1'.
                -  hosts: The list of names of all containers on the container network,
                    sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
                    for a three-node cluster.

        input_data_config (dict[string, object]): the contents from /opt/ml/input/config/inputdataconfig.json.
            For example, suppose that you specify three data channels (train, evaluation, and
            validation) in your request. This dictionary will contain:

            {'train': {
                'ContentType':  'trainingContentType',
                'TrainingInputMode': 'File',
                'S3DistributionType': 'FullyReplicated',
                'RecordWrapperType': 'None'
            },
            'evaluation' : {
                'ContentType': 'evalContentType',
                'TrainingInputMode': 'File',
                'S3DistributionType': 'FullyReplicated',
                'RecordWrapperType': 'None'
            },
            'validation': {
                'TrainingInputMode': 'File',
                'S3DistributionType': 'FullyReplicated',
                'RecordWrapperType': 'None'
            }}

            You can find more information about /opt/ml/input/config/inputdataconfig.json here:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig

        output_data_dir (str): The dir to write non-model training artifacts (e.g. evaluation results) which will be
            retained by SageMaker, e.g. /opt/ml/output/data. As your algorithm runs in a container, it generates
            output including the status of the training job and model and output artifacts. Your algorithm should
            write this information to the this directory.

        hosts (list[str]): The list of names of all containers on the container network, sorted lexicographically.
            For example, `['algo-1', 'algo-2', 'algo-3']` for a three-node cluster.

        channel_input_dirs (dict[string, string]): containing the data channels and the directories where the
            training data was saved. When you run training, you can partition your training data into different logical
            'channels'. Depending on your problem, some common channel ideas are: 'train', 'test',
            'evaluation' or 'images','labels'.

            The format of channel_input_dir is as follows:
                - `channel`(str) - the name of the channel defined in the input_data_config.
                - `training data path`(str) - the path to the directory where the training data is
                saved.
        framework_module (str):  Name of the framework module and entry point. For example:
            my_module:main

        network_interface_name (str): Name of the network interface used for distributed training

        job_name (str): The name of the current training job
    """

    def __init__(self, resource_config=None, input_data_config=None, hyperparameters=None):
        super(TrainingEnv, self).__init__()

        resource_config = resource_config or read_resource_config()
        current_host = resource_config['current_host']
        hosts = resource_config['hosts']
        input_data_config = input_data_config or read_input_data_config()

        all_hyperparameters = hyperparameters or read_hyperparameters()
        split_result = _mapping.split_by_criteria(all_hyperparameters,
                                                  keys=_params.SAGEMAKER_HYPERPARAMETERS,
                                                  prefix=_params.SAGEMAKER_PREFIX)

        sagemaker_hyperparameters = split_result.included
        additional_framework_parameters = {
            k: sagemaker_hyperparameters[k] for k in sagemaker_hyperparameters.keys()
            if k not in _params.SAGEMAKER_HYPERPARAMETERS
        }

        sagemaker_region = sagemaker_hyperparameters.get(_params.REGION_NAME_PARAM,
                                                         boto3.session.Session().region_name)

        os.environ[_params.JOB_NAME_ENV] = sagemaker_hyperparameters.get(_params.JOB_NAME_PARAM, '')
        os.environ[_params.CURRENT_HOST_ENV] = current_host
        os.environ[_params.REGION_NAME_ENV] = sagemaker_region or ''

        self._hosts = hosts

        # eth0 is the default network interface defined by SageMaker with VPC support and local mode.
        # ethwe is the current network interface defined by SageMaker training, it will be changed
        # to eth0 in the short future.
        self._network_interface_name = resource_config.get('network_interface_name', 'eth0')

        self._hyperparameters = split_result.excluded
        self._additional_framework_parameters = additional_framework_parameters
        self._resource_config = resource_config
        self._input_data_config = input_data_config
        self._output_data_dir = output_data_dir
        self._output_intermediate_dir = output_intermediate_dir
        self._channel_input_dirs = {channel: channel_path(channel) for channel in input_data_config}
        self._current_host = current_host

        # override base class attributes
        if self._module_name is None:
            self._module_name = str(sagemaker_hyperparameters.get(_params.USER_PROGRAM_PARAM, None))
        self._user_entry_point = self._user_entry_point or sagemaker_hyperparameters.get(
            _params.USER_PROGRAM_PARAM)

        self._module_dir = str(sagemaker_hyperparameters.get(_params.SUBMIT_DIR_PARAM, code_dir))
        self._log_level = sagemaker_hyperparameters.get(_params.LOG_LEVEL_PARAM, logging.INFO)
        self._sagemaker_s3_output = sagemaker_hyperparameters.get(_params.S3_OUTPUT_LOCATION_PARAM,
                                                                  None)
        self._framework_module = os.environ.get(_params.FRAMEWORK_TRAINING_MODULE_ENV, None)

        self._input_dir = input_dir
        self._input_config_dir = input_config_dir
        self._output_dir = output_dir
        self._job_name = os.environ.get(_params.TRAINING_JOB_ENV.upper(), None)

        self._master_hostname = list(hosts)[0]
        self._is_master = current_host == self._master_hostname

    @property
    def is_master(self):  # type: () -> bool
        """Returns True if host is master
        """
        return self._is_master

    @property
    def master_hostname(self):  # type: () -> str
        """Returns the hostname of the master node
        """
        return self._master_hostname

    @property
    def job_name(self):  # type: () -> str
        """The name of the current training job.

        Returns:
            str: the training job name.
        """
        return self._job_name

    @property
    def additional_framework_parameters(self):  # type: () -> dict
        """The dict of additional framework hyperparameters. All the hyperparameters prefixed with 'sagemaker_' but
            not in SAGEMAKER_HYPERPARAMETERS will be included here.

        Returns:
            dict: additional framework hyperparameters, SageMaker Python SDK adds hyperparameters with a prefix
            **sagemaker_** during training. These hyperparameters are framework independent settings and are not
            defined by the user.
        """
        return self._additional_framework_parameters

    def sagemaker_s3_output(self):  # type: () -> str
        """S3 output directory location provided by the user.

        Returns:
            str: S3 location.
        """
        return self._sagemaker_s3_output

    def to_cmd_args(self):
        """Command line arguments representation of the training environment.

        Returns:
            (list): List of cmd arguments
        """
        return _mapping.to_cmd_args(self.hyperparameters)

    def to_env_vars(self):
        """Environment variable representation of the training environment

        Returns:
            dict: an instance of dictionary
        """

        env = {
            'hosts': self.hosts, 'network_interface_name': self.network_interface_name,
            'hps': self.hyperparameters, 'user_entry_point': self.user_entry_point,
            'framework_params': self.additional_framework_parameters,
            'resource_config': self.resource_config, 'input_data_config': self.input_data_config,
            'output_data_dir': self.output_data_dir,
            'channels': sorted(self.channel_input_dirs.keys()),
            'current_host': self.current_host, 'module_name': self.module_name,
            'log_level': self.log_level,
            'framework_module': self.framework_module, 'input_dir': self.input_dir,
            'input_config_dir': self.input_config_dir, 'output_dir': self.output_dir,
            'num_cpus': self.num_cpus,
            'num_gpus': self.num_gpus, 'model_dir': self.model_dir, 'module_dir': self.module_dir,
            'training_env': dict(self), 'user_args': self.to_cmd_args(),
            'output_intermediate_dir': self.output_intermediate_dir
        }

        for name, path in self.channel_input_dirs.items():
            env['channel_%s' % name] = path

        for key, value in self.hyperparameters.items():
            env['hp_%s' % key] = value

        return _mapping.to_env_vars(env)

    @property
    def hosts(self):  # type: () -> list
        """The list of names of all containers on the container network, sorted lexicographically.
                For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.
        Returns:
              list[str]: all the hosts in the training network.
        """
        return self._hosts

    @property
    def channel_input_dirs(self):  # type: () -> dict
        """A dict[str, str] containing the data channels and the directories where the training
        data was saved.
        When you run training, you can partition your training data into different logical "channels".
        Depending on your problem, some common channel ideas are: "train", "test", "evaluation"
            or "images',"labels".
        The format of channel_input_dir is as follows:
            - `channel`[key](str) - the name of the channel defined in the input_data_config.
            - `training data path`[value](str) - the path to the directory where the training data is saved.
        Returns:
            dict[str, str] with the information about the channels.
        """
        return self._channel_input_dirs

    @property
    def network_interface_name(self):  # type: () -> str
        """Name of the network interface used for distributed training
        Returns:
              str: name of the network interface, for example, 'ethwe'
        """
        return self._network_interface_name

    @property
    def input_dir(self):  # type: () -> str
        """The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
        and configuration files before and during training.
        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`input_data_dir`)
        Returns:
            str: the path of the input directory, e.g. /opt/ml/input/
        """
        return self._input_dir

    @property
    def input_config_dir(self):  # type: () -> str
        """The directory where standard SageMaker configuration files are located, e.g. /opt/ml/input/config/.
        SageMaker training creates the following files in this folder when training starts:
            - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
                request available in this file.
            - `inputdataconfig.json`: You specify data channel information in the InputDataConfig parameter
                in a CreateTrainingJob request. Amazon SageMaker makes this information available
                in this file.
            - `resourceconfig.json`: name of the current host and all host containers in the training
        More information about this files can be find here:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
        Returns:
            str: the path of the input directory, e.g. /opt/ml/input/config/
        """
        return self._input_config_dir

    @property
    def output_dir(self):  # type: () -> str
        """The directory where training success/failure indications will be written, e.g. /opt/ml/output.
        To save non-model artifacts check `output_data_dir`.
        Returns:
            str: the path to the output directory, e.g. /opt/ml/output/.
        """
        return self._output_dir

    @property
    def hyperparameters(self):  # type: () -> dict
        """The dict of hyperparameters that were passed to the training job.
        Returns:
            dict[str, object]: An instance of `HyperParameters` containing the training job
                                                    hyperparameters.
        """
        return self._hyperparameters

    @property
    def resource_config(self):  # type: () -> dict
        """A dictionary with the contents from /opt/ml/input/config/resourceconfig.json.
                It has the following keys:
                    - current_host: The name of the current container on the container network.
                        For example, 'algo-1'.
                    -  hosts: The list of names of all containers on the container network, sorted lexicographically.
                        For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.
                Returns:
                    dict[str, str or list(str)]
        """
        return self._resource_config

    @property
    def input_data_config(self):  # type: () -> dict
        """A dictionary with the contents from /opt/ml/input/config/inputdataconfig.json.
                For example, suppose that you specify three data channels (train, evaluation, and validation) in
                your request. This dictionary will contain:
                ```{"train": {
                        "ContentType":  "trainingContentType",
                        "TrainingInputMode": "File",
                        "S3DistributionType": "FullyReplicated",
                        "RecordWrapperType": "None"
                    },
                    "evaluation" : {
                        "ContentType": "evalContentType",
                        "TrainingInputMode": "File",
                        "S3DistributionType": "FullyReplicated",
                        "RecordWrapperType": "None"
                    },
                    "validation": {
                        "TrainingInputMode": "File",
                        "S3DistributionType": "FullyReplicated",
                        "RecordWrapperType": "None"
                    }
                 } ```
                You can find more information about /opt/ml/input/config/inputdataconfig.json here:
                    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig
                Returns:
                    dict[str, dict[str, str]]
        """
        return self._input_data_config

    @property
    def output_data_dir(self):  # type: () -> str
        """The dir to write non-model training artifacts (e.g. evaluation results) which will be retained
        by SageMaker, e.g. /opt/ml/output/data/{current_host}.
        As your algorithm runs in a container, it generates output including the status of the
        training job and model and output artifacts. Your algorithm should write this information
        to the this directory.
        Returns:
            str: the path to output data directory, e.g. /opt/ml/output/data/algo-1.
        """
        return self._output_data_dir

    @property
    def output_intermediate_dir(self):  # type: () -> str
        """The directory for intermediate output artifacts that should be synced to S3.
        Any files written to this directory will be uploaded to S3 by a background process
        while training is in progress, but only if sagemaker_s3_output was specified.
        Returns:
            str: the path to the intermediate output directory, e.g. /opt/ml/output/intermediate.
        """
        return self._output_intermediate_dir

    @property
    def framework_module(self):  # type: () -> str
        """Returns:
            str: Name of the framework module and entry point. For example:
                my_module:main"""
        return self._framework_module


class ServingEnv(_Env):
    """Provides access to aspects of the serving environment relevant to serving containers, including
       system characteristics, environment variables and configuration settings.

       The ServingEnv is a read-only snapshot of the container environment. It does not contain any form of state.

       It is a dictionary like object, allowing any builtin function that works with dictionary.

       Example on how to print the state of the container:
           >>> from sagemaker_containers import _env

           >>> print(str(_env.ServingEnv()))
       Example on how a script can use training environment:
           >>>ServingEnv = _env.ServingEnv()


        Attributes:
            use_nginx (bool): Whether to use nginx as a reverse proxy.
            model_server_timeout (int): Timeout in seconds for the model server.
            model_server_workers (int): Number of worker processes the model server will use.
            framework_module (str):  Name of the framework module and entry point. For example:
                my_module:main
            default_accept (str): The desired default MIME type of the inference in the response
                as specified in the user-supplied SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT environment
                variable. Otherwise, returns 'application/json' by default.
                For example: application/json
            http_port (str): Port that SageMaker will use to handle invocations and pings against the
                running Docker container. Default is 8080. For example: 8080
            safe_port_range (str): HTTP port range that can be used by customers to avoid collisions
                with the HTTP port specified by SageMaker for handling pings and invocations.
                For example: 1111-2222
    """

    def __init__(self):
        super(ServingEnv, self).__init__()

        use_nginx = util.strtobool(os.environ.get(_params.USE_NGINX_ENV, 'true')) == 1
        model_server_timeout = int(os.environ.get(_params.MODEL_SERVER_TIMEOUT_ENV, '60'))
        model_server_workers = int(os.environ.get(_params.MODEL_SERVER_WORKERS_ENV, num_cpus()))
        framework_module = os.environ.get(_params.FRAMEWORK_SERVING_MODULE_ENV, None)
        default_accept = os.environ.get(_params.DEFAULT_INVOCATIONS_ACCEPT_ENV, _content_types.JSON)
        http_port = os.environ.get(_params.SAGEMAKER_BIND_TO_PORT_ENV, '8080')
        safe_port_range = os.environ.get(_params.SAGEMAKER_SAFE_PORT_RANGE_ENV)

        self._use_nginx = use_nginx
        self._model_server_timeout = model_server_timeout
        self._model_server_workers = model_server_workers
        self._framework_module = framework_module
        self._default_accept = default_accept
        self._http_port = http_port
        self._safe_port_range = safe_port_range

    @property
    def use_nginx(self):  # type: () -> bool
        """Returns:
            bool: whether to use nginx as a reverse proxy. Default: True"""
        return self._use_nginx

    @property
    def model_server_timeout(self):  # type: () -> int
        """Returns:
            int: Timeout in seconds for the model server. This is passed over to gunicorn, from the docs:
                Workers silent for more than this many seconds are killed and restarted. Our default value is 60.
                If ``use_nginx`` is True, then this same value will be used for nginx's proxy_read_timeout."""
        return self._model_server_timeout

    @property
    def model_server_workers(self):  # type: () -> int
        """Returns:
            int: Number of worker processes the model server will use"""
        return self._model_server_workers

    @property
    def framework_module(self):  # type: () -> str
        """Returns:
            str: Name of the framework module and entry point. For example:
                my_module:main"""
        return self._framework_module

    @property
    def default_accept(self):  # type: () -> str
        """Returns:
            str: The desired MIME type of the inference in the response. For example: application/json.
                Default: application/json"""
        return self._default_accept

    @property
    def http_port(self):  # type: () -> str
        """Returns:
            str: HTTP port that SageMaker will use to handle invocations and pings against the running
                Docker container. Default is 8080. For example: 8080"""
        return self._http_port

    @property
    def safe_port_range(self):  # type: () -> str
        """Returns:
            str: HTTP port range that can be used by customers to avoid collisions with the HTTP port
                specified by SageMaker for handling pings and invocations. For example: 1111-2222"""
        return self._safe_port_range


def write_env_vars(env_vars=None):  # type: (dict) -> None
    """Write the dictionary env_vars in the system, as environment variables.

    Args:
        env_vars ():

    Returns:

    """
    env_vars = env_vars or {}
    env_vars['PYTHONPATH'] = ':'.join(sys.path)

    for name, value in env_vars.items():
        os.environ[name] = value
