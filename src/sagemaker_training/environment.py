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
"""This module contains functionality related to the container environment.
This includes constants, utility functions, and classes which provide access
to relevant aspects of the environment (including system characteristics,
filesystem locations, environment variables, and configuration files).
"""
from __future__ import absolute_import

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

from sagemaker_training import logging_config, mapping, params

logger = logging_config.get_logger()

SAGEMAKER_BASE_PATH = os.path.join("/opt", "ml")  # type: str
BASE_PATH_ENV = "SAGEMAKER_BASE_DIR"  # type: str

HYPERPARAMETERS_FILE = "hyperparameters.json"  # type: str
RESOURCE_CONFIG_FILE = "resourceconfig.json"  # type: str
INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"  # type: str


def _write_json(obj, path):  # type: (object, str) -> None
    """Write a serializeable object as a JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f)


def _is_training_path_configured():  # type: () -> bool
    """Check if the tree structure with data and configuration files used for training
    exists.

    When a SageMaker Training Job is created, the Docker container that will be used for
    training is executed with the folder /opt/ml attached. The /opt/ml folder contains
    data and configurations files necessary for training.

    Outside SageMaker, the environment variable SAGEMAKER_BASE_DIR defines the location
    of the base folder.

    This function checks whether /opt/ml exists or if the base folder variable exists.

    Returns:
        (bool): Whether the training path is configured.

    """
    return os.path.exists(SAGEMAKER_BASE_PATH) or BASE_PATH_ENV in os.environ


def _set_base_path_env():  # type: () -> None
    """Set the environment variable SAGEMAKER_BASE_DIR as
    ~/sagemaker_local/jobs/{timestamp}/opt/ml
    """
    timestamp = str(time.time())
    local_config_dir = os.path.join(
        os.path.expanduser("~"), "sagemaker_local", "jobs", timestamp, "opt", "ml"
    )

    logger.info("Setting environment variable SAGEMAKER_BASE_DIR as %s ." % local_config_dir)
    os.environ[BASE_PATH_ENV] = local_config_dir


_is_path_configured = _is_training_path_configured()

if not _is_path_configured:
    logger.info("Directory /opt/ml does not exist.")
    _set_base_path_env()

base_dir = os.environ.get(BASE_PATH_ENV, SAGEMAKER_BASE_PATH)  # type: str

code_dir = os.path.join(base_dir, "code")
"""str: the path of the user's code directory, e.g., /opt/ml/code/"""

model_dir = os.path.join(base_dir, "model")  # type: str
"""str: the directory where models should be saved, e.g., /opt/ml/model/"""

input_dir = os.path.join(base_dir, "input")  # type: str
"""str: the path of the input directory, e.g. /opt/ml/input/

The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
        and configuration files before and during training.
        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`_input_data_dir`)
        Returns:
            str: the path of the input directory, e.g. /opt/ml/input/
"""

_input_data_dir = os.path.join(input_dir, "data")  # type: str

input_config_dir = os.path.join(input_dir, "config")  # type: str
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

output_dir = os.path.join(base_dir, "output")  # type: str
"""str: the path to the output directory, e.g. /opt/ml/output/.

The directory where training success/failure indications will be written, e.g. /opt/ml/output.
To save non-model artifacts check `output_data_dir`.
Returns:
    str: the path to the output directory, e.g. /opt/ml/output/.
"""

output_data_dir = os.path.join(output_dir, "data")  # type: str

output_intermediate_dir = os.path.join(output_dir, "intermediate")  # type: str
"""str: the path to the intermediate output directory, e.g. /opt/ml/output/intermediate.

The directory special behavior is to move artifacts from the training instance to
s3 directory during training.
Returns:
    str: the path to the intermediate output directory, e.g. /opt/ml/output/intermediate.
"""

hyperparameters_file_dir = os.path.join(input_config_dir, HYPERPARAMETERS_FILE)  # type: str
input_data_config_file_dir = os.path.join(input_config_dir, INPUT_DATA_CONFIG_FILE)  # type: str
resource_config_file_dir = os.path.join(input_config_dir, RESOURCE_CONFIG_FILE)  # type: str


def _create_training_directories():
    """Create the directory structure and files necessary for training under the base path."""
    logger.info("Creating a new training folder under %s ." % base_dir)

    os.makedirs(model_dir)
    os.makedirs(input_config_dir)
    os.makedirs(output_data_dir)

    _write_json({}, hyperparameters_file_dir)
    _write_json({}, input_data_config_file_dir)

    host_name = socket.gethostname()
    resources_dict = {
        "current_host": host_name,
        "hosts": [host_name],
        "current_instance_group": "homogeneousCluster",
        "instance_groups": [
            {
                "instance_group_name": "homogeneousCluster",
                "instance_type": "local",
                "hosts": [host_name],
            }
        ],
    }
    _write_json(resources_dict, resource_config_file_dir)


if not _is_path_configured:
    _create_training_directories()


def _create_code_dir():  # type: () -> None
    """Create /opt/ml/code when the module is imported."""
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
    with open(path, "r") as f:
        return json.load(f)


def read_hyperparameters():  # type: () -> dict
    """Read the hyperparameters from /opt/ml/input/config/hyperparameters.json.

    For more information about hyperparameters.json:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-hyperparameters

    Returns:
         (dict[string, object]): A dictionary containing the hyperparameters.
    """
    hyperparameters = _read_json(hyperparameters_file_dir)

    deserialized_hps = {}

    for k, v in hyperparameters.items():
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            logger.info(
                "Failed to parse hyperparameter %s value %s to Json.\n"
                "Returning the value itself",
                k,
                v,
            )

        deserialized_hps[k] = v

    return deserialized_hps


def read_resource_config():  # type: () -> dict
    """Read the resource configuration from /opt/ml/input/config/resourceconfig.json.

    For more information about resourceconfig.json:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-dist-training

    Returns:
        resource_config (dict[string, object]): the contents from /opt/ml/input/config/resourceconfig.json.
                        It has the following keys:
                            - current_host: The name of the current container on the container
                                            network. For example, 'algo-1'.
                            - current_instance_type: Type of EC2 instance
                            - hosts: The list of names of all nodes on the container
                                      network, sorted lexicographically. For example,
                                      `['algo-1', 'algo-2', 'algo-3']` for a three-node cluster.
                            - current_instance_group: Name of the current instance group
                            - instance_groups: List of instance group dicts containing info about
                                      instance_type, hosts list and group name
                            - network_interface_name: Name of network interface exposed to container
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
        https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

    Returns:
            input_data_config (dict[string, object]): Contents from /opt/ml/input/config/inputdataconfig.json.
    """
    return _read_json(input_data_config_file_dir)


def channel_path(channel):  # type: (str) -> str
    """Return the directory containing the channel data file(s) which is:
    - <self.base_dir>/input/data/<channel>

    For more information about channels: https://docs.aws.amazon.com/sagemaker/latest/dg/API_Channel.html

    Returns:
        str: The input data directory for the specified channel.
    """
    return os.path.join(_input_data_dir, channel)


def num_neurons():  # type: () -> int
    """Return the number of neuron cores available in the current container.

    Returns:
        int: Number of Neuron Cores available in the current container.
    """
    try:
        cmd = shlex.split("neuron-ls -j")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
        j = json.loads(output)
        neuron_cores = 0
        for item in j:
            neuron_cores += item.get("nc_count", 0)
        logger.info(f"Found {neuron_cores} neurons on this instance")
        return neuron_cores
    except OSError:
        logger.info("No Neurons detected (normal if no neurons installed)")
        return 0
    except (subprocess.CalledProcessError) as e:
        if e.output is not None:
            try:
                msg = e.output.decode("utf-8").partition("error=")[2]
                logger.info(
                    "No Neurons detected (normal if no neurons installed). \
                            If neuron installed then {}".format(
                        msg
                    )
                )
            except AttributeError:
                logger.info("No Neurons detected (normal if no neurons installed)")
        else:
            logger.info("No Neurons detected (normal if no neurons installed)")

        return 0


def num_gpus():  # type: () -> int
    """Return the number of GPUs available in the current container.

    Returns:
        int: Number of GPUs available in the current container.
    """
    try:
        cmd = shlex.split("nvidia-smi --list-gpus")
        output = subprocess.check_output(cmd).decode("utf-8")
        return sum([1 for x in output.split("\n") if x.startswith("GPU ")])
    except (OSError, subprocess.CalledProcessError):
        logger.info("No GPUs detected (normal if no gpus installed)")
        return 0


def num_cpus():  # type: () -> int
    """Return the number of CPUs available in the current container.

    Returns:
        int: Number of CPUs available in the current container.
    """
    return multiprocessing.cpu_count()


def validate_smddpmprun():  # type: () -> bool
    """Whether smddpmprun is installed.

    Returns:
        bool: True if both are installed
    """
    try:
        output = subprocess.run(
            ["which", "smddpmprun"],
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout != ""
    except subprocess.CalledProcessError:
        return False


class Environment(mapping.MappingMixin):  # pylint:disable=too-many-public-methods
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and
    configuration settings.

    The Environment is a read-only snapshot of the container environment during training. It does
    not contain any form of state.

    It is a dictionary like object, allowing any builtin function that works with dictionary.

    Example on how a script can use training environment:
            >>>from sagemaker_training import environment

            >>>env = environment.Environment()

            get the path of the channel 'training' from the inputdataconfig.json file
            >>>training_dir = environment.channel_input_dirs['training']

            get the hyperparameter 'training_data_file' from hyperparameters.json file
            >>>file_name = environment.hyperparameters['training_data_file']

            get the folder where the model should be saved
            >>>model_dir = environment.model_dir

            >>>data = np.load(os.path.join(training_dir, file_name))
            >>>x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])
            >>>model = ResNet50(weights='imagenet')
            ...
            >>>model.fit(x_train, y_train)

            save the model in the end of training
            >>>model.save(os.path.join(model_dir, 'saved_model'))

    Attributes:
        current_host (str): The name of the current container on the container network. For
                            example, 'algo-1'.
        module_name (str): The name of the user provided module.
        module_dir (str): The full path location of the user provided module.
        input_dir (str): The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves
                         input data and configuration files before and during training. The input
                         data directory has the following subdirectories:
                         config (`input_config_dir`) and data (`_input_data_dir`)

        input_config_dir (str): The directory where standard SageMaker configuration files are
                                located, e.g. /opt/ml/input/config/.

            SageMaker training creates the following files in this folder when training starts:
                - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a
                                          CreateTrainingJob request available in this file.
                - `inputdataconfig.json`: You specify data channel information in the
                                          InputDataConfig parameter in a CreateTrainingJob request.
                                          Amazon SageMaker makes this information available in this
                                          file.
                - `resourceconfig.json`: name of the current host and all host containers in the
                                         training

            More information about these files can be find here:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

        model_dir (str):  the directory where models should be saved, e.g., /opt/ml/model/

        output_dir (str): The directory where training success/failure indications will be written,
                          e.g. /opt/ml/output. To save non-model artifacts check `output_data_dir`.

        hyperparameters (dict[string, object]): An instance of `HyperParameters` containing the
                                                training job hyperparameters.

        resource_config (dict[string, object]): the contents from
                                                /opt/ml/input/config/resourceconfig.json.
            It has the following keys:
                - current_host: The name of the current container on the container network.
                    For example, 'algo-1'.
                -  hosts: The list of names of all containers on the container network,
                    sorted lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']`
                    for a three-node cluster.
                -  current_instance_group: Name of the current instance group
                - instance_groups: List of instance group dicts containing info about
                            instance_type, hosts list and group name
                - network_interface_name: Name of network interface exposed to container

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
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

        output_data_dir (str): The dir to write non-model training artifacts (e.g. evaluation
                               results) which will be retained by SageMaker,
                               e.g. /opt/ml/output/data. As your algorithm runs in a container,
                               it generates output including the status of the training job and
                               model and output artifacts. Your algorithm should write this
                               information to the this directory.

        hosts (list[str]): The list of names of all containers on the container network, sorted
                           lexicographically. For example, `['algo-1', 'algo-2', 'algo-3']` for
                           a three-node cluster.

        channel_input_dirs (dict[string, string]): containing the data channels and the directories
                                                   where the training data was saved. When you run
                                                   training, you can partition your training data
                                                   into different logical 'channels'. Depending on
                                                   your problem, some common channel ideas are:
                                                   'train', 'test', 'evaluation' or 'images',
                                                   'labels'.

            The format of channel_input_dir is as follows:
                - `channel`(str) - the name of the channel defined in the input_data_config.
                - `training data path`(str) - the path to the directory where the training data is
                saved.
        framework_module (str):  Name of the framework module and entry point. For example:
            my_module:main

        network_interface_name (str): Name of the network interface used for distributed training.

        job_name (str): The name of the current training job.
    """

    def __init__(self, resource_config=None, input_data_config=None, hyperparameters=None):
        """Initialize a read-only snapshot of the container environment.

        Args:
            resource_config (dict[string, object]): The contents from
                /opt/ml/input/config/resourceconfig.json.
                It has the following keys:
                    - current_host: The name of the current container on the container
                                    network. For example, 'algo-1'.
                    - current_instance_type: Type of EC2 instance
                    - hosts: The list of names of all nodes on the container
                                network, sorted lexicographically. For example,
                                `['algo-1', 'algo-2', 'algo-3']` for a three-node cluster.
                    - current_instance_group: Name of the current instance group
                    - instance_groups: List of instance group dicts containing info about
                                instance_type, hosts list and group name
                    - network_interface_name: Name of network interface exposed to container

            input_data_config (dict[string, object]): The contents from /opt/ml/input/config/inputdataconfig.json.
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
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig

            hyperparameters (dict[string, object]): An instance of `HyperParameters` containing the
                training job hyperparameters.
        """
        current_host = os.environ.get(params.CURRENT_HOST_ENV)
        module_name = os.environ.get(params.USER_PROGRAM_ENV, None)
        module_dir = os.environ.get(params.SUBMIT_DIR_ENV, code_dir)
        log_level = int(os.environ.get(params.LOG_LEVEL_ENV, logging.INFO))

        self._num_gpus = num_gpus()
        self._num_cpus = num_cpus()
        self._num_neurons = num_neurons()
        self._module_name = module_name
        self._user_entry_point = module_name
        self._module_dir = module_dir
        self._log_level = log_level
        self._model_dir = model_dir

        resource_config = resource_config or read_resource_config()
        input_data_config = input_data_config or read_input_data_config()
        all_hyperparameters = hyperparameters or read_hyperparameters()

        hosts = resource_config["hosts"]
        current_instance_type = resource_config.get("current_instance_type", "local")
        current_instance_group = resource_config.get("current_group_name", "homogeneousCluster")
        current_host = resource_config["current_host"]

        self._current_host = current_host
        self._current_instance_type = current_instance_type
        self._current_instance_group = current_instance_group

        split_result = mapping.split_by_criteria(
            all_hyperparameters,
            keys=params.SAGEMAKER_HYPERPARAMETERS,
            prefix=params.SAGEMAKER_PREFIX,
        )
        sagemaker_hyperparameters = split_result.included

        additional_framework_parameters = {
            k: sagemaker_hyperparameters[k]
            for k in sagemaker_hyperparameters.keys()
            if k not in params.SAGEMAKER_HYPERPARAMETERS
        }

        sagemaker_region = sagemaker_hyperparameters.get(
            params.REGION_NAME_PARAM, boto3.session.Session().region_name
        )

        os.environ[params.JOB_NAME_ENV] = sagemaker_hyperparameters.get(params.JOB_NAME_PARAM, "")
        os.environ[params.CURRENT_HOST_ENV] = current_host
        os.environ[params.REGION_NAME_ENV] = sagemaker_region or ""

        # hosts comprises of instances from all the groups
        self._hosts = hosts

        # eth0 is the default network interface defined by SageMaker with VPC support and
        # local mode.
        # ethwe is the current network interface defined by SageMaker training, it will be
        # changed to eth0 in the short future.
        self._network_interface_name = resource_config.get("network_interface_name", "eth0")

        self._hyperparameters = split_result.excluded
        self._additional_framework_parameters = additional_framework_parameters
        self._resource_config = resource_config
        self._input_data_config = input_data_config
        self._output_data_dir = output_data_dir
        self._output_intermediate_dir = output_intermediate_dir
        self._channel_input_dirs = {channel: channel_path(channel) for channel in input_data_config}

        # override base class attributes
        if self._module_name is None:
            self._module_name = str(sagemaker_hyperparameters.get(params.USER_PROGRAM_PARAM, None))
        self._user_entry_point = self._user_entry_point or sagemaker_hyperparameters.get(
            params.USER_PROGRAM_PARAM
        )

        self._module_dir = str(sagemaker_hyperparameters.get(params.SUBMIT_DIR_PARAM, code_dir))
        self._log_level = sagemaker_hyperparameters.get(params.LOG_LEVEL_PARAM, logging.INFO)
        self._sagemaker_s3_output = sagemaker_hyperparameters.get(
            params.S3_OUTPUT_LOCATION_PARAM, None
        )
        self._framework_module = os.environ.get(params.FRAMEWORK_TRAINING_MODULE_ENV, None)

        self._input_dir = input_dir
        self._input_config_dir = input_config_dir
        self._output_dir = output_dir
        self._job_name = os.environ.get(params.TRAINING_JOB_ENV.upper(), None)

        # Heterogeneous cluster changes - get the instance group related information
        current_instance_group_hosts = self.get_current_instance_group_hosts()
        instance_groups = self.get_instance_groups()
        instance_groups_dict = self.get_instance_groups_dict()
        distribution_instance_groups = self._additional_framework_parameters.get(
            "sagemaker_distribution_instance_groups",
            self.get_distribution_instance_groups_from_resource_config(),
        )
        self._distribution_instance_groups = distribution_instance_groups
        distribution_hosts = self.get_distribution_hosts()

        self._current_instance_group_hosts = current_instance_group_hosts
        self._instance_groups = instance_groups
        self._instance_groups_dict = instance_groups_dict
        self._distribution_hosts = distribution_hosts
        is_hetero = bool(len(self._instance_groups) > 1)
        self._is_hetero = is_hetero
        master_hostname = self.get_master_hostname()
        self._master_hostname = master_hostname
        self._is_master = current_host == self._master_hostname
        self._distribution_enabled = bool(
            self._current_instance_group in self._distribution_instance_groups
        )

        mp_parameters = os.environ.get(params.SM_HP_MP_PARAMETERS)
        self._is_modelparallel_enabled = mp_parameters and mp_parameters != "{}"
        self._is_smddpmprun_installed = validate_smddpmprun()

    @property
    def current_instance_type(self):
        """
        Return current instance type
        """
        return self._current_instance_type

    @property
    def is_hetero(self):
        """
        Return if current mode is hetero
        """
        return self._is_hetero

    @property
    def current_instance_group(self):
        """
        Return name of the current instance group
        """
        return self._current_instance_group

    @property
    def instance_groups(self):
        """
        Return list of all instance groups
        """
        return self._instance_groups

    @property
    def instance_groups_dict(self):
        """
        Return dict of all instance groups
        """
        return self._instance_groups_dict

    @property
    def current_instance_group_hosts(self):
        """
        Return hosts in the current instance group
        """
        return self._current_instance_group_hosts

    @property
    def distribution_hosts(self):
        """
        Return list of hosts on which distribution will be applied
        """
        return self._distribution_hosts

    @property
    def distribution_instance_groups(self):
        """Return list of instance groups which have distribution"""
        return self._distribution_instance_groups

    @property
    def master_hostname(self):
        """Return master hostname"""
        return self._master_hostname

    @property
    def model_dir(self):  # type: () -> str
        """The directory where models should be saved.

        Returns:
            str: The directory where models should be saved, e.g., /opt/ml/model/
        """
        return self._model_dir

    @property
    def current_host(self):  # type: () -> str
        """The name of the current container on the container network. For example, 'algo-1'.

        Returns:
            str: Current host.
        """
        return self._current_host

    @property
    def num_gpus(self):  # type: () -> int
        """The number of GPUs available in the current container.

        Returns:
            int: Number of GPUs available in the current container.
        """
        return self._num_gpus

    @property
    def num_neurons(self):  # type: () -> int
        """The number of Neuron Cores available in the current container.

        Returns:
            int: Number of Neuron Cores available in the current container.
        """
        return self._num_neurons

    @property
    def num_cpus(self):  # type: () -> int
        """The number of CPUs available in the current container.

        Returns:
            int: Number of CPUs available in the current container.
        """
        return self._num_cpus

    @property
    def module_name(self):  # type: () -> str
        """The name of the user provided module.

        Returns:
            str: Name of the user provided module.
        """
        return self._parse_module_name(self._module_name)

    @property
    def module_dir(self):  # type: () -> str
        """The full path location of the user provided module.

        Returns:
            str: Full path location of the user provided module.
        """
        return self._module_dir

    @property
    def log_level(self):  # type: () -> int
        """Environment logging level.

        Returns:
            int: Environment logging level.
        """
        return self._log_level

    @property
    def user_entry_point(self):  # type: () -> str
        """The name of provided user entry point.

        Returns:
            str: The name of provided user entry point.
        """
        return self._user_entry_point

    @staticmethod
    def _parse_module_name(program_param):
        """Given a module name or a script name, Returns the module name.
        This function is used for backwards compatibility.

        Args:
            program_param (str): Module or script name.

        Returns:
            str: Module name.
        """
        if program_param and program_param.endswith(".py"):
            return program_param[:-3]
        return program_param

    @property
    def is_master(self):  # type: () -> bool
        """Returns True if host is master."""
        return self._is_master

    @property
    def job_name(self):  # type: () -> str
        """The name of the current training job.

        Returns:
            str: The training job name.
        """
        return self._job_name

    @property
    def additional_framework_parameters(self):  # type: () -> dict
        """The dict of additional framework hyperparameters. All the hyperparameters prefixed with
        'sagemaker_' but not in SAGEMAKER_HYPERPARAMETERS will be included here.

        Returns:
            dict: Additional framework hyperparameters, SageMaker Python SDK adds hyperparameters
                  with a prefix **sagemaker_** during training. These hyperparameters are
                  framework independent settings and are not defined by the user.
        """
        return self._additional_framework_parameters

    def get_distribution_instance_groups_from_resource_config(self):
        """If non heterogeneous cluster mode is used, instance_groups inside distribution is a noop
        We populate the sagemaker_distribution_instance_groups with current instance group name ~
        homogeneousCluster
        """
        # pylint: disable=too-many-boolean-expressions
        distribution_instance_groups = []
        current_instance_group = self.resource_config.get(
            "current_group_name", "homogeneousCluster"
        )
        if (
            self._additional_framework_parameters.get("sagemaker_mpi_enabled", False)
            or self._additional_framework_parameters.get(
                "sagemaker_parameter_server_enabled", False
            )
            or self._additional_framework_parameters.get(
                "sagemaker_distributed_dataparallel_enabled", False
            )
            or self._additional_framework_parameters.get("sagemaker_pytorch_ddp_enabled", False)
            or self._additional_framework_parameters.get(
                "sagemaker_pytorch_xla_multi_worker_enabled", False
            )
            or self._additional_framework_parameters.get(
                "sagemaker_multi_worker_mirrored_strategy_enabled", False
            )
            or self._additional_framework_parameters.get(
                "sagemaker_torch_distributed_enabled", False
            )
        ):
            distribution_instance_groups.append(current_instance_group)
        return distribution_instance_groups

    def get_current_instance_group(self):
        """
        Get the current instance group name
        """
        return self.resource_config["current_instance_group"]

    def get_distribution_hosts(self):
        """
        Get the list of all hosts in all distribution instance groups
        """
        distribution_hosts = []
        instance_groups_config = self._resource_config.get("instance_groups", [])
        if instance_groups_config:
            for group in instance_groups_config:
                if group["instance_group_name"] in self._distribution_instance_groups:
                    distribution_hosts.extend(group["hosts"])
        else:
            # local mode
            distribution_hosts = self.hosts.copy()
        return distribution_hosts

    def get_current_instance_group_hosts(self):
        """
        Get the list of hosts in the current instance group
        """
        instance_groups_config = self._resource_config.get("instance_groups", [])
        for group in instance_groups_config:
            if self._current_instance_group == group["instance_group_name"]:
                return group["hosts"]
        return []

    def get_instance_groups(self):  # type: () -> list
        """
        List of instance groups provided for the job
        """
        instance_groups = []
        instance_groups_config = self._resource_config.get("instance_groups", [])
        # log missing instance groups and return empty list
        if not instance_groups_config:
            logger.info("instance_groups entry not present in resource_config")

        for group in instance_groups_config:
            instance_groups.append(group["instance_group_name"])
        return instance_groups

    def get_instance_groups_dict(self):
        """
        Dictionaty of instance groups with group_names as keys
        """
        instance_groups_dict = {}
        instance_groups_config = self._resource_config.get("instance_groups", [])
        for group in instance_groups_config:
            instance_groups_dict[group["instance_group_name"]] = group
        return instance_groups_dict

    def get_master_hostname(self):
        """
        Get the master hostname from the list of hosts in the distribution instance groups
        """
        if self._distribution_hosts:
            return list(self._distribution_hosts)[0]
        # if no distribution found
        return list(self._hosts)[0]

    def sagemaker_s3_output(self):  # type: () -> str
        """S3 output directory location provided by the user.

        Returns:
            str: S3 location.
        """
        return self._sagemaker_s3_output

    def to_cmd_args(self):
        """Command line arguments representation of the training environment.

        Returns:
            (list): List of cmd arguments.
        """
        return mapping.to_cmd_args(self.hyperparameters)

    def to_env_vars(self):
        """Environment variable representation of the training environment.

        Returns:
            dict: An instance of dictionary.
        """

        env = {
            "hosts": self.hosts,
            "network_interface_name": self.network_interface_name,
            "hps": self.hyperparameters,
            "user_entry_point": self.user_entry_point,
            "framework_params": self.additional_framework_parameters,
            "resource_config": self.resource_config,
            "input_data_config": self.input_data_config,
            "output_data_dir": self.output_data_dir,
            "channels": sorted(self.channel_input_dirs.keys()),
            "current_host": self.current_host,
            "current_instance_type": self.current_instance_type,
            "current_instance_group": self.current_instance_group,
            "current_instance_group_hosts": self.current_instance_group_hosts,
            "instance_groups": self.instance_groups,
            "instance_groups_dict": self.instance_groups_dict,
            "distribution_instance_groups": self.distribution_instance_groups,
            "is_hetero": self.is_hetero,
            "module_name": self.module_name,
            "log_level": self.log_level,
            "framework_module": self.framework_module,
            "input_dir": self.input_dir,
            "input_config_dir": self.input_config_dir,
            "output_dir": self.output_dir,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "num_neurons": self.num_neurons,
            "model_dir": self.model_dir,
            "module_dir": self.module_dir,
            "training_env": dict(self),
            "user_args": self.to_cmd_args(),
            "output_intermediate_dir": self.output_intermediate_dir,
        }

        for name, path in self.channel_input_dirs.items():
            env["channel_%s" % name] = path

        for key, value in self.hyperparameters.items():
            env["hp_%s" % key] = value

        return mapping.to_env_vars(env)

    @property
    def hosts(self):  # type: () -> list
        """The list of names of all containers on the container network, sorted lexicographically.
                For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.

        Returns:
              list[str]: All the hosts in the training network.
        """
        return self._hosts

    @property
    def channel_input_dirs(self):  # type: () -> dict
        """A dict[str, str] containing the data channels and the directories where the training
        data was saved.
        When you run training, you can partition your training data into different logical
        "channels".
        Depending on your problem, some common channel ideas are: "train", "test", "evaluation"
            or "images',"labels".
        The format of channel_input_dir is as follows:
            - `channel`[key](str) - the name of the channel defined in the input_data_config.
            - `training data path`[value](str) - the path to the directory where the training
                                                 data is saved.

        Returns:
            dict[str, str] With the information about the channels.
        """
        return self._channel_input_dirs

    @property
    def network_interface_name(self):  # type: () -> str
        """Name of the network interface used for distributed training.

        Returns:
              str: Name of the network interface, for example, 'ethwe'.
        """
        return self._network_interface_name

    @property
    def input_dir(self):  # type: () -> str
        """The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
        and configuration files before and during training.
        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`input_data_dir`)

        Returns:
            str: The path of the input directory, e.g. /opt/ml/input/.
        """
        return self._input_dir

    @property
    def input_config_dir(self):  # type: () -> str
        """The directory where standard SageMaker configuration files are located, e.g.
        /opt/ml/input/config/.
        SageMaker training creates the following files in this folder when training starts:
            - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a
                                      CreateTrainingJob request available in this file.
            - `inputdataconfig.json`: You specify data channel information in the
                                      InputDataConfig parameter in a CreateTrainingJob request.
                                      Amazon SageMaker makes this information available in this
                                      file.
            - `resourceconfig.json`: name of the current host and all host containers in the
                                     training More information about this files can be find here:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

        Returns:
            str: The path of the input directory, e.g. /opt/ml/input/config/.
        """
        return self._input_config_dir

    @property
    def output_dir(self):  # type: () -> str
        """The directory where training success/failure indications will be written,
        e.g. /opt/ml/output.
        To save non-model artifacts check `output_data_dir`.

        Returns:
            str: The path to the output directory, e.g. /opt/ml/output/.
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
            - current_host: The name of the current container on the container
                        network. For example, 'algo-1'.
            - current_instance_type: Type of EC2 instance
            - hosts: The list of names of all nodes on the container
                        network, sorted lexicographically. For example,
                        `['algo-1', 'algo-2', 'algo-3']` for a three-node cluster.
            - current_instance_group: Name of the current instance group
            - instance_groups: List of instance group dicts containing info about
                        instance_type, hosts list and group name
            - network_interface_name: Name of network interface exposed to container

        Returns:
            dict[str, str or list(str)]
        """
        return self._resource_config

    @property
    def input_data_config(self):  # type: () -> dict
        """A dictionary with the contents from /opt/ml/input/config/inputdataconfig.json.

        For example, suppose that you specify three data channels (train,
        evaluation, and validation) in your request. This dictionary will contain:
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
        """The dir to write non-model training artifacts (e.g. evaluation results) which will be
        retained by SageMaker, e.g. /opt/ml/output/data/{current_host}.
        As your algorithm runs in a container, it generates output including the status of the
        training job and model and output artifacts. Your algorithm should write this information
        to the this directory.

        Returns:
            str: The path to output data directory, e.g. /opt/ml/output/data/algo-1.
        """
        return self._output_data_dir

    @property
    def output_intermediate_dir(self):  # type: () -> str
        """The directory for intermediate output artifacts that should be synced to S3.
        Any files written to this directory will be uploaded to S3 by a background process
        while training is in progress, but only if sagemaker_s3_output was specified.

        Returns:
            str: The path to the intermediate output directory, e.g. /opt/ml/output/intermediate.
        """
        return self._output_intermediate_dir

    @property
    def framework_module(self):  # type: () -> str
        """Name of the framework module and entry point.

        Returns:
            str: Name of the framework module and entry point. For example:
                my_module:main
        """
        return self._framework_module

    @property
    def is_modelparallel_enabled(self):  # type: () -> bool
        """Whether SM ModelParallel is enabled.

        Returns:
            bool: True if SM ModelParallel is enabled
        """
        return self._is_modelparallel_enabled

    @property
    def is_smddpmprun_installed(self):  # type: () -> bool
        """Whether smddpmprun is installed.

        Returns:
            bool: True if both are installed
        """
        return self._is_smddpmprun_installed


def write_env_vars(env_vars=None):  # type: (dict) -> None
    """Write the dictionary env_vars in the system, as environment variables.

    Args:
        env_vars (dict): A dictionary of environment variables.
    """
    env_vars = env_vars or {}
    env_vars["PYTHONPATH"] = ":".join(sys.path)

    for name, value in env_vars.items():
        os.environ[name] = value
