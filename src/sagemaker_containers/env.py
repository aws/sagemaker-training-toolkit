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

import contextlib
from distutils import util
import json
import logging
import multiprocessing
import os
import shlex
import shutil
import subprocess
import tempfile


from sagemaker_containers import mapping

logger = logging.getLogger(__name__)

BASE_PATH_ENV = 'BASE_PATH'  # type: str
CURRENT_HOST_ENV = 'CURRENT_HOST'  # type: str

BASE_PATH = os.environ.get(BASE_PATH_ENV, os.path.join('/opt', 'ml'))  # type: str

MODEL_PATH = os.path.join(BASE_PATH, 'model')  # type: str
INPUT_PATH = os.path.join(BASE_PATH, 'input')  # type: str
INPUT_DATA_PATH = os.path.join(INPUT_PATH, 'data')  # type: str
INPUT_CONFIG_PATH = os.path.join(INPUT_PATH, 'config')  # type: str
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')  # type: str
OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, 'data')  # type: str

HYPERPARAMETERS_FILE = 'hyperparameters.json'  # type: str
RESOURCE_CONFIG_FILE = 'resourceconfig.json'  # type: str
INPUT_DATA_CONFIG_FILE = 'inputdataconfig.json'  # type: str

HYPERPARAMETERS_PATH = os.path.join(INPUT_CONFIG_PATH, HYPERPARAMETERS_FILE)  # type: str
INPUT_DATA_CONFIG_FILE_PATH = os.path.join(INPUT_CONFIG_PATH, INPUT_DATA_CONFIG_FILE)  # type: str
RESOURCE_CONFIG_PATH = os.path.join(INPUT_CONFIG_PATH, RESOURCE_CONFIG_FILE)  # type: str

USER_PROGRAM_PARAM = 'sagemaker_program'  # type: str
USER_PROGRAM_ENV = USER_PROGRAM_PARAM.upper()  # type: str

SUBMIT_DIR_PARAM = 'sagemaker_submit_directory'  # type: str
SUBMIT_DIR_ENV = SUBMIT_DIR_PARAM.upper()  # type: str

ENABLE_METRICS_PARAM = 'sagemaker_enable_cloudwatch_metrics'  # type: str
ENABLE_METRICS_ENV = ENABLE_METRICS_PARAM.upper()  # type: str

LOG_LEVEL_PARAM = 'sagemaker_container_log_level'  # type: str
LOG_LEVEL_ENV = LOG_LEVEL_PARAM.upper()  # type: str

JOB_NAME_PARAM = 'sagemaker_job_name'  # type: str
JOB_NAME_ENV = JOB_NAME_PARAM.upper()  # type: str

DEFAULT_MODULE_NAME_PARAM = 'default_user_module_name'  # type: str
REGION_NAME_PARAM = 'sagemaker_region'  # type: str
REGION_NAME_ENV = REGION_NAME_PARAM.upper()  # type: str

MODEL_SERVER_WORKERS_ENV = 'SAGEMAKER_MODEL_SERVER_WORKERS'  # type: str
MODEL_SERVER_TIMEOUT_ENV = 'SAGEMAKER_MODEL_SERVER_TIMEOUT'  # type: str
USE_NGINX_ENV = 'SAGEMAKER_USE_NGINX'  # type: str
FRAMEWORK_SERVING_MODULE_ENV = 'SAGEMAKER_SERVING_MODULE'  # type: str
FRAMEWORK_TRAINING_MODULE_ENV = 'SAGEMAKER_TRAINING_MODULE'  # type: str

SAGEMAKER_HYPERPARAMETERS = (USER_PROGRAM_PARAM, SUBMIT_DIR_PARAM, ENABLE_METRICS_PARAM, REGION_NAME_PARAM,
                             LOG_LEVEL_PARAM, JOB_NAME_PARAM, DEFAULT_MODULE_NAME_PARAM)  # type: set


def read_json(path):  # type: (str) -> dict
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
    hyperparameters = read_json(HYPERPARAMETERS_PATH)

    try:
        return {k: json.loads(v) for k, v in hyperparameters.items()}
    except (ValueError, TypeError):  # pragma: py2 no cover
        logger.warning("Failed to parse hyperparameters' values to Json. Returning the hyperparameters instead:")
        return hyperparameters


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
    return read_json(RESOURCE_CONFIG_PATH)


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
    return read_json(INPUT_DATA_CONFIG_FILE_PATH)


def channel_path(channel):  # type: (str) -> str
    """ Returns the directory containing the channel data file(s) which is:
    - <self.base_dir>/input/data/<channel>

    For more information about channels: https://docs.aws.amazon.com/sagemaker/latest/dg/API_Channel.html

    Returns:
        str: The input data directory for the specified channel.
    """
    return os.path.join(INPUT_DATA_PATH, channel)


def gpu_count():  # type: () -> int
    """The number of gpus available in the current container.

    Returns:
        int: number of gpus available in the current container.
    """
    try:
        cmd = shlex.split('nvidia-smi --list-gpus')
        output = subprocess.check_output(cmd).decode('utf-8')
        return sum([1 for x in output.split('\n') if x.startswith('GPU ')])
    except (OSError, subprocess.CalledProcessError):
        logger.warning('No GPUs detected (normal if no gpus installed)')
        return 0


def cpu_count():  # type: () -> int
    """The number of cpus available in the current container.

    Returns:
        int: number of cpus available in the current container.
    """
    return multiprocessing.cpu_count()


class Env(mapping.MappingMixin):
    """Base Class which provides access to aspects of the environment including
    system characteristics, filesystem locations, environment variables and configuration settings.

    The Env is a read-only snapshot of the container environment. It does not contain any form of state.
    It is a dictionary like object, allowing any builtin function that works with dictionary.

    Attributes:
            current_host (str): The name of the current container on the container network. For example, 'algo-1'.
            num_gpus (int): The number of gpus available in the current container.
            num_cpus (int): The number of cpus available in the current container.
            module_name (str): The name of the user provided module.
            module_dir (str): The full path location of the user provided module.
    """

    def __init__(self):
        current_host = os.environ.get(CURRENT_HOST_ENV)
        module_name = os.environ.get(USER_PROGRAM_ENV, None)
        module_dir = os.environ.get(SUBMIT_DIR_ENV, None)
        enable_metrics = util.strtobool(os.environ.get(ENABLE_METRICS_ENV, 'false')) == 1
        log_level = os.environ.get(LOG_LEVEL_ENV, logging.INFO)

        self._current_host = current_host
        self._num_gpus = gpu_count()
        self._num_cpus = cpu_count()
        self._module_name = module_name
        self._module_dir = module_dir
        self._enable_metrics = enable_metrics
        self._log_level = log_level
        self._model_dir = MODEL_PATH

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
    def enable_metrics(self):  # type: () -> bool
        """Whether metrics should be executed in the environment or not.

        Returns:
            bool: representing whether metrics should be executed or not.
        """
        return self._enable_metrics

    @property
    def log_level(self):  # type: () -> int
        """Environment logging level.

        Returns:
            int: environment logging level.
        """
        return self._log_level

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


class TrainingEnv(Env):
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.

    The TrainingEnv is a read-only snapshot of the container environment during training. It does not contain any form
    of state.

    It is a dictionary like object, allowing any builtin function that works with dictionary.

    Example on how to print the state of the container:
        ```
        >>> from sagemaker_containers import env

        >>> print(str(env.TrainingEnv()))
        ```
    Example on how a script can use training environment:
        ```
            >>>from sagemaker_containers import env

            >>>train_env = env.TrainingEnv()

            get the path of the channel 'training' from the inputdataconfig.json file
            >>>training_dir = train_env.channel_input_dirs['training']

            get a the hyperparameter 'training_data_file' from hyperparameters.json file
            >>>file_name = train_env.hyperparameters['training_data_file']

            get the folder where the model should be saved
            >>>model_dir = train_env.model_dir
            >>>data = np.load(os.path.join(training_dir, file_name))
            >>>x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])
            >>>model = ResNet50(weights='imagenet')
            ...
            >>>model.fit(x_train, y_train)

            save the model in the end of training
            >>>model.save(os.path.join(model_dir, 'saved_model'))
        ```

    Attributes:
        input_dir (str): The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
            and configuration files before and during training. The input data directory has the
            following subdirectories: config (`input_config_dir`) and data (`input_data_dir`)

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
    """

    def __init__(self):
        super(TrainingEnv, self).__init__()

        resource_config = read_resource_config()
        current_host = resource_config['current_host']
        hosts = resource_config['hosts']
        network_interface_name = resource_config.get('network_interface_name', 'ethwe')
        input_data_config = read_input_data_config()

        all_hyperparameters = read_hyperparameters()
        split_result = mapping.split_by_criteria(all_hyperparameters, SAGEMAKER_HYPERPARAMETERS)

        sagemaker_hyperparameters = split_result.included

        sagemaker_region = sagemaker_hyperparameters[REGION_NAME_PARAM]
        os.environ[JOB_NAME_ENV] = sagemaker_hyperparameters[JOB_NAME_PARAM]
        os.environ[CURRENT_HOST_ENV] = current_host
        os.environ[REGION_NAME_ENV] = sagemaker_region

        self._hosts = hosts
        self._network_interface_name = network_interface_name
        self._input_dir = INPUT_PATH
        self._input_config_dir = INPUT_CONFIG_PATH
        self._output_dir = OUTPUT_PATH
        self._hyperparameters = split_result.excluded
        self._resource_config = resource_config
        self._input_data_config = input_data_config
        self._output_data_dir = os.path.join(OUTPUT_DATA_PATH, current_host)
        self._channel_input_dirs = {channel: channel_path(channel) for channel in input_data_config}
        self._current_host = current_host

        # override base class attributes
        self._module_name = str(sagemaker_hyperparameters[USER_PROGRAM_PARAM])
        self._module_dir = str(sagemaker_hyperparameters[SUBMIT_DIR_PARAM])
        self._enable_metrics = sagemaker_hyperparameters.get(ENABLE_METRICS_PARAM, False)
        self._log_level = sagemaker_hyperparameters.get(LOG_LEVEL_PARAM, logging.INFO)
        self._framework_module = os.environ.get(FRAMEWORK_TRAINING_MODULE_ENV, None)

    @property
    def hosts(self):  # type: () -> list
        """The list of names of all containers on the container network, sorted lexicographically.
                For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.
        Returns:
              list[str]: all the hosts in the training network.
        """
        return self._hosts

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
    def framework_module(self):  # type: () -> str
        """Returns:
            str: Name of the framework module and entry point. For example:
                my_module:main"""
        return self._framework_module


class ServingEnv(Env):
    """Provides access to aspects of the serving environment relevant to serving containers, including
       system characteristics, environment variables and configuration settings.

       The ServingEnv is a read-only snapshot of the container environment. It does not contain any form of state.

       It is a dictionary like object, allowing any builtin function that works with dictionary.

       Example on how to print the state of the container:
           >>> from sagemaker_containers import env

           >>> print(str(env.ServingEnv()))
       Example on how a script can use training environment:
           >>>serving_env = env.ServingEnv()


        Attributes:
            use_nginx (bool): Whether to use nginx as a reverse proxy.
            model_server_timeout (int): Timeout in seconds for the model server.
            model_server_workers (int): Number of worker processes the model server will use.
            framework_module (str):  Name of the framework module and entry point. For example:
                my_module:main
    """

    def __init__(self):
        super(ServingEnv, self).__init__()

        use_nginx = util.strtobool(os.environ.get(USE_NGINX_ENV, 'true')) == 1
        model_server_timeout = int(os.environ.get(MODEL_SERVER_TIMEOUT_ENV, '60'))
        model_server_workers = int(os.environ.get(MODEL_SERVER_WORKERS_ENV, cpu_count()))
        framework_module = os.environ.get(FRAMEWORK_SERVING_MODULE_ENV, None)

        self._use_nginx = use_nginx
        self._model_server_timeout = model_server_timeout
        self._model_server_workers = model_server_workers
        self._framework_module = framework_module

    @property
    def use_nginx(self):  # type: () -> bool
        """Returns:
            bool: whether to use nginx as a reverse proxy. Default: True"""
        return self._use_nginx

    @property
    def model_server_timeout(self):  # type: () -> int
        """Returns:
            int: Timeout in seconds for the model server. This is passed over to gunicorn, from the docs:
                Workers silent for more than this many seconds are killed and restarted. Our default value is 60."""
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


@contextlib.contextmanager
def tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the context exits.

    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix, otherwise there will be no
                        suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix; otherwise,
                        a default prefix is used.
        dir (str):  If dir is specified, the file will be created in that directory; otherwise, a default directory is
                        used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)


def write_file(path, data, mode='w'):  # type: (str, str, str) -> None
    """Write data to a file.

    Args:
        path (str): path to the file.
        data (str): data to be written to the file.
        mode (str): mode which the file will be open.
    """
    with open(path, mode) as f:
        f.write(data)
