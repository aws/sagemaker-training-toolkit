#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import importlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import pkg_resources

import container_support as cs

logger = logging.getLogger(__name__)


class ContainerEnvironment(object):
    """Provides access to common aspects of the container environment, including
    important system characteristics, filesystem locations, and configuration settings.
    """

    BASE_DIRECTORY = "/opt/ml"
    USER_SCRIPT_NAME_PARAM = "sagemaker_program"
    USER_REQUIREMENTS_FILE_PARAM = "sagemaker_requirements"
    USER_SCRIPT_ARCHIVE_PARAM = "sagemaker_submit_directory"
    CLOUDWATCH_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
    CONTAINER_LOG_LEVEL_PARAM = "sagemaker_container_log_level"
    JOB_NAME_PARAM = "sagemaker_job_name"
    CURRENT_HOST_ENV = "CURRENT_HOST"
    JOB_NAME_ENV = "JOB_NAME"
    USE_NGINX_ENV = "SAGEMAKER_USE_NGINX"
    SAGEMAKER_REGION_PARAM_NAME = 'sagemaker_region'

    def __init__(self, base_dir=BASE_DIRECTORY):
        self.base_dir = base_dir
        "The current root directory for SageMaker interactions (``/opt/ml`` when running in SageMaker)."

        self.model_dir = os.path.join(base_dir, "model")
        "The directory to write model artifacts to so they can be handed off to SageMaker."

        self.code_dir = os.path.join(base_dir, "code")
        "The directory where user-supplied code will be staged."

        self.available_cpus = self._get_available_cpus()
        "The number of cpus available in the current container."

        self.available_gpus = self._get_available_gpus()
        "The number of gpus available in the current container."

        # subclasses will override
        self.user_script_name = None
        "The filename of the python script that contains user-supplied training/hosting code."

        # subclasses will override
        self.user_requirements_file = None
        "The filename of the text file that contains user-supplied dependencies required to be installed by pip"

        # subclasses will override
        self.user_script_archive = None
        "The S3 location of the python code archive that contains user-supplied training/hosting code"

        self.enable_cloudwatch_metrics = False
        "Report system metrics to CloudWatch? (default = False)"

        # subclasses will override
        self.container_log_level = None
        "The logging level for the root logger."

        # subclasses will override
        self.sagemaker_region = None
        "The current AWS region."

    def download_user_module(self):
        """Download user-supplied python archive from S3.
        """
        tmp = os.path.join(tempfile.gettempdir(), "script.tar.gz")
        cs.download_s3_resource(self.user_script_archive, tmp)
        cs.untar_directory(tmp, self.code_dir)

    def import_user_module(self):
        """Import user-supplied python module.
        """
        sys.path.insert(0, self.code_dir)

        script = self.user_script_name
        if script.endswith(".py"):
            script = script[:-3]

        user_module = importlib.import_module(script)
        return user_module

    def pip_install_requirements(self):
        """Install user-supplied requirements with pip.
        """
        if not self.user_requirements_file:
            return

        requirements_file = os.path.join(self.code_dir, self.user_requirements_file)
        if os.path.exists(requirements_file):
            logger.info('current Python environment:\n{}'.format(subprocess.check_output(['pip', 'freeze'])))

            logger.info('installing requirements in {} via pip'.format(requirements_file))
            output = subprocess.check_output(['pip', 'install', '-r', requirements_file])
            logger.info(output)
        else:
            logger.warn('Requirements file:{} was not found'.format(requirements_file))

    def start_metrics_if_enabled(self):
        if self.enable_cloudwatch_metrics:
            logger.info("starting metrics service")
            telegraf_conf = pkg_resources.resource_filename('container_support', 'etc/telegraf.conf')
            subprocess.Popen(['telegraf', '--config', telegraf_conf])

    @staticmethod
    def load_framework():
        """Import the deep learning framework needed for the current training job.
        """
        # TODO less atrocious implementation -- perhaps set in env or hyperparameters?
        try:
            return importlib.import_module('mxnet_container')
        except ImportError:
            return importlib.import_module('tf_container')

    @staticmethod
    def _get_available_cpus():
        return multiprocessing.cpu_count()

    @staticmethod
    def _get_available_gpus():
        gpus = 0
        try:
            output = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode('utf-8')
            gpus = sum([1 for x in output.split('\n') if x.startswith('GPU ')])
        except Exception as e:
            logger.debug("exception listing gpus (normal if no nvidia gpus installed): %s" % str(e))

        return gpus

    @staticmethod
    def _load_config(path):
        with open(path, 'r') as f:
            return json.load(f)


class TrainingEnvironment(ContainerEnvironment):
    """Provides access to aspects of the container environment relevant to training jobs.
    """
    HYPERPARAMETERS_FILE = "hyperparameters.json"
    RESOURCE_CONFIG_FILE = "resourceconfig.json"
    INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"
    S3_URI_PARAM = 'sagemaker_s3_uri'
    TRAINING_JOB_ENV = 'training_job_name'

    def __init__(self, base_dir=ContainerEnvironment.BASE_DIRECTORY):
        super(TrainingEnvironment, self).__init__(base_dir)
        self.input_dir = os.path.join(self.base_dir, "input")
        "The base directory for training data and configuration files."

        self.input_config_dir = os.path.join(self.input_dir, "config")
        "The directory where standard SageMaker configuration files are located."

        self.output_dir = os.path.join(self.base_dir, "output")
        "The directory where training success/failure indications will be written."

        self.resource_config = self._load_config(os.path.join(
            self.input_config_dir, TrainingEnvironment.RESOURCE_CONFIG_FILE))
        "dict of resource configuration settings."

        self.hyperparameters = self._load_hyperparameters(
            os.path.join(self.input_config_dir, TrainingEnvironment.HYPERPARAMETERS_FILE))
        "dict of hyperparameters that were passed to the CreateTrainingJob API."

        self.current_host = self.resource_config.get('current_host', '')
        "The hostname of the current container."

        self.hosts = self.resource_config.get('hosts', [])
        "The list of hostnames available to the current training job."

        self.output_data_dir = os.path.join(
            self.output_dir,
            "data",
            self.current_host if len(self.hosts) > 1 else '')
        "The dir to write non-model training artifacts (e.g. evaluation results) which will be retained by SageMaker. "

        self.job_name = os.environ.get(TrainingEnvironment.TRAINING_JOB_ENV.upper(), None)
        "The name of the current training job"

        # TODO validate docstring
        self.channels = self._load_config(
            os.path.join(self.input_config_dir, TrainingEnvironment.INPUT_DATA_CONFIG_FILE))
        "dict of training input data channel name to directory with the input files for that channel."

        # TODO validate docstring
        self.channel_dirs = {channel: self._get_channel_dir(channel) for channel in self.channels}

        self.user_script_name = self.hyperparameters.get(ContainerEnvironment.USER_SCRIPT_NAME_PARAM, '')
        self.user_requirements_file = self.hyperparameters.get(ContainerEnvironment.USER_REQUIREMENTS_FILE_PARAM, None)
        self.user_script_archive = self.hyperparameters.get(ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM, '')

        self.enable_cloudwatch_metrics = self.hyperparameters.get(ContainerEnvironment.CLOUDWATCH_METRICS_PARAM, False)
        self.container_log_level = self.hyperparameters.get(ContainerEnvironment.CONTAINER_LOG_LEVEL_PARAM)

        os.environ[ContainerEnvironment.JOB_NAME_ENV] = self.hyperparameters.get(
            ContainerEnvironment.JOB_NAME_PARAM, '')
        os.environ[ContainerEnvironment.CURRENT_HOST_ENV] = self.current_host

        self.sagemaker_region = self.hyperparameters[ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME]
        os.environ[ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME.upper()] = self.sagemaker_region

    def _load_hyperparameters(self, path):
        serialized = self._load_config(path)
        return self._deserialize_hyperparameters(serialized)

    # TODO expecting serialized hyperparams might break containers that aren't launched by python sdk
    @staticmethod
    def _deserialize_hyperparameters(hp):
        hyperparameter_dict = {}

        for (k, v) in hp.items():
            # Tuning jobs inject a hyperparameter that does not conform to the JSON format
            if k == '_tuning_objective_metric':
                if v.startswith('"') and v.endswith('"'):
                    v = v.strip('"')
                hyperparameter_dict[k] = v
            else:
                hyperparameter_dict[k] = json.loads(v)

        return hyperparameter_dict

    def write_success_file(self):
        TrainingEnvironment.ensure_directory(self.output_dir)
        path = os.path.join(self.output_dir, 'success')
        open(path, 'w').close()

    @staticmethod
    def write_failure_file(message, base_dir=None):
        base_dir = base_dir or ContainerEnvironment.BASE_DIRECTORY
        output_dir = os.path.join(base_dir, "output")
        TrainingEnvironment.ensure_directory(output_dir)
        with open(os.path.join(output_dir, 'failure'), 'a') as fd:
            fd.write(message)

    @staticmethod
    def ensure_directory(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _get_channel_dir(self, channel):
        """ Returns the directory containing the channel data file(s).

        This is either:

        - <self.base_dir>/input/data/<channel> OR
        - <self.base_dir>/input/data/<channel>/<channel_s3_suffix>

        Where channel_s3_suffix is the hyperparameter value with key <S3_URI_PARAM>_<channel>.

        The first option is returned if <self.base_dir>/input/data/<channel>/<channel_s3_suffix>
        does not exist in the file-system or <S3_URI_PARAM>_<channel> does not exist in
        self.hyperparmeters. Otherwise, the second option is returned.

        TODO: Refactor once EASE downloads directly into /opt/ml/input/data/<channel>
        TODO: Adapt for Pipe Mode

        Returns:
            (str) The input data directory for the specified channel.
        """
        channel_s3_uri_param = "{}_{}".format(TrainingEnvironment.S3_URI_PARAM, channel)
        if channel_s3_uri_param in self.hyperparameters:
            channel_s3_suffix = self.hyperparameters.get(channel_s3_uri_param)
            channel_dir = os.path.join(self.input_dir, 'data', channel, channel_s3_suffix)
            if os.path.exists(channel_dir):
                return channel_dir
        return os.path.join(self.input_dir, 'data', channel)


class HostingEnvironment(ContainerEnvironment):
    """ Provides access to aspects of the container environment relevant to hosting jobs.
    """

    MODEL_SERVER_WORKERS_PARAM = 'SAGEMAKER_MODEL_SERVER_WORKERS'
    MODEL_SERVER_TIMEOUT_PARAM = "SAGEMAKER_MODEL_SERVER_TIMEOUT"

    def __init__(self, base_dir=ContainerEnvironment.BASE_DIRECTORY):
        super(HostingEnvironment, self).__init__(base_dir)
        self.model_server_timeout = os.environ.get(HostingEnvironment.MODEL_SERVER_TIMEOUT_PARAM, 60)
        self.user_script_name = os.environ.get(ContainerEnvironment.USER_SCRIPT_NAME_PARAM.upper(), None)
        self.user_requirements_file = os.environ.get(ContainerEnvironment.USER_REQUIREMENTS_FILE_PARAM.upper(), None)
        self.user_script_archive = os.environ.get(ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM.upper(), None)

        self.enable_cloudwatch_metrics = os.environ.get(
            ContainerEnvironment.CLOUDWATCH_METRICS_PARAM.upper(), 'false').lower() == 'true'

        self.use_nginx = os.environ.get(ContainerEnvironment.USE_NGINX_ENV, 'true') == 'true'
        "Use nginx as front-end HTTP server instead of gunicorn."

        self.model_server_workers = int(os.environ.get(
            HostingEnvironment.MODEL_SERVER_WORKERS_PARAM,
            self.available_cpus))
        "The number of model server processes to run concurrently."

        self.container_log_level = int(os.environ[ContainerEnvironment.CONTAINER_LOG_LEVEL_PARAM.upper()])

        self.sagemaker_region = os.environ[ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME.upper()]
        os.environ[ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME.upper()] = self.sagemaker_region

        os.environ[ContainerEnvironment.JOB_NAME_ENV] = os.environ.get(
            ContainerEnvironment.JOB_NAME_PARAM.upper(), '')


def configure_logging():
    format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'
    default_level = logging.INFO

    level = None

    for c in [HostingEnvironment, TrainingEnvironment]:
        try:
            level = int(c().container_log_level)
            break
        except:  # noqa
            pass

    logging.basicConfig(format=format, level=level or default_level)

    if not level:
        logging.warn("error reading log_level, using INFO")

    if not level or level >= logging.INFO:
        logging.getLogger("boto3").setLevel(logging.WARNING)
