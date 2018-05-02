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
import itertools
import json
import logging
import os

from mock import Mock, patch
import pytest
import six

import sagemaker_containers as smc
import test

RESOURCE_CONFIG = dict(current_host='algo-1', hosts=['algo-1', 'algo-2', 'algo-3'])

INPUT_DATA_CONFIG = {
    'train': {
        'ContentType': 'trainingContentType',
        'TrainingInputMode': 'File',
        'S3DistributionType': 'FullyReplicated',
        'RecordWrapperType': 'None'
    },
    'validation': {
        'TrainingInputMode': 'File',
        'S3DistributionType': 'FullyReplicated',
        'RecordWrapperType': 'None'
    }}

USER_HYPERPARAMETERS = dict(batch_size=32, learning_rate=.001)
SAGEMAKER_HYPERPARAMETERS = {'sagemaker_region': 'us-west-2', 'default_user_module_name': 'net',
                             'sagemaker_job_name': 'sagemaker-training-job', 'sagemaker_program': 'main.py',
                             'sagemaker_submit_directory': 'imagenet', 'sagemaker_enable_cloudwatch_metrics': True,
                             'sagemaker_container_log_level': logging.WARNING}

ALL_HYPERPARAMETERS = dict(itertools.chain(USER_HYPERPARAMETERS.items(), SAGEMAKER_HYPERPARAMETERS.items()))


def test_read_json():
    test.write_json(ALL_HYPERPARAMETERS, smc.environment.HYPERPARAMETERS_PATH)

    assert smc.environment.read_json(smc.environment.HYPERPARAMETERS_PATH) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        smc.environment.read_json('non-existent.json')


def test_read_hyperparameters():
    test.write_json(ALL_HYPERPARAMETERS, smc.environment.HYPERPARAMETERS_PATH)

    assert smc.environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_key_serialized_hyperparameters():
    key_serialized_hps = {k: json.dumps(v) for k, v in ALL_HYPERPARAMETERS.items()}
    test.write_json(key_serialized_hps, smc.environment.HYPERPARAMETERS_PATH)

    assert smc.environment.read_hyperparameters() == ALL_HYPERPARAMETERS


@patch('sagemaker_containers.environment.read_json', lambda x: {'a': 1})
@patch('json.loads')
def test_read_exception(loads):
    loads.side_effect = ValueError('Unable to read.')

    with pytest.raises(ValueError) as e:
        smc.environment.read_hyperparameters()
    assert 'Unable to read.' in str(e)


def test_resource_config():
    test.write_json(RESOURCE_CONFIG, smc.environment.RESOURCE_CONFIG_PATH)

    assert smc.environment.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config():
    test.write_json(INPUT_DATA_CONFIG, smc.environment.INPUT_DATA_CONFIG_FILE_PATH)

    assert smc.environment.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs():
    input_data_path = smc.environment.INPUT_DATA_PATH
    assert smc.environment.channel_path('evaluation') == os.path.join(input_data_path, 'evaluation')
    assert smc.environment.channel_path('training') == os.path.join(input_data_path, 'training')


@patch('subprocess.check_output', lambda s: six.b('GPU 0\nGPU 1'))
def test_gpu_count_in_gpu_instance():
    assert smc.environment.gpu_count() == 2


@patch('multiprocessing.cpu_count', lambda: OSError())
def test_gpu_count_in_cpu_instance():
    assert smc.environment.gpu_count() == 0


@patch('multiprocessing.cpu_count', lambda: 2)
def test_cpu_count():
    assert smc.environment.cpu_count() == 2


@pytest.fixture(name='training_environment')
def create_training_environment():
    with patch('sagemaker_containers.environment.read_resource_config', lambda: RESOURCE_CONFIG), \
         patch('sagemaker_containers.environment.read_input_data_config', lambda: INPUT_DATA_CONFIG), \
         patch('sagemaker_containers.environment.read_hyperparameters', lambda: ALL_HYPERPARAMETERS), \
         patch('sagemaker_containers.environment.cpu_count', lambda: 8), \
         patch('sagemaker_containers.environment.gpu_count', lambda: 4):
        session_mock = Mock()
        session_mock.region_name = 'us-west-2'
        return smc.environment.TrainingEnvironment(session=session_mock)


@pytest.fixture(name='serving_environment')
def create_serving_environment():
    with patch('sagemaker_containers.environment.cpu_count', lambda: 8), \
         patch('sagemaker_containers.environment.gpu_count', lambda: 4):

        os.environ[smc.environment.USE_NGINX_ENV] = 'false'
        os.environ[smc.environment.MODEL_SERVER_TIMEOUT_ENV] = '20'
        os.environ[smc.environment.CURRENT_HOST_ENV] = 'algo-1'
        os.environ[smc.environment.USER_PROGRAM_ENV] = 'main.py'
        os.environ[smc.environment.SUBMIT_DIR_ENV] = 'my_dir'
        os.environ[smc.environment.ENABLE_METRICS_ENV] = 'true'
        os.environ[smc.environment.REGION_NAME_ENV] = 'us-west-2'
        return smc.environment.ServingEnvironment(session=Mock())


def test_train_environment_create(training_environment):
    assert training_environment.num_gpu == 4
    assert training_environment.num_cpu == 8
    assert training_environment.input_dir.endswith('/opt/ml/input')
    assert training_environment.input_config_dir.endswith('/opt/ml/input/config')
    assert training_environment.model_dir.endswith('/opt/ml/model')
    assert training_environment.output_dir.endswith('/opt/ml/output')
    assert training_environment.hyperparameters == USER_HYPERPARAMETERS
    assert training_environment.resource_config == RESOURCE_CONFIG
    assert training_environment.input_data_config == INPUT_DATA_CONFIG
    assert training_environment.output_data_dir.endswith('/opt/ml/output/data')
    assert training_environment.hosts == RESOURCE_CONFIG['hosts']
    assert training_environment.channel_input_dirs['train'].endswith('/opt/ml/input/data/train')
    assert training_environment.channel_input_dirs['validation'].endswith('/opt/ml/input/data/validation')
    assert training_environment.current_host == RESOURCE_CONFIG['current_host']
    assert training_environment.module_name == 'main'
    assert training_environment.module_dir == 'imagenet'
    assert training_environment.enable_metrics
    assert training_environment.log_level == logging.WARNING


def test_serve_environment_create(serving_environment):
    assert serving_environment.num_gpu == 4
    assert serving_environment.num_cpu == 8
    assert serving_environment.use_nginx is False
    assert serving_environment.model_server_timeout == 20
    assert serving_environment.model_server_workers == 8
    assert serving_environment.module_name == 'main'
    assert serving_environment.enable_metrics


def test_train_environment_properties(training_environment):
    assert training_environment.properties() == ['channel_input_dirs', 'current_host', 'enable_metrics', 'hosts',
                                                 'hyperparameters', 'input_config_dir', 'input_data_config',
                                                 'input_dir', 'log_level', 'model_dir', 'module_dir', 'module_name',
                                                 'num_cpu', 'num_gpu', 'output_data_dir', 'output_dir',
                                                 'resource_config']


def test_serve_environment_properties(serving_environment):
    assert serving_environment.properties() == ['current_host', 'enable_metrics', 'log_level', 'model_server_timeout',
                                                'model_server_workers', 'module_dir', 'module_name', 'num_cpu',
                                                'num_gpu', 'use_nginx']


@patch('sagemaker_containers.environment.cpu_count', lambda: 8)
@patch('sagemaker_containers.environment.gpu_count', lambda: 4)
def test_environment_dictionary():
    session_mock = Mock()
    session_mock.region_name = 'us-west-2'
    os.environ[smc.environment.USER_PROGRAM_ENV] = 'my_app.py'
    env = smc.environment.Environment(session=session_mock)

    assert len(env) == len(env.properties())

    assert env['num_gpu'] == 4
    assert env['num_cpu'] == 8
    assert env['module_name'] == 'my_app'
    assert env['enable_metrics']
    assert env['log_level'] == logging.INFO


def test_environment_dictionary_get_exception(serving_environment):
    with pytest.raises(KeyError) as e:
        serving_environment['non_existent_field']

    assert str(e.value.args[0]) == 'Trying to access invalid key non_existent_field'


@pytest.mark.parametrize('sagemaker_program', ['program.py', 'program'])
def test_environment_module_name(sagemaker_program):
    session_mock = Mock()
    session_mock.region_name = 'us-west-2'
    os.environ[smc.environment.USER_PROGRAM_ENV] = sagemaker_program
    env = smc.environment.Environment(session=session_mock)
    assert env.module_name == 'program'


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_temporary_directory(rmtree, mkdtemp):
    with smc.environment.tmpdir():
        mkdtemp.assert_called()
    rmtree.assert_called()


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_temporary_directory_with_args(rmtree, mkdtemp):
    with smc.environment.tmpdir('suffix', 'prefix', '/tmp'):
        mkdtemp.assert_called_with(dir='/tmp', prefix='prefix', suffix='suffix')
    rmtree.assert_called()
