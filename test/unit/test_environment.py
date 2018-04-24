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

from mock import Mock, patch
import pytest
from six import b
from six.moves import reload_module

import sagemaker_containers.environment as environment

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


@pytest.fixture(name='opt_ml_path')
def override_opt_ml_path(tmpdir):
    opt_ml = tmpdir.mkdir('opt').mkdir('ml')
    with patch.dict('os.environ', {'BASE_PATH': str(opt_ml)}):
        reload_module(environment)
        yield opt_ml
    reload_module(environment)


@pytest.fixture(name='input_path')
def override_input_path(opt_ml_path):
    return opt_ml_path.mkdir('input')


@pytest.fixture(name='input_config_path')
def override_input_config_path(input_path):
    return input_path.mkdir('config')


@pytest.fixture(name='input_data_path')
def override_input_data_path(input_path):
    return input_path.mkdir('data')


def test_read_json(tmpdir):
    path_obj = tmpdir.join('hyperparameters.json')
    json_dump(ALL_HYPERPARAMETERS, tmpdir.join('hyperparameters.json'))

    assert environment.read_json(str(path_obj)) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        environment.read_json('non-existent.json')


def test_read_hyperparameters(input_config_path):
    json_dump(ALL_HYPERPARAMETERS, input_config_path.join('hyperparameters.json'))

    assert environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_key_serialized_hyperparameters(input_config_path):
    key_serialized_hps = {k: json.dumps(v) for k, v in ALL_HYPERPARAMETERS.items()}
    json_dump(key_serialized_hps, input_config_path.join('hyperparameters.json'))

    assert environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_split_hyperparameters_only_provided_by_user():
    assert environment.split_hyperparameters(USER_HYPERPARAMETERS) == ({}, USER_HYPERPARAMETERS)


def test_split_hyperparameters_only_provided_by_sagemaker():
    assert environment.split_hyperparameters(SAGEMAKER_HYPERPARAMETERS) == (SAGEMAKER_HYPERPARAMETERS, {})


def test_split_hyperparameters():
    assert environment.split_hyperparameters(ALL_HYPERPARAMETERS) == (SAGEMAKER_HYPERPARAMETERS, USER_HYPERPARAMETERS)


def test_resource_config(input_config_path):
    json_dump(RESOURCE_CONFIG, input_config_path.join('resourceconfig.json'))

    assert environment.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config(input_config_path):
    json_dump(INPUT_DATA_CONFIG, input_config_path.join('inputdataconfig.json'))

    assert environment.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs(input_data_path):
    assert environment.channel_path('evaluation') == str(input_data_path.join('evaluation'))
    assert environment.channel_path('training') == str(input_data_path.join('training'))


@patch('subprocess.check_output', lambda s: b('GPU 0\nGPU 1'))
def test_gpu_count_in_gpu_instance():
    assert environment.gpu_count() == 2


@patch('multiprocessing.cpu_count', lambda: OSError())
def test_gpu_count_in_cpu_instance():
    assert environment.gpu_count() == 0


@patch('multiprocessing.cpu_count', lambda: 2)
def test_cpu_count():
    assert environment.cpu_count() == 2


@patch('sagemaker_containers.environment.read_resource_config', lambda: RESOURCE_CONFIG)
@patch('sagemaker_containers.environment.read_input_data_config', lambda: INPUT_DATA_CONFIG)
@patch('sagemaker_containers.environment.read_hyperparameters', lambda: ALL_HYPERPARAMETERS)
@patch('sagemaker_containers.environment.cpu_count', lambda: 8)
@patch('sagemaker_containers.environment.gpu_count', lambda: 4)
def test_environment_create():
    env = environment.Environment.create(session=Mock())

    assert env.num_gpu == 4
    assert env.num_cpu == 8
    assert env.input_dir == '/opt/ml/input'
    assert env.input_config_dir == '/opt/ml/input/config'
    assert env.model_dir == '/opt/ml/model'
    assert env.output_dir == '/opt/ml/output'
    assert env.hyperparameters == USER_HYPERPARAMETERS
    assert env.resource_config == RESOURCE_CONFIG
    assert env.input_data_config == INPUT_DATA_CONFIG
    assert env.output_data_dir == '/opt/ml/output/data'
    assert env.hosts == RESOURCE_CONFIG['hosts']
    assert env.channel_input_dirs['train'] == '/opt/ml/input/data/train'
    assert env.channel_input_dirs['validation'] == '/opt/ml/input/data/validation'
    assert env.current_host == RESOURCE_CONFIG['current_host']
    assert env.module_name == 'main.py'
    assert env.module_dir == 'imagenet'
    assert env.enable_metrics
    assert env.log_level == logging.WARNING


def json_dump(data, path_obj):  # type: (object, py.path.local) -> None
    """Writes JSON serialized data to the local file system path

    Args:
        data (object): object to be serialized
        path_obj (py.path.local): path.local object of the file to be written
    """
    path_obj.write(json.dumps(data))
