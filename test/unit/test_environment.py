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
import six

import sagemaker_containers as smc
from test.conftest import json_dump

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


def test_read_json(tmpdir):
    path_obj = tmpdir.join('hyperparameters.json')
    json_dump(ALL_HYPERPARAMETERS, tmpdir.join('hyperparameters.json'))

    assert smc.environment.read_json(str(path_obj)) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        smc.environment.read_json('non-existent.json')


def test_read_hyperparameters(input_config_path):
    json_dump(ALL_HYPERPARAMETERS, input_config_path.join('hyperparameters.json'))

    assert smc.environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_key_serialized_hyperparameters(input_config_path):
    key_serialized_hps = {k: json.dumps(v) for k, v in ALL_HYPERPARAMETERS.items()}
    json_dump(key_serialized_hps, input_config_path.join('hyperparameters.json'))

    assert smc.environment.read_hyperparameters() == ALL_HYPERPARAMETERS


@patch('sagemaker_containers.environment.read_json', lambda x: {'a': 1})
@patch('json.loads')
def test_read_exception(loads):
    loads.side_effect = ValueError('Unable to read.')

    with pytest.raises(ValueError) as e:
        smc.environment.read_hyperparameters()
    assert 'Unable to read.' in str(e)


def test_resource_config(input_config_path):
    json_dump(RESOURCE_CONFIG, input_config_path.join('resourceconfig.json'))

    assert smc.environment.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config(input_config_path):
    json_dump(INPUT_DATA_CONFIG, input_config_path.join('inputdataconfig.json'))

    assert smc.environment.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs(input_data_path):
    assert smc.environment.channel_path('evaluation') == str(input_data_path.join('evaluation'))
    assert smc.environment.channel_path('training') == str(input_data_path.join('training'))


@patch('subprocess.check_output', lambda s: six.b('GPU 0\nGPU 1'))
def test_gpu_count_in_gpu_instance():
    assert smc.environment.gpu_count() == 2


@patch('multiprocessing.cpu_count', lambda: OSError())
def test_gpu_count_in_cpu_instance():
    assert smc.environment.gpu_count() == 0


@patch('multiprocessing.cpu_count', lambda: 2)
def test_cpu_count():
    assert smc.environment.cpu_count() == 2


@pytest.fixture(name='environment')
def create_environment():
    with patch('sagemaker_containers.environment.read_resource_config', lambda: RESOURCE_CONFIG), \
         patch('sagemaker_containers.environment.read_input_data_config', lambda: INPUT_DATA_CONFIG), \
         patch('sagemaker_containers.environment.read_hyperparameters', lambda: ALL_HYPERPARAMETERS), \
         patch('sagemaker_containers.environment.cpu_count', lambda: 8), \
         patch('sagemaker_containers.environment.gpu_count', lambda: 4):
        return smc.Environment.create(session=Mock())


def test_environment_create(environment):
    assert environment.num_gpu == 4
    assert environment.num_cpu == 8
    assert environment.input_dir == '/opt/ml/input'
    assert environment.input_config_dir == '/opt/ml/input/config'
    assert environment.model_dir == '/opt/ml/model'
    assert environment.output_dir == '/opt/ml/output'
    assert environment.hyperparameters == USER_HYPERPARAMETERS
    assert environment.resource_config == RESOURCE_CONFIG
    assert environment.input_data_config == INPUT_DATA_CONFIG
    assert environment.output_data_dir == '/opt/ml/output/data'
    assert environment.hosts == RESOURCE_CONFIG['hosts']
    assert environment.channel_input_dirs['train'] == '/opt/ml/input/data/train'
    assert environment.channel_input_dirs['validation'] == '/opt/ml/input/data/validation'
    assert environment.current_host == RESOURCE_CONFIG['current_host']
    assert environment.module_name == 'main'
    assert environment.module_dir == 'imagenet'
    assert environment.enable_metrics
    assert environment.log_level == logging.WARNING


def test_environment_properties(environment):
    assert environment.properties() == ['channel_input_dirs', 'current_host', 'enable_metrics', 'hosts',
                                        'hyperparameters', 'input_config_dir', 'input_data_config', 'input_dir',
                                        'log_level', 'model_dir', 'module_dir', 'module_name', 'num_cpu', 'num_gpu',
                                        'output_data_dir', 'output_dir', 'resource_config']


def test_environment_dictionary(environment):
    assert len(environment) == len(environment.properties())

    assert environment['num_gpu'] == 4
    assert environment['num_cpu'] == 8
    assert environment['input_dir'] == '/opt/ml/input'
    assert environment['input_config_dir'] == '/opt/ml/input/config'
    assert environment['model_dir'] == '/opt/ml/model'
    assert environment['output_dir'] == '/opt/ml/output'
    assert environment['hyperparameters'] == USER_HYPERPARAMETERS
    assert environment['resource_config'] == RESOURCE_CONFIG
    assert environment['input_data_config'] == INPUT_DATA_CONFIG
    assert environment['output_data_dir'] == '/opt/ml/output/data'
    assert environment['hosts'] == RESOURCE_CONFIG['hosts']
    assert environment['channel_input_dirs']['train'] == '/opt/ml/input/data/train'
    assert environment['channel_input_dirs']['validation'] == '/opt/ml/input/data/validation'
    assert environment['current_host'] == RESOURCE_CONFIG['current_host']
    assert environment['module_name'] == 'main'
    assert environment['module_dir'] == 'imagenet'
    assert environment['enable_metrics']
    assert environment['log_level'] == logging.WARNING


def test_environment_dictionary_get_exception(environment):
    with pytest.raises(KeyError) as e:
        environment['non_existent_field']

    assert str(e.value.args[0]) == 'Trying to access invalid key non_existent_field'


@pytest.mark.parametrize('sagemaker_program', ['program.py', 'program'])
def test_environment_module_name(sagemaker_program, environment):
    env_dict = dict(environment)
    del env_dict['module_name']

    env = smc.environment.Environment(module_name=sagemaker_program, **env_dict)
    assert env.module_name == 'program'
