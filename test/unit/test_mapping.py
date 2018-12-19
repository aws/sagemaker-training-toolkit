# -*- coding: utf-8 -*-
#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import pytest

import sagemaker_containers.beta.framework as framework


@pytest.mark.parametrize('dictionary, keys, expected', [
    ({}, (), ({}, {})),
    ({'x': 1, 'y': 2}, 'x', ({'x': 1}, {'y': 2})),
    ({'x': 1, 'y': 2}, (), ({}, {'x': 1, 'y': 2})),
    ({'x': 1, 'y': 2}, ('x', 'y'), ({'x': 1, 'y': 2}, {}))
])
def test_split_by_criteria_with_keys(dictionary, keys, expected):
    assert framework.mapping.split_by_criteria(dictionary, keys=keys) == expected


@pytest.mark.parametrize('dictionary, keys, prefix, expected', [
    ({'x': 1, 'y': 2}, 'y', 'x', ({'x': 1, 'y': 2}, {}))
])
def test_split_by_criteria_with_keys_and_criteria(dictionary, keys, prefix, expected):
    assert framework.mapping.split_by_criteria(dictionary, keys=keys, prefix=prefix) == expected


@pytest.mark.parametrize('dictionary, prefix, expected', [
    ({}, (), ({}, {})),
    ({'sagemaker_x': 1, 'y': 2}, ('sagemaker',), ({'sagemaker_x': 1}, {'y': 2})),
    ({'sagemaker_x': 1, 'y': 2}, ('something_else',), ({}, {'sagemaker_x': 1, 'y': 2})),
    ({'sagemaker_x': 1, 'y': 2}, ('y',), ({'y': 2}, {'sagemaker_x': 1}))
])
def test_split_by_criteria_with_prefix(dictionary, prefix, expected):
    assert framework.mapping.split_by_criteria(dictionary, prefix=prefix) == expected


class ProcessEnvironment(framework.mapping.MappingMixin):
    @property
    def a(self):
        return 1

    @property
    def b(self):
        return 2

    def d(self):
        return 23

    def __init__(self):
        self.c = 3


def test_mapping_mixin():
    p = ProcessEnvironment()

    assert p['a'] == 1
    assert len(p) == 2
    assert p['b'] == 2
    assert str(p) in ("{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}")


@pytest.mark.parametrize('property, error, msg', [
    ('c', AttributeError, "type object 'ProcessEnvironment' has no attribute 'c'"),
    ('d', KeyError, 'Trying to access non property d'),
    ('non_existent_field', AttributeError,
     "type object 'ProcessEnvironment' has no attribute 'non_existent_field'")
])
def test_mapping_throws_exception_trying_to_access_non_properties(property, error, msg):
    with pytest.raises(error) as e:
        ProcessEnvironment()[property]

    assert str(e.value.args[0]) == msg


@pytest.mark.parametrize(
    'target, expected',
    [({'da-sh': '1', 'un_der': '2', 'un-sh': '3', 'da_der': '2'},
      [u'--da-sh', u'1', u'--da_der', u'2', u'--un-sh', u'3', u'--un_der', u'2']),

     ({},
      []),

     ({'': ''},
      [u'', u'']),

     ({'unicode': u'¡ø', 'bytes': b'2', 'floats': 4., 'int': 2},
      [u'--bytes', u'2', u'--floats', u'4.0', u'--int', u'2', u'--unicode', u'¡ø']),

     ({'U': u'1', 'b': b'2', 'T': '', '': '42'},
      ['', '42', '-T', '', '-U', '1', '-b', '2']),

     ({'nested': ['1', ['2', '3', [['6']]]]},
      ['--nested', "['1', ['2', '3', [['6']]]]"]),

     ({'map': {'a': [1, 3, 4]}, 'channel_dirs': {'train': 'foo', 'eval': 'bar'}},
      ['--channel_dirs', 'eval=bar,train=foo', '--map', 'a=[1, 3, 4]']),

     ({'truthy': True, 'falsy': False},
      ['--falsy', 'False', '--truthy', 'True'])

     ])
def test_to_cmd_args(target, expected):
    actual = framework.mapping.to_cmd_args(target)

    assert actual == expected


@pytest.mark.parametrize('target, expected', [
    ({'model_dir': '/opt/ml/model', 'OUTPUT_DIR': '/opt/ml/output'},
     {'SM_MODEL_DIR': '/opt/ml/model', 'SM_OUTPUT_DIR': '/opt/ml/output'}),

    ({}, {}),

    ({'': None}, {u'': u''}),

    ({'bytes': b'2', 'floats': 4.0, 'int': 2, 'unicode': '¡ø'},
     {'SM_BYTES': '2', 'SM_FLOATS': '4.0', 'SM_INT': '2', 'SM_UNICODE': '¡ø'}),

    ({'nested': ['1', ['2', '3', [['6']]]]},
     {'SM_NESTED': '["1",["2","3",[["6"]]]]'}),

    ({'channel_dirs': {'eval': 'bar', 'train': 'foo'}, 'map': {'a': [1, 3, 4]}},
     {'SM_CHANNEL_DIRS': '{"eval":"bar","train":"foo"}', 'SM_MAP': '{"a":[1,3,4]}'}),

    ({'truthy': True, 'falsy': False},
     {'SM_FALSY': 'false', 'SM_TRUTHY': 'true'})
])
def test_to_env_vars(target, expected):
    actual = framework.mapping.to_env_vars(target)

    assert actual == expected


def test_env_vars_round_trip():
    hyperparameters = {
        'loss': 'SGD',
        'sagemaker_program': 'user_script.py',
        'epochs': 10,
        'batch_size': 64,
        'precision': 5.434322,
        'sagemaker_region': 'us-west-2',
        'sagemaker_job_name': 'horovod-training-job',
        'sagemaker_submit_directory': 's3/something'
    }

    resource_config = {
        'current_host': 'algo-1',
        'hosts': ['algo-1', 'algo-2', 'algo-3']
    }

    input_data_config = {
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
        }
    }

    os.environ[framework.params.FRAMEWORK_TRAINING_MODULE_ENV] = 'test.functional.simple_framework:train'

    training_env = framework.training_env(resource_config=resource_config,
                                          input_data_config=input_data_config,
                                          hyperparameters=hyperparameters)

    os.environ[framework.params.FRAMEWORK_TRAINING_MODULE_ENV] = ''

    args = framework.mapping.to_cmd_args(training_env.hyperparameters)

    env_vars = training_env.to_env_vars()
    env_vars['SM_USER_ARGS'] = ' '.join(args)

    assert env_vars['SM_OUTPUT_DATA_DIR'] == training_env.output_data_dir
    assert env_vars['SM_INPUT_DATA_CONFIG'] == '{"train":{"ContentType":"trainingContentType",' \
                                               '"RecordWrapperType":"None","S3DistributionType":"FullyReplicated",' \
                                               '"TrainingInputMode":"File"},"validation":{"RecordWrapperType":"None",' \
                                               '"S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}'
    assert env_vars['SM_NETWORK_INTERFACE_NAME'] == 'eth0'
    assert env_vars['SM_LOG_LEVEL'] == '20'
    assert env_vars['SM_INPUT_DIR'].endswith('/opt/ml/input')
    assert env_vars['SM_NUM_CPUS'] == str(training_env.num_cpus)
    assert env_vars['SM_HP_BATCH_SIZE'] == '64'
    assert env_vars['SM_CHANNEL_TRAIN'].endswith('/opt/ml/input/data/train')
    assert env_vars['SM_CHANNEL_VALIDATION'].endswith('/opt/ml/input/data/validation')
    assert env_vars['SM_HP_EPOCHS'] == '10'
    assert env_vars['SM_HPS'] == '{"batch_size":64,"epochs":10,"loss":"SGD","precision":5.434322}'
    assert env_vars['SM_HP_PRECISION'] == '5.434322'
    assert env_vars['SM_RESOURCE_CONFIG'] == '{"current_host":"algo-1","hosts":["algo-1","algo-2","algo-3"]}'
    assert env_vars['SM_MODULE_NAME'] == 'user_script'
    assert env_vars['SM_INPUT_CONFIG_DIR'].endswith('/opt/ml/input/config')
    assert env_vars['SM_USER_ARGS'] == '--batch_size 64 --epochs 10 --loss SGD --precision 5.434322'
    assert env_vars['SM_OUTPUT_DIR'].endswith('/opt/ml/output')
    assert env_vars['SM_MODEL_DIR'].endswith('/opt/ml/model')
    assert env_vars['SM_HOSTS'] == '["algo-1","algo-2","algo-3"]'
    assert env_vars['SM_NUM_GPUS'] == str(training_env.num_gpus)
    assert env_vars['SM_MODULE_DIR'] == 's3/something'
    assert env_vars['SM_CURRENT_HOST'] == 'algo-1'
    assert env_vars['SM_CHANNELS'] == '["train","validation"]'
    assert env_vars['SM_HP_LOSS'] == 'SGD'
    assert env_vars['SM_FRAMEWORK_MODULE'] == 'test.functional.simple_framework:train'

    assert all(x in env_vars['SM_TRAINING_ENV'] for x in (training_env.properties()))
