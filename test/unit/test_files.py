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
import logging
import os

from mock import mock_open, patch
import pytest
import six

from sagemaker_containers import _env, _files
import test

builtins_open = '__builtin__.open' if six.PY2 else 'builtins.open'

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
    test.write_json(ALL_HYPERPARAMETERS, _env.hyperparameters_file_dir)

    assert _files.read_json(_env.hyperparameters_file_dir) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        _files.read_json('non-existent.json')


def test_read_file():
    test.write_json('test', _env.hyperparameters_file_dir)

    assert _files.read_file(_env.hyperparameters_file_dir) == '\"test\"'


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_tmpdir(rmtree, mkdtemp):
    with _files.tmpdir():
        mkdtemp.assert_called()
    rmtree.assert_called()


@patch('tempfile.mkdtemp')
@patch('shutil.rmtree')
def test_tmpdir_with_args(rmtree, mkdtemp):
    with _files.tmpdir('suffix', 'prefix', '/tmp'):
        mkdtemp.assert_called_with(dir='/tmp', prefix='prefix', suffix='suffix')
    rmtree.assert_called()


@patch(builtins_open, mock_open())
def test_write_file():
    _files.write_file('/tmp/my-file', '42')
    open.assert_called_with('/tmp/my-file', 'w')
    open().write.assert_called_with('42')

    _files.write_file('/tmp/my-file', '42', 'a')
    open.assert_called_with('/tmp/my-file', 'a')
    open().write.assert_called_with('42')


@patch(builtins_open, mock_open())
def test_write_success_file():
    file_path = os.path.join(_env.output_dir, 'success')
    empty_msg = ''
    _files.write_success_file()
    open.assert_called_with(file_path, 'w')
    open().write.assert_called_with(empty_msg)


@patch(builtins_open, mock_open())
def test_write_failure_file():
    file_path = os.path.join(_env.output_dir, 'failure')
    failure_msg = 'This is a failure'
    _files.write_failure_file(failure_msg)
    open.assert_called_with(file_path, 'w')
    open().write.assert_called_with(failure_msg)


@patch('sagemaker_containers._files.s3_download')
@patch('os.path.isdir', lambda x: True)
@patch('shutil.rmtree')
@patch('shutil.move')
def test_download_and_and_extract_source_dir(move, rmtree, s3_download):
    uri = _env.channel_path('code')
    _files.download_and_extract(uri, 'train.sh', _env.code_dir)
    s3_download.assert_not_called()

    rmtree.assert_any_call(_env.code_dir)
    move.assert_called_with(uri, _env.code_dir)


@patch('sagemaker_containers._files.s3_download')
@patch('os.path.isdir', lambda x: False)
@patch('shutil.copy2')
def test_download_and_and_extract_file(copy, s3_download):
    uri = _env.channel_path('code')
    _files.download_and_extract(uri, 'train.sh', _env.code_dir)

    s3_download.assert_not_called()
    copy.assert_called_with(uri, os.path.join(_env.code_dir, 'train.sh'))
