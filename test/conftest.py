# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import json
import logging
import os
import time

import boto3
from mock import patch
import numpy as np
import pytest
import sagemaker
import six

import sagemaker_containers.environment as environment

logger = logging.getLogger(__name__)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture(scope='session', name='sagemaker_session')
def create_sagemaker_session():
    boto_session = boto3.Session(region_name=DEFAULT_REGION)

    return sagemaker.Session(boto_session=boto_session)


@pytest.fixture(name='opt_ml_path')
def override_opt_ml_path(tmpdir):
    input_data = tmpdir.mkdir('input')
    input_data.mkdir('config')
    input_data.mkdir('data')
    tmpdir.mkdir('model')

    with patch.dict('os.environ', {'BASE_PATH': str(tmpdir)}):
        six.moves.reload_module(environment)
        yield tmpdir
    six.moves.reload_module(environment)


@pytest.fixture(name='input_path')
def override_input_path(opt_ml_path):
    return opt_ml_path.join('input')


@pytest.fixture(name='input_config_path')
def override_input_config_path(input_path):
    return input_path.join('config')


@pytest.fixture(name='input_data_path')
def override_input_data_path(input_path):
    return input_path.join('data')


def json_dump(data, path_obj):  # type: (object, py.path.local) -> None
    """Writes JSON serialized data to the local file system path

    Args:
        data (object): object to be serialized
        path_obj (py.path.local): path.local object of the file to be written
    """
    path_obj.write(json.dumps(data))


@pytest.fixture(name='upload_script')
def fixture_upload_script(tmpdir, sagemaker_session, test_bucket):
    s3_key_prefix = os.path.join('test', 'sagemaker-containers', str(time.time()))

    def upload_script_fn(name, directory=None):
        directory = directory or str(tmpdir)
        session = sagemaker_session.boto_session
        uploaded_code = sagemaker.fw_utils.tar_and_upload_dir(script=name, session=session, bucket=test_bucket,
                                                              s3_key_prefix=s3_key_prefix, directory=directory)
        return uploaded_code.s3_prefix

    return upload_script_fn


@pytest.fixture
def create_channel(input_data_path):
    def create_channel_fn(channel, file_name, data):
        np.savez(str(input_data_path.mkdir(channel).join(file_name)), **data)

    return create_channel_fn


@pytest.fixture(name='test_bucket', scope='session')
def create_test_bucket(sagemaker_session):
    return sagemaker_session.default_bucket()


@pytest.fixture(name='create_script')
def fixture_create_script(tmpdir):
    def create_script_fn(name, content):
        content = [content] if isinstance(content, six.string_types) else content

        tmpdir.join(name).write(os.linesep.join(content))

    return create_script_fn


@pytest.fixture
def create_training(create_script, upload_script, input_config_path):
    def create_training_fn(script_name, script, hyperparameters, resource_config, input_data_config):
        create_script(script_name, script)
        hyperparameters['sagemaker_submit_directory'] = upload_script(name=script_name)

        json_dump(hyperparameters, input_config_path.join('hyperparameters.json'))
        json_dump(resource_config, input_config_path.join('resourceconfig.json'))
        json_dump(input_data_config, input_config_path.join('inputdataconfig.json'))

    return create_training_fn
