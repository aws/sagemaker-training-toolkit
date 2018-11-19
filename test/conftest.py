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
from __future__ import absolute_import

import json
import logging
import os
import shutil
import socket

from mock import patch
import pytest

from sagemaker_containers import _env

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


def _write_json(obj, path):  # type: (object, str) -> None
    with open(path, 'w') as f:
        json.dump(obj, f)


@pytest.fixture(autouse=True)
def create_base_path():

    yield str(os.environ[_env.BASE_PATH_ENV])

    shutil.rmtree(os.environ[_env.BASE_PATH_ENV])

    os.makedirs(_env.model_dir)
    os.makedirs(_env.input_config_dir)
    os.makedirs(_env.code_dir)
    os.makedirs(_env.output_data_dir)

    _write_json({}, _env.hyperparameters_file_dir)
    _write_json({}, _env.input_data_config_file_dir)
    host_name = socket.gethostname()

    resources_dict = {
        "current_host": host_name,
        "hosts":        [host_name]
    }
    _write_json(resources_dict, _env.resource_config_file_dir)


@pytest.fixture(autouse=True)
def patch_exit_process():
    def _exit(error_code):
        if error_code:
            raise ValueError(error_code)

    with patch('sagemaker_containers._trainer._exit_processes', _exit):
        yield _exit
