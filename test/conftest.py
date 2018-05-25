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

import logging
import os
import shutil

from mock import patch
import pytest

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture(autouse=True)
def create_base_path():
    from sagemaker_containers import _env

    os.makedirs(_env.model_dir)
    os.makedirs(_env.input_config_dir)
    os.makedirs(_env._output_data_dir)

    yield str(os.environ['base_dir'])

    shutil.rmtree(os.environ['base_dir'])


@pytest.fixture(autouse=True)
def patch_exit_process():
    def _exit(error_code):
        if error_code:
            raise ValueError(error_code)

    with patch('sagemaker_containers._trainer._exit_processes', _exit):
        yield _exit
