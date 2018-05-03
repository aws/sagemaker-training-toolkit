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
import logging
import os
import shutil

import pytest

logger = logging.getLogger(__name__)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture(autouse=True)
def create_base_path():
    from sagemaker_containers import env

    os.makedirs(env.MODEL_PATH)
    os.makedirs(env.INPUT_CONFIG_PATH)
    os.makedirs(env.OUTPUT_DATA_PATH)

    yield str(os.environ['BASE_PATH'])

    shutil.rmtree(os.environ['BASE_PATH'])
