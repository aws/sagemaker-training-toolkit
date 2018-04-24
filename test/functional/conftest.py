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
import time

import boto3
import pytest
from sagemaker import fw_utils, Session
import six

logger = logging.getLogger(__name__)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

DEFAULT_REGION = 'us-west-2'


@pytest.fixture(name='test_bucket', scope='session')
def create_test_bucket(sagemaker_session):
    return sagemaker_session.default_bucket()


@pytest.fixture
def upload_script(tmpdir, sagemaker_session, test_bucket):
    s3_key_prefix = os.path.join('test', 'sagemaker-containers', str(time.time()))

    def upload_script_fn(name):
        session = sagemaker_session.boto_session
        uploaded_code = fw_utils.tar_and_upload_dir(script=name, session=session, bucket=test_bucket,
                                                    s3_key_prefix=s3_key_prefix, directory=str(tmpdir))
        return uploaded_code.s3_prefix

    return upload_script_fn


@pytest.fixture
def create_script(tmpdir):
    def create_script_fn(name, content):
        content = [content] if isinstance(content, six.string_types) else content

        tmpdir.join(name).write(os.linesep.join(content))

    return create_script_fn


@pytest.fixture(scope='session', name='sagemaker_session')
def create_sagemaker_session():
    boto_session = boto3.Session(region_name=DEFAULT_REGION)

    return Session(boto_session=boto_session)
