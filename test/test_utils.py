#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os

import pytest
from mock import patch, call

import container_support as cs


def test_parse_s3_url_invalid():
    with pytest.raises(ValueError):
        cs.parse_s3_url("nots3://blah/blah")


def test_parse_s3_url():
    assert ("bucket", "key") == cs.parse_s3_url("s3://bucket/key")


def test_parse_s3_url_no_key():
    assert ("bucket", "") == cs.parse_s3_url("s3://bucket/")


def test_download_s3():
    with patch('boto3.resource') as patched:
        with patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}):
            assert cs.download_s3_resource("s3://bucket/key", "target") == "target"
            assert [call('s3', region_name='us-west-2'),
                    call().Bucket('bucket'),
                    call().Bucket().download_file('key', 'target')] == patched.mock_calls


def test_download_s3_with_alternate_region_env_var():
    with patch('boto3.resource') as patched:
        with patch.dict(os.environ, {'SAGEMAKER_REGION': 'us-west-2'}):
            assert cs.download_s3_resource("s3://bucket/key", "target") == "target"
            assert [call('s3', region_name='us-west-2'),
                    call().Bucket('bucket'),
                    call().Bucket().download_file('key', 'target')] == patched.mock_calls


def test_untar_directory():
    with patch('container_support.utils.open', create=True) as mocked_open, patch('tarfile.open') as mocked_tarfile:
        cs.untar_directory('a/b/c', 'd/e/f')
        assert call('a/b/c', 'rb') in mocked_open.mock_calls
        assert call().__enter__().extractall(path='d/e/f') in mocked_tarfile.mock_calls
