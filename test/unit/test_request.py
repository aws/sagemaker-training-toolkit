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

from mock import patch
import numpy as np
import pytest

from sagemaker_training import content_types, encoders, worker
import test


@pytest.mark.parametrize("content_type_header", ["ContentType", "Content-Type"])
def test_request(content_type_header):
    headers = {content_type_header: content_types.JSON, "Accept": content_types.CSV}

    request = worker.Request(test.environ(data="42", headers=headers))

    assert request.content_type == content_types.JSON
    assert request.accept == content_types.CSV
    assert request.content == "42"

    headers = {content_type_header: content_types.NPY, "Accept": content_types.CSV}
    request = worker.Request(
        test.environ(data=encoders.encode([6, 9.3], content_types.NPY), headers=headers)
    )

    assert request.content_type == content_types.NPY
    assert request.accept == content_types.CSV

    result = encoders.decode(request.data, content_types.NPY)
    np.testing.assert_array_equal(result, np.array([6, 9.3]))


@patch("sagemaker_training.env.ServingEnv")
def test_request_without_accept(serving_env):
    serving_env.default_accept = "application/json"

    request = worker.Request(test.environ(), serving_env=serving_env)
    assert request.accept == "application/json"


@patch("sagemaker_training.env.ServingEnv")
def test_request_with_accept_any(serving_env):
    serving_env.default_accept = "application/NPY"

    request = worker.Request(
        test.environ(headers={"Accept": content_types.ANY}), serving_env=serving_env
    )

    assert request.accept == "application/NPY"


@patch("sagemaker_training.env.ServingEnv")
def test_request_with_accept(serving_env):
    serving_env.default_accept = "application/NPY"

    request = worker.Request(
        test.environ(headers={"Accept": content_types.CSV}), serving_env=serving_env
    )

    assert request.accept == "text/csv"


@pytest.mark.parametrize("content_type_header", ["ContentType", "Content-Type"])
def test_request_content_type(content_type_header):
    response = test.request(headers={content_type_header: content_types.CSV})
    assert response.content_type == content_types.CSV

    response = worker.Request(test.environ(headers={"ContentType": content_types.NPY}))
    assert response.content_type == content_types.NPY
