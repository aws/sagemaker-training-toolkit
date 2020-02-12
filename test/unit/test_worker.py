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
import warnings

from mock import MagicMock, patch, PropertyMock
import pytest
from six.moves import http_client, range

from sagemaker_training import content_types, worker


def test_default_ping_fn():
    assert worker.default_healthcheck_fn().status_code == http_client.OK


@pytest.mark.parametrize(
    "module_name, expected_name", [("test_module", "test_module"), (None, "user_program")]
)
@patch("sagemaker_training.env.ServingEnv.module_name", PropertyMock(return_value="user_program"))
def test_worker(module_name, expected_name):
    app = worker.Worker(transform_fn=MagicMock().transform, module_name=module_name)
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == []
    assert app.request_class == worker.Request


@pytest.mark.parametrize(
    "module_name, expected_name", [("test_module", "test_module"), (None, "user_program")]
)
@patch("sagemaker_training.env.ServingEnv.module_name", PropertyMock(return_value="user_program"))
def test_worker_with_initialize(module_name, expected_name):
    mock = MagicMock()
    app = worker.Worker(
        transform_fn=mock.transform, initialize_fn=mock.initialize, module_name=module_name
    )
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == [mock.initialize]
    assert app.request_class == worker.Request


@pytest.mark.parametrize("content_type", [content_types.JSON, content_types.ANY])
def test_invocations(content_type):
    def transform_fn():
        return worker.Response(response="fake data", mimetype=content_type)

    app = worker.Worker(transform_fn=transform_fn, module_name="test_module")

    with app.test_client() as client:
        for _ in range(9):
            response = client.post("/invocations")
            assert response.status_code == http_client.OK
            assert response.get_data().decode("utf-8") == "fake data"
            assert response.mimetype == content_type


def test_ping():
    app = worker.Worker(transform_fn=MagicMock(), module_name="test_module")

    with app.test_client() as client:
        for _ in range(9):
            response = client.get("/ping")
            assert response.status_code == http_client.OK
            assert response.mimetype == content_types.JSON


def test_response_accept_deprecated():
    def transform_fn():
        return worker.Response(response="fake data", accept="deprecated accept arg")

    app = worker.Worker(transform_fn=transform_fn, module_name="test_module")
    with app.test_client() as client:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.post("/invocations")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


def test_no_execution_parameters_fn():
    app = worker.Worker(transform_fn=MagicMock(), module_name="test_module")
    with app.test_client() as client:
        response = client.get("/execution-parameters")
        assert response.status_code == http_client.NOT_FOUND


def test_user_execution_parameters_fn():
    expected_json = {"some-json": 1}
    expected_response_string = json.dumps(expected_json)
    test_execution_parameters_fn = MagicMock(return_value=expected_response_string)

    app = worker.Worker(
        transform_fn=MagicMock(),
        module_name="test_module",
        execution_parameters_fn=test_execution_parameters_fn,
    )

    with app.test_client() as client:
        response = client.get("/execution-parameters")
        assert response.status_code == http_client.OK
        assert response.get_data().decode("utf-8") == expected_response_string
        assert json.loads(response.get_data().decode("utf-8")) == expected_json
