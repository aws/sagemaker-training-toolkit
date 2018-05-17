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

from mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest
from six.moves import range

from sagemaker_containers import content_types, encoders, status_codes, worker
import test


def test_default_ping_fn():
    assert worker.default_healthcheck_fn().status_code == status_codes.OK


@pytest.fixture(name='flask')
def patch_flask():
    property_mock = PropertyMock(return_value='user_program')
    with patch('flask.Flask') as flask, \
            patch('sagemaker_containers.env.ServingEnv.module_name',
                  property_mock):
        yield flask


@pytest.mark.parametrize('module_name, expected_name', [('test_module', 'test_module'), (None, 'user_program')])
@patch('sagemaker_containers.env.ServingEnv.module_name', PropertyMock(return_value='user_program'))
def test_worker(module_name, expected_name):
    app = worker.Worker(transform_fn=MagicMock().transform, module_name=module_name)
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == []
    assert app.request_class == worker.Request


@pytest.mark.parametrize('module_name, expected_name', [('test_module', 'test_module'), (None, 'user_program')])
@patch('sagemaker_containers.env.ServingEnv.module_name', PropertyMock(return_value='user_program'))
def test_worker_with_initialize(module_name, expected_name):
    mock = MagicMock()
    app = worker.Worker(transform_fn=mock.transform, initialize_fn=mock.initialize, module_name=module_name)
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == [mock.initialize]
    assert app.request_class == worker.Request


def test_invocations():
    def transform_fn():
        return worker.Response(response='fake data', accept=content_types.JSON)

    app = worker.Worker(transform_fn=transform_fn, module_name='test_module')

    with app.test_client() as client:
        for _ in range(9):
            response = client.post('/invocations')
            assert response.status_code == status_codes.OK
            assert response.get_data().decode('utf-8') == 'fake data'
            assert response.mimetype == content_types.JSON


def test_ping():
    app = worker.Worker(transform_fn=MagicMock(), module_name='test_module')

    with app.test_client() as client:
        for _ in range(9):
            response = client.get('/ping')
            assert response.status_code == status_codes.OK
            assert response.mimetype == content_types.JSON


def test_request():
    request = test.request(data='42')

    assert request.content_type == content_types.JSON
    assert request.accept == content_types.JSON
    assert request.content == '42'

    request = test.request(data=encoders.encode([6, 9.3], content_types.NPY),
                           content_type=content_types.NPY,
                           accept=content_types.CSV)

    assert request.content_type == content_types.NPY
    assert request.accept == content_types.CSV

    result = encoders.decode(request.data, content_types.NPY)
    np.testing.assert_array_equal(result, np.array([6, 9.3]))


def test_request_content_type():
    response = test.request(content_type=content_types.CSV)
    assert response.content_type == content_types.CSV

    response = test.request(headers={'ContentType': content_types.NPY})
    assert response.content_type == content_types.NPY
