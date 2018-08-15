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
from six.moves import http_client, range

from sagemaker_containers import _content_types, _encoders, _worker
import test


def test_default_ping_fn():
    assert _worker.default_healthcheck_fn().status_code == http_client.OK


@pytest.fixture(name='flask')
def patch_flask():
    property_mock = PropertyMock(return_value='user_program')
    with patch('flask.Flask') as flask, \
            patch('sagemaker_containers._env.ServingEnv.module_name',
                  property_mock):
        yield flask


@pytest.mark.parametrize('module_name, expected_name', [('test_module', 'test_module'), (None, 'user_program')])
@patch('sagemaker_containers._env.ServingEnv.module_name', PropertyMock(return_value='user_program'))
def test_worker(module_name, expected_name):
    app = _worker.Worker(transform_fn=MagicMock().transform, module_name=module_name)
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == []
    assert app.request_class == _worker.Request


@pytest.mark.parametrize('module_name, expected_name', [('test_module', 'test_module'), (None, 'user_program')])
@patch('sagemaker_containers._env.ServingEnv.module_name', PropertyMock(return_value='user_program'))
def test_worker_with_initialize(module_name, expected_name):
    mock = MagicMock()
    app = _worker.Worker(transform_fn=mock.transform, initialize_fn=mock.initialize, module_name=module_name)
    assert app.import_name == expected_name
    assert app.before_first_request_funcs == [mock.initialize]
    assert app.request_class == _worker.Request


def test_invocations():
    def transform_fn():
        return _worker.Response(response='fake data', accept=_content_types.JSON)

    app = _worker.Worker(transform_fn=transform_fn, module_name='test_module')

    with app.test_client() as client:
        for _ in range(9):
            response = client.post('/invocations')
            assert response.status_code == http_client.OK
            assert response.get_data().decode('utf-8') == 'fake data'
            assert response.mimetype == _content_types.JSON


def test_ping():
    app = _worker.Worker(transform_fn=MagicMock(), module_name='test_module')

    with app.test_client() as client:
        for _ in range(9):
            response = client.get('/ping')
            assert response.status_code == http_client.OK
            assert response.mimetype == _content_types.JSON


def test_request():
    request = test.request(data='42')

    assert request.content_type == _content_types.JSON
    assert request.accept == _content_types.JSON
    assert request.content == '42'

    request = test.request(data=_encoders.encode([6, 9.3], _content_types.NPY),
                           content_type=_content_types.NPY,
                           accept=_content_types.CSV)

    assert request.content_type == _content_types.NPY
    assert request.accept == _content_types.CSV

    result = _encoders.decode(request.data, _content_types.NPY)
    np.testing.assert_array_equal(result, np.array([6, 9.3]))


def test_request_content_type():
    response = test.request(content_type=_content_types.CSV)
    assert response.content_type == _content_types.CSV

    response = test.request(headers={'ContentType': _content_types.NPY})
    assert response.content_type == _content_types.NPY
