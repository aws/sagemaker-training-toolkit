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

from mock import patch, PropertyMock
import pytest
from six.moves import range

from sagemaker_containers import content_types, status_codes, worker


class Transformer(object):
    def __init__(self):
        self.calls = dict(initialize=0, transform=0)

    def initialize(self):
        self.calls['initialize'] += 1

    def transform(self):
        self.calls['transform'] += 1
        return worker.Response(json.dumps(self.calls), content_types.JSON)


def test_worker_with_initialize():
    transformer = Transformer()

    with worker.Worker(transform_fn=transformer.transform,
                       initialize_fn=transformer.initialize,
                       module_name='worker_with_initialize').test_client() as client:
        assert client.application.import_name == 'worker_with_initialize'

        assert client.get('/ping').status_code == status_codes.OK

        for _ in range(9):
            response = client.post('/invocations')
            assert response.status_code == status_codes.OK

        response = client.post('/invocations')
        assert json.loads(response.get_data(as_text=True)) == dict(initialize=1, transform=10)
        assert response.mimetype == content_types.JSON


@patch('sagemaker_containers.env.ServingEnv.module_name', PropertyMock(return_value='user_program'))
@pytest.mark.parametrize('module_name,expected_name', [('my_module', 'my_module'), (None, 'user_program')])
def test_worker(module_name, expected_name):
    transformer = Transformer()

    with worker.Worker(transform_fn=transformer.transform,
                       module_name=module_name).test_client() as client:
        assert client.application.import_name == expected_name

        assert client.get('/ping').status_code == status_codes.OK

        for _ in range(9):
            response = client.post('/invocations')
            assert response.status_code == status_codes.OK

        response = client.post('/invocations')
        assert json.loads(response.get_data(as_text=True)) == dict(initialize=0, transform=10)
        assert response.mimetype == content_types.JSON


def test_worker_with_custom_ping():
    transformer = Transformer()

    def custom_ping():
        return 'ping', status_codes.ACCEPTED

    with worker.Worker(transform_fn=transformer.transform,
                       healthcheck_fn=custom_ping,
                       module_name='custom_ping').test_client() as client:
        response = client.get('/ping')
        assert response.status_code == status_codes.ACCEPTED
        assert response.get_data(as_text=True) == 'ping'
