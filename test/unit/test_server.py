# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from mock import call, Mock, patch, PropertyMock
import pytest

from sagemaker_containers import _env, _server


FIRST_PORT = '1111'
LAST_PORT = '2222'
SAFE_PORT_RANGE = '{}-{}'.format(FIRST_PORT, LAST_PORT)


@patch.object(_env.ServingEnv, 'model_server_workers', PropertyMock(return_value=2))
@patch.object(_env.ServingEnv, 'model_server_timeout', PropertyMock(return_value=100))
@patch.object(_env.ServingEnv, 'use_nginx', PropertyMock(return_value=False))
@patch('sagemaker_containers._env.num_gpus', lambda: 0)
@patch('os.wait', lambda: (-1, 0))
@patch('subprocess.Popen')
def test_start_no_nginx(popen):
    popen.return_value.pid = -1
    calls = [call(
        ['gunicorn',
         '--timeout', '100',
         '-k', 'gevent',
         '-b', '0.0.0.0:8080',
         '--worker-connections', '2000',
         '-w', '2',
         '--log-level', 'info',
         'my_module'])]

    _server.start('my_module')
    popen.assert_has_calls(calls)


@patch.object(_env.ServingEnv, 'model_server_workers', PropertyMock(return_value=2))
@patch.object(_env.ServingEnv, 'model_server_timeout', PropertyMock(return_value=100))
@patch.object(_env.ServingEnv, 'use_nginx', PropertyMock(return_value=True))
@patch('sagemaker_containers._env.num_gpus', lambda: 0)
@patch('sagemaker_containers._server.nginx_config_file', '/tmp/nginx.conf')
@patch('sagemaker_containers._server.nginx_config_template_file', '/tmp/nginx.conf.template')
@patch('sagemaker_containers._files.read_file', lambda x: 'random_string')
@patch('sagemaker_containers._files.write_file', Mock())
@patch('os.wait', lambda: (-1, 0))
@patch('subprocess.Popen')
def test_start_with_nginx(popen):
    popen.return_value.pid = -1
    calls = [
        call(['nginx', '-c', '/tmp/nginx.conf']),
        call(['gunicorn',
              '--timeout', '100',
              '-k', 'gevent',
              '-b', 'unix:/tmp/gunicorn.sock',
              '--worker-connections', '2000',
              '-w', '2',
              '--log-level', 'info',
              'my_module'])
    ]
    _server.start('my_module')
    popen.assert_has_calls(calls)


def test_next_safe_port_first():
    safe_port = _server.next_safe_port(SAFE_PORT_RANGE)
    assert safe_port == FIRST_PORT


def test_next_safe_port_after():
    safe_port = _server.next_safe_port(SAFE_PORT_RANGE, FIRST_PORT)
    next_safe_port = str(int(FIRST_PORT) + 1)

    assert safe_port == next_safe_port


def test_next_safe_port_greater_than_range_exception():
    current_port = str(int(LAST_PORT) + 1)

    with pytest.raises(ValueError):
        _server.next_safe_port(SAFE_PORT_RANGE, current_port)


def test_next_safe_port_less_than_range_exception():
    current_port = str(int(FIRST_PORT) - 100)

    with pytest.raises(ValueError):
        _server.next_safe_port(SAFE_PORT_RANGE, current_port)
