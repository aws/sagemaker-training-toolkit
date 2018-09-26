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
import json
import signal

from mock import patch
import pytest

from container_support.serving import (Server,
                                       Transformer,
                                       UnsupportedContentTypeError,
                                       UnsupportedAcceptTypeError,
                                       UnsupportedInputShapeError)

JSON_CONTENT_TYPE = "application/json"
JSON_DATA = json.dumps([1, 2])


@pytest.fixture(scope="module")
def app():
    s = Server("test", Transformer())
    s.app.testing = True
    a = s.app.test_client()
    yield a


def test_app_healthcheck(app):
    result = app.get("/ping")
    assert 200 == result.status_code
    assert 0 == len(result.data)


def test_app_invoke(app):
    data = '{"k1":"v1", "k2":"v2"}'
    result = app.post("/invocations",
                      data=data,
                      headers={"ContentType": JSON_CONTENT_TYPE,
                               "Accept": JSON_CONTENT_TYPE})

    assert 200 == result.status_code
    assert data == result.data.decode('utf-8')
    assert "application/json" == result.content_type


@pytest.mark.parametrize("header", ["ContentType", "Content-Type"])
def test_app_invoke_error(header):
    def f(*args):
        raise Exception("error")

    error_server = Server("error", Transformer(f))
    error_server.app.testing = True
    error_app = error_server.app.test_client()

    result = error_app.post("/invocations",
                            data='{}',
                            headers={header: JSON_CONTENT_TYPE})

    assert 500 == result.status_code
    assert 0 == len(result.data)


@pytest.mark.parametrize("header", ["ContentType", "Content-Type"])
def test_invoke_default_container_handlers(header, app):
    with patch('json.dumps') as patched:
        patched.return_value = JSON_DATA
        headers = {header: JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE}
        result = app.post("/invocations", data=JSON_DATA, headers=headers)

    assert 200 == result.status_code
    assert JSON_DATA == result.data.decode('utf-8')
    assert JSON_CONTENT_TYPE == result.content_type


@pytest.mark.parametrize("header", ["ContentType", "Content-Type"])
def test_transform_with_unsupported_content_type(header):
    transformer = Transformer(_unsupported_content_type_transform)
    new_server = Server("testing_input", transformer)
    new_server.app.testing = True
    new_app = new_server.app.test_client()

    result = new_app.post("/invocations", data='{}', headers={header: "application/some"})

    assert 415 == result.status_code
    assert "some" in str(result.data)


@pytest.mark.parametrize("header", ["ContentType", "Content-Type"])
def test_output_with_unsupported_content_type(header):
    transformer = Transformer(_unsupported_accept_transform)
    new_server = Server("testing_output", transformer)
    new_server.app.testing = True
    new_app = new_server.app.test_client()

    headers = {header: JSON_CONTENT_TYPE, "Accept": "application/other"}
    result = new_app.post("/invocations", data=JSON_DATA, headers=headers)

    assert 406 == result.status_code
    assert "other" in str(result.data)


def test_transform_with_unsupported_input_shape_type():
    transformer = Transformer(_unsupported_input_shape_transform)
    new_server = Server("testing_input_shape", transformer)
    new_server.app.testing = True
    new_app = new_server.app.test_client()

    result = new_app.post("/invocations", data='{}')

    assert 412 == result.status_code


def test_unsupported_content_type_exception():
    with pytest.raises(UnsupportedContentTypeError) as u:
        raise UnsupportedContentTypeError('some')

    assert "some" in str(u.value.message)


def test_unsupported_accept_type_exception():
    with pytest.raises(UnsupportedAcceptTypeError) as u:
        raise UnsupportedAcceptTypeError('some')

    assert "some" in str(u.value.message)


@patch('os.path.exists')
@patch('container_support.HostingEnvironment')
def test_download_user_module_already_exists(env, os_path_exists):
    env.code_dir.return_value = "code_dir"
    env.user_script_name.return_value = "script"
    os_path_exists.return_value = True
    Server._download_user_module_internal(env)
    env.download_user_module.assert_not_called()


@patch('os.path')
@patch('container_support.HostingEnvironment')
def test_download_user_module(env, os_path):
    env.code_dir.return_value = "code_dir"
    env.user_script_name.return_value = "script"
    os_path.exists.return_value = False
    Server._download_user_module_internal(env)
    env.download_user_module.assert_called_with()


@patch('shutil.rmtree')
@patch('os.path')
@patch('container_support.HostingEnvironment')
def test_download_user_module_exception(env, os_path, rmtree):
    env.code_dir.return_value = "code_dir"
    env.user_script_name.return_value = "script"
    os_path.exists.return_value = False
    env.download_user_module.side_effect = Exception("error")

    with pytest.raises(Exception):
        Server._download_user_module_internal(env)

    rmtree.assert_called_with(env.code_dir)


@patch('os.kill')
@patch('sys.exit')
def test_sigterm_hander(exit, kill):
    Server._sigterm_handler(1, 2)

    kill.assert_called_with(1, signal.SIGQUIT)
    kill.assert_called_with(2, signal.SIGTERM)
    exit.assert_called_with(0)


@patch('container_support.Server._download_user_module')
@patch('subprocess.check_call')
@patch('subprocess.Popen')
@patch('signal.signal')
@patch('container_support.HostingEnvironment')
@patch('container_support.ContainerEnvironment')
@patch.dict(os.environ, {'SAGEMAKER_CONTAINER_LOG_LEVEL': '20', 'SAGEMAKER_REGION': 'us-west-2'})
def test_server_start(ContainerEnvironment, HostingEnvironment, signal, Popen, check_call, _download_user_module):
    with pytest.raises(OSError):
        Server.start()

    env = HostingEnvironment()
    env.start_metrics_if_enabled.assert_called()
    env.pip_install_requirements.assert_called()
    ContainerEnvironment.load_framework().load_dependencies.assert_called()


@patch('os.kill')
@patch('sys.exit')
def test_sigterm_hander(exit, kill):
    Server._sigterm_handler(0, 2)

    kill.assert_called_with(2, signal.SIGTERM)
    exit.assert_called_with(0)


def test_unsupported_input_shape_exception():
    assert "2" in UnsupportedInputShapeError(2).message


def _unsupported_content_type_transform(data, input_content_type, output_content_type):
    raise UnsupportedContentTypeError(input_content_type)


def _unsupported_accept_transform(data, input_content_type, output_content_type):
    raise UnsupportedAcceptTypeError(output_content_type)


def _unsupported_input_shape_transform(data, input_content_type, output_content_type):
    raise UnsupportedInputShapeError(3)

