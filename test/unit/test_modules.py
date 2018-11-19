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
from __future__ import absolute_import

import contextlib
import os
import sys

from mock import call, patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, _files, _modules, _params

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('boto3.resource', autospec=True)
@pytest.mark.parametrize('url,bucket_name,key,dst',
                         [('S3://my-bucket/path/to/my-file', 'my-bucket', 'path/to/my-file', '/tmp/my-file'),
                          ('s3://my-bucket/my-file', 'my-bucket', 'my-file', '/tmp/my-file')])
def test_s3_download(resource, url, bucket_name, key, dst):
    region = 'us-west-2'
    os.environ[_params.REGION_NAME_ENV] = region

    _files.s3_download(url, dst)

    chain = call('s3', region_name=region).Bucket(bucket_name).download_file(key, dst)
    assert resource.mock_calls == chain.call_list()


def test_s3_download_wrong_scheme():
    with pytest.raises(ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"):
        _files.s3_download('c://my-bucket/my-file', '/tmp/file')


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install(check_error):
    path = 'c://sagemaker-pytorch-container'
    _modules.install(path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError, cwd=path)

    with patch('os.path.exists', return_value=True):
        _modules.install(path)

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'], _errors.InstallModuleError, cwd=path)


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_fails(check_error):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        _modules.install('git://aws/container-support')


@patch('sys.executable', None)
def test_install_no_python_executable():
    with pytest.raises(RuntimeError) as e:
        _modules.install('git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@contextlib.contextmanager
def patch_tmpdir():
    yield '/tmp'


@patch('importlib.import_module')
def test_exists(import_module):
    assert _modules.exists('my_module')

    import_module.side_effect = ImportError()

    assert not _modules.exists('my_module')


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        _modules.run('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


@patch('sagemaker_containers._process.python_executable')
@patch('sagemaker_containers._process.check_error')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run(log_script_invocation,  check_error, executable):
    _modules.run('pytest', ['--version'])

    expected_cmd = [executable(), '-m', 'pytest', '--version']
    log_script_invocation.assert_called_with(expected_cmd, {})
    check_error.assert_called_with(expected_cmd, _errors.ExecuteUserScriptError)


@patch('sagemaker_containers._process.python_executable')
@patch('sagemaker_containers._process.create')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_no_wait(log_script_invocation,  create, executable):
    _modules.run('pytest', ['--version'], {'PYPATH': '/opt/ml/code'}, wait=False)

    expected_cmd = [executable(), '-m', 'pytest', '--version']
    log_script_invocation.assert_called_with(expected_cmd, {'PYPATH': '/opt/ml/code'})
    create.assert_called_with(expected_cmd, _errors.ExecuteUserScriptError)


@pytest.mark.parametrize('wait, cache', [[True, False], [True, False]])
@patch('sagemaker_containers._modules.run')
@patch('sagemaker_containers._modules.install')
@patch('sagemaker_containers._env.write_env_vars')
@patch('sagemaker_containers._files.download_and_extract')
def test_run_module_wait(download_and_extract, write_env_vars, install, run, wait, cache):
    with pytest.warns(DeprecationWarning):
        _modules.run_module(uri='s3://url', args=['42'], wait=wait, cache=cache)
        module_name = 'default_user_module_name'

        download_and_extract.assert_called_with('s3://url', module_name, _env.code_dir)
        write_env_vars.assert_called_with({})
        install.assert_called_with(_env.code_dir)

        run.assert_called_with('default_user_module_name', ['42'], {}, True)


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers._modules.install')
@patch('importlib.import_module')
@patch('six.moves.reload_module')
def test_import_module(reload, import_module, install, download_and_extract):

    _modules.import_module('s3://bucket/my-module')

    download_and_extract.assert_called_with('s3://bucket/my-module', 'default_user_module_name', _env.code_dir)
    install.assert_called_with(_env.code_dir)
    reload.assert_called_with(import_module(_modules.DEFAULT_MODULE_NAME))
