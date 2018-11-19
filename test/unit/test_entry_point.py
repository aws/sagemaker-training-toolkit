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

import os
import sys

from mock import patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, entry_point

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@pytest.fixture
def entry_point_type_module():
    with patch('os.listdir', lambda x: ('setup.py',)):
        yield


@pytest.fixture(autouse=True)
def entry_point_type_script():
    with patch('os.listdir', lambda x: ()):
        yield


@pytest.fixture()
def has_requirements():
    with patch('os.path.exists', lambda x: x.endswith('requirements.txt')):
        yield


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_module(check_error, entry_point_type_module):
    path = 'c://sagemaker-pytorch-container'
    entry_point.install('python_module.py', path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError, cwd=path)

    with patch('os.path.exists', return_value=True):
        entry_point.install('python_module.py', path)

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'], _errors.InstallModuleError, cwd=path)


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_script(check_error, entry_point_type_module, has_requirements):
    path = 'c://sagemaker-pytorch-container'
    entry_point.install('train.py', path)

    with patch('os.path.exists', return_value=True):
        entry_point.install(path, 'python_module.py')


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_fails(check_error, entry_point_type_module):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        entry_point.install('git://aws/container-support', 'script')


@patch('sys.executable', None)
def test_install_no_python_executable(has_requirements, entry_point_type_module):
    with pytest.raises(RuntimeError) as e:
        entry_point.install('train.py', 'git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_bash(log, popen, entry_point_type_script):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entry_point._call('launcher.sh', ['--lr', '13'])

    cmd = ['/bin/sh', '-c', './launcher.sh --lr 13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_python(log, popen, entry_point_type_script):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entry_point._call('launcher.py', ['--lr', '13'])

    cmd = [sys.executable, 'launcher.py', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_module(log, popen, entry_point_type_module):
    with pytest.raises(_errors.ExecuteUserScriptError):
        entry_point._call('module.py', ['--lr', '13'])

    cmd = [sys.executable, '-m', 'module', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ)
    log.assert_called_with(cmd, {})


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        entry_point._call('wrong module')

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entry_point._call')
@patch('os.chmod')
def test_run_module_wait(chmod, call, download_and_extract):
    entry_point.run(uri='s3://url', user_entry_point='launcher.sh', args=['42'])

    download_and_extract.assert_called_with('s3://url', 'launcher.sh', _env.code_dir)
    call.assert_called_with('launcher.sh', ['42'], {}, True)
    chmod.assert_called_with(os.path.join(_env.code_dir, 'launcher.sh'), 511)


@patch('sagemaker_containers._files.download_and_extract')
@patch('sagemaker_containers.entry_point._call')
def test_run_module_no_wait(call, download_and_extract, entry_point_type_module):
    with pytest.raises(_errors.InstallModuleError):
        entry_point.run(uri='s3://url', user_entry_point='default_user_module_name', args=['42'], wait=False)

        download_and_extract.assert_called_with('s3://url', 'default_user_module_name', _env.code_dir)
        call.assert_called_with('default_user_module_name', ['42'], {}, False)
