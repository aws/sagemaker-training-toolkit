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
import subprocess
import sys

from mock import MagicMock, patch
import pytest

from sagemaker_containers import _env, _errors, _process


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


def test_python_executable_exception():
    with patch('sys.executable', None):
        with pytest.raises(RuntimeError):
            _process.python_executable()


@patch('subprocess.Popen', MagicMock(side_effect=ValueError('FAIL')))
def test_create_error():
    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.create(['run'], _errors.ExecuteUserScriptError)


@patch('subprocess.Popen')
def test_check_error(popen):
    process = MagicMock(wait=MagicMock(return_value=0))
    popen.return_value = process

    assert process == _process.check_error(['run'], _errors.ExecuteUserScriptError)


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_bash(log, popen, entry_point_type_script):
    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.ProcessRunner('launcher.sh', ['--lr', '13'], {}).run()

    cmd = ['/bin/sh', '-c', './launcher.sh --lr 13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ, stderr=None)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_python(log, popen, entry_point_type_script):
    popen().communicate.return_value = (None, 0)

    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.ProcessRunner('launcher.py', ['--lr', '13'], {}).run(capture_error=True)

    cmd = [sys.executable, 'launcher.py', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ,
                             stderr=subprocess.PIPE)
    log.assert_called_with(cmd, {})


@patch('subprocess.Popen')
@patch('sagemaker_containers._logging.log_script_invocation')
def test_run_module(log, popen, entry_point_type_module):
    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.ProcessRunner('module.py', ['--lr', '13'], {}).run()

    cmd = [sys.executable, '-m', 'module', '--lr', '13']
    popen.assert_called_with(cmd, cwd=_env.code_dir, env=os.environ,
                             stderr=None)
    log.assert_called_with(cmd, {})


@patch('sagemaker_containers.training_env', lambda: {})
def test_run_error():
    with pytest.raises(_errors.ExecuteUserScriptError) as e:
        _process.ProcessRunner('wrong module', [], {}).run()

    message = str(e.value)
    assert 'ExecuteUserScriptError:' in message
