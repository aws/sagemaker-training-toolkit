# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from io import StringIO

import os
import subprocess
import sys
import unittest
from mock import ANY, MagicMock, patch
import pytest
import nest_asyncio
from sagemaker_training import environment, errors, process


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@pytest.fixture
def entry_point_type_module():
    with patch("os.listdir", lambda x: ("setup.py",)):
        yield


@pytest.fixture(autouse=True)
def entry_point_type_script():
    with patch("os.listdir", lambda x: ()):
        yield


@pytest.fixture()
def has_requirements():
    with patch("os.path.exists", lambda x: x.endswith("requirements.txt")):
        yield


def test_python_executable_exception():
    with patch("sys.executable", None):
        with pytest.raises(RuntimeError):
            process.python_executable()


@patch("subprocess.Popen", MagicMock(side_effect=ValueError("FAIL")))
def test_create_error():
    with pytest.raises(errors.ExecuteUserScriptError):
        process.create(["run"], errors.ExecuteUserScriptError, 1)


@patch("subprocess.Popen")
def test_check_error(popen):
    test_process = MagicMock(wait=MagicMock(return_value=0))
    popen.return_value = test_process

    assert test_process == process.check_error(
        ["run"], errors.ExecuteUserScriptError, 1
    )


@patch("subprocess.Popen")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_bash(log, popen, entry_point_type_script):
    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("launcher.sh", ["--lr", "1 3"], {}, 1).run()

    cmd = ["/bin/sh", "-c", "./launcher.sh --lr '1 3'"]
    popen.assert_called_with(cmd, cwd=environment.code_dir, env=os.environ, stderr=None)
    log.assert_called_with(cmd, {})


@patch("subprocess.Popen")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_python(log, popen, entry_point_type_script):
    popen().communicate.return_value = (None, b"this is stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("launcher.py", ["--lr", "13"], {}, 1).run(
            capture_error=False
        )

    cmd = [sys.executable, "launcher.py", "--lr", "13"]
    popen.assert_called_with(
        cmd, cwd=environment.code_dir, env=os.environ, stderr=subprocess.PIPE
    )
    log.assert_called_with(cmd, {})


@patch("subprocess.Popen")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_module(log, popen, entry_point_type_module):
    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("module.py", ["--lr", "13"], {}, 1).run()

    cmd = [sys.executable, "-m", "module", "--lr", "13"]
    popen.assert_called_with(cmd, cwd=environment.code_dir, env=os.environ, stderr=None)
    log.assert_called_with(cmd, {})


@patch("sagemaker_training.environment.Environment", lambda: {})
def test_run_error():
    with pytest.raises(errors.ExecuteUserScriptError) as e:
        process.ProcessRunner("wrong module", [], {}, 1).run()

    message = str(e.value)
    assert "ExecuteUserScriptError:" in message


@pytest.mark.asyncio
async def test_watch(event_loop):
    expected_outcome = "[1, mpirank:2, algo-2]<stdout>:This is stdout"

    def write_to_stream():
        print("[1,2]<stdout>:This is stdout")

    with patch("sys.stdout", new=StringIO()) as mock_stdout:
        write_to_stream()
        output = await process.watch(mock_stdout.getvalue(), 2)
        assert output.return_value == expected_outcome
