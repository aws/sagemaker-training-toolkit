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

import asyncio
import os
import sys

from mock import ANY, MagicMock, patch
import pytest

from sagemaker_training import environment, errors, process


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class AsyncMock1(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock1, self).__call__(*args, **kwargs)


class AsyncMock2(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock1, self).__call__(*args, **kwargs)


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


@patch("subprocess.Popen")
def test_check_error(popen):
    test_process = MagicMock(wait=MagicMock(return_value=0))
    popen.return_value = test_process

    assert test_process == process.check_error(["run"], errors.ExecuteUserScriptError, 1)


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
async def test_watch(event_loop, capsys):
    num_processes_per_host = 8
    expected_stream = "[1,mpirank:10,algo-2]<stdout>:This is stdout\n"
    expected_stream += "[1,mpirank:10,algo-2]<stderr>:This is stderr\n"
    expected_stream += (
        "[1,mpirank:0,algo-1]<stderr>:FileNotFoundError: [Errno 2] No such file or directory\n"
    )
    expected_errmsg = ":FileNotFoundError: [Errno 2] No such file or directory\n"

    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,10]<stdout>:This is stdout\n")
    stream.feed_data(b"[1,10]<stderr>:This is stderr\n")
    stream.feed_data(b"[1,0]<stderr>:FileNotFoundError: [Errno 2] No such file or directory")
    stream.feed_eof()

    output = await process.watch(stream, num_processes_per_host)
    captured_stream = capsys.readouterr()
    assert captured_stream.out == expected_stream
    assert output == expected_errmsg


@patch("asyncio.run", AsyncMock(side_effect=ValueError("FAIL")))
def test_create_error():
    with pytest.raises(errors.ExecuteUserScriptError):
        process.create(["run"], errors.ExecuteUserScriptError, 1)


@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_shell")
@pytest.mark.asyncio
async def test_run_async(async_shell, async_gather):
    processes_per_host = 2
    async_gather.return_value = "test"
    cmd = ["python3", "launcher.py", "--lr", "13"]
    rc, output, proc = await process.run_async(
        cmd,
        processes_per_host,
        env=os.environ,
        stderr=asyncio.subprocess.PIPE,
        cwd=environment.code_dir,
    )
    async_shell.assert_called_once()
    async_gather.assert_called_once()
    async_shell.assert_called_with(
        " ".join(cmd),
        stdout=asyncio.subprocess.PIPE,
        env=ANY,
        cwd=ANY,
        stderr=asyncio.subprocess.PIPE,
    )
    assert output == "test"


@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_shell")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_python(log, async_shell, async_gather, entry_point_type_script, event_loop):

    async_gather.return_value = ("stdout", "stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        rc, output, proc = process.ProcessRunner("launcher.py", ["--lr", "13"], {}, 2).run(
            capture_error=True
        )
        assert output == "stderr"

    cmd = [sys.executable, "launcher.py", "--lr", "13"]
    async_shell.assert_called_once()
    async_gather.assert_called_once()
    async_shell.assert_called_with(
        " ".join(cmd),
        cwd=environment.code_dir,
        env=os.environ,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    log.assert_called_with(cmd, {})
