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
import json
import logging
import os
import sys

from mock import ANY, MagicMock, patch
import pytest
import six

from sagemaker_training import environment, errors, process


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class AsyncMock1(MagicMock):
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

    assert test_process == process.check_error(
        ["run"], errors.ExecuteUserScriptError, 1, capture_error=False
    )


@patch("subprocess.Popen")
def test_check_error_smtrainingcompilerconfigurationerror(popen):
    test_process = MagicMock(wait=MagicMock(return_value=0))
    popen.return_value = test_process

    assert test_process == process.check_error(
        ["run"], errors.SMTrainingCompilerConfigurationError, 1, capture_error=False
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
def test_run_module(log, popen, entry_point_type_module):
    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("module.py", ["--lr", "13"], {}, 1).run()

    cmd = [sys.executable, "-m", "module", "--lr", "13"]
    popen.assert_called_with(cmd, cwd=environment.code_dir, env=os.environ, stderr=None)
    log.assert_called_with(cmd, {})


@patch("sagemaker_training.environment.Environment", lambda: {})
def test_run_error():
    with pytest.raises(errors.ExecuteUserScriptError) as e:
        process.ProcessRunner("wrong_module.sh", [], {}, 1).run()

    message = str(e.value)
    assert "ExecuteUserScriptError:" in message


@pytest.mark.parametrize(
    "entry_point",
    [
        "train.sh; touch /tmp/pwned",
        'train.sh"; touch /tmp/pwned; echo "',
        "train.sh`touch /tmp/pwned`",
        "train.sh$(touch /tmp/pwned)",
        "train.sh | touch /tmp/pwned",
        "train.sh && touch /tmp/pwned",
        "train.sh\ntouch /tmp/pwned",
        "train.sh $HOME",
        "wrong module",
        "../../../bin/sh",
        "./train.sh",
        "/bin/sh",
        "",
    ],
)
@patch("sagemaker_training.environment.Environment", lambda: {})
def test_create_command_rejects_unsafe_entry_point(entry_point):
    """An entry point that a shell would reinterpret must be rejected, not quoted."""
    runner = process.ProcessRunner(entry_point, [], {}, 1)
    with pytest.raises(errors.ClientError):
        runner._create_command()


@pytest.mark.parametrize(
    "entry_point", ["train.sh", "run_training.sh", "my-script_v2.sh", "bin/train.sh", "a.b+c.sh"]
)
@patch("sagemaker_training.environment.Environment", lambda: {})
def test_create_command_accepts_safe_entry_point(entry_point):
    """Ordinary shell entry points keep working and are not mangled."""
    runner = process.ProcessRunner(entry_point, ["--epochs", "10"], {}, 1)

    assert runner._create_command() == [
        "/bin/sh",
        "-c",
        "./%s --epochs 10" % entry_point,
    ]


@pytest.mark.asyncio
async def test_watch(event_loop, capsys):
    num_processes_per_host = 8
    expected_stream = "[1,mpirank:10,algo-2]<stdout>:This is stdout\n"
    expected_stream += "[1,mpirank:10,algo-2]<stderr>:This is stderr\n"
    expected_stream += (
        "[1,mpirank:0,algo-1]<stderr>:FileNotFoundError: [Errno 2] No such file or directory\n"
    )
    expected_errmsg = "FileNotFoundError: [Errno 2] No such file or directory\n"

    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,10]<stdout>:This is stdout\n")
    stream.feed_data(b"[1,10]<stderr>:This is stderr\n")
    stream.feed_data(b"[1,0]<stderr>:FileNotFoundError: [Errno 2] No such file or directory")
    stream.feed_eof()

    output = await process.watch(stream, num_processes_per_host)
    captured_stream = capsys.readouterr()
    assert captured_stream.out == expected_stream
    assert output == expected_errmsg


@pytest.mark.asyncio
async def test_watch_custom_error(event_loop, capsys):
    num_processes_per_host = 8
    expected_stream = "[1,mpirank:10,algo-2]<stdout>:This is stdout\n"
    expected_stream += "[1,mpirank:10,algo-2]<stderr>:This is stderr\n"
    expected_stream += "[1,mpirank:0,algo-1]<stderr>:SMDDPNCCLError: unhandled cuda error\n"
    expected_errmsg = "SMDDPNCCLError: unhandled cuda error\n"

    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,10]<stdout>:This is stdout\n")
    stream.feed_data(b"[1,10]<stderr>:This is stderr\n")
    stream.feed_data(b"[1,0]<stderr>:SMDDPNCCLError: unhandled cuda error")
    stream.feed_eof()

    error_classes = ["SMDDPNCCLError"]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    captured_stream = capsys.readouterr()
    assert captured_stream.out == expected_stream
    assert output == expected_errmsg

    # test errors piped in stdout
    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,0]<stdout>:SMDDPNCCLError: unhandled cuda error")
    stream.feed_eof()

    error_classes = ["SMDDPNCCLError"]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    assert output == expected_errmsg

    # test single item
    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,0]<stdout>:SMDDPNCCLError: unhandled cuda error")
    stream.feed_eof()
    error_classes = "SMDDPNCCLError"
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    assert output == expected_errmsg

    # test internal error
    expected_errmsg = "ImportModuleError: module does not exist\n"
    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,0]<stderr>:ImportModuleError: module does not exist")
    stream.feed_eof()
    error_classes = [errors.ImportModuleError]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    assert output == expected_errmsg


@pytest.mark.asyncio
async def test_watch_debugger_error(event_loop, capsys):
    num_processes_per_host = 8
    expected_stream = "[1,mpirank:10,algo-2]<stdout>:This is stdout\n"
    expected_stream += "[1,mpirank:10,algo-2]<stderr>:This is stderr\n"
    expected_stream += "[1,mpirank:0,algo-1]<stderr>:SMDebugError: debugger exception raised\n"
    expected_errmsg = "SMDebugError: debugger exception raised\n"

    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,10]<stdout>:This is stdout\n")
    stream.feed_data(b"[1,10]<stderr>:This is stderr\n")
    stream.feed_data(b"[1,0]<stderr>:SMDebugError: debugger exception raised")
    stream.feed_eof()

    error_classes = ["SMDebugError"]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    captured_stream = capsys.readouterr()
    assert captured_stream.out == expected_stream
    assert output == expected_errmsg

    # test errors piped in stdout
    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,0]<stdout>:SMDebugError: debugger exception raised")
    stream.feed_eof()

    error_classes = ["SMDebugError"]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    assert output == expected_errmsg

    # test single item
    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,0]<stdout>:SMDebugError: debugger exception raised")
    stream.feed_eof()
    error_classes = "SMDebugError"
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    assert output == expected_errmsg


def test_get_tensorflow_exception_error(event_loop, caplog):
    with caplog.at_level(logging.INFO):
        process.get_tensorflow_exception_classes()
        expected_errmsg = "Exceptions not imported for SageMaker TF as Tensorflow is not installed."
        assert expected_errmsg in caplog.text


@pytest.mark.asyncio
async def test_watch_special_characters(event_loop, capsys):
    num_processes_per_host = 8
    expected_stream = "[1,mpirank:10,algo-2]<stdout>:This is stdout with character �\n"
    expected_stream += "[1,mpirank:10,algo-2]<stderr>:This is stderr with character �\n"
    expected_stream += (
        "[1,mpirank:0,algo-1]<stderr>:ExecuteUserScriptError: [Errno 2] Invalid character �\n"
    )
    expected_errmsg = "ExecuteUserScriptError: [Errno 2] Invalid character �\n"

    stream = asyncio.StreamReader()
    stream.feed_data(b"[1,10]<stdout>:This is stdout with character \x83\n")
    stream.feed_data(b"[1,10]<stderr>:This is stderr with character \x83\n")
    stream.feed_data(b"[1,0]<stderr>:ExecuteUserScriptError: [Errno 2] Invalid character \x83")
    stream.feed_eof()

    error_classes = ["ExecuteUserScriptError"]
    output = await process.watch(stream, num_processes_per_host, error_classes=error_classes)
    captured_stream = capsys.readouterr()
    assert captured_stream.out == expected_stream
    assert output == expected_errmsg


@patch("asyncio.run", AsyncMock(side_effect=ValueError("FAIL")))
def test_create_error():
    with pytest.raises(errors.ExecuteUserScriptError):
        process.create(["run"], errors.ExecuteUserScriptError, 1)


@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
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
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        env=ANY,
        cwd=ANY,
        stderr=asyncio.subprocess.PIPE,
    )
    assert output == "test"


@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
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
        *cmd,
        cwd=environment.code_dir,
        env=os.environ,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    log.assert_called_with(cmd, {})


# Argument values containing characters that are significant to a shell. Each must reach
# the user script as a single, verbatim argument.
_SPECIAL_CHARACTER_VALUES = [
    "1; touch /tmp/marker",
    "1 && touch /tmp/marker",
    "1 | touch /tmp/marker",
    "$(touch /tmp/marker)",
    "`touch /tmp/marker`",
    '1"; touch /tmp/marker; echo "',
    "1'; touch /tmp/marker; echo '",
    "1\ntouch /tmp/marker",
    "$HOME",
    "a b",
    '{"key": "value with spaces"}',
]


@pytest.mark.parametrize("value", _SPECIAL_CHARACTER_VALUES)
@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_python_special_character_hyperparameter_capture_error(
    log, async_exec, async_gather, entry_point_type_script, event_loop, value
):
    """A Python entry point's hyperparameter is passed as one argv element on the
    capture_error (asyncio) path."""
    async_gather.return_value = ("stdout", "stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("train.py", ["--lr", value], {}, 1).run(capture_error=True)

    async_exec.assert_called_once()
    positional = async_exec.call_args[0]
    assert positional == (sys.executable, "train.py", "--lr", value)


@pytest.mark.parametrize("value", _SPECIAL_CHARACTER_VALUES)
@patch("subprocess.Popen")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_python_special_character_hyperparameter_popen(
    log, popen, entry_point_type_script, value
):
    """Same guarantee on the non-capture (Popen) path."""
    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("train.py", ["--lr", value], {}, 1).run(capture_error=False)

    popen.assert_called_once()
    assert popen.call_args[0][0] == [sys.executable, "train.py", "--lr", value]


@pytest.mark.parametrize("value", _SPECIAL_CHARACTER_VALUES)
@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_module_special_character_hyperparameter(
    log, async_exec, async_gather, entry_point_type_module, event_loop, value
):
    """Python package entry points get the same treatment."""
    async_gather.return_value = ("stdout", "stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("module.py", ["--lr", value], {}, 1).run(capture_error=True)

    assert async_exec.call_args[0] == (sys.executable, "-m", "module", "--lr", value)


@pytest.mark.parametrize(
    "entry_point",
    [
        "train.py; touch /tmp/marker; echo .py",
        "train.py && touch /tmp/marker #.py",
        "$(touch /tmp/marker).py",
        "`touch /tmp/marker`.py",
        'train.py"; touch /tmp/marker; echo ".py',
        "train.py | touch /tmp/marker; echo .py",
    ],
)
@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_python_special_character_entry_point_is_single_argv(
    log, async_exec, async_gather, entry_point_type_script, event_loop, entry_point
):
    """A Python entry point name containing special characters is handed to the
    interpreter as a single file name argument."""
    async_gather.return_value = ("stdout", "stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner(entry_point, [], {}, 1).run(capture_error=True)

    assert async_exec.call_args[0] == (sys.executable, entry_point)


@pytest.mark.parametrize("value", _SPECIAL_CHARACTER_VALUES)
@patch("asyncio.gather", new_callable=AsyncMock1)
@patch("asyncio.create_subprocess_exec")
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_run_bash_special_character_hyperparameter_capture_error(
    log, async_exec, async_gather, entry_point_type_script, event_loop, value
):
    """The shell entry point keeps exactly one shell level: argv is ['/bin/sh', '-c', s]
    and s carries the hyperparameter shlex-quoted for that single parse."""
    async_gather.return_value = ("stdout", "stderr")

    with pytest.raises(errors.ExecuteUserScriptError):
        process.ProcessRunner("train.sh", ["--lr", value], {}, 1).run(capture_error=True)

    positional = async_exec.call_args[0]
    assert positional[:2] == ("/bin/sh", "-c")
    assert len(positional) == 3
    assert positional[2] == "./train.sh --lr %s" % six.moves.shlex_quote(value)
    assert not positional[2].startswith('"')


def _write_script(directory, name, body, executable=False):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(body)
    if executable:
        os.chmod(path, 0o755)
    return path


# Real-process check that argument values and entry point names are passed through
# verbatim. Each case would create a marker file if the value were interpreted rather
# than passed as data.
_END_TO_END_SPECIAL_CHARACTER_CASES = [
    ("train.py", ["--lr", "1; touch {marker}"]),
    ("train.py", ["--lr", "1 && touch {marker}"]),
    ("train.py", ["--out", "$(touch {marker})"]),
    ("train.py", ["--out", "`touch {marker}`"]),
    ("train.py", ["--out", '1"; touch {marker}; echo "']),
    ("train.py", ["--out", "1'; touch {marker}; echo '"]),
    ("train.py; touch {marker}; echo .py", []),
    ("train.py && touch {marker} #.py", []),
    ("train.sh", ["--lr", "1; touch {marker}"]),
    ("train.sh", ["--out", "$(touch {marker})"]),
    ("train.sh", ["--out", '1"; touch {marker}; echo "']),
]


@pytest.mark.parametrize("capture_error", [True, False])
@pytest.mark.parametrize("entry_point, args", _END_TO_END_SPECIAL_CHARACTER_CASES)
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_special_character_values_are_passed_as_data(
    log, tmpdir, entry_point_type_script, entry_point, args, capture_error
):
    code_dir = str(tmpdir.mkdir("code"))
    marker = os.path.join(str(tmpdir), "marker")
    entry_point = entry_point.format(marker=marker)
    args = [arg.format(marker=marker) for arg in args]

    # Scripts exit non-zero so the runner raises and we do not depend on success.
    _write_script(code_dir, "train.py", "import sys\nprint(sys.argv[1:])\nsys.exit(3)\n")
    _write_script(code_dir, "train.sh", '#!/bin/sh\necho "$@"\nexit 3\n', executable=True)

    with patch.object(environment, "code_dir", code_dir):
        with pytest.raises(errors.ExecuteUserScriptError):
            process.ProcessRunner(entry_point, args, {}, 1).run(capture_error=capture_error)

    assert not os.path.exists(
        marker
    ), "value in %r / %r was interpreted rather than passed as data" % (
        entry_point,
        args,
    )


@pytest.mark.parametrize("capture_error", [True, False])
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_hyperparameters_reach_python_entry_point_verbatim(
    log, tmpdir, entry_point_type_script, capture_error
):
    """Values with spaces, quotes and JSON survive end to end unchanged."""
    code_dir = str(tmpdir.mkdir("code"))
    received = os.path.join(str(tmpdir), "argv.json")
    _write_script(
        code_dir,
        "train.py",
        "import json, sys\n"
        "with open(sys.argv[1], 'w') as f:\n"
        "    json.dump(sys.argv[2:], f)\n",
    )
    args = [received, "--name", "a b", "--json", '{"k": "v w"}', "--q", "it's", "--s", "$HOME;x"]

    with patch.object(environment, "code_dir", code_dir):
        process.ProcessRunner("train.py", args, {}, 1).run(capture_error=capture_error)

    with open(received) as handle:
        assert json.load(handle) == args[1:]


@pytest.mark.parametrize("capture_error", [True, False])
@patch("sagemaker_training.logging_config.log_script_invocation")
def test_hyperparameters_reach_shell_entry_point_verbatim(
    log, tmpdir, entry_point_type_script, capture_error
):
    """The shell entry point still receives each argument intact through its one shell level."""
    code_dir = str(tmpdir.mkdir("code"))
    received = os.path.join(str(tmpdir), "argv.txt")
    _write_script(
        code_dir,
        "train.sh",
        '#!/bin/sh\nout="$1"; shift\nfor a in "$@"; do printf "%s\\n" "$a" >> "$out"; done\n',
        executable=True,
    )
    args = [received, "--name", "a b", "--json", '{"k": "v w"}', "--q", "it's", "--s", "$HOME;x"]

    with patch.object(environment, "code_dir", code_dir):
        process.ProcessRunner("train.sh", args, {}, 1).run(capture_error=capture_error)

    with open(received) as handle:
        assert handle.read().splitlines() == args[1:]
