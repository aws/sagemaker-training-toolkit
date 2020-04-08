# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from mock import call, MagicMock, patch, PropertyMock
import pytest
from six import PY2

from sagemaker_training import entry_point, env, errors, process, runner

builtins_open = "__builtin__.open" if PY2 else "builtins.open"


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


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.prepare")
@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_module(check_error, download_and_extract, prepare, entry_point_type_module):
    uri = "s3://foo/bar"
    path = "c://sagemaker-pytorch-container"
    entry_point.install(uri, "python_module.py", path)

    cmd = [sys.executable, "-m", "pip", "install", "."]
    check_error.assert_called_with(cmd, errors.InstallModuleError, capture_error=False, cwd=path)

    with patch("os.path.exists", return_value=True):
        entry_point.install(uri, "python_module.py", path)

        check_error.assert_called_with(
            cmd + ["-r", "requirements.txt"],
            errors.InstallModuleError,
            cwd=path,
            capture_error=False,
        )


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.prepare")
@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_script(
    check_error, download_and_extract, prepare, entry_point_type_module, has_requirements
):
    uri = "s3://foo/bar"
    path = "c://sagemaker-pytorch-container"
    entry_point.install(uri, "train.py", path)

    with patch("os.path.exists", return_value=True):
        entry_point.install(uri, path, "python_module.py")


@patch("sagemaker_training.files.s3_download")
@patch("tarfile.open")
@patch("sagemaker_training.modules.prepare")
@patch("sagemaker_training.modules.install")
@patch("os.chmod")
def test_install_local_directory(chmod, install, prepare, tarfile, s3_download):
    uri = "/opt/ml/input/data/code/sourcedir.tar.gz"
    entry_point.install(uri)

    s3_download.assert_not_called()
    tarfile.assert_called_with(name="/opt/ml/input/data/code/sourcedir.tar.gz", mode="r:gz")


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.prepare")
@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_fails(check_error, download_and_extract, prepare, entry_point_type_module):
    check_error.side_effect = errors.ClientError()
    with pytest.raises(errors.ClientError):
        entry_point.install("s3://foo/bar", "git://aws/container-support", "script")


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.prepare")
@patch("sys.executable", None)
@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_no_python_executable(
    check_error, download_and_extract, prepare, has_requirements, entry_point_type_module
):
    with pytest.raises(RuntimeError) as e:
        entry_point.install("s3://foo/bar", "train.py", "git://aws/container-support")
    assert str(e.value) == "Failed to retrieve the real path for the Python executable binary"


@patch("sagemaker_training.files.download_and_extract")
@patch("os.chmod")
@patch("sagemaker_training.process.check_error", autospec=True)
@patch("socket.gethostbyname")
def test_run_module_wait(gethostbyname, check_error, chmod, download_and_extract):
    runner_mock = MagicMock(spec=process.ProcessRunner)

    entry_point.run(
        uri="s3://url",
        user_entry_point="launcher.sh",
        args=["42"],
        capture_error=True,
        runner_type=runner_mock,
    )

    download_and_extract.assert_called_with("s3://url", env.code_dir)
    runner_mock.run.assert_called_with(True, True)
    chmod.assert_called_with(os.path.join(env.code_dir, "launcher.sh"), 511)


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.install")
@patch.object(
    env.TrainingEnv, "hosts", return_value=["algo-1", "algo-2"], new_callable=PropertyMock
)
@patch("socket.gethostbyname")
def test_run_calls_hostname_resolution(gethostbyname, install, hosts, download_and_extract):
    runner_mock = MagicMock(spec=process.ProcessRunner)
    entry_point.run(
        uri="s3://url", user_entry_point="launcher.py", args=["42"], runner_type=runner_mock
    )

    gethostbyname.assert_called_with("algo-2")
    gethostbyname.assert_any_call("algo-1")


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.install")
@patch.object(
    env.TrainingEnv, "hosts", return_value=["algo-1", "algo-2"], new_callable=PropertyMock
)
@patch("socket.gethostbyname")
def test_run_waits_hostname_resolution(gethostbyname, hosts, install, download_and_extract):

    gethostbyname.side_effect = [ValueError(), ValueError(), True, True]

    runner_mock = MagicMock(spec=process.ProcessRunner)
    entry_point.run(
        uri="s3://url", user_entry_point="launcher.py", args=["42"], runner_type=runner_mock
    )

    gethostbyname.assert_has_calls([call("algo-1"), call("algo-1"), call("algo-1"), call("algo-2")])


@patch("sagemaker_training.files.download_and_extract")
@patch("os.chmod")
@patch("socket.gethostbyname")
def test_run_module_no_wait(gethostbyname, chmod, download_and_extract):
    runner_mock = MagicMock(spec=process.ProcessRunner)

    module_name = "default_user_module_name"
    entry_point.run(
        uri="s3://url",
        user_entry_point=module_name,
        args=["42"],
        wait=False,
        runner_type=runner_mock,
    )

    runner_mock.run.assert_called_with(False, False)


@patch("sys.path")
@patch("sagemaker_training.runner.get")
@patch("sagemaker_training.files.download_and_extract")
@patch("os.chmod")
@patch("socket.gethostbyname")
def test_run_module_with_env_vars(gethostbyname, chmod, download_and_extract, get_runner, sys_path):
    module_name = "default_user_module_name"
    args = ["--some-arg", "42"]
    entry_point.run(
        uri="s3://url", user_entry_point=module_name, args=args, env_vars={"FOO": "BAR"}
    )

    expected_env_vars = {"FOO": "BAR", "PYTHONPATH": ""}
    get_runner.assert_called_with(
        runner.ProcessRunnerType, module_name, args, expected_env_vars, None
    )


@patch("sys.path")
@patch("sagemaker_training.runner.get")
@patch("sagemaker_training.files.download_and_extract")
@patch("os.chmod")
@patch("socket.gethostbyname")
def test_run_module_with_extra_opts(
    gethostbyname, chmod, download_and_extract, get_runner, sys_path
):
    module_name = "default_user_module_name"
    args = ["--some-arg", "42"]
    extra_opts = {"foo": "bar"}

    entry_point.run(uri="s3://url", user_entry_point=module_name, args=args, extra_opts=extra_opts)
    get_runner.assert_called_with(runner.ProcessRunnerType, module_name, args, {}, extra_opts)
