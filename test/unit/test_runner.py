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

from mock import MagicMock, patch
import pytest

from sagemaker_training import mpi, process, runner

USER_SCRIPT = "script"
CMD_ARGS = ["--some-arg", 42]
ENV_VARS = {"FOO": "BAR"}
DEFAULT_PROC_PER_HOST = 1

NCCL_DEBUG_MPI_OPT = "-X NCCL_DEBUG=WARN"
MPI_OPTS = {
    "sagemaker_mpi_num_of_processes_per_host": 2,
    "sagemaker_mpi_num_processes": 4,
    "sagemaker_mpi_custom_mpi_options": NCCL_DEBUG_MPI_OPT,
}


@pytest.mark.parametrize(
    "runner_class", [process.ProcessRunner, mpi.MasterRunner, mpi.WorkerRunner]
)
def test_get_runner_returns_runnner_itself(runner_class):
    runner_mock = MagicMock(spec=runner_class)

    assert runner.get(runner_mock) == runner_mock


@patch("sagemaker_training.environment.Environment")
def test_get_runner_by_process_returns_runnner(training_env):
    test_runner = runner.get(runner.ProcessRunnerType)

    assert isinstance(test_runner, process.ProcessRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()


@patch("sagemaker_training.environment.Environment")
def test_get_runner_by_process_with_extra_args(training_env):
    test_runner = runner.get(runner.ProcessRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS)

    assert isinstance(test_runner, process.ProcessRunner)

    assert test_runner._user_entry_point == USER_SCRIPT
    assert test_runner._args == CMD_ARGS
    assert test_runner._env_vars == ENV_VARS

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()


@patch("sagemaker_training.environment.Environment")
def test_get_runner_by_mpi_returns_runnner(training_env):
    training_env().num_gpus = 0

    test_runner = runner.get(runner.MPIRunnerType)

    assert isinstance(test_runner, mpi.MasterRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()

    training_env().is_master = False
    test_runner = runner.get(runner.MPIRunnerType)

    assert isinstance(test_runner, mpi.WorkerRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()


@patch("sagemaker_training.environment.Environment")
def test_runnner_with_default_cpu_processes_per_host(training_env):
    training_env().additional_framework_parameters = dict()
    training_env().num_gpus = 0

    test_runner = runner.get(runner.MPIRunnerType)

    assert isinstance(test_runner, mpi.MasterRunner)
    assert test_runner._processes_per_host == 1


@patch("sagemaker_training.environment.Environment")
def test_runnner_with_default_gpu_processes_per_host(training_env):
    training_env().additional_framework_parameters = dict()
    training_env().num_gpus = 2

    test_runner = runner.get(runner.MPIRunnerType)

    assert isinstance(test_runner, mpi.MasterRunner)
    assert test_runner._processes_per_host == 2


@patch("sagemaker_training.environment.Environment")
def test_get_runner_by_mpi_with_extra_args(training_env):
    training_env().num_gpus = 0

    test_runner = runner.get(runner.MPIRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS, MPI_OPTS)

    assert isinstance(test_runner, mpi.MasterRunner)

    assert test_runner._user_entry_point == USER_SCRIPT
    assert test_runner._args == CMD_ARGS
    assert test_runner._env_vars == ENV_VARS
    assert test_runner._processes_per_host == 2
    assert test_runner._num_processes == 4
    assert test_runner._custom_mpi_options == NCCL_DEBUG_MPI_OPT

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()
    training_env().additional_framework_parameters.assert_not_called()

    training_env().is_master = False
    test_runner = runner.get(runner.MPIRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS)

    assert isinstance(test_runner, mpi.WorkerRunner)

    assert test_runner._user_entry_point == USER_SCRIPT
    assert test_runner._args == CMD_ARGS
    assert test_runner._env_vars == ENV_VARS

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()


def test_get_runner_invalid_identifier():
    with pytest.raises(ValueError):
        runner.get(42)
