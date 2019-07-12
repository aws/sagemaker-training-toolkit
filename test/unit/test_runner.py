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

from mock import MagicMock, patch
import pytest

from sagemaker_containers import _mpi, _process, _runner

USER_SCRIPT = 'script'
CMD_ARGS = ['--some-arg', 42]
ENV_VARS = {'FOO': 'BAR'}

NCCL_DEBUG_MPI_OPT = '-X NCCL_DEBUG=WARN'
MPI_OPTS = {
    'sagemaker_mpi_num_of_processes_per_host': 2,
    'sagemaker_mpi_num_processes': 4,
    'sagemaker_mpi_custom_mpi_options': NCCL_DEBUG_MPI_OPT
}


@pytest.mark.parametrize('runner_class', [_process.ProcessRunner, _mpi.MasterRunner, _mpi.WorkerRunner])
def test_get_runner_returns_runnner_itself(runner_class):
    runner = MagicMock(spec=runner_class)

    assert _runner.get(runner) == runner


@patch('sagemaker_containers.training_env')
def test_get_runner_by_process_returns_runnner(training_env):
    runner = _runner.get(_runner.ProcessRunnerType)

    assert isinstance(runner, _process.ProcessRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()


@patch('sagemaker_containers.training_env')
def test_get_runner_by_process_with_extra_args(training_env):
    runner = _runner.get(_runner.ProcessRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS)

    assert isinstance(runner, _process.ProcessRunner)

    assert runner._user_entry_point == USER_SCRIPT
    assert runner._args == CMD_ARGS
    assert runner._env_vars == ENV_VARS

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()


@patch('sagemaker_containers.training_env')
def test_get_runner_by_mpi_returns_runnner(training_env):
    training_env().num_gpus = 0

    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.MasterRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()

    training_env().is_master = False
    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.WorkerRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()


@patch('sagemaker_containers.training_env')
def test_runnner_with_default_cpu_processes_per_host(training_env):
    training_env().additional_framework_parameters = dict()
    training_env().num_gpus = 0

    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.MasterRunner)
    assert runner._process_per_host == 1


@patch('sagemaker_containers.training_env')
def test_runnner_with_default_gpu_processes_per_host(training_env):
    training_env().additional_framework_parameters = dict()
    training_env().num_gpus = 2

    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.MasterRunner)
    assert runner._process_per_host == 2


@patch('sagemaker_containers.training_env')
def test_get_runner_by_mpi_with_extra_args(training_env):
    training_env().num_gpus = 0

    runner = _runner.get(_runner.MPIRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS, MPI_OPTS)

    assert isinstance(runner, _mpi.MasterRunner)

    assert runner._user_entry_point == USER_SCRIPT
    assert runner._args == CMD_ARGS
    assert runner._env_vars == ENV_VARS
    assert runner._process_per_host == 2
    assert runner._num_processes == 4
    assert runner._custom_mpi_options == NCCL_DEBUG_MPI_OPT

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()
    training_env().additional_framework_parameters.assert_not_called()

    training_env().is_master = False
    runner = _runner.get(_runner.MPIRunnerType, USER_SCRIPT, CMD_ARGS, ENV_VARS)

    assert isinstance(runner, _mpi.WorkerRunner)

    assert runner._user_entry_point == USER_SCRIPT
    assert runner._args == CMD_ARGS
    assert runner._env_vars == ENV_VARS

    training_env().to_cmd_args.assert_not_called()
    training_env().to_env_vars.assert_not_called()
    training_env().user_entry_point.assert_not_called()


def test_get_runner_invalid_identifier():
    with pytest.raises(ValueError):
        _runner.get(42)
