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

from mock import MagicMock, patch
import pytest

from sagemaker_containers import _mpi, _process, _runner


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
def test_get_runner_by_mpi_returns_runnner(training_env):
    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.MasterRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()

    training_env().is_master = False
    runner = _runner.get(_runner.MPIRunnerType)

    assert isinstance(runner, _mpi.WorkerRunner)
    training_env().to_cmd_args.assert_called()
    training_env().to_env_vars.assert_called()


def test_get_runner_invalid_identifier():
    with pytest.raises(ValueError):
        _runner.get(42)
