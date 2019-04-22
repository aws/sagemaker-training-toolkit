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

import enum
from typing import Dict, List  # noqa ignore=F401 imported but unused

import sagemaker_containers
from sagemaker_containers import _mpi, _params, _process


class RunnerType(enum.Enum):
    MPI = 'MPI'
    Process = 'Process'


ProcessRunnerType = RunnerType.Process
MPIRunnerType = RunnerType.MPI


def get(identifier, user_entry_point=None, args=None, env_vars=None):
    # type: (RunnerType, str, List[str], Dict[str]) -> _process.Runner
    if isinstance(identifier, _process.ProcessRunner):
        return identifier
    else:
        return _get_by_runner_type(identifier, user_entry_point, args, env_vars)


def _get_by_runner_type(identifier, user_entry_point=None, args=None, env_vars=None):
    env = sagemaker_containers.training_env()
    user_entry_point = user_entry_point or env.user_entry_point
    args = args or env.to_cmd_args()
    env_vars = env_vars or env.to_env_vars()

    if identifier is RunnerType.MPI and env.is_master:
        processes_per_host = env.additional_framework_parameters.get(_params.MPI_PROCESSES_PER_HOST,
                                                                     1)
        custom_mpi_options = env.additional_framework_parameters.get(_params.MPI_CUSTOM_OPTIONS, '')

        return _mpi.MasterRunner(user_entry_point,
                                 args,
                                 env_vars,
                                 env.master_hostname,
                                 env.hosts,
                                 processes_per_host,
                                 custom_mpi_options,
                                 env.network_interface_name)
    elif identifier is RunnerType.MPI:
        return _mpi.WorkerRunner(user_entry_point,
                                 args,
                                 env_vars,
                                 env.master_hostname)
    elif identifier is RunnerType.Process:
        return _process.ProcessRunner(user_entry_point,
                                      args,
                                      env_vars)
    else:
        raise ValueError('Invalid identifier %s' % identifier)
