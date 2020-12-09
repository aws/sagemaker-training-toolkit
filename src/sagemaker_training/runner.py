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
"""This module contains functionality to get process runners based on the
runner type.
"""
from __future__ import absolute_import

import enum

from sagemaker_training import environment, mpi, params, process, smdataparallel


class RunnerType(enum.Enum):
    """Enumerated type consisting of valid types of runners."""

    MPI = "MPI"
    Process = "Process"
    SMDataParallel = "SMDataParallel"


ProcessRunnerType = RunnerType.Process
MPIRunnerType = RunnerType.MPI
SMDataParallelRunnerType = RunnerType.SMDataParallel


def get(identifier, user_entry_point=None, args=None, env_vars=None, extra_opts=None):
    """Get the process runner based on the runner type.

    Args:
        identifier (RunnerType or process.ProcessRunner): The type of runner to get.
        user_entry_point (str): The name of the user entry point.
        args ([str]): A list of arguments to include when executing the entry point.
        env_vars (dict(str,str)): A dictionary of environment variables.
        extra_opts (dict): A dictionary of extra arguments for MPI.

    Returns:
        process.Runner: The process.
    """
    if isinstance(identifier, process.ProcessRunner):
        return identifier
    else:
        return _get_by_runner_type(identifier, user_entry_point, args, env_vars, extra_opts)


def _get_by_runner_type(
    identifier, user_entry_point=None, args=None, env_vars=None, extra_opts=None
):
    env = environment.Environment()
    user_entry_point = user_entry_point or env.user_entry_point
    args = args or env.to_cmd_args()
    env_vars = env_vars or env.to_env_vars()

    if identifier is RunnerType.SMDataParallel and env.is_master:
        mpi_args = extra_opts or {}
        return smdataparallel.SMDataParallelRunner(
            user_entry_point,
            args,
            env_vars,
            env.master_hostname,
            env.hosts,
            env.network_interface_name,
        )
    elif identifier is RunnerType.SMDataParallel:
        return mpi.WorkerRunner(user_entry_point, args, env_vars, env.master_hostname)
    elif identifier is RunnerType.MPI and env.is_master:
        mpi_args = extra_opts or {}

        # Default to single process for CPU
        default_processes_per_host = env.num_gpus if env.num_gpus > 0 else 1
        processes_per_host = _mpi_param_value(
            mpi_args, env, params.MPI_PROCESSES_PER_HOST, default_processes_per_host
        )
        num_processes = _mpi_param_value(mpi_args, env, params.MPI_NUM_PROCESSES)
        custom_mpi_options = _mpi_param_value(mpi_args, env, params.MPI_CUSTOM_OPTIONS, "")

        return mpi.MasterRunner(
            user_entry_point,
            args,
            env_vars,
            env.master_hostname,
            env.hosts,
            processes_per_host,
            custom_mpi_options,
            env.network_interface_name,
            num_processes=num_processes,
        )
    elif identifier is RunnerType.MPI:
        return mpi.WorkerRunner(user_entry_point, args, env_vars, env.master_hostname)
    elif identifier is RunnerType.Process:
        return process.ProcessRunner(user_entry_point, args, env_vars)
    else:
        raise ValueError("Invalid identifier %s" % identifier)


def _mpi_param_value(mpi_args, env, param_name, default=None):
    return mpi_args.get(param_name) or env.additional_framework_parameters.get(param_name, default)
