# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Contains functionality related to Sagemaker provided health checks"""
import argparse
from inspect import getfile, isclass
import gethostname
import json
import logging
import os
import subprocess
import paramiko
import time

from sagemaker_training import (
    environment,
    errors,
    logging_config,
    process,
    SM_EFA_NCCL_INSTANCES,
    timeout,
)

logger = logging_config.get_logger()
logging.getLogger("paramiko").setLevel(logging.INFO)

class HealthCheckRunner(process.ProcessRunner):
    """Configure and execute Sagemaker provided health check
       This includes setup of Nvidia DCGM test, NCCL test and EFA connectivity checker.
    """
    def __init__(
        self,
        user_entry_point,
        args,
        env_vars,
        master_hostname,
        host_list,
        processes_per_host,
        timeout_in_mins=60,
    ):
    """Initialize a HealthCheckRunner.

    HealthCheckRunner will help run the Sagemaker provided health check and collect results among hosts.

    Args:
        user_entry_point (str): The name of the user entry point.
        args ([str]): A list of arguments to include when executing the entry point.
        env_vars (Dict[str, str]): A dictionary of environment variables.
        master_hostname (str): The master hostname.
        host_list ([str]): A list of hosts.
        processes_per_host (int): number of processes run on each host
        timeout_in_mins (int): The number of seconds to wait for workers. Defaults to 60 minutes.
    """
        super(SMDataParallelRunner, self).__init__(
            user_entry_point, args, env_vars, processes_per_host
        )
        self._master_hostname = master_hostname
        self._host_list = host_list
        self._processes_per_host = processes_per_host
        self.timeout_in_seconds = timeout_in_seconds

    def create_dcgm_diag_command(
        self,
        dcgm_diag_level,
        dcgm_diag_args,
        enable_fail_early,
        timeout_in_mins,
    ):
        if enable_fail_early:
            dcgm_diag_cmd = ['sudo', 'dcgmi', 'diag', '-r', str(dcgm_diag_level), '--fail-early'] + dcgm_diag_args
        else:
            dcgm_diag_cmd = ['sudo', 'dcgmi', 'diag', '-r', str(dcgm_diag_level)] + dcgm_diag_args

        return dcgm_diag_cmd

    def create_nccl_test_command(
        self,
        nccl_test_algo,
        nccl_test_args,
        check_mode,
        host_list,
        timeout_in_mins
    ):
        num_hosts = len(host_list)

        if check_mode == 'local':
            nccl_test_cmd = [nccl_test_algo] + nccl_test_args
        else if check_mode == 'cluster':
            nccl_test_cmd = ['mpirun', '-H'] + host_list + ['-np', str(num_hosts), nccl_test_algo, nccl_test_args]

        return nccl_test_cmd

    def create_efa_checker_command(
        self,
        check_mode,
        host_list,
        timeout_in_mins
    ):
        if check_mode == 'local':
            efa_checker_cmd = ['efa_checker_single_node', '--no_instance_id', '--verbose'] 
        else if check_mode == 'cluster':
            efa_checker_cmd = ['mpirun', '-N', '1', '-x FI_EFA_FORK_SAFE=1', '--hostfile', 'hosts.txt', 'efa_checker_multi_node']
        
        return efa_checker_cmd   

    def exec_cmd_sync(cmd: str)
    """Execute health check cmd in a blocking call
       It is used to run health checks that only takes a few minutes, e.g. the EFA connectivity check and NCCL local test
    """    
    try:
        result = subprocess.run(efa_local_checker_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Execute command '{cmd}' returned error: {e.output}")
        print(e.output)
