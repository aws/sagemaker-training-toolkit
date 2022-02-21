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
"""This module contains functionality Contains functionality 
related to SM PyTorch Vanilla Distributed Data Parallel Training.
Refer: https://pytorch.org/docs/stable/distributed.html#debugging-torch-distributed-applications
"""
from __future__ import absolute_import

import paramiko
import sys

from sagemaker_training import (
    _entry_point_type,
    environment,
    errors,
    logging_config,
)

#TODO: This will be moved to torch elastic distributed launcher
PYTORCH_DIST_MODULE = "torch.distributed.launch"
logger = logging_config.get_logger()

def python_executable():
    """Return the real path for the Python executable, if it exists.
    Return RuntimeError otherwise.

    Returns:
        (str): The real path of the current Python executable.
    """
    if not sys.executable:
        raise RuntimeError("Failed to retrieve the real path for the Python executable binary")
    return sys.executable

class VanillaDDPRunner():

    def __init__(self, user_entry_point, args, env_vars, master_hostname, hosts, current_host, processes_per_host):
        """Initialize a Native PT DDP ProcessRunner, which is responsible for executing 
        the user entry point within a process.

        Args:
            user_entry_point (str): The name of the user entry point.
            args ([str]): A list of arguments to include when executing the entry point.
            env_vars (dict(str,str)): A dictionary of environment variables.
        """
        self._user_entry_point = user_entry_point
        self._args = args
        self._env_vars = env_vars
        self._master_hostname = master_hostname
        self._hosts = hosts
        self._current_host = current_host
        self._processes_per_host = processes_per_host

    def _python_command(self):  # pylint: disable=no-self-use
        return [python_executable()]
    
    def _get_vanilla_ddp_command(self):
        """
        Based on the number of hosts, vanilla ddp command differs.
        """
        entrypoint_type = _entry_point_type.get(environment.code_dir, self._user_entry_point)

        if entrypoint_type is _entry_point_type.PYTHON_PROGRAM:
            num_hosts = len(self._hosts)
            current_host = self._current_host
            current_node_rank = self._hosts.index(current_host)
            master_port = 55555

            pt_dist_cmd = f" {PYTORCH_DIST_MODULE} --nproc_per_node {self._processes_per_host} "
            multinode_options = f" --nnodes {num_hosts} --node_rank {current_node_rank} "

            final_cmd = self._python_command() + " -m " + pt_dist_cmd 

            if num_hosts > 1:
                master_options = f" --master_addr {self._master_hostname} --master_port {master_port} "
                final_cmd += multinode_options + master_options

            final_cmd += self._user_entry_point + self._args
            return final_cmd
        else:
            logger.error('Unknown entry point type for this distribution')

    def run(self, capture_error=True):
        """
        Run the process.

        Args:
            capture_error (bool): A boolean indicating whether to direct stderr to a stream
                that can later be read. Defaults to True.
        Returns:
            process (subprocess.Popen): The spawned process.
        """
        cmd = self._get_vanilla_ddp_command()

        logging_config.log_script_invocation(cmd, self._env_vars)

        process = process.check_error(
            cmd,
            errors.ExecuteUserScriptError,
            self._processes_per_host,
            capture_error=capture_error,
            cwd=environment.code_dir,
        )
        return process

def _can_connect(host, port=55555):
    # type: (str, int) -> bool
    """Check if the connection to provided ``host`` and ``port`` is possible.

    Args:
        host (str): Hostname for the host to check connection.
        port (int): Port name of the host to check connection on.
    """
    try:
        logger.debug("Testing connection to host %s at port %s", host, port)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port)
        logger.info("Can connect to host %s at port %s", host, port)
        return True
    except Exception:  # pylint: disable=broad-except
        logger.info("Cannot connect to host %s at port %s. Retrying...", host, port)
        return False
    finally:
        client.close()
        logger.info("Connection closed")