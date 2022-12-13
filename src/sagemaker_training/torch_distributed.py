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
"""This module contains functionality related to Torch Distributed Elastic Runner.
Refer: https://pytorch.org/docs/stable/elastic/run.html
"""
from __future__ import absolute_import

import os

from sagemaker_training import (
    _entry_point_type,
    environment,
    errors,
    logging_config,
    process,
    SM_EFA_NCCL_INSTANCES,
    SM_EFA_RDMA_INSTANCES,
)

TORCH_DISTRIBUTED_MODULE = "torchrun"
MASTER_PORT = "7777"

logger = logging_config.get_logger()


class TorchDistributedRunner(process.ProcessRunner):
    """Runner responsible for preparing Pytorch distributed data parallel training"""

    def __init__(
        self,
        user_entry_point,
        args,
        env_vars,
        processes_per_host,
        master_hostname,
        hosts,
        current_host,
        network_interface_name,
        instance_type="ml.trn1.2xlarge",
    ):
        """Initialize a Native PT Launcher, which is responsible for executing
        the user entry point within a process.

        Args:
            user_entry_point (str): The name of the user entry point.
            args ([str]): A list of arguments to include when executing the entry point.
            env_vars (dict(str,str)): A dictionary of environment variables.
        """
        super(TorchDistributedRunner, self).__init__(
            user_entry_point, args, env_vars, processes_per_host
        )

        self._master_hostname = master_hostname
        self._hosts = hosts
        self._current_host = current_host
        self._network_interface_name = network_interface_name
        self._instance_type = instance_type

    def _setup(self):
        logger.info("Starting distributed training through torchrun")
        # EFA settings
        if self._instance_type in SM_EFA_NCCL_INSTANCES:
            # Enable EFA use
            os.environ["FI_PROVIDER"] = "efa"
        if self._instance_type in SM_EFA_RDMA_INSTANCES:
            # Use EFA's RDMA functionality for one-sided and two-sided transfer
            os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = str(self._network_interface_name)

    def _create_command(self):
        """
        Based on the number of hosts, torchrun command differs.
        Currently the elasticity feture of torchrun is not yet supported.
        """
        self._setup()
        entrypoint_type = _entry_point_type.get(environment.code_dir, self._user_entry_point)

        if entrypoint_type is _entry_point_type.PYTHON_PACKAGE:
            raise errors.ClientError(
                "Python packages are not supported for torch_distributed. "
                "Please use a python script as the entry-point"
            )

        if entrypoint_type is _entry_point_type.PYTHON_PROGRAM:
            num_hosts = len(self._hosts)
            torchrun_cmd = []

            node_options = [
                TORCH_DISTRIBUTED_MODULE,
                "--nnodes",
                str(num_hosts),
                "--nproc_per_node",
                str(self._processes_per_host),
            ]

            torchrun_cmd += node_options

            multinode_options = [
                "--master_addr",
                str(self._master_hostname),
                "--master_port",
                MASTER_PORT,
                "--node_rank",
                str(self._hosts.index(self._current_host)),
            ]

            if num_hosts > 1:
                torchrun_cmd += multinode_options

            # match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)

            torchrun_cmd.append(str(self._user_entry_point))
            torchrun_cmd += self._args
            return torchrun_cmd
        else:
            raise errors.ClientError("Unsupported entry point type for torch_distributed")

    def run(self, capture_error=True, wait=True):
        """
        Run the process.

        Args:
            capture_error (bool): A boolean indicating whether to direct stderr to a stream
                that can later be read. Defaults to True.
        Returns:
            process (subprocess.Popen): The spawned process.
        """
        cmd = self._create_command()
        logging_config.log_script_invocation(cmd, self._env_vars)
        if wait:
            process_spawned = process.check_error(
                cmd,
                errors.ExecuteUserScriptError,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        else:
            process_spawned = process.create(
                cmd,
                errors.ExecuteUserScriptError,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        return process_spawned
