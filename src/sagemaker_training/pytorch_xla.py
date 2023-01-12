# Copyright 2018-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains functionality related to distributed training using
PT-XLA (PyTorch - Accelerated Linear Algebra)."""
from __future__ import absolute_import

import os

from sagemaker_training import (
    _entry_point_type,
    environment,
    errors,
    logging_config,
    process,
)


logger = logging_config.get_logger()


class PyTorchXLARunner(process.ProcessRunner):
    """Responsible for PT-XLA distributed training."""

    MESH_SERVICE_PORT = 53957
    WORKER_PORT = 43857

    def __init__(
        self,
        user_entry_point,
        args,
        env_vars,
        processes_per_host,
        master_hostname,
        current_host,
        hosts,
        num_gpus,
    ):
        """Initialize a PyTorchXLARunner, which is responsible for distributed
        training with PT-XLA.

        Args:
            user_entry_point (str): The name of the user entry point.
            args ([str]): A list of arguments to include when executing the entry point.
            env_vars (dict(str,str)): A dictionary of environment variables.
            master_hostname (str): The master hostname.
            current_host (str): The current hostname.
            hosts ([str]): A list of hosts.
            num_gpus (int): The number of GPUs available per host.
        """

        super(PyTorchXLARunner, self).__init__(user_entry_point, args, env_vars, processes_per_host)

        self._master_hostname = master_hostname
        self._current_host = current_host
        self._hosts = hosts
        self._num_gpus = num_gpus

        self._num_hosts = len(self._hosts)
        self._rank = self._hosts.index(self._current_host)

    def _setup(self):  # type: () -> None
        logger.info("Starting distributed training through PT-XLA Runtime.")
        self._check_compatibility()

        # Set NCCL logging to info to debug customer issues
        os.environ["NCCL_DEBUG"] = "info"

        # Use `simple` protocol to handle the out-of-order data delivery from EFA
        os.environ["NCCL_PROTO"] = "simple"

        # Use GPU RDMA when available (available only in p4d.24xlarge)
        os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"

        # Use multiple connections per GPU to better saturate the EFA bandwidth
        os.environ["OFI_NCCL_NIC_DUP_CONNS"] = str(self._num_gpus)

        # Set cluster configuration for XLA runtime
        os.environ["XRT_HOST_ORDINAL"] = str(self._rank)
        os.environ["XRT_SHARD_WORLD_SIZE"] = str(self._num_hosts)
        address = "localservice:{};{}:" + str(self.WORKER_PORT)
        os.environ["XRT_WORKERS"] = "|".join(
            [address.format(i, host) for i, host in enumerate(self._hosts)]
        )
        os.environ["GPU_NUM_DEVICES"] = str(self._num_gpus)
        if self._num_hosts > 1:
            os.environ[
                "XRT_MESH_SERVICE_ADDRESS"
            ] = f"{self._master_hostname}:{self.MESH_SERVICE_PORT}"

        logger.info("Completed environment setup for distributed training through PT-XLA Runtime.")

    def _create_command(self):
        entrypoint_type = _entry_point_type.get(environment.code_dir, self._user_entry_point)

        if entrypoint_type is _entry_point_type.PYTHON_PACKAGE:
            raise errors.SMTrainingCompilerConfigurationError(
                "Distributed Training through PT-XLA is not supported for Python packages. "
                "Please use a python script as the entry-point"
            )
        if entrypoint_type is _entry_point_type.PYTHON_PROGRAM:
            return self._pytorch_xla_command() + [self._user_entry_point] + self._args
        else:
            raise errors.SMTrainingCompilerConfigurationError(
                "Distributed Training through PT-XLA is only supported for Python scripts. "
                "Please use a python script as the entry-point"
            )

    def _pytorch_xla_command(self):
        return self._python_command() + [
            "-m",
            "torch_xla.distributed.xla_spawn",
            "--num_gpus",
            str(self._num_gpus),
        ]

    def _check_compatibility(self):
        self._check_processor_compatibility()
        self._check_for_torch_xla()
        self._check_for_sagemaker_integration()

    def _check_for_sagemaker_integration(self):
        # pylint: disable=no-self-use
        try:
            import torch_xla.distributed.xla_spawn  # pylint: disable=unused-import # noqa: F401
        except ModuleNotFoundError as exception:
            raise errors.SMTrainingCompilerConfigurationError(
                "Unable to find SageMaker integration code in PT-XLA. "
                "AWS SageMaker adds custom code on top of open source "
                "PT-XLA to provide platform specific "
                "optimizations. These SageMaker specific binaries are"
                " shipped as part of our Deep Learning Containers."
                " Please refer to "
                "https://github.com/aws/deep-learning-containers"
                "/blob/master/available_images.md"
            ) from exception

    def _check_for_torch_xla(self):
        # pylint: disable=no-self-use
        try:
            import torch_xla  # pylint: disable=unused-import # noqa: F401
        except ModuleNotFoundError as exception:
            raise errors.SMTrainingCompilerConfigurationError(
                "Unable to find PT-XLA in the execution environment. "
                "This distribution mechanism requires PT-XLA to be available"
                " in the execution environment. "
                "SageMaker Training Compiler provides ready-to-use containers with PT-XLA. "
                "Please refer to https://github.com/aws/deep-learning-containers"
                "/blob/master/available_images.md "
            ) from exception

    def _check_processor_compatibility(self):
        if not self._num_gpus > 0:
            raise errors.SMTrainingCompilerConfigurationError(
                "Distributed training through PT-XLA is only supported for GPUs."
            )
