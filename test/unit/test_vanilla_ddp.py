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

import asyncio
import os
import sys

from mock import ANY, MagicMock, patch
import pytest

from sagemaker_training import environment, vanilla_ddp
from test.unit.test_mpi import MockSSHClient


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@patch("asyncio.gather", new_callable=AsyncMock)
@patch("os.path.exists")
@patch("sagemaker_training.process.python_executable", return_value="usr/bin/python3")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("asyncio.create_subprocess_shell")
@patch("sagemaker_training.environment.Environment")
def test_vanilla_ddp_run_multi_node_python(
    training_env,
    async_shell,
    policy,
    ssh_client,
    python_executable,
    path_exists,
    async_gather,
    event_loop,
):
    with patch.dict(os.environ, clear=True):
        hosts = ["algo-1", "algo-2"]
        master_hostname = hosts[0]
        num_processes_per_host = 8

        vanilla_ddp_runner = vanilla_ddp.VanillaDDPRunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={
                "SM_TRAINING_ENV": '{"additional_framework_parameters":{"sagemaker_instance_type":"ml.p3.16xlarge"}}'
            },
            processes_per_host=num_processes_per_host,
            master_hostname=master_hostname,
            hosts=hosts,
            current_host="algo-1",
        )

        process = vanilla_ddp_runner.run()

        # ssh_client().load_system_host_keys.assert_called()
        # ssh_client().set_missing_host_key_policy.assert_called_with(policy())
        # ssh_client().connect.assert_called_with("algo-2", port=22)
        # ssh_client().close.assert_called()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            "8",
            "--nnodes",
            "2",
            "--node_rank",
            "0",
            "--master_addr",
            "algo-1",
            "--master_port",
            "55555",
            "train.py",
            "-v",
            "--lr",
            "35",
        ]
        async_shell.assert_called_with(
            " ".join(cmd),
            cwd=environment.code_dir,
            env=ANY,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        async_shell.assert_called_once()
        async_gather.assert_called_once()
        assert process == async_shell.return_value


@patch("asyncio.gather", new_callable=AsyncMock)
@patch("os.path.exists")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("asyncio.create_subprocess_shell")
@patch("sagemaker_training.environment.Environment")
def test_vanilla_ddp_run_single_node_python(
    training_env, async_shell, policy, ssh_client, path_exists, async_gather, event_loop
):
    with patch.dict(os.environ, clear=True):
        hosts = ["algo-1"]
        master_hostname = hosts[0]
        num_processes_per_host = 8
        host_list = hosts

        vanilla_ddp_runner = vanilla_ddp.VanillaDDPRunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={
                "SM_TRAINING_ENV": '{"additional_framework_parameters":{"sagemaker_instance_type":"ml.p4d.24xlarge"}}'
            },
            master_hostname=master_hostname,
            hosts=host_list,
            current_host="algo-1",
            processes_per_host=num_processes_per_host,
        )

        process = vanilla_ddp_runner.run()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            "8",
            "train.py",
            "-v",
            "--lr",
            "35",
        ]
        async_shell.assert_called_with(
            " ".join(cmd),
            env=ANY,
            cwd=environment.code_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async_shell.assert_called_once()
        async_gather.assert_called_once()
        assert process == async_shell.return_value


@patch("sagemaker_training.logging_config.log_script_invocation")
def test_connection(log):
    with pytest.raises(Exception):
        vanilla_ddp._can_connect("test_host")
        log.assert_called_with("Cannot connect to host test_host at port 55555. Retrying...")
