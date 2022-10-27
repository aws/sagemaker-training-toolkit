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
from __future__ import absolute_import

import asyncio
import json
import os

from mock import ANY, MagicMock, patch
import pytest

from sagemaker_training import environment
from sagemaker_training.errors import ClientError
from sagemaker_training.torch_distributed import TorchDistributedRunner


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@pytest.fixture()
def cluster(cluster_size):
    return [f"algo-{i+1}" for i in range(cluster_size)]


@pytest.fixture()
def master(cluster):
    return cluster[0]


@pytest.fixture()
def cluster_size():
    return 2


@pytest.fixture()
def instance_type():
    return "ml.trn1.2xlarge"


@pytest.fixture()
def num_neurons(instance_type):
    if instance_type in [
        "ml.trn1.2xlarge",
    ]:
        return 2
    elif instance_type in [
        "ml.trn1.32xlarge",
    ]:
        return 32


@pytest.fixture
def entry_point_type_module():
    with patch("os.listdir", lambda x: ("setup.py",)):
        yield


@patch.dict(os.environ, {}, clear=True)
class TestTorchDistributedRunner:
    @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge"])
    @pytest.mark.parametrize("cluster_size", [2])
    def test_setup(self, cluster, cluster_size, master, instance_type, num_neurons, *patches):
        for rank, current_host in enumerate(cluster):
            print(f"Testing as host {rank+1} in cluster of size {cluster_size}")
            runner = TorchDistributedRunner(
                user_entry_point="train.py",
                args=["-v", "--lr", "35"],
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {
                            "additional_framework_parameters": {
                                "sagemaker_instance_type": instance_type
                            }
                        }
                    ),
                },
                master_hostname=master,
                hosts=cluster,
                current_host=current_host,
                processes_per_host=num_neurons,
                network_interface_name="eth0",
            )
            runner._check_compatibility = lambda: None
            runner._setup()

    @pytest.mark.parametrize("instance_type", ["ml.trn1.32xlarge"])
    @pytest.mark.parametrize("cluster_size", [4])
    def test_setup_trn1_efa(
        self, cluster, cluster_size, master, instance_type, num_neurons, *patches
    ):
        for rank, current_host in enumerate(cluster):
            print(f"Testing as host {rank+1} in cluster of size {cluster_size}")
            runner = TorchDistributedRunner(
                user_entry_point="train.py",
                args=["-v", "--lr", "35"],
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {
                            "additional_framework_parameters": {
                                "sagemaker_instance_type": instance_type
                            }
                        }
                    ),
                },
                master_hostname=master,
                hosts=cluster,
                current_host=current_host,
                processes_per_host=num_neurons,
                network_interface_name="eth0",
                instance_type=instance_type,
            )
            runner._check_compatibility = lambda: None
            runner._setup()
            assert os.environ["FI_EFA_USE_DEVICE_RDMA"] == "1"
            assert os.environ["FI_PROVIDER"] == "efa"

    @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge", "ml.trn1.32xlarge"])
    @pytest.mark.parametrize("cluster_size", [1, 1])
    def test_create_singlenode_command_with_py_script(
        self, cluster, cluster_size, master, instance_type, num_neurons, *patches
    ):
        training_args = ["-v", "--lr", "35"]
        training_script = "train.py"
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f"Testing as host {rank+1} in cluster of size {cluster_size}")
            runner = TorchDistributedRunner(
                user_entry_point=training_script,
                args=training_args,
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {
                            "additional_framework_parameters": {
                                "sagemaker_instance_type": instance_type
                            }
                        }
                    ),
                },
                master_hostname=master,
                hosts=cluster,
                current_host=current_host,
                processes_per_host=num_neurons,
                network_interface_name="eth0",
            )
            received_command = runner._create_command()
            expected_command = [
                "torchrun",
                "--nnodes",
                str(len(cluster)),
                "--nproc_per_node",
                str(num_neurons),
                training_script,
            ] + training_args
            assert received_command[0].split("/")[-1] == expected_command[0]
            assert received_command[1:] == expected_command[1:]

    @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge", "ml.trn1.32xlarge"])
    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_create_multinode_command_with_py_script(
        self, cluster, cluster_size, master, instance_type, num_neurons, *patches
    ):
        training_args = ["-v", "--lr", "35"]
        training_script = "train.py"
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f"Testing as host {rank+1} in cluster of size {cluster_size}")
            runner = TorchDistributedRunner(
                user_entry_point=training_script,
                args=training_args,
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {
                            "additional_framework_parameters": {
                                "sagemaker_instance_type": instance_type
                            }
                        }
                    ),
                },
                master_hostname=master,
                hosts=cluster,
                current_host=current_host,
                processes_per_host=num_neurons,
                network_interface_name="eth0",
            )
            received_command = runner._create_command()
            expected_command = [
                "torchrun",
                "--nnodes",
                str(len(cluster)),
                "--nproc_per_node",
                str(num_neurons),
                "--master_addr",
                str(master),
                "--master_port",
                "7777",
                "--node_rank",
                str(cluster.index(current_host)),
                training_script,
            ] + training_args
            assert received_command[0].split("/")[-1] == expected_command[0]
            assert received_command[1:] == expected_command[1:]

    @patch("asyncio.gather", new_callable=AsyncMock)
    @patch("asyncio.create_subprocess_shell")
    @patch("sagemaker_training.environment.Environment")
    @patch("subprocess.run")
    def test_run_multinode_job_with_py_script(
        self,
        subprocess_run,
        training_env,
        async_shell,
        async_gather,
    ):
        with patch.dict(os.environ, clear=True):
            hosts = ["algo-1", "algo-2"]
            current_host = "algo-1"
            master_hostname = hosts[0]
            num_hosts = len(hosts)
            num_processes_per_host = 32
            network_interface_name = "eth0"
            torch_distributed_runner = TorchDistributedRunner(
                user_entry_point="train.py",
                args=["-v", "--lr", "35"],
                env_vars={
                    "SM_TRAINING_ENV": '{"additional_framework_parameters":{"sagemaker_instance_type":"ml.trn1.32xlarge"}}'
                },
                processes_per_host=num_processes_per_host,
                master_hostname=master_hostname,
                hosts=hosts,
                current_host="algo-1",
                network_interface_name=network_interface_name,
            )
        _, _, process = torch_distributed_runner.run(wait=False)
        cmd = [
            "torchrun",
            "--nnodes",
            str(num_hosts),
            "--nproc_per_node",
            str(num_processes_per_host),
            "--master_addr",
            str(master_hostname),
            "--master_port",
            "7777",
            "--node_rank",
            str(hosts.index(current_host)),
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
        subprocess_run.assert_not_called()

    @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge", "ml.trn1.32xlarge"])
    @pytest.mark.parametrize("cluster_size", [2, 4])
    def test_create_command_with_shell_script(
        self, cluster, cluster_size, master, instance_type, num_neurons, *patches
    ):
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f"Testing as host {rank + 1} in cluster of size {cluster_size}")
            runner = TorchDistributedRunner(
                user_entry_point="train.sh",
                args=["-v", "--lr", "35"],
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {
                            "additional_framework_parameters": {
                                "sagemaker_instance_type": instance_type
                            }
                        }
                    ),
                },
                master_hostname=master,
                hosts=cluster,
                current_host=current_host,
                processes_per_host=num_neurons,
                network_interface_name="eth0",
            )
            with pytest.raises(ClientError) as err:
                runner._create_command()
            assert "Unsupported entry point type for torch_distributed" in str(err)

    @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge"])
    @pytest.mark.parametrize("cluster_size", [2])
    def test_raises_error_with_python_package(
        self,
        entry_point_type_module,
        cluster,
        cluster_size,
        master,
        instance_type,
        num_neurons,
        *patches,
    ):

        with patch("sagemaker_training.environment.code_dir", entry_point_type_module):
            for current_host in cluster:
                rank = cluster.index(current_host)
                print(f"Testing as host {rank + 1} in cluster of size {cluster_size}")
                runner = TorchDistributedRunner(
                    user_entry_point="train.py",
                    args=["-v", "--lr", "35"],
                    env_vars={
                        "SM_TRAINING_ENV": json.dumps(
                            {
                                "additional_framework_parameters": {
                                    "sagemaker_instance_type": instance_type
                                }
                            }
                        ),
                    },
                    master_hostname=master,
                    hosts=cluster,
                    current_host=current_host,
                    processes_per_host=num_neurons,
                    network_interface_name="eth0",
                )
                with pytest.raises(ClientError) as err:
                    runner._create_command()
                assert (
                    "Python packages are not supported for torch_distributed. "
                    "Please use a python script as the entry-point"
                ) in str(err)
