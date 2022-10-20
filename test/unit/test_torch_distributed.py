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

import json
import os

from mock import patch
import pytest

from sagemaker_training.errors import ClientError
from sagemaker_training.torch_distributed import TorchDistributedRunner


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


@patch.dict(os.environ, {}, clear=True)
# @pytest.mark.parametrize("instance_type", ["ml.trn1.2xlarge", "ml.trn1.32xlarge"])
# @pytest.mark.parametrize("cluster_size", [2, 4])
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
