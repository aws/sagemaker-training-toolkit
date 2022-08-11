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
from sagemaker_training.pytorch_xla import PyTorchXLARunner


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
    return "ml.p3.16xlarge"


@pytest.fixture()
def num_gpus(instance_type):
    if instance_type in [
        "ml.p3.16xlarge",
    ]:
        return 8
    elif instance_type in [
        "ml.p3.2xlarge",
    ]:
        return 1
    elif instance_type in [
        "ml.c5.4xlarge",
    ]:
        return 0


@patch.dict(os.environ, {}, clear=True)
@pytest.mark.parametrize("instance_type", ["ml.p3.16xlarge", "ml.p3.2xlarge"])
@pytest.mark.parametrize("cluster_size", [1, 4])
class TestPyTorchXLARunner:
    def test_setup(self, cluster, cluster_size, master, instance_type, num_gpus, *patches):
        for rank, current_host in enumerate(cluster):
            print(f"Testing as host {rank+1}/{cluster_size}")
            runner = PyTorchXLARunner(
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
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            runner._check_compatibility = lambda: None
            runner._setup()
            assert os.environ["XRT_HOST_ORDINAL"] == str(rank)
            assert os.environ["XRT_SHARD_WORLD_SIZE"] == str(cluster_size)
            assert os.environ["XRT_WORKERS"] == "|".join(
                [
                    f"localservice:{i};{host}:{PyTorchXLARunner.WORKER_PORT}"
                    for i, host in enumerate(cluster)
                ]
            )
            assert os.environ["GPU_NUM_DEVICES"] == str(num_gpus)
            if cluster_size > 1:
                assert (
                    os.environ["XRT_MESH_SERVICE_ADDRESS"]
                    == f"{master}:{PyTorchXLARunner.MESH_SERVICE_PORT}"
                )

    def test_create_command_with_py_script(
        self, cluster, cluster_size, master, instance_type, num_gpus, *patches
    ):
        training_args = ["-v", "--lr", "35"]
        training_script = "train.py"
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f"Testing as host {rank+1}/{cluster_size}")
            runner = PyTorchXLARunner(
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
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            received_command = runner._create_command()
            expected_command = [
                "python",
                "-m",
                "torch_xla.distributed.xla_spawn",
                "--num_gpus",
                str(num_gpus),
                training_script,
            ] + training_args
            assert received_command[0].split("/")[-1] == expected_command[0]
            assert received_command[1:] == expected_command[1:]

    def test_create_command_with_shell_script(
        self, cluster, cluster_size, master, instance_type, num_gpus, *patches
    ):
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f"Testing as host {rank + 1}/{cluster_size}")
            runner = PyTorchXLARunner(
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
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            with pytest.raises(ClientError) as err:
                runner._create_command()
            assert "Please use a python script" in str(err)

    def test_check_gpu_compatibility(
        self, cluster, cluster_size, master, instance_type, num_gpus, *patches
    ):
        for rank, current_host in enumerate(cluster):
            print(f"Testing as host {rank + 1}/{cluster_size}")
            runner = PyTorchXLARunner(
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
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            runner._check_processor_compatibility()

    def test_check_env_compatibility(
        self, cluster, cluster_size, master, instance_type, num_gpus, *patches
    ):
        for rank, current_host in enumerate(cluster):
            print(f"Testing as host {rank + 1}/{cluster_size}")
            runner = PyTorchXLARunner(
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
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            with pytest.raises(ClientError) as err:
                runner._check_for_torch_xla()
            assert "mechanism requires PT-XLA" in str(err)
            with pytest.raises(ClientError) as err:
                runner._check_for_sagemaker_integration()
            assert "Unable to find SageMaker integration code" in str(err)


def test_check_cpu_compatibility(cluster, cluster_size, master, *patches):
    for rank, current_host in enumerate(cluster):
        print(f"Testing as host {rank + 1}/{cluster_size}")
        runner = PyTorchXLARunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={
                "SM_TRAINING_ENV": json.dumps(
                    {
                        "additional_framework_parameters": {
                            "sagemaker_instance_type": "ml.c5.4xlarge"
                        }
                    }
                ),
            },
            processes_per_host=1,
            master_hostname=master,
            current_host=current_host,
            hosts=cluster,
            num_gpus=0,
        )
        with pytest.raises(ValueError) as err:
            runner._check_processor_compatibility()
        assert "only supported for GPU" in str(err)
