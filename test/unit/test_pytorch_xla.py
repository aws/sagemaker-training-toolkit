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

import pytest
from mock import ANY, MagicMock, patch
import os
import json

from sagemaker_training import pytorch_xla


@pytest.fixture(autouse=True)
def cluster(cluster_size):
    return [f"algo-{i+1}" for i in range(cluster_size)]


@pytest.fixture(autouse=True)
def master(cluster):
    return cluster[0]


@pytest.fixture(autouse=True)
def cluster_size():
    return 2


@pytest.fixture(autouse=True)
def instance_type():
    return 'ml.p3.16xlarge'


@pytest.fixture(autouse=True)
def num_gpus(instance_type):
    if  instance_type in [
        'ml.p3.16xlarge',
        ]:
        return 8
    elif instance_type in [
        'ml.p3.2xlarge',
        ]:
        return 1


@patch.dict(os.environ, {}, clear=True)
@pytest.mark.parametrize('instance_type', ['ml.p3.16xlarge', 'ml.p3.2xlarge'])
@pytest.mark.parametrize('cluster_size', [1, 4])
class TestPyTorchXLARunner:
    @patch('sagemaker_training.pytorch_xla.PyTorchXLARunner.__check_compatibility')
    def test_setup(self, *patches):
        for current_host in cluster:
            rank = cluster.index(current_host)
            print(f'Testing as host {rank+1}/{cluster_size}')
            runner = pytorch_xla.PyTorchXLARunner(
                user_entry_point="train.sh",
                args=["-v", "--lr", "35"],
                env_vars={
                    "SM_TRAINING_ENV": json.dumps(
                        {"additional_framework_parameters":
                             {"sagemaker_instance_type": instance_type}}
                    ),
                },
                processes_per_host=num_gpus,
                master_hostname=master,
                current_host=current_host,
                hosts=cluster,
                num_gpus=num_gpus,
            )
            runner.setup()
            assert os.environ["XRT_HOST_ORDINAL"] == str(rank)
            assert os.environ["XRT_SHARD_WORLD_SIZE"] == str(cluster_size)
            assert os.environ["XRT_WORKERS"] = "|".join(['localservice:{};{}:{}'.format(i, host, pytorch_xla.PyTorchXLARunner.WORKER_PORT) for i, host in enumerate(cluster)])
            assert os.environ["GPU_NUM_DEVICES"] = str(num_gpus)
            if cluster_size > 1:
                assert os.environ["XRT_MESH_SERVICE_ADDRESS"] = f"{master}:{pytorch_xla.PyTorchXLARunner.MESH_SERVICE_PORT}"


    def test_command(self):
        raise NotImplementedError()


    def test_compatibility(self):
        raise NotImplementedError()

