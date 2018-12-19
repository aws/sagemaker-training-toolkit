# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import os
import shutil
import subprocess

import pytest
from sagemaker.estimator import Framework

logging.basicConfig(level=logging.INFO)

dir_path = os.path.realpath(__file__)
root_dir = os.path.realpath(os.path.join(dir_path, '..', '..', '..'))
source_dir = os.path.realpath(os.path.join(dir_path, '..', '..', 'resources', 'openmpi'))


class CustomEstimator(Framework):

    def create_model(self, **kwargs):
        raise NotImplementedError('This methos is not supported.')


@pytest.mark.skip(reason="waiting for local mode fix on  "
                         "https://github.com/aws/sagemaker-python-sdk/pull/559")
def test_mpi(tmpdir):

    estimator = CustomEstimator(entry_point='launcher.sh',
                                image_name=build_mpi_image(tmpdir),
                                role='SageMakerRole',
                                train_instance_count=2,
                                source_dir=source_dir,
                                train_instance_type='local',
                                hyperparameters={
                                    'sagemaker_mpi_enabled': True,
                                    'sagemaker_mpi_custom_mpi_options': '-verbose',
                                    'sagemaker_network_interface_name': 'eth0'
                                })

    estimator.fit()


def build_mpi_image(tmpdir):
    tmp = str(tmpdir)

    subprocess.check_call(['python', 'setup.py', 'sdist'], cwd=root_dir)

    for file in os.listdir(os.path.join(root_dir, 'dist')):
        shutil.copy2(os.path.join(root_dir, 'dist', file), tmp)

    shutil.copy2(os.path.join(source_dir, 'Dockerfile'), tmp)

    imagename = 'openmpi'
    subprocess.check_call(['docker', 'build', '-t', imagename, '.'], cwd=tmp)

    return imagename
