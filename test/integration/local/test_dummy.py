# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import subprocess
import sys

import pytest
from sagemaker.estimator import Estimator


@pytest.fixture(scope="module", autouse=True)
def container():
    try:
        command = (
            "docker run --name sagemaker-training-toolkit-test "
            "sagemaker-training-toolkit-test:dummy train"
        )

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        yield proc.pid

    finally:
        subprocess.check_call("docker rm -f sagemaker-training-toolkit-test".split())


def test_install_requirements(capsys):
    estimator = Estimator(
        image_uri="sagemaker-training-toolkit-test:dummy",
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
    )

    estimator.fit()

    stdout = capsys.readouterr().out

    assert "Installing collected packages: pyfiglet" in stdout
    assert "Successfully installed pyfiglet-0.8.post1" in stdout
    assert "Reporting training SUCCESS" in stdout
