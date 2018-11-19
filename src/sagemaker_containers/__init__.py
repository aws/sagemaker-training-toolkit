# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


def training_env():  # type: () -> _env.TrainingEnv
    """Create a TrainingEnv.

    Returns:
        TrainingEnv: an instance of TrainingEnv
    """

    from sagemaker_containers import _env

    return _env.TrainingEnv(
        resource_config=_env.read_resource_config(),
        input_data_config=_env.read_input_data_config(),
        hyperparameters=_env.read_hyperparameters())
