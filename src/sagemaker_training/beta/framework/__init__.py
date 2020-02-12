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
"""BETA module containing volatile or experimental code."""
from __future__ import absolute_import

# flake8: noqa ignore=F401 imported but unused
import sagemaker_training
from sagemaker_training import content_types
from sagemaker_training import encoders
from sagemaker_training import errors
from sagemaker_training import env
from sagemaker_training import mpi
from sagemaker_training import files
from sagemaker_training import functions
from sagemaker_training import _logging as logging
from sagemaker_training import mapping
from sagemaker_training import modules
from sagemaker_training import entry_point
from sagemaker_training import params
from sagemaker_training import process
from sagemaker_training import _runner as runner
from sagemaker_training import server
from sagemaker_training import trainer
from sagemaker_training import transformer
from sagemaker_training import worker


def training_env(resource_config=None, input_data_config=None, hyperparameters=None):
    """Placeholder docstring"""

    resource_config = resource_config or env.read_resource_config()
    input_data_config = input_data_config or env.read_input_data_config()
    hyperparameters = hyperparameters or env.read_hyperparameters()

    return env.TrainingEnv(
        resource_config=resource_config,
        input_data_config=input_data_config,
        hyperparameters=hyperparameters,
    )
