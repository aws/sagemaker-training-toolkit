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
from __future__ import absolute_import

import os

import sagemaker_containers
from sagemaker_containers.beta.framework import functions, modules


def train():
    training_env = sagemaker_containers.training_env()

    script = modules.import_module(training_env.module_dir, training_env.module_name)

    model = script.train(**functions.matching_args(script.train, training_env))

    if model:
        if hasattr(script, 'save'):
            script.save(model, training_env.model_dir)
        else:
            model_file = os.path.join(training_env.model_dir, 'saved_model')
            model.save(model_file)
