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
import importlib

from sagemaker_containers import env


def train():
    training_env = env.TrainingEnv()

    # TODO: iquintero - add error handling for ImportError to let the user know
    # if the framework module is not defined.
    framework_name, entry_point = training_env.framework_module.split(':')
    framework = importlib.import_module(framework_name)
    entry = getattr(framework, entry_point)
    entry()
