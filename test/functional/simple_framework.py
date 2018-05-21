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
import os

from sagemaker_containers import env, functions, modules


def train():
    training_env = env.training_env()

    script = modules.import_module_from_s3(training_env.module_dir, training_env.module_name, False)

    model = script.train(**functions.matching_args(script.train, env.environment_mapping()))

    if model:
        if hasattr(script, 'save'):
            script.save(model, env.model_dir)
        else:
            model_file = os.path.join(env.model_dir, 'saved_model')
            model.save(model_file)
