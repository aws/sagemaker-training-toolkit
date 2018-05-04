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
from mock import Mock, patch

from sagemaker_containers import trainer


class TrainingEnv(Mock):
    framework_module = 'my_framework:train'


@patch('importlib.import_module')
@patch('sagemaker_containers.env.TrainingEnv', new_callable=TrainingEnv)
def test_train(training_env, import_module):
    framework = Mock()
    import_module.return_value = framework
    trainer.train()

    import_module.assert_called_with('my_framework')
    framework.train.assert_called()
