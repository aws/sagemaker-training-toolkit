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
from mock import patch, PropertyMock

from sagemaker_training import env
from sagemaker_training.cli import train as train_cli


@patch.object(env.ServingEnv, "framework_module", PropertyMock(return_value="my_flask_app"))
@patch("sagemaker_training.trainer.train")
def test_entry_point(train):
    train_cli.main()
    train.assert_called()
