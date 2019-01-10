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
import json
import os
import shutil
import sys

import pytest

import gethostname
from sagemaker_containers import _errors, _process

OPT_ML = "/opt/ml"
INPUT_CONFIG = "/opt/ml/input/config/"


@pytest.fixture()
def opt_ml_input_config():
    if os.path.exists(OPT_ML):
        shutil.rmtree(OPT_ML)

    try:
        os.makedirs(INPUT_CONFIG)

        yield INPUT_CONFIG

    finally:
        shutil.rmtree(OPT_ML)


@pytest.mark.parametrize('content,value', [
    [{'channel': 'training', 'current_host': 'algo-5', 'File': 'pipe'}, 'algo-5'],
    [{'current_host': 'algo-1-thse'}, 'algo-1-thse']])
def test_gethostname_resource_config_set(content, value, opt_ml_input_config):
    with open("/opt/ml/input/config/resourceconfig.json", 'w') as f:
        json.dump(content, f)

    assert gethostname.call(30)


def test_gethostname_with_env_not_set(opt_ml_input_config):
    py_cmd = "import gethostname\nassert gethostname.call(30) == 'algo-9'"

    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.check_error([sys.executable, '-c', py_cmd], _errors.ExecuteUserScriptError)
