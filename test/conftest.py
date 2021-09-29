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

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys

from mock import patch
import pytest

from sagemaker_training import environment

logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARN)

DEFAULT_REGION = "us-west-2"


def _write_json(obj, path):  # type: (object, str) -> None
    with open(path, "w") as f:
        json.dump(obj, f)


@pytest.fixture(autouse=True)
def create_base_path():

    yield str(os.environ[environment.BASE_PATH_ENV])

    shutil.rmtree(os.environ[environment.BASE_PATH_ENV])

    os.makedirs(environment.model_dir)
    os.makedirs(environment.input_config_dir)
    os.makedirs(environment.code_dir)
    os.makedirs(environment.output_data_dir)

    _write_json({}, environment.hyperparameters_file_dir)
    _write_json({}, environment.input_data_config_file_dir)
    host_name = socket.gethostname()

    resources_dict = {"current_host": host_name, "hosts": [host_name]}
    _write_json(resources_dict, environment.resource_config_file_dir)


@pytest.fixture(autouse=True)
def patch_exit_process():
    def _exit(error_code):
        if error_code:
            raise ValueError(error_code)

    with patch("sagemaker_training.trainer._exit_processes", _exit):
        yield _exit


@pytest.fixture(autouse=True)
def fix_protobuf_installation_for_python_2():
    # Python 2 requires an __init__.py at every level,
    # but protobuf doesn't honor that, so we create the file ourselves.
    # https://stackoverflow.com/a/45141001
    if sys.version_info.major == 2:
        protobuf_info = subprocess.check_output("pip show protobuf".split())
        site_packages = re.match(r"[\S\s]*Location: (.*)\s", protobuf_info).group(1)
        with open(os.path.join(site_packages, "google", "__init__.py"), "w"):
            pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
