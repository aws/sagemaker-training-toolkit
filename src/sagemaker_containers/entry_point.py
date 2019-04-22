# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sys
from typing import Dict, List  # noqa ignore=F401 imported but unused

from sagemaker_containers import _entry_point_type, _env, _files, _modules, _runner


def run(uri,
        user_entry_point,
        args,
        env_vars=None,
        wait=True,
        capture_error=False,
        runner=_runner.ProcessRunnerType):
    # type: (str, str, List[str], Dict[str, str], bool, bool, _runner.RunnerType) -> None
    """Download, prepare and executes a compressed tar file from S3 or provided directory as an user
    entrypoint. Runs the user entry point, passing env_vars as environment variables and args as command
    arguments.

    If the entry point is:
        - A Python package: executes the packages as >>> env_vars python -m module_name + args
        - A Python script: executes the script as >>> env_vars python module_name + args
        - Any other: executes the command as >>> env_vars /bin/sh -c ./module_name + args

    Example:
         >>>import sagemaker_containers
         >>>from sagemaker_containers.beta.framework import entry_point

         >>>env = sagemaker_containers.training_env()
         {'channel-input-dirs': {'training': '/opt/ml/input/training'}, 'model_dir': '/opt/ml/model', ...}


         >>>hyperparameters = env.hyperparameters
         {'batch-size': 128, 'model_dir': '/opt/ml/model'}

         >>>args = mapping.to_cmd_args(hyperparameters)
         ['--batch-size', '128', '--model_dir', '/opt/ml/model']

         >>>env_vars = mapping.to_env_vars()
         ['SAGEMAKER_CHANNELS':'training', 'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
         'MODEL_DIR':'/opt/ml/model', ...}

         >>>entry_point.run('user_script', args, env_vars)
         SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
         SAGEMAKER_MODEL_DIR=/opt/ml/model python -m user_script --batch-size 128 --model_dir /opt/ml/model

     Args:
        user_entry_point (str): name of the user provided entry point
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        uri (str): the location of the module.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.

     """
    env_vars = env_vars or {}
    env_vars = env_vars.copy()

    _files.download_and_extract(uri, user_entry_point, _env.code_dir)

    install(user_entry_point, _env.code_dir, capture_error)

    _env.write_env_vars(env_vars)

    return _runner.get(runner, user_entry_point, args, env_vars).run(wait, capture_error)


def install(name, dst, capture_error=False):
    """Install the user provided entry point to be executed as follow:
        - add the path to sys path
        - if the user entry point is a command, gives exec permissions to the script

    Args:
        name (str): name of the script or module.
        dst (str): path to directory with the script or module.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    if dst not in sys.path:
        sys.path.insert(0, dst)

    entrypoint_type = _entry_point_type.get(dst, name)
    if entrypoint_type is _entry_point_type.PYTHON_PACKAGE:
        _modules.install(dst, capture_error)
    if entrypoint_type is _entry_point_type.COMMAND:
        os.chmod(os.path.join(dst, name), 511)
