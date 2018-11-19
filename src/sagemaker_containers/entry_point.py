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

import enum
import os
import sys

from sagemaker_containers import _env, _errors, _files, _logging, _modules, _process


def run(uri, user_entry_point, args, env_vars=None, wait=True):
    # type: (str, str, list, dict, bool) -> subprocess.Popen
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
        wait (bool): If True, holds the process executing the user entry-point.
                     If False, returns the process that is executing it.
     """
    env_vars = env_vars or {}
    env_vars = env_vars.copy()

    _files.download_and_extract(uri, user_entry_point, _env.code_dir)

    install(user_entry_point, _env.code_dir)

    _env.write_env_vars(env_vars)

    return _call(user_entry_point, args, env_vars, wait)


def install(name, dst):
    """Install the user provided entry point to be executed as follow:
        - add the path to sys path
        - if the user entry point is a command, gives exec permissions to the script

    Args:
        name (str): name of the script or module.
        dst (str): path to directory with the script or module.
    """
    if dst not in sys.path:
        sys.path.insert(0, dst)

    entrypoint_type = entry_point_type(dst, name)
    if entrypoint_type is EntryPointType.PYTHON_PACKAGE:
        _modules.install(dst)
    if entrypoint_type is EntryPointType.COMMAND:
        os.chmod(os.path.join(dst, name), 511)


def _call(user_entry_point, args=None, env_vars=None, wait=True):  # type: (str, list, dict, bool) -> Popen
    args = args or []
    env_vars = env_vars or {}

    entrypoint_type = entry_point_type(_env.code_dir, user_entry_point)

    if entrypoint_type is EntryPointType.PYTHON_PACKAGE:
        cmd = [_process.python_executable(), '-m', user_entry_point.replace('.py', '')] + args
    elif entrypoint_type is EntryPointType.PYTHON_PROGRAM:
        cmd = [_process.python_executable(), user_entry_point] + args
    else:
        cmd = ['/bin/sh', '-c', './%s %s' % (user_entry_point, ' '.join(args))]

    _logging.log_script_invocation(cmd, env_vars)

    if wait:
        return _process.check_error(cmd, _errors.ExecuteUserScriptError)

    else:
        return _process.create(cmd, _errors.ExecuteUserScriptError)


class EntryPointType(enum.Enum):
    PYTHON_PACKAGE = 'PYTHON_PACKAGE'
    PYTHON_PROGRAM = 'PYTHON_PROGRAM'
    COMMAND = 'COMMAND'


def entry_point_type(path, name):  # type: (str, str) -> EntryPointType
    """
    Args:
        path (string): Directory where the entry point is located
        name (string): Name of the entry point file

    Returns:
        (EntryPointType): The type of the entry point
    """
    if 'setup.py' in os.listdir(path):
        return EntryPointType.PYTHON_PACKAGE
    elif name.endswith('.py'):
        return EntryPointType.PYTHON_PROGRAM
    else:
        return EntryPointType.COMMAND
