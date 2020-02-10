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
"""Placeholder docstring"""
from __future__ import absolute_import

import os
import socket
import sys
from typing import Dict, List  # noqa ignore=F401 imported but unused

from retrying import retry

from sagemaker_training import _entry_point_type, _env, _files, _modules, _runner


def run(
    uri,
    user_entry_point,
    args,
    env_vars=None,
    wait=True,
    capture_error=False,
    runner=_runner.ProcessRunnerType,
    extra_opts=None,
):
    # type: (str, str, List[str], Dict[str, str], bool, bool, _runner.RunnerType,Dict[str, str]) -> None  # pylint: disable=line-too-long # noqa ignore=E501
    """Download, prepare and executes a compressed tar file from S3 or provided directory as an user
    entrypoint. Runs the user entry point, passing env_vars as environment variables and args
    as command arguments.

    If the entry point is:
        - A Python package: executes the packages as >>> env_vars python -m module_name + args
        - A Python script: executes the script as >>> env_vars python module_name + args
        - Any other: executes the command as >>> env_vars /bin/sh -c ./module_name + args

    Example:
         >>>import sagemaker_training
         >>>from sagemaker_training.beta.framework import entry_point

         >>>env = sagemaker_training.training_env()
         {'channel-input-dirs': {'training': '/opt/ml/input/training'},
          'model_dir': '/opt/ml/model', ...}


         >>>hyperparameters = env.hyperparameters
         {'batch-size': 128, 'model_dir': '/opt/ml/model'}

         >>>args = mapping.to_cmd_args(hyperparameters)
         ['--batch-size', '128', '--model_dir', '/opt/ml/model']

         >>>env_vars = mapping.to_env_vars()
         ['SAGEMAKER_CHANNELS':'training', 'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
         'MODEL_DIR':'/opt/ml/model', ...}

         >>>entry_point.run('user_script', args, env_vars)
         SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
         SAGEMAKER_MODEL_DIR=/opt/ml/model python -m user_script --batch-size 128
                             --model_dir /opt/ml/model

    Args:
        uri (str): the location of the module.
        user_entry_point (str): name of the user provided entry point
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written (default: None).
        wait (bool): If the user entry point should be run to completion before this method returns
            (default: True).
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
        runner (sagemaker_training.beta.framework.runner.RunnerType): the type of runner object to
            be created (default: sagemaker_training.beta.framework.runner.ProcessRunnerType).
        extra_opts (dict): Additional options for running the entry point (default: None).
            Currently, this only applies for MPI.

    Returns:
        sagemaker_training.beta.framework.process.ProcessRunner: the runner object responsible for
            executing the entry point.
    """
    env_vars = env_vars or {}
    env_vars = env_vars.copy()

    _files.download_and_extract(uri, _env.code_dir)

    install(user_entry_point, _env.code_dir, capture_error)

    _env.write_env_vars(env_vars)

    _wait_hostname_resolution()

    return _runner.get(runner, user_entry_point, args, env_vars, extra_opts).run(
        wait, capture_error
    )


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


@retry(stop_max_delay=1000 * 60 * 15, wait_exponential_multiplier=100, wait_exponential_max=30000)
def _dns_lookup(host):
    """ Retrying dns lookup on host """
    return socket.gethostbyname(host)


def _wait_hostname_resolution():
    """Wait for the hostname resolution of the container. This is known behavior as the cluster
    boots up and has been documented here:
     https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-dist-training
    """
    for host in _env.TrainingEnv().hosts:
        _dns_lookup(host)
