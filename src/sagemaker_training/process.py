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
import subprocess
import sys
from typing import Dict, List, Mapping  # noqa ignore=F401 imported but unused

import six

from sagemaker_training import entry_point_type, env, errors, logging_config


def create(cmd, error_class, cwd=None, capture_error=False, **kwargs):
    """Create subprocess.Popen object for the given command.

    Args:
        cmd (list): The command to be run.
        error_class (cls): The class to use when raising an exception.
        cwd (str): The location from which to run the command (default: None).
            If None, this defaults to the ``code_dir`` of the environment.
        capture_error (bool): whether or not to direct stderr to a stream
            that can later be read (default: False).
        **kwargs: Extra arguments that are passed to the subprocess.Popen constructor.

    Returns:
        subprocess.Popen: the process for the given command

    Raises:
        error_class: if there is an exception raised when creating the process
    """
    try:
        stderr = subprocess.PIPE if capture_error else None
        return subprocess.Popen(
            cmd, env=os.environ, cwd=cwd or env.code_dir, stderr=stderr, **kwargs
        )
    except Exception as e:  # pylint: disable=broad-except
        six.reraise(error_class, error_class(e), sys.exc_info()[2])


def check_error(cmd, error_class, capture_error=False, **kwargs):
    # type: (List[str], type, bool, Mapping[str, object]) -> subprocess.Popen
    """Run a commmand, raising an exception if there is an error.

    Args:
        cmd (list): The command to be run.
        error_class (cls): The class to use when raising an exception.
        capture_error (bool): whether or not to include stderr in
            the exception message (default: False). In either case,
            stderr is streamed to the process's output.
        **kwargs: Extra arguments that are passed to the subprocess.Popen constructor.

    Returns:
        subprocess.Popen: the process for the given command

    Raises:
        error_class: if there is an exception raised when creating the process
    """
    process = create(cmd, error_class, capture_error=capture_error, **kwargs)

    if capture_error:
        _, stderr = process.communicate()
        return_code = process.poll()
    else:
        stderr = None
        return_code = process.wait()

    if return_code:
        raise error_class(return_code=return_code, cmd=" ".join(cmd), output=stderr)
    return process


def python_executable():
    """Returns the real path for the Python executable, if it exists.
    Returns RuntimeError otherwise.

    Returns:
        (str): the real path of the current Python executable
    """
    if not sys.executable:
        raise RuntimeError("Failed to retrieve the real path for the Python executable binary")
    return sys.executable


class ProcessRunner(object):
    """Responsible to execute the user entrypoint whithin a process.
    """

    def __init__(self, user_entry_point, args, env_vars):
        # type: (str, List[str], Dict[str, str]) -> None
        self._user_entry_point = user_entry_point
        self._args = args
        self._env_vars = env_vars

    def _create_command(self):
        """Placeholder docstring"""
        entrypoint_type = entry_point_type.get(env.code_dir, self._user_entry_point)

        if entrypoint_type is entry_point_type.PYTHON_PACKAGE:
            entry_module = self._user_entry_point.replace(".py", "")
            return self._python_command() + ["-m", entry_module] + self._args
        elif entrypoint_type is entry_point_type.PYTHON_PROGRAM:
            return self._python_command() + [self._user_entry_point] + self._args
        else:
            args = [
                six.moves.shlex_quote(arg)  # pylint: disable=too-many-function-args
                for arg in self._args
            ]
            return ["/bin/sh", "-c", "./%s %s" % (self._user_entry_point, " ".join(args))]

    def _python_command(self):  # pylint: disable=no-self-use
        """Placeholder docstring"""
        return [python_executable()]

    def _setup(self):
        """Placeholder docstring"""

    def _tear_down(self):
        """Placeholder docstring"""

    def run(self, wait=True, capture_error=False):
        """Placeholder docstring"""
        self._setup()

        cmd = self._create_command()

        logging_config.log_script_invocation(cmd, self._env_vars)

        if wait:
            process = check_error(
                cmd, errors.ExecuteUserScriptError, capture_error=capture_error, cwd=env.code_dir
            )
        else:
            process = create(
                cmd, errors.ExecuteUserScriptError, capture_error=capture_error, cwd=env.code_dir
            )

        self._tear_down()

        return process
