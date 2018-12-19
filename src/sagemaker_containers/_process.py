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

import os
import subprocess
import sys
from typing import Dict, List, Mapping  # noqa ignore=F401 imported but unused

import six

from sagemaker_containers import _entry_point_type, _env, _errors, _logging


def create(cmd, error_class, cwd=None, capture_error=False, **kwargs):
    try:
        stderr = subprocess.PIPE if capture_error else None
        return subprocess.Popen(cmd, env=os.environ, cwd=cwd or _env.code_dir, stderr=stderr, **kwargs)
    except Exception as e:
        six.reraise(error_class, error_class(e), sys.exc_info()[2])


def check_error(cmd,
                error_class,
                capture_error=False,
                **kwargs):
    # type: (List[str], type, bool, Mapping[str, object]) -> subprocess.Popen
    process = create(cmd, error_class, capture_error=capture_error, **kwargs)

    if capture_error:
        _, stderr = process.communicate()
        return_code = process.poll()
    else:
        stderr = None
        return_code = process.wait()

    if return_code:
        raise error_class(return_code=return_code, cmd=' '.join(cmd), output=stderr)
    return process


def python_executable():
    """Returns the real path for the Python executable, if it exists. Returns RuntimeError otherwise.

    Returns:
        (str): the real path of the current Python executable
    """
    if not sys.executable:
        raise RuntimeError('Failed to retrieve the real path for the Python executable binary')
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
        entrypoint_type = _entry_point_type.get(_env.code_dir, self._user_entry_point)

        if entrypoint_type is _entry_point_type.PYTHON_PACKAGE:
            return [python_executable(), '-m',
                    self._user_entry_point.replace('.py', '')] + self._args
        elif entrypoint_type is _entry_point_type.PYTHON_PROGRAM:
            return [python_executable(), self._user_entry_point] + self._args
        else:
            return ['/bin/sh', '-c', './%s %s' % (self._user_entry_point, ' '.join(self._args))]

    def _setup(self):
        pass

    def _tear_down(self):
        pass

    def run(self, wait=True, capture_error=False):
        self._setup()

        cmd = self._create_command()

        _logging.log_script_invocation(cmd, self._env_vars)

        if wait:
            process = check_error(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)
        else:
            process = create(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)

        self._tear_down()

        return process
