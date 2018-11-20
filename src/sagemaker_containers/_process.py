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

import six

from sagemaker_containers import _env


def create(cmd, error_class, cwd=None, capture_error=False, **kwargs):
    try:
        stderr = subprocess.PIPE if capture_error else None
        return subprocess.Popen(cmd, env=os.environ, cwd=cwd or _env.code_dir, stderr=stderr, **kwargs)
    except Exception as e:
        six.reraise(error_class, error_class(e), sys.exc_info()[2])


def check_error(cmd, error_class, capture_error=False, **kwargs):
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
