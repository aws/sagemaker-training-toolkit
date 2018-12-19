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
import enum
import os


class _EntryPointType(enum.Enum):
    PYTHON_PACKAGE = 'PYTHON_PACKAGE'
    PYTHON_PROGRAM = 'PYTHON_PROGRAM'
    COMMAND = 'COMMAND'


PYTHON_PACKAGE = _EntryPointType.PYTHON_PACKAGE
PYTHON_PROGRAM = _EntryPointType.PYTHON_PROGRAM
COMMAND = _EntryPointType.COMMAND


def get(path, name):  # type: (str, str) -> _EntryPointType
    """
    Args:
        path (string): Directory where the entry point is located
        name (string): Name of the entry point file

    Returns:
        (_EntryPointType): The type of the entry point
    """
    if 'setup.py' in os.listdir(path):
        return _EntryPointType.PYTHON_PACKAGE
    elif name.endswith('.py'):
        return _EntryPointType.PYTHON_PROGRAM
    else:
        return _EntryPointType.COMMAND
