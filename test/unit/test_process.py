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

from mock import MagicMock, patch
import pytest

from sagemaker_containers import _errors, _process


def test_python_executable_exception():
    with patch('sys.executable', None):
        with pytest.raises(RuntimeError):
            _process.python_executable()


@patch('subprocess.Popen', MagicMock(side_effect=ValueError('FAIL')))
def test_create_error():
    with pytest.raises(_errors.ExecuteUserScriptError):
        _process.create(['run'], _errors.ExecuteUserScriptError)


@patch('subprocess.Popen')
def test_check_error(popen):
    process = MagicMock(wait=MagicMock(return_value=0))
    popen.return_value = process

    assert process == _process.check_error(['run'], _errors.ExecuteUserScriptError)
