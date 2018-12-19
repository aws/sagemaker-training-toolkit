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

from mock import patch
import pytest

from sagemaker_containers import _entry_point_type


@pytest.fixture
def entry_point_type_module():
    with patch('os.listdir', lambda x: ('setup.py',)):
        yield


@pytest.fixture(autouse=True)
def entry_point_type_script():
    with patch('os.listdir', lambda x: ()):
        yield


@pytest.fixture()
def has_requirements():
    with patch('os.path.exists', lambda x: x.endswith('requirements.txt')):
        yield


def test_get_package(entry_point_type_module):
    assert _entry_point_type.get('bla', 'program.py') == _entry_point_type.PYTHON_PACKAGE


def test_get_command(entry_point_type_script):
    assert _entry_point_type.get('bla', 'program.sh') == _entry_point_type.COMMAND


def test_get_program():
    assert _entry_point_type.get('bla', 'program.py') == _entry_point_type.PYTHON_PROGRAM
