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

from sagemaker_containers import modules
import test

content = ['from distutils.core import setup\n',
           'setup(name="test_script", py_modules=["test_script"])']

SETUP = test.File('setup.py', content)

USER_SCRIPT = test.File('test_script.py', 'def validate(): return True')


def test_download_and_import_module():
    user_module = test.UserModule(USER_SCRIPT).add_file(SETUP).upload()

    module = modules.download_and_import(user_module.url, 'test_script')

    assert module.validate()


def test_download_and_import_script():
    user_module = test.UserModule(USER_SCRIPT).upload()

    module = modules.download_and_import(user_module.url, 'test_script')

    assert module.validate()


content = ['import os',
           'def validate():',
           '    return os.path.exist("requirements.txt")']

USER_SCRIPT_WITH_REQUIREMENTS = test.File('test_script.py', content)

REQUIREMENTS_FILE = test.File('requirements.txt', ['keras', 'h5py'])


def test_download_and_import_script_with_requirements():
    user_module = test.UserModule(USER_SCRIPT_WITH_REQUIREMENTS).add_file(REQUIREMENTS_FILE).upload()

    module = modules.download_and_import(user_module.url, 'test_script')

    assert module.validate()
