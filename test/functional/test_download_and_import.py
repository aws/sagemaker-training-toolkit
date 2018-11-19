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

import shlex
import subprocess
import textwrap

import pytest

from sagemaker_containers.beta.framework import errors, modules
import test

data = ['from distutils.core import setup\n',
        'setup(name="my_test_script", py_modules=["my_test_script"])']

SETUP_FILE = test.File('setup.py', data)

USER_SCRIPT_FILE = test.File('my_test_script.py', 'def validate(): return True')


@pytest.fixture(name='user_module_name')
def erase_user_module():
    user_module = 'my_test_script'
    yield user_module
    try:
        subprocess.check_call(shlex.split('pip uninstall -y --quiet %s' % user_module))
    except subprocess.CalledProcessError:
        pass


def test_import_module(user_module_name):
    user_module = test.UserModule(USER_SCRIPT_FILE).add_file(SETUP_FILE).upload()

    module = modules.import_module(user_module.url, user_module_name)

    assert module.validate()


def test_import_module_with_s3_script(user_module_name):
    user_module = test.UserModule(USER_SCRIPT_FILE).add_file(SETUP_FILE).upload()

    module = modules.import_module(user_module.url, user_module_name, cache=False)

    assert module.validate()


def test_import_module_with_local_script(user_module_name, tmpdir):
    tmp_code_dir = str(tmpdir)

    test.UserModule(USER_SCRIPT_FILE).add_file(SETUP_FILE).create_tmp_dir_with_files(tmp_code_dir)

    module = modules.import_module(tmp_code_dir, user_module_name, cache=False)

    assert module.validate()


data = textwrap.dedent("""
            from pyfiglet import Figlet

            def say():
                return Figlet().renderText('SageMaker').strip()

""")

USER_SCRIPT_WITH_REQUIREMENTS = test.File('my_test_script.py', data)

REQUIREMENTS_FILE = test.File('requirements.txt', 'pyfiglet')


def test_import_module_with_s3_script_with_requirements(user_module_name):
    user_module = test.UserModule(USER_SCRIPT_WITH_REQUIREMENTS).add_file(SETUP_FILE)
    user_module = user_module.add_file(REQUIREMENTS_FILE).upload()

    module = modules.import_module(user_module.url, user_module_name, cache=False)

    assert module.say() == """
 ____                   __  __       _.............
/ ___|  __ _  __ _  ___|  \/  | __ _| | _____ _ __.
\___ \ / _` |/ _` |/ _ \ |\/| |/ _` | |/ / _ \ '__|
 ___) | (_| | (_| |  __/ |  | | (_| |   <  __/ |...
|____/ \__,_|\__, |\___|_|  |_|\__,_|_|\_\___|_|...
             |___/.................................
""".replace('.', ' ').strip()  # noqa W605


data = ['raise ValueError("this script does not work")']
USER_SCRIPT_WITH_ERROR = test.File('my_test_script.py', data)


def test_import_module_with_s3_script_with_error(user_module_name):
    user_module = test.UserModule(USER_SCRIPT_WITH_ERROR).add_file(SETUP_FILE).upload()

    with pytest.raises(errors.ImportModuleError):
        modules.import_module(user_module.url, user_module_name, cache=False)
