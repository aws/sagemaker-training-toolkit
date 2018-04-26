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

from sagemaker_containers import modules


def test_download_and_import_module(upload_script, create_script):

    create_script('my_script.py', 'def validate(): return True')

    content = ['from distutils.core import setup',
               "setup(name='my_script', py_modules=['my_script'])"]

    create_script('setup.py', content)

    url = upload_script('my_script.py')

    module = modules.download_and_import(url, 'my_script')

    assert module.validate()


def test_download_and_import_script(upload_script, create_script):

    create_script('my_script.py', 'def validate(): return True')

    url = upload_script('my_script.py')

    module = modules.download_and_import(url, 'my_script')

    assert module.validate()


def test_download_and_import_script_with_requirements(upload_script, create_script):
    script = os.linesep.join(['import os',
                              'def validate():',
                              '    return os.path.exist("requirements.txt")'])

    create_script('my_script.py', script)
    create_script('requirements.txt', 'keras\nh5py')

    url = upload_script('my_script.py')

    module = modules.download_and_import(url, 'my_script')

    assert module.validate()
