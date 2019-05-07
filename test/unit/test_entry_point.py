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
from __future__ import absolute_import

import os
import sys

from mock import MagicMock, patch
import pytest
from six import PY2

from sagemaker_containers import _env, _errors, _process, _runner, entry_point

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


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


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_module(check_error, entry_point_type_module):
    path = 'c://sagemaker-pytorch-container'
    entry_point.install('python_module.py', path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_error.assert_called_with(cmd, _errors.InstallModuleError,
                                   capture_error=False, cwd=path)

    with patch('os.path.exists', return_value=True):
        entry_point.install('python_module.py', path)

        check_error.assert_called_with(cmd + ['-r', 'requirements.txt'],
                                       _errors.InstallModuleError, cwd=path,
                                       capture_error=False)


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_script(check_error, entry_point_type_module, has_requirements):
    path = 'c://sagemaker-pytorch-container'
    entry_point.install('train.py', path)

    with patch('os.path.exists', return_value=True):
        entry_point.install(path, 'python_module.py')


@patch('sagemaker_containers._process.check_error', autospec=True)
def test_install_fails(check_error, entry_point_type_module):
    check_error.side_effect = _errors.ClientError()
    with pytest.raises(_errors.ClientError):
        entry_point.install('git://aws/container-support', 'script')


@patch('sys.executable', None)
def test_install_no_python_executable(has_requirements, entry_point_type_module):
    with pytest.raises(RuntimeError) as e:
        entry_point.install('train.py', 'git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@patch('sagemaker_containers._files.download_and_extract')
@patch('os.chmod')
def test_run_module_wait(chmod, download_and_extract):
    runner = MagicMock(spec=_process.ProcessRunner)
    entry_point.run(uri='s3://url', user_entry_point='launcher.sh', args=['42'],
                    capture_error=True, runner=runner)

    download_and_extract.assert_called_with('s3://url', _env.code_dir)
    runner.run.assert_called_with(True, True)
    chmod.assert_called_with(os.path.join(_env.code_dir, 'launcher.sh'), 511)


@patch('sagemaker_containers._files.download_and_extract')
@patch('os.chmod')
def test_run_module_no_wait(chmod, download_and_extract):
    runner = MagicMock(spec=_process.ProcessRunner)

    module_name = 'default_user_module_name'
    entry_point.run(uri='s3://url', user_entry_point=module_name, args=['42'], wait=False, runner=runner)

    runner.run.assert_called_with(False, False)


@patch('sys.path')
@patch('sagemaker_containers._runner.get')
@patch('sagemaker_containers._files.download_and_extract')
@patch('os.chmod')
def test_run_module_with_env_vars(chmod, download_and_extract, get_runner, sys_path):
    module_name = 'default_user_module_name'
    args = ['--some-arg', '42']
    entry_point.run(uri='s3://url', user_entry_point=module_name, args=args, env_vars={'FOO': 'BAR'})

    expected_env_vars = {'FOO': 'BAR', 'PYTHONPATH': ''}
    get_runner.assert_called_with(_runner.ProcessRunnerType, module_name, args, expected_env_vars, None)


@patch('sys.path')
@patch('sagemaker_containers._runner.get')
@patch('sagemaker_containers._files.download_and_extract')
@patch('os.chmod')
def test_run_module_with_extra_opts(chmod, download_and_extract, get_runner, sys_path):
    module_name = 'default_user_module_name'
    args = ['--some-arg', '42']
    extra_opts = {'foo': 'bar'}

    entry_point.run(uri='s3://url', user_entry_point=module_name, args=args, extra_opts=extra_opts)
    get_runner.assert_called_with(_runner.ProcessRunnerType, module_name, args, {}, extra_opts)
