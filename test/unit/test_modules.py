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

import contextlib
import importlib
import os
import subprocess
import sys
import tarfile
import textwrap

from mock import call, mock_open, patch
import pytest
from six import PY2

from sagemaker_containers import errors, modules
import test

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('boto3.resource', autospec=True)
@pytest.mark.parametrize('url,bucket_name,key,dst', [
    ('S3://my-bucket/path/to/my-file', 'my-bucket', 'path/to/my-file', '/tmp/my-file'),
    ('s3://my-bucket/my-file', 'my-bucket', 'my-file', '/tmp/my-file')
])
def test_s3_download(resource, url, bucket_name, key, dst):
    modules.s3_download(url, dst)

    chain = call('s3').Bucket(bucket_name).download_file(key, dst)
    assert resource.mock_calls == chain.call_list()


@patch(builtins_open, mock_open())
@patch('os.path.exists', lambda x: False)
def test_prepare():
    modules.prepare('c:/path/to/', 'my-module')

    open.assert_any_call('c:/path/to/setup.py', 'w')
    open.assert_any_call('c:/path/to/setup.cfg', 'w')
    open.assert_any_call('c:/path/to/MANIFEST.in', 'w')

    content = textwrap.dedent("""
    from setuptools import setup

    setup(packages=[''],
          name="my-module",
          version='1.0.0',
          include_package_data=True)
    """)

    open().write.assert_any_call(content)

    content = textwrap.dedent("""
    [wheel]
    universal = 1
    """)
    open().write.assert_any_call(content)

    content = textwrap.dedent("""
    recursive-include . *

    recursive-exclude . __pycache__*
    recursive-exclude . *.pyc
    recursive-exclude . *.pyo
    """)
    open().write.assert_any_call(content)


@patch(builtins_open, mock_open())
@patch('os.path.exists', lambda x: True)
def test_prepare_already_prepared():
    modules.prepare('c:/path/to/', 'my-module')
    open.assert_not_called()


def test_s3_download_wrong_scheme():
    with pytest.raises(ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"):
        modules.s3_download('c://my-bucket/my-file', '/tmp/file')


@patch('subprocess.check_call', autospec=True)
def test_install(check_call):
    path = 'c://sagemaker-pytorch-container'
    modules.install(path)

    cmd = [sys.executable, '-m', 'pip', 'install', '-U', '.']
    check_call.assert_called_with(cmd, cwd=path)

    with patch('os.path.exists', return_value=True):
        modules.install(path)

        check_call.assert_called_with(cmd + ['-r', 'requirements.txt'], cwd=path)


@patch('subprocess.check_call')
def test_install_fails(check_call):
    check_call.side_effect = subprocess.CalledProcessError(1, 'returned non-zero exit status 1')
    with pytest.raises(errors.InstallModuleError) as e:
        modules.install('git://aws/container-support')
    assert type(e.value.args[0]) == subprocess.CalledProcessError


@patch('sys.executable', None)
def test_install_no_python_executable():
    with pytest.raises(RuntimeError) as e:
        modules.install('git://aws/container-support')
    assert str(e.value) == 'Failed to retrieve the real path for the Python executable binary'


@contextlib.contextmanager
def patch_tmpdir():
    yield '/tmp'


@patch('importlib.import_module')
def test_exists(import_module):
    assert modules.exists('my_module')

    import_module.side_effect = ImportError()

    assert not modules.exists('my_module')


class TestDownloadAndImport(test.TestBase):
    patches = [
        patch('sagemaker_containers.env.tmpdir', new=patch_tmpdir),
        patch('sagemaker_containers.modules.prepare', autospec=True),
        patch('sagemaker_containers.modules.install', autospec=True),
        patch('sagemaker_containers.modules.s3_download', autospec=True),
        patch('sagemaker_containers.modules.exists', autospec=True),
        patch('tarfile.open', autospec=True),
        patch('importlib.import_module', autospec=True),
        patch('six.moves.reload_module', autospec=True),
        patch('os.makedirs', autospec=True)]

    def test_without_cache(self):
        with tarfile.open() as tar_file:
            module = modules.download_and_import('s3://bucket/my-module', cache=False)

            assert module == importlib.import_module(modules.DEFAULT_MODULE_NAME)

            modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')
            os.makedirs.assert_called_with('/tmp/module_dir')

            tar_file.extractall.assert_called_with(path='/tmp/module_dir')
            modules.prepare.assert_called_with('/tmp/module_dir', modules.DEFAULT_MODULE_NAME)
            modules.install.assert_called_with('/tmp/module_dir')

    def test_with_cache_and_module_already_installed(self):
        with tarfile.open() as tar_file:
            modules.exists.return_value = True

            module = modules.download_and_import('s3://bucket/my-module', cache=True)

            assert module == importlib.import_module(modules.DEFAULT_MODULE_NAME)

            modules.s3_download.return_value.assert_not_called()
            os.makedirs.return_value.assert_not_called()

            tar_file.extractall.return_value.assert_not_called()
            modules.prepare.return_value.assert_not_called()
            modules.install.return_value.assert_not_called()

    def test_default_name(self):
        with tarfile.open() as tar_file:
            modules.exists.return_value = False

            module = modules.download_and_import('s3://bucket/my-module', cache=True)

            assert module == importlib.import_module(modules.DEFAULT_MODULE_NAME)

            modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')
            os.makedirs.assert_called_with('/tmp/module_dir')

            tar_file.extractall.assert_called_with(path='/tmp/module_dir')
            modules.prepare.assert_called_with('/tmp/module_dir', modules.DEFAULT_MODULE_NAME)
            modules.install.assert_called_with('/tmp/module_dir')

    def test_any_name(self):
        with tarfile.open() as tar_file:
            modules.exists.return_value = False

            module = modules.download_and_import('s3://bucket/my-module', 'another_module_name', cache=True)

            assert module == importlib.import_module('another_module_name')

            modules.s3_download.assert_called_with('s3://bucket/my-module', '/tmp/tar_file')
            os.makedirs.assert_called_with('/tmp/module_dir')

            tar_file.extractall.assert_called_with(path='/tmp/module_dir')
            modules.prepare.assert_called_with('/tmp/module_dir', 'another_module_name')
            modules.install.assert_called_with('/tmp/module_dir')
