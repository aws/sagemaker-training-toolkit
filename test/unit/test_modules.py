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
import os
import sys
import textwrap

from mock import call, mock_open, patch
import pytest

from sagemaker_training import environment, errors, files, modules, params

builtins_open = "builtins.open"


@patch("boto3.resource", autospec=True)
@pytest.mark.parametrize(
    "url,bucket_name,key,dst,endpoint",
    [
        ("S3://my-bucket/path/to/my-file", "my-bucket", "path/to/my-file", "/tmp/my-file", None),
        ("s3://my-bucket/my-file", "my-bucket", "my-file", "/tmp/my-file", "http://localhost:9000"),
    ],
)
def test_s3_download(resource, url, bucket_name, key, dst, endpoint):
    region = "us-west-2"
    os.environ[params.REGION_NAME_ENV] = region
    if endpoint is not None:
        os.environ[params.S3_ENDPOINT_URL] = endpoint

    files.s3_download(url, dst)

    chain = (
        call("s3", region_name=region, endpoint_url=endpoint)
        .Bucket(bucket_name)
        .download_file(key, dst)
    )
    assert resource.mock_calls == chain.call_list()


def test_s3_download_wrong_scheme():
    with pytest.raises(
        ValueError, message="Expecting 's3' scheme, got: c in c://my-bucket/my-file"
    ):
        files.s3_download("c://my-bucket/my-file", "/tmp/file")


@patch("sagemaker_training.process.check_error", autospec=True)
def test_install(check_error):
    path = "c://sagemaker-pytorch-container"
    modules.install(path)

    cmd = [sys.executable, "-m", "pip", "install", "."]
    check_error.assert_called_with(cmd, errors.InstallModuleError, 1, cwd=path, capture_error=False)

    with patch("os.path.exists", return_value=True):
        modules.install(path)

        check_error.assert_called_with(
            cmd + ["-r", "requirements.txt"],
            errors.InstallModuleError,
            1,
            capture_error=False,
            cwd=path,
        )


@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_requirements(check_error):
    path = "c://sagemaker-pytorch-container"

    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]

    with patch("os.path.exists", return_value=True):
        modules.install_requirements(path)

        check_error.assert_called_with(
            cmd, errors.InstallRequirementsError, 1, cwd=path, capture_error=False
        )


@patch("sagemaker_training.process.check_error", autospec=True)
def test_install_fails(check_error):
    check_error.side_effect = errors.ClientError()
    with pytest.raises(errors.ClientError):
        modules.install("git://aws/container-support")


@patch("sys.executable", None)
def test_install_no_python_executable():
    with pytest.raises(RuntimeError) as e:
        modules.install("git://aws/container-support")
    assert str(e.value) == "Failed to retrieve the real path for the Python executable binary"


@contextlib.contextmanager
def patch_tmpdir():
    yield "/tmp"


@patch(builtins_open, mock_open())
@patch("os.path.exists", lambda x: False)
def test_prepare():
    modules.prepare("c:/path/to/", "my-module")

    open.assert_any_call("c:/path/to/setup.py", "w")
    open.assert_any_call("c:/path/to/setup.cfg", "w")
    open.assert_any_call("c:/path/to/MANIFEST.in", "w")

    data = textwrap.dedent(
        """
    from setuptools import setup
    setup(packages=[''],
          name="my-module",
          version='1.0.0',
          include_package_data=True)
    """
    )

    open().write.assert_any_call(data)

    data = textwrap.dedent(
        """
    [wheel]
    universal = 1
    """
    )
    open().write.assert_any_call(data)

    data = textwrap.dedent(
        """
    recursive-include . *
    recursive-exclude . __pycache__*
    recursive-exclude . *.pyc
    recursive-exclude . *.pyo
    """
    )
    open().write.assert_any_call(data)


@patch(builtins_open, mock_open())
@patch("os.path.exists", lambda x: True)
def test_prepare_already_prepared():
    modules.prepare("c:/path/to/", "my-module")
    open.assert_not_called()


@patch("importlib.import_module")
def test_exists(import_module):
    assert modules.exists("my_module")

    import_module.side_effect = ImportError()

    assert not modules.exists("my_module")


@patch("sagemaker_training.files.download_and_extract")
@patch("sagemaker_training.modules.install")
@patch("importlib.import_module")
@patch("six.moves.reload_module")
def test_import_module(reload, import_module, install, download_and_extract):

    modules.import_module("s3://bucket/my-module")

    download_and_extract.assert_called_with("s3://bucket/my-module", environment.code_dir)
    install.assert_called_with(environment.code_dir)
    reload.assert_called_with(import_module(modules.DEFAULT_MODULE_NAME))


@patch("sagemaker_training.files.s3_download")
@patch("tarfile.open")
@patch("sagemaker_training.modules.prepare")
@patch("sagemaker_training.modules.install")
@patch("importlib.import_module")
@patch("six.moves.reload_module")
def test_import_module_local_directory(
    reload, import_module, install, prepare, tarfile, s3_download
):
    uri = "/opt/ml/input/data/code/sourcedir.tar.gz"
    modules.import_module(uri)

    s3_download.assert_not_called()
    tarfile.assert_called_with(name="/opt/ml/input/data/code/sourcedir.tar.gz", mode="r:gz")
    prepare.assert_called_once()
    install.assert_called_once()
