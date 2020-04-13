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
import itertools
import logging
import os
import tarfile

from mock import mock_open, patch
import pytest
import six

from sagemaker_training import environment, files
import test

builtins_open = "__builtin__.open" if six.PY2 else "builtins.open"

RESOURCE_CONFIG = dict(current_host="algo-1", hosts=["algo-1", "algo-2", "algo-3"])

INPUT_DATA_CONFIG = {
    "train": {
        "ContentType": "trainingContentType",
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
    "validation": {
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
}

USER_HYPERPARAMETERS = dict(batch_size=32, learning_rate=0.001)
SAGEMAKER_HYPERPARAMETERS = {
    "sagemaker_region": "us-west-2",
    "default_user_module_name": "net",
    "sagemaker_job_name": "sagemaker-training-job",
    "sagemaker_program": "main.py",
    "sagemaker_submit_directory": "imagenet",
    "sagemaker_enable_cloudwatch_metrics": True,
    "sagemaker_container_log_level": logging.WARNING,
}

ALL_HYPERPARAMETERS = dict(
    itertools.chain(USER_HYPERPARAMETERS.items(), SAGEMAKER_HYPERPARAMETERS.items())
)


def test_read_json():
    test.write_json(ALL_HYPERPARAMETERS, environment.hyperparameters_file_dir)

    assert files.read_json(environment.hyperparameters_file_dir) == ALL_HYPERPARAMETERS


def test_read_json_throws_exception():
    with pytest.raises(IOError):
        files.read_json("non-existent.json")


def test_read_file():
    test.write_json("test", environment.hyperparameters_file_dir)

    assert files.read_file(environment.hyperparameters_file_dir) == '"test"'


@patch("tempfile.mkdtemp")
@patch("shutil.rmtree")
def test_tmpdir(rmtree, mkdtemp):
    with files.tmpdir():
        mkdtemp.assert_called()
    rmtree.assert_called()


@patch("tempfile.mkdtemp")
@patch("shutil.rmtree")
def test_tmpdir_with_args(rmtree, mkdtemp):
    with files.tmpdir("suffix", "prefix", "/tmp"):
        mkdtemp.assert_called_with(dir="/tmp", prefix="prefix", suffix="suffix")
    rmtree.assert_called()


@patch(builtins_open, mock_open())
def test_write_file():
    files.write_file("/tmp/my-file", "42")
    open.assert_called_with("/tmp/my-file", "w")
    open().write.assert_called_with("42")

    files.write_file("/tmp/my-file", "42", "a")
    open.assert_called_with("/tmp/my-file", "a")
    open().write.assert_called_with("42")


@patch(builtins_open, mock_open())
def test_write_success_file():
    file_path = os.path.join(environment.output_dir, "success")
    empty_msg = ""
    files.write_success_file()
    open.assert_called_with(file_path, "w")
    open().write.assert_called_with(empty_msg)


@patch(builtins_open, mock_open())
def test_write_failure_file():
    file_path = os.path.join(environment.output_dir, "failure")
    failure_msg = "This is a failure"
    files.write_failure_file(failure_msg)
    open.assert_called_with(file_path, "w")
    open().write.assert_called_with(failure_msg)


@patch("sagemaker_training.files.s3_download")
@patch("os.path.isdir", lambda x: True)
@patch("shutil.rmtree")
@patch("shutil.copytree")
def test_download_and_extract_source_dir(copy, rmtree, s3_download):
    uri = environment.channel_path("code")
    files.download_and_extract(uri, environment.code_dir)
    s3_download.assert_not_called()

    rmtree.assert_any_call(environment.code_dir)
    copy.assert_called_with(uri, environment.code_dir)


@patch("sagemaker_training.files.s3_download")
@patch("os.path.isdir", lambda x: False)
@patch("shutil.copy2")
def test_download_and_extract_file(copy, s3_download):
    uri = __file__
    files.download_and_extract(uri, environment.code_dir)

    s3_download.assert_not_called()
    copy.assert_called_with(uri, environment.code_dir)


@patch("sagemaker_training.files.s3_download")
@patch("os.path.isdir", lambda x: False)
@patch("tarfile.TarFile.extractall")
def test_download_and_extract_tar(extractall, s3_download):
    t = tarfile.open(name="test.tar.gz", mode="w:gz")
    t.close()
    uri = t.name
    files.download_and_extract(uri, environment.code_dir)

    s3_download.assert_not_called()
    extractall.assert_called_with(path=environment.code_dir)

    os.remove(uri)
