# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains utilities related to reading, writing, and downloading
files and directories.
"""
from __future__ import absolute_import

import contextlib
import json
import os
import shutil
import tarfile
import tempfile

import boto3
from six.moves.urllib import parse

from sagemaker_training import environment, logging_config, params

logger = logging_config.get_logger()


def write_success_file():  # type: () -> None
    """Create a file 'success' when training is successful. This file doesn't need to
    have any content.
    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    """
    file_path = os.path.join(environment.output_dir, "success")
    empty_content = ""
    write_file(file_path, empty_content)


def write_failure_file(failure_msg):  # type: (str) -> None
    """Create a file 'failure' if training fails after all algorithm output (for example,
    logging) completes, the failure description should be written to this file. In a
    DescribeTrainingJob response, Amazon SageMaker returns the first 1024 characters from
    this file as FailureReason.
    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The description of failure.
    """
    file_path = os.path.join(environment.output_dir, "failure")

    # Write failure file only if it does not exist
    if not os.path.exists(file_path):
        write_file(file_path, failure_msg)
    else:
        logger.info("Failure file exists. Skipping creation....")


@contextlib.contextmanager
def tmpdir(suffix="", prefix="tmp", directory=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the
    context exits.

    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix,
                       otherwise there will be no suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix;
                       otherwise, a default prefix is used.
        directory (str):  If directory is specified, the file will be created in that directory;
                    otherwise, a default directory is used.
    Returns:
        str: Path to the directory.
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory)
    yield tmp
    shutil.rmtree(tmp)


def write_file(path, data, mode="w"):  # type: (str, str, str) -> None
    """Write data to a file.

    Args:
        path (str): Path to the file.
        data (str): Data to be written to the file.
        mode (str): Mode which the file will be open.
    """
    with open(path, mode) as f:
        f.write(data)


def read_file(path, mode="r"):
    """Read data from a file.

    Args:
        path (str): Path to the file.
        mode (str): mode which the file will be open.

    Returns:
    """
    with open(path, mode) as f:
        return f.read()


def read_json(path):  # type: (str) -> dict
    """Read a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        (dict[object, object]): A dictionary representation of the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def download_and_extract(uri, path):  # type: (str, str) -> None
    """Download, prepare and install a compressed tar file from S3 or local directory as
    an entry point.

    SageMaker Python SDK saves the user provided entry points as compressed tar files in S3

    Args:
        uri (str): the location of the entry point.
        path (bool): The path where the script will be installed. It will not download and
                     install the if the path already has the user entry point.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.listdir(path):
        with tmpdir() as tmp:
            if uri.startswith("s3://"):
                dst = os.path.join(tmp, "tar_file")
                s3_download(uri, dst)

                with tarfile.open(name=dst, mode="r:gz") as t:
                    t.extractall(path=path)

            elif os.path.isdir(uri):
                if uri == path:
                    return
                if os.path.exists(path):
                    shutil.rmtree(path)
                shutil.copytree(uri, path)
            elif tarfile.is_tarfile(uri):
                with tarfile.open(name=uri, mode="r:gz") as t:
                    t.extractall(path=path)
            else:
                shutil.copy2(uri, path)


def s3_download(url, dst):  # type: (str, str) -> None
    """Download a file from S3.

    Args:
        url (str): the s3 url of the file.
        dst (str): the destination where the file will be saved.
    """
    url = parse.urlparse(url)

    if url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    bucket, key = url.netloc, url.path.lstrip("/")

    region = os.environ.get("AWS_REGION", os.environ.get(params.REGION_NAME_ENV))
    endpoint_url = os.environ.get(params.S3_ENDPOINT_URL, None)
    s3 = boto3.resource("s3", region_name=region, endpoint_url=endpoint_url)

    s3.Bucket(bucket).download_file(key, dst)
