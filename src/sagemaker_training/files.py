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


def _normalize_member_name(name):  # type: (str) -> str
    """Rewrite an archive member name so it cannot escape its extraction directory.

    Strips any drive letter, leading separators and ``.``/``..`` segments, leaving a
    relative path that is always anchored inside the destination. ``../../etc/passwd``
    becomes ``etc/passwd``; ``/etc/passwd`` becomes ``etc/passwd``.

    Args:
        name (str): The member name as recorded in the archive.

    Returns:
        str: A relative path safe to join onto the destination directory, or an
            empty string if nothing is left after normalization.
    """
    # Archives are POSIX-separated, but a Windows-authored tar may carry backslashes
    # or a drive letter; treat both separators so neither survives normalization.
    candidate = name.replace("\\", "/")
    candidate = os.path.splitdrive(candidate)[1]

    safe_segments = []
    for segment in candidate.split("/"):
        if segment in ("", ".", os.pardir):
            continue
        safe_segments.append(segment)
    return "/".join(safe_segments)


def _sanitize_member(member, destination):
    # type: (tarfile.TarInfo, str) -> tarfile.TarInfo
    """Normalize a member's path and drop it if it still cannot be extracted safely.

    Two layers, in order:

    1. Normalization -- the member name is rewritten to stay inside ``destination``,
       so traversal cannot escape by construction rather than by rejection.
    2. Validation -- the normalized target is re-checked against ``destination``, and
       members that cannot be represented safely (special files, links pointing
       outside, names that normalize away entirely) are skipped.

    Args:
        member (tarfile.TarInfo): The member to sanitize. Mutated in place.
        destination (str): The real path of the extraction directory.

    Returns:
        tarfile.TarInfo: The sanitized member, or ``None`` to skip it.
    """
    original_name = member.name
    normalized = _normalize_member_name(original_name)

    if not normalized:
        logger.warning(
            "Skipping archive member %r: no usable path remains after normalization.",
            original_name,
        )
        return None

    if normalized != original_name:
        logger.warning(
            "Archive member %r would have been extracted outside the code directory; "
            "normalized to %r.",
            original_name,
            normalized,
        )
    member.name = normalized

    # Layer 2: nothing should escape after normalization -- verify rather than assume.
    target = os.path.realpath(os.path.join(destination, member.name))
    if target != destination and not target.startswith(destination + os.sep):
        logger.warning(
            "Skipping archive member %r: it still resolves outside the code directory.",
            original_name,
        )
        return None

    # Device and FIFO members have no legitimate place in a source bundle.
    if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
        logger.warning(
            "Skipping archive member %r: special files are not supported.", original_name
        )
        return None

    if member.issym() or member.islnk():
        link_base = destination if member.islnk() else os.path.dirname(target)
        link_target = os.path.realpath(os.path.join(link_base, member.linkname))
        if link_target != destination and not link_target.startswith(destination + os.sep):
            logger.warning(
                "Skipping archive member %r: link target %r points outside the code directory.",
                original_name,
                member.linkname,
            )
            return None

    return member


def _safe_extractall(tar, path):  # type: (tarfile.TarFile, str) -> None
    """Extract a tar archive, normalizing member paths so none escape ``path``.

    Implements the two-layer defence agreed on P427358576: normalize each member's
    path so traversal cannot escape, then validate the normalized result. A hostile
    archive extracts inside the code directory with a warning rather than failing the
    job, so a malformed customer bundle does not become a training failure.

    Where the interpreter provides it (Python 3.12, backported to 3.8.17, 3.9.17,
    3.10.12 and 3.11.4), the stdlib ``data`` filter also runs, which strips setuid,
    setgid, sticky bits and ownership metadata. On older interpreters the member list
    is sanitized directly, since ``filter`` is not accepted there.

    Args:
        tar (tarfile.TarFile): The open archive.
        path (str): The directory members must stay within.
    """
    destination = os.path.realpath(path)

    if hasattr(tarfile, "data_filter"):

        def _filter(member, dest_path):
            sanitized = _sanitize_member(member, destination)
            if sanitized is None:
                return None
            # Metadata hardening from the stdlib filter. The path is already safe, so
            # the traversal checks inside data_filter cannot trip on our own output.
            return tarfile.data_filter(sanitized, dest_path)

        tar.extractall(path=path, filter=_filter)
        return

    sanitized_members = []
    for member in tar.getmembers():
        sanitized = _sanitize_member(member, destination)
        if sanitized is not None:
            sanitized_members.append(sanitized)
    tar.extractall(path=path, members=sanitized_members)


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
        logger.info(f"Provided path: {path}  is empty, unzipping")
        with tmpdir() as tmp:
            if uri.startswith("s3://"):
                dst = os.path.join(tmp, "tar_file")
                s3_download(uri, dst)

                with tarfile.open(name=dst, mode="r:gz") as t:
                    _safe_extractall(t, path)

            elif os.path.isdir(uri):
                if uri == path:
                    return
                if os.path.exists(path):
                    shutil.rmtree(path)
                shutil.copytree(uri, path)
            elif tarfile.is_tarfile(uri):
                with tarfile.open(name=uri, mode="r:gz") as t:
                    _safe_extractall(t, path)
            else:
                shutil.copy2(uri, path)
    else:
        logger.info(f"Provided path: {path} is not empty, abandoning unzipping sourcedir.tar.gz")


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
