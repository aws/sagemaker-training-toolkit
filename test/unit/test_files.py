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
    extractall.assert_called_once()
    kwargs = extractall.call_args[1]
    assert kwargs["path"] == environment.code_dir
    if hasattr(tarfile, "data_filter"):
        # Members are sanitized through a callable filter, which also applies the
        # stdlib data filter for metadata hardening.
        assert callable(kwargs["filter"])
    else:
        # `filter` is not accepted before 3.12, so a sanitized member list is passed.
        assert "filter" not in kwargs
        assert "members" in kwargs

    os.remove(uri)


def _tar_with_member(tar_path, arcname, payload_dir, extra=None):
    """Build a tar containing a benign entry plus one member named `arcname`."""
    benign = os.path.join(payload_dir, "train.sh")
    with open(benign, "w") as f:
        f.write("#!/bin/sh\necho ok\n")
    os.chmod(benign, 0o755)
    with tarfile.open(tar_path, "w:gz") as t:
        t.add(benign, arcname="train.sh")
        if arcname:
            t.add(benign, arcname=arcname)
        for name in extra or []:
            t.add(benign, arcname=name)


def _no_stray_files_outside(root, extract_dir):
    """Every file under `root` must live inside `extract_dir` (or be input we created)."""
    strays = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if full.startswith(extract_dir + os.sep):
                continue
            if "payload" in full or full.endswith(".tar.gz"):
                continue  # test inputs
            strays.append(full)
    return strays


def test_download_and_extract_normalizes_path_traversal(tmpdir):
    """A ../ member is re-anchored inside the destination instead of escaping it."""
    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    marker_name = "ESCAPED.txt"
    marker = os.path.join(str(tmpdir), marker_name)

    depth = len(os.path.abspath(extract_dir).strip(os.sep).split(os.sep))
    traversal = os.path.join(*([os.pardir] * depth)) + marker
    tar_path = os.path.join(str(tmpdir), "evil.tar.gz")
    _tar_with_member(tar_path, traversal, payload_dir)

    # Normalization means the archive still extracts -- no job-breaking exception.
    files.download_and_extract(tar_path, extract_dir)

    assert not os.path.exists(marker), "path traversal escaped the destination directory"
    assert not _no_stray_files_outside(str(tmpdir), extract_dir)
    # The member landed inside, with the traversal segments stripped.
    relocated = os.path.join(extract_dir, *marker.strip(os.sep).split(os.sep))
    assert os.path.exists(relocated), "normalized member was not extracted inside the dest"
    assert os.path.exists(os.path.join(extract_dir, "train.sh"))


def test_download_and_extract_normalizes_absolute_member(tmpdir):
    """An absolute-path member is re-anchored inside the destination."""
    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    marker = os.path.join(str(tmpdir), "ABS_ESCAPED.txt")

    benign = os.path.join(payload_dir, "train.sh")
    with open(benign, "w") as f:
        f.write("#!/bin/sh\necho ok\n")
    tar_path = os.path.join(str(tmpdir), "abs.tar.gz")
    with tarfile.open(tar_path, "w:gz") as t:
        info = t.gettarinfo(benign, arcname=marker)
        with open(benign, "rb") as fh:
            t.addfile(info, fh)

    files.download_and_extract(tar_path, extract_dir)

    assert not os.path.exists(marker), "absolute path member escaped the destination"
    relocated = os.path.join(extract_dir, *marker.strip(os.sep).split(os.sep))
    assert os.path.exists(relocated), "normalized absolute member was not extracted inside"


def test_download_and_extract_preserves_executable_bit(tmpdir):
    """The hardened extraction must not strip the exec bit off a shell entry point."""
    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    tar_path = os.path.join(str(tmpdir), "benign.tar.gz")
    _tar_with_member(tar_path, None, payload_dir, extra=["nested/dir/helper.sh"])

    files.download_and_extract(tar_path, extract_dir)

    extracted = os.path.join(extract_dir, "train.sh")
    assert os.path.exists(extracted)
    assert os.access(extracted, os.X_OK), "entry point lost its executable bit"
    assert os.path.exists(os.path.join(extract_dir, "nested", "dir", "helper.sh"))


def test_download_and_extract_skips_symlink_escape(tmpdir):
    """A symlink pointing outside the destination is dropped, not followed."""
    extract_dir = os.path.join(str(tmpdir), "code")
    tar_path = os.path.join(str(tmpdir), "symlink.tar.gz")

    info = tarfile.TarInfo("escape")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    with tarfile.open(tar_path, "w:gz") as t:
        t.addfile(info)

    files.download_and_extract(tar_path, extract_dir)

    assert not os.path.exists(os.path.join(extract_dir, "escape")), "escaping symlink extracted"


def test_download_and_extract_normalizes_dot_dot_only_member(tmpdir):
    """A member named only of traversal segments normalizes away and is skipped."""
    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    tar_path = os.path.join(str(tmpdir), "dots.tar.gz")
    _tar_with_member(tar_path, "../..", payload_dir)

    files.download_and_extract(tar_path, extract_dir)

    assert os.path.exists(os.path.join(extract_dir, "train.sh"))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("train.sh", "train.sh"),
        ("nested/dir/train.sh", "nested/dir/train.sh"),
        ("../../etc/passwd", "etc/passwd"),
        ("/etc/passwd", "etc/passwd"),
        ("./train.sh", "train.sh"),
        ("a/../../b/train.sh", "a/b/train.sh"),
        ("..", ""),
        ("../..", ""),
        ("", ""),
        ("..\\..\\windows\\evil.txt", "windows/evil.txt"),
    ],
)
def test_normalize_member_name(name, expected):
    """Normalization always yields a relative path anchored inside the destination."""
    assert files._normalize_member_name(name) == expected


# --- pre-3.12 fallback: `filter` is not accepted, members are sanitized directly ---


def test_fallback_normalizes_path_traversal(tmpdir, monkeypatch):
    """Force the pre-3.12 code path and confirm traversal is still contained.

    Interpreters without tarfile.data_filter (Python < 3.8.17/3.9.17/3.10.12/3.11.4)
    cannot pass filter=, so _safe_extractall sanitizes the member list instead. This
    test removes data_filter so that branch runs on any interpreter.
    """
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    assert not hasattr(tarfile, "data_filter")

    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    marker = os.path.join(str(tmpdir), "FALLBACK_ESCAPED.txt")

    depth = len(os.path.abspath(extract_dir).strip(os.sep).split(os.sep))
    traversal = os.path.join(*([os.pardir] * depth)) + marker
    tar_path = os.path.join(str(tmpdir), "evil-fallback.tar.gz")
    _tar_with_member(tar_path, traversal, payload_dir)

    files.download_and_extract(tar_path, extract_dir)

    assert not os.path.exists(marker), "fallback path let traversal escape"
    relocated = os.path.join(extract_dir, *marker.strip(os.sep).split(os.sep))
    assert os.path.exists(relocated)


def test_fallback_allows_benign(tmpdir, monkeypatch):
    """The pre-3.12 fallback must not reject or mangle ordinary archives."""
    monkeypatch.delattr(tarfile, "data_filter", raising=False)

    extract_dir = os.path.join(str(tmpdir), "code")
    payload_dir = os.path.join(str(tmpdir), "payload")
    os.makedirs(payload_dir)
    tar_path = os.path.join(str(tmpdir), "benign-fallback.tar.gz")
    _tar_with_member(tar_path, None, payload_dir, extra=["nested/dir/helper.sh"])

    files.download_and_extract(tar_path, extract_dir)

    assert os.path.exists(os.path.join(extract_dir, "train.sh"))
    assert os.path.exists(os.path.join(extract_dir, "nested", "dir", "helper.sh"))
    assert os.access(os.path.join(extract_dir, "train.sh"), os.X_OK)


def test_fallback_skips_symlink_escape(tmpdir, monkeypatch):
    """The pre-3.12 fallback must drop symlinks pointing outside the destination."""
    monkeypatch.delattr(tarfile, "data_filter", raising=False)

    extract_dir = os.path.join(str(tmpdir), "code")
    tar_path = os.path.join(str(tmpdir), "symlink-fallback.tar.gz")

    info = tarfile.TarInfo("escape")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    with tarfile.open(tar_path, "w:gz") as t:
        t.addfile(info)

    files.download_and_extract(tar_path, extract_dir)

    assert not os.path.exists(os.path.join(extract_dir, "escape"))
