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
import time

import boto3
from botocore.exceptions import ClientError
import numpy as np

from sagemaker_containers import _env, _files, _intermediate_output
import test

intermediate_path = _env.output_intermediate_dir
bucket = test.default_bucket()
bucket_uri = 's3://{}'.format(bucket)
region = test.DEFAULT_REGION


def _timestamp():
    return time.strftime("%Y-%m-%d-%H-%M-%S".format(time.gmtime(time.time())))


def test_intermediate_upload():
    os.environ['TRAINING_JOB_NAME'] = _timestamp()
    p = _intermediate_output.start_sync(bucket_uri, region)

    file1 = os.path.join(intermediate_path, 'file1.txt')
    write_file(file1, 'file1!')

    os.makedirs(os.path.join(intermediate_path, 'dir1', 'dir2', 'dir3'))

    dir1 = os.path.join(intermediate_path, 'dir1')
    dir2 = os.path.join(dir1, 'dir2')
    dir3 = os.path.join(dir2, 'dir3')
    file2 = os.path.join(dir1, 'file2.txt')
    file3 = os.path.join(dir2, 'file3.txt')
    file4 = os.path.join(dir3, 'file4.txt')
    write_file(file2, 'dir1_file2!')
    write_file(file3, 'dir2_file3!')
    write_file(file4, 'dir1_file4!')

    dir_to_delete1 = os.path.join(dir1, 'dir4')
    file_to_delete1 = os.path.join(dir_to_delete1, 'file_to_delete1.txt')
    os.makedirs(dir_to_delete1)
    write_file(file_to_delete1, 'file_to_delete1!')
    os.remove(file_to_delete1)
    os.removedirs(dir_to_delete1)

    file_to_delete2_but_copy = os.path.join(intermediate_path, 'file_to_delete2_but_copy.txt')
    write_file(file_to_delete2_but_copy, 'file_to_delete2!')
    time.sleep(1)
    os.remove(file_to_delete2_but_copy)

    file_to_modify1 = os.path.join(dir3, 'file_to_modify1.txt')
    write_file(file_to_modify1, 'dir3_file_to_modify1_1!')
    write_file(file_to_modify1, 'dir3_file_to_modify1_2!')
    write_file(file_to_modify1, 'dir3_file_to_modify1_3!')
    content_to_assert = 'dir3_file_to_modify1_4!'
    write_file(file_to_modify1, content_to_assert)

    # the last file to be moved
    file5 = os.path.join(intermediate_path, 'file5.txt')
    write_file(file5, 'file5!')

    _files.write_success_file()

    p.join()

    # shouldn't be moved
    file6 = os.path.join(intermediate_path, 'file6.txt')
    write_file(file6, 'file6!')

    # assert that all files that should be under intermediate are still there
    assert os.path.exists(file1)
    assert os.path.exists(file2)
    assert os.path.exists(file3)
    assert os.path.exists(file4)
    assert os.path.exists(file5)
    assert os.path.exists(file6)
    assert os.path.exists(file_to_modify1)
    # and all the deleted folders and files aren't there
    assert not os.path.exists(dir_to_delete1)
    assert not os.path.exists(file_to_delete1)
    assert not os.path.exists(file_to_delete2_but_copy)

    # assert files exist in S3
    key_prefix = os.path.join(os.environ.get('TRAINING_JOB_NAME'), 'output', 'intermediate')
    client = boto3.client('s3', region)
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file1, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file2, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file3, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file4, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file5, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file_to_modify1, intermediate_path)))
    deleted_file = os.path.join(key_prefix,
                                os.path.relpath(file_to_delete2_but_copy, intermediate_path))
    assert _file_exists_in_s3(client, deleted_file)
    assert not _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(dir_to_delete1, intermediate_path)))
    assert not _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file_to_delete1, intermediate_path)))
    assert not _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file6, intermediate_path)))

    # check that modified file has
    s3 = boto3.resource('s3', region_name=region)
    key = os.path.join(key_prefix, os.path.relpath(file_to_modify1, intermediate_path))
    modified_file = os.path.join(_env.output_dir, 'modified_file.txt')
    s3.Bucket(bucket).download_file(key, modified_file)
    with open(modified_file) as f:
        content = f.read()
        assert content == content_to_assert


def test_nested_delayed_file():
    os.environ['TRAINING_JOB_NAME'] = _timestamp()
    p = _intermediate_output.start_sync(bucket_uri, region)

    os.makedirs(os.path.join(intermediate_path, 'dir1'))
    dir1 = os.path.join(intermediate_path, 'dir1')

    time.sleep(3)

    os.makedirs(os.path.join(dir1, 'dir2'))
    dir2 = os.path.join(dir1, 'dir2')

    time.sleep(3)

    file1 = os.path.join(dir2, 'file1.txt')
    write_file(file1, 'file1')

    os.makedirs(os.path.join(intermediate_path, 'dir3'))
    dir3 = os.path.join(intermediate_path, 'dir3')

    time.sleep(3)

    file2 = os.path.join(dir3, 'file2.txt')
    write_file(file2, 'file2')

    _files.write_success_file()
    p.join()

    # assert that all files that should be under intermediate are still there
    assert os.path.exists(file1)
    assert os.path.exists(file2)

    # assert file exist in S3
    key_prefix = os.path.join(os.environ.get('TRAINING_JOB_NAME'), 'output', 'intermediate')
    client = boto3.client('s3', region)
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file1, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file2, intermediate_path)))


def test_large_files():
    os.environ['TRAINING_JOB_NAME'] = _timestamp()
    p = _intermediate_output.start_sync(bucket_uri, region)

    file_size = 1024 * 256 * 17  # 17MB

    file = os.path.join(intermediate_path, 'file.npy')
    _generate_large_npy_file(file_size, file)

    file_to_modify = os.path.join(intermediate_path, 'file_to_modify.npy')
    _generate_large_npy_file(file_size, file_to_modify)
    content_to_assert = _generate_large_npy_file(file_size, file_to_modify)

    _files.write_failure_file('Failure!!')
    p.join()

    assert os.path.exists(file)
    assert os.path.exists(file_to_modify)

    key_prefix = os.path.join(os.environ.get('TRAINING_JOB_NAME'), 'output', 'intermediate')
    client = boto3.client('s3', region)
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file, intermediate_path)))
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file_to_modify, intermediate_path)))

    # check that modified file has
    s3 = boto3.resource('s3', region_name=region)
    key = os.path.join(key_prefix, os.path.relpath(file_to_modify, intermediate_path))
    modified_file = os.path.join(_env.output_dir, 'modified_file.npy')
    s3.Bucket(bucket).download_file(key, modified_file)
    assert np.array_equal(np.load(modified_file), content_to_assert)


def write_file(path, data, mode='w'):
    with open(path, mode) as f:
        f.write(data)


def _generate_large_npy_file(size, file_path):
    letters = np.array(list(chr(ord('a') + i) for i in range(26)))
    content = np.random.choice(letters, size)
    np.save(file_path, content)
    return content


def _file_exists_in_s3(client, key):
    """return the key's size if it exist, else None"""
    try:
        obj = client.head_object(Bucket=bucket, Key=key)
        return obj['ContentLength']
    except ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise
