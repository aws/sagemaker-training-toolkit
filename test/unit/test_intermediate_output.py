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

from inotify_simple import Event, flags
from mock import MagicMock, patch
import pytest

from sagemaker_containers import _env, _files, _intermediate_output

REGION = 'us-west'
S3_BUCKET = 's3://mybucket/'


def test_accept_file_output_no_process():
    intemediate_sync = _intermediate_output.start_sync(
        'file://my/favorite/file', REGION)
    assert intemediate_sync is None


def test_wrong_output():
    with pytest.raises(ValueError) as e:
        _intermediate_output.start_sync('tcp://my/favorite/url', REGION)
    assert 'Expecting \'s3\' scheme' in str(e)


@patch('inotify_simple.INotify', MagicMock())
def test_daemon_process():
    intemediate_sync = _intermediate_output.start_sync(S3_BUCKET, REGION)
    assert intemediate_sync.daemon is True


@patch('boto3.client', MagicMock())
@patch('shutil.copy2')
@patch('inotify_simple.INotify')
@patch('boto3.s3.transfer.S3Transfer.upload_file')
@patch('multiprocessing.Process')
def test_non_write_ignored(process_mock, upload_file, inotify_mock, copy2):
    process = process_mock.return_value
    inotify = inotify_mock.return_value

    inotify.add_watch.return_value = 1
    mask = flags.CREATE
    for flag in flags:
        if flag is not flags.CLOSE_WRITE and flag is not flags.ISDIR:
            mask = mask | flag
    inotify.read.return_value = [Event(1, mask, 'cookie', 'file_name')]

    def watch():
        call = process_mock.call_args
        args, kwargs = call
        _intermediate_output._watch(kwargs['args'][0], kwargs['args'][1],
                                    kwargs['args'][2], kwargs['args'][3])

    process.start.side_effect = watch

    _files.write_success_file()
    _intermediate_output.start_sync(S3_BUCKET, REGION)

    inotify.add_watch.assert_called()
    inotify.read.assert_called()
    copy2.assert_not_called()
    upload_file.assert_not_called()


@patch('boto3.client', MagicMock())
@patch('shutil.copy2')
@patch('inotify_simple.INotify')
@patch('boto3.s3.transfer.S3Transfer.upload_file')
@patch('multiprocessing.Process')
def test_modification_triggers_upload(process_mock, upload_file, inotify_mock, copy2):
    process = process_mock.return_value
    inotify = inotify_mock.return_value

    inotify.add_watch.return_value = 1
    inotify.read.return_value = [Event(1, flags.CLOSE_WRITE, 'cookie', 'file_name')]

    def watch():
        call = process_mock.call_args
        args, kwargs = call
        _intermediate_output._watch(kwargs['args'][0], kwargs['args'][1],
                                    kwargs['args'][2], kwargs['args'][3])

    process.start.side_effect = watch

    _files.write_success_file()
    _intermediate_output.start_sync(S3_BUCKET, REGION)

    inotify.add_watch.assert_called()
    inotify.read.assert_called()
    copy2.assert_called()
    upload_file.assert_called()


@patch('boto3.client', MagicMock())
@patch('shutil.copy2')
@patch('inotify_simple.INotify')
@patch('boto3.s3.transfer.S3Transfer.upload_file')
@patch('multiprocessing.Process')
def test_new_folders_are_watched(process_mock, upload_file, inotify_mock, copy2):
    process = process_mock.return_value
    inotify = inotify_mock.return_value

    new_dir = 'new_dir'
    new_dir_path = os.path.join(_env.output_intermediate_dir, new_dir)
    inotify.add_watch.return_value = 1
    inotify.read.return_value = [Event(1, flags.CREATE | flags.ISDIR, 'cookie', new_dir)]

    def watch():
        os.makedirs(new_dir_path)

        call = process_mock.call_args
        args, kwargs = call
        _intermediate_output._watch(kwargs['args'][0], kwargs['args'][1],
                                    kwargs['args'][2], kwargs['args'][3])

    process.start.side_effect = watch

    _files.write_success_file()
    _intermediate_output.start_sync(S3_BUCKET, REGION)

    watch_flags = flags.CLOSE_WRITE | flags.CREATE
    inotify.add_watch.assert_any_call(_env.output_intermediate_dir, watch_flags)
    inotify.add_watch.assert_any_call(new_dir_path, watch_flags)
    inotify.read.assert_called()
    copy2.assert_not_called()
    upload_file.assert_not_called()
