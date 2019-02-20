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
from __future__ import absolute_import

import concurrent.futures as futures
import multiprocessing
import os
import shutil
import time

import boto3
import boto3.s3.transfer as s3transfer
import inotify_simple
from six.moves.urllib.parse import urlparse

from sagemaker_containers import _env, _logging

logger = _logging.get_logger()

intermediate_path = _env.output_intermediate_dir  # type: str
failure_file_path = os.path.join(_env.output_dir, 'failure')  # type: str
success_file_path = os.path.join(_env.output_dir, 'success')  # type: str
tmp_dir_path = os.path.join(intermediate_path, '.tmp.sagemaker_s3_sync')  # type: str


def _timestamp():
    """Return a timestamp with microsecond precision."""
    moment = time.time()
    moment_us = repr(moment).split('.')[1]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_us), time.gmtime(moment))


def _upload_to_s3(s3_uploader, relative_path, file_path, filename):
    try:
        key = os.path.join(s3_uploader['key_prefix'], relative_path, filename)
        s3_uploader['transfer'].upload_file(file_path, s3_uploader['bucket'], key)
    except FileNotFoundError:  # noqa ignore=F821
        # Broken link or deleted
        pass
    except Exception:
        logger.exception('Failed to upload file to s3.')
    finally:
        # delete the original file
        if os.path.exists(file_path):
            os.remove(file_path)


def _copy_file(executor, s3_uploader, relative_path, filename):
    try:
        src = os.path.join(intermediate_path, relative_path, filename)
        dst = os.path.join(tmp_dir_path, relative_path, '{}.{}'.format(_timestamp(), filename))
        shutil.copy2(src, dst)
        executor.submit(_upload_to_s3, s3_uploader, relative_path, dst, filename)
    except FileNotFoundError:  # noqa ignore=F821
        # Broken link or deleted
        pass
    except Exception:
        logger.exception('Failed to copy file to the temporarily directory.')


def _watch(inotify, watchers, watch_flags, s3_uploader):
    """As soon as a user is done with a file under `/opt/ml/output/intermediate`
    we would get notified by using inotify. We would copy this file under
    `/opt/ml/output/intermediate/.tmp.sagemaker_s3_sync` folder preserving
    the same folder structure to prevent it from being further modified.
    As we copy the file we would add timestamp with microseconds precision
    to avoid modification during s3 upload.
    After that we copy the file to s3 in a separate Thread.
    We keep the queue of the files we need to move as FIFO.
    """
    # initialize a thread pool with 1 worker
    # to be used for uploading files to s3 in a separate thread
    executor = futures.ThreadPoolExecutor(max_workers=1)

    last_pass_done = False
    stop_file_exists = False

    # after we see stop file do one additional pass to make sure we didn't miss anything
    while not last_pass_done:
        # wait for any events in the directory for 1 sec and then re-check exit conditions
        for event in inotify.read(timeout=1000):
            for flag in inotify_simple.flags.from_mask(event.mask):
                # if new directory was created traverse the directory tree to recursively add all
                # created folders to the watchers list.
                # Upload files to s3 if there any files.
                # There is a potential race condition if upload the file and the see a notification
                # for it which should cause any problems because when we copy files to temp dir
                # we add a unique timestamp up to microseconds.
                if flag is inotify_simple.flags.ISDIR and inotify_simple.flags.CREATE & event.mask:
                    path = os.path.join(intermediate_path, watchers[event.wd], event.name)
                    for folder, dirs, files in os.walk(path):
                        wd = inotify.add_watch(folder, watch_flags)
                        relative_path = os.path.relpath(folder, intermediate_path)
                        watchers[wd] = relative_path
                        tmp_sub_folder = os.path.join(tmp_dir_path, relative_path)
                        if not os.path.exists(tmp_sub_folder):
                            os.makedirs(tmp_sub_folder)
                        for file in files:
                            _copy_file(executor, s3_uploader, relative_path, file)
                elif flag is inotify_simple.flags.CLOSE_WRITE:
                    _copy_file(executor, s3_uploader, watchers[event.wd], event.name)

        last_pass_done = stop_file_exists
        stop_file_exists = os.path.exists(success_file_path) or os.path.exists(failure_file_path)

    # wait for all the s3 upload tasks to finish and shutdown the executor
    executor.shutdown(wait=True)


def start_sync(s3_output_location, region):
    """Starts intermediate folder sync which copies files from 'opt/ml/output/intermediate'
    directory to the provided s3 output location as files created or modified.
    If files are deleted it doesn't delete them from s3.

    It starts intermediate folder behavior as a daemonic process and
    only if the directory doesn't exists yet, if it does - it indicates
    that platform is taking care of syncing files to S3 and container should not interfere.

    Args:
        s3_output_location (str): name of the script or module.
        region (str): the location of the module.

    Returns:
        (multiprocessing.Process): the intermediate output sync daemonic process.
    """
    if not s3_output_location or os.path.exists(intermediate_path):
        logger.debug('Could not initialize intermediate folder sync to s3.')
        return

    # create intermediate and intermediate_tmp directories
    os.makedirs(intermediate_path)
    os.makedirs(tmp_dir_path)

    # configure unique s3 output location similar to how SageMaker platform does it
    # or link it to the local output directory
    url = urlparse(s3_output_location)
    if url.scheme == 'file':
        logger.debug('Local directory is used for output. No need to sync any intermediate output.')
        return
    elif url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    # create s3 transfer client
    client = boto3.client('s3', region)
    s3_transfer = s3transfer.S3Transfer(client)
    s3_uploader = {
        'transfer': s3_transfer,
        'bucket': url.netloc,
        'key_prefix': os.path.join(url.path.lstrip('/'), os.environ.get('TRAINING_JOB_NAME', ''),
                                   'output', 'intermediate'),
    }

    # Add intermediate folder to the watch list
    inotify = inotify_simple.INotify()
    watch_flags = inotify_simple.flags.CLOSE_WRITE | inotify_simple.flags.CREATE
    watchers = {}
    wd = inotify.add_watch(intermediate_path, watch_flags)
    watchers[wd] = ''

    # start subprocess to sync any files from intermediate folder to s3
    p = multiprocessing.Process(target=_watch, args=[inotify, watchers, watch_flags, s3_uploader])
    # Make the process daemonic as a safety switch to prevent training job from hanging forever
    # in case if something goes wrong and main container process exits in an unexpected way
    p.daemon = True
    p.start()
    return p
