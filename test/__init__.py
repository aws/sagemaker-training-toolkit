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
import collections
import json
import logging
import os
import tarfile
import tempfile
import time

import boto3
import pytest
import sagemaker
import six

# loading base path before loading the environment so all the environment paths are loaded properly
os.environ['BASE_PATH'] = os.path.join(tempfile.mkdtemp(), 'opt', 'ml')
DEFAULT_REGION = 'us-west-2'

from sagemaker_containers import env  # noqa ignore=E402 module level import not at top of file

DEFAULT_CONFIG = dict(ContentType="application/x-numpy", TrainingInputMode="File",
                      S3DistributionType="FullyReplicated", RecordWrapperType="None")

DEFAULT_HYPERPARAMETERS = dict(sagemaker_region='us-west-2', sagemaker_job_name='sagemaker-training-job',
                               sagemaker_enable_cloudwatch_metrics=False, sagemaker_container_log_level=logging.WARNING)


def sagemaker_session(region_name=DEFAULT_REGION):  # type: (str) -> sagemaker.Session
    return sagemaker.Session(boto3.Session(region_name=region_name))


def default_bucket(session=None):  # type: (sagemaker.Session) -> str
    session = session or sagemaker_session()
    return session.default_bucket()


def write_json(obj, path):  # type: (object, str) -> None
    """Serialize ``obj`` as JSON in the ``path`` file.

    Args:
        obj (object): Object to be serialized
        path (str): Path to JSON file
    """
    with open(path, 'w') as f:
        json.dump(obj, f)


def prepare(user_module, hyperparameters, channels, current_host='algo-1', hosts=None):
    # type: (UserModule, dict, list, str, list) -> None
    hosts = hosts or ['algo-1']

    user_module.upload()

    create_hyperparameters_config(hyperparameters, user_module.url)
    create_resource_config(current_host, hosts)
    create_input_data_config(channels)


def hyperparameters(**kwargs):  # type: (...) -> dict
    default_hyperparameters = DEFAULT_HYPERPARAMETERS.copy()

    default_hyperparameters.update(kwargs)
    return default_hyperparameters


def create_resource_config(current_host, hosts):  # type: (str, list) -> None
    write_json(dict(current_host=current_host, hosts=hosts), env.RESOURCE_CONFIG_PATH)


def create_input_data_config(channels):  # type: (list) -> None
    input_data_config = {channel.name: channel.config for channel in channels}

    write_json(input_data_config, env.INPUT_DATA_CONFIG_FILE_PATH)


def create_hyperparameters_config(hyperparameters, submit_dir, sagemaker_hyperparameters=None):
    # type: (dict, str, dict) -> None
    all_hyperparameters = {env.SUBMIT_DIR_PARAM: submit_dir}
    all_hyperparameters.update(sagemaker_hyperparameters or DEFAULT_HYPERPARAMETERS.copy())
    all_hyperparameters.update(hyperparameters)

    write_json(all_hyperparameters, env.HYPERPARAMETERS_PATH)


File = collections.namedtuple('File', ['name', 'content'])  # type: (str, str or list) -> File


class UserModule(object):

    def __init__(self, main_file, key=None, bucket=None, session=None):
        # type: (File, str, str, sagemaker.Session) -> None
        session = session or sagemaker_session()
        self._s3 = session.boto_session.resource('s3')
        self.bucket = bucket or default_bucket(session)
        self.key = key or os.path.join('test', 'sagemaker-containers', str(time.time()), 'sourcedir.tar.gz')
        self._files = [main_file]

    def add_file(self, file):  # type: (File) -> UserModule
        self._files.append(file)
        return self

    @property
    def url(self):  # type: () -> str
        return os.path.join('s3://', self.bucket, self.key)

    def upload(self):  # type: () -> UserModule
        with env.tmpdir() as tmpdir:
            tar_name = os.path.join(tmpdir, 'sourcedir.tar.gz')
            with tarfile.open(tar_name, mode='w:gz') as tar:
                for _file in self._files:
                    name = os.path.join(tmpdir, _file.name)
                    with open(name, 'w+') as f:

                        if isinstance(_file.content, six.string_types):
                            content = _file.content
                        else:
                            content = '\n'.join(_file.content)

                        f.write(content)
                    tar.add(name=name, arcname=_file.name)

            self._s3.Object(self.bucket, self.key).upload_file(tar_name)
        return self


class Channel(collections.namedtuple('Channel', ['name', 'config'])):  # type: (str, dict) -> Channel

    def __new__(cls, name, config=None):
        config = DEFAULT_CONFIG.copy().update(config or {})
        return super(Channel, cls).__new__(cls, name=name, config=config)

    @staticmethod
    def create(name, config=None):  # type: (str, dict) -> Channel
        channel = Channel(name, config)
        channel.make_directory()
        return channel

    def make_directory(self):  # type: () -> None
        os.makedirs(self.path)

    @property
    def path(self):  # type: () -> str
        return os.path.join(env.INPUT_DATA_PATH, self.name)


class TestBase(object):
    patches = []

    @pytest.fixture(autouse=True)
    def set_up(self):

        for _patch in self.patches:
            _patch.start()

        yield

        for _patch in self.patches:
            _patch.stop()
