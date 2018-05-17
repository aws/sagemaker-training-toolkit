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

import importlib
import logging
import os
import shlex
import subprocess
import sys
import tarfile
import textwrap

import boto3
import six
from six.moves.urllib.parse import urlparse

from sagemaker_containers import env, errors

logger = logging.getLogger(__name__)

DEFAULT_MODULE_NAME = 'default_user_module_name'


def s3_download(url, dst):  # type: (str, str) -> None
    """Download a file from S3.

    Args:
        url (str): the s3 url of the file.
        dst (str): the destination where the file will be saved.
    """
    url = urlparse(url)

    if url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, dst))

    bucket, key = url.netloc, url.path.lstrip('/')

    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(key, dst)


def prepare(path, name):  # type: (str, str) -> None
    """Prepare a Python script (or module) to be imported as a module.

    If the script does not contain a setup.py file, it creates a minimal setup.

    Args:
        path (str): path to directory with the script or module.
        name (str): name of the script or module.
    """
    setup_path = os.path.join(path, 'setup.py')
    if not os.path.exists(setup_path):
        data = textwrap.dedent("""
        from setuptools import setup

        setup(packages=[''],
              name="%s",
              version='1.0.0',
              include_package_data=True)
        """ % name)

        logging.info('Module %s does not provide a setup.py. \nGenerating setup.py' % name)

        env.write_file(setup_path, data)

        data = textwrap.dedent("""
        [wheel]
        universal = 1
        """)

        logging.info('Generating setup.cfg')

        env.write_file(os.path.join(path, 'setup.cfg'), data)

        data = textwrap.dedent("""
        recursive-include . *

        recursive-exclude . __pycache__*
        recursive-exclude . *.pyc
        recursive-exclude . *.pyo
        """)

        logging.info('Generating MANIFEST.in')

        env.write_file(os.path.join(path, 'MANIFEST.in'), data)


def install(path):  # type: (str) -> None
    """Install a Python module in the executing Python environment.

    Args:
        path (str):  Real path location of the Python module.
    """
    if not sys.executable:
        raise RuntimeError('Failed to retrieve the real path for the Python executable binary')
    try:
        cmd = '%s -m pip install -U . ' % sys.executable

        if os.path.exists(os.path.join(path, 'requirements.txt')):
            cmd += '-r requirements.txt'

        subprocess.check_call(shlex.split(cmd), cwd=path)
    except subprocess.CalledProcessError as e:
        six.raise_from(errors.InstallModuleError(e), e)


def exists(name):  # type: (str) -> bool
    """Return True if the module exists. Return False otherwise.

    Args:
        name (str): module name.

    Returns:
        (bool): boolean indicating if the module exists or not.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True


def download_and_import(url, name=DEFAULT_MODULE_NAME, cache=True):  # type: (str, str, bool) -> module
    """Download, prepare and install a compressed tar file from S3 as a module.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.

    This function downloads this compressed file, transforms it as a module, and installs it.

    Args:
        name (str): name of the script or module.
        url (str): the s3 url of the file.
        cache (bool): default True. It will not download and install the module again if it is already installed.

    Returns:
        (module): the imported module
    """
    should_use_cache = cache and exists(name)

    if not should_use_cache:
        with env.tmpdir() as tmpdir:
            dst = os.path.join(tmpdir, 'tar_file')
            s3_download(url, dst)

            module_path = os.path.join(tmpdir, 'module_dir')

            os.makedirs(module_path)

            with tarfile.open(name=dst, mode='r:gz') as t:
                t.extractall(path=module_path)

                prepare(module_path, name)

                install(module_path)

    try:
        module = importlib.import_module(name)
        six.moves.reload_module(module)

        return module
    except Exception as e:
        six.raise_from(errors.ImportModuleError(e), e)
