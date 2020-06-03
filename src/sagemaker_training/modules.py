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
"""This module contains functionality related to preparing, installing,
and importing Python modules.
"""
from __future__ import absolute_import

import importlib
import os
import shlex
import sys
import textwrap

import six

from sagemaker_training import environment, errors, files, logging_config, process

logger = logging_config.get_logger()

DEFAULT_MODULE_NAME = "default_user_module_name"


def exists(name):  # type: (str) -> bool
    """Return True if the module exists. Return False otherwise.

    Args:
        name (str): Module name.

    Returns:
        (bool): Boolean indicating if the module exists or not.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True


def has_requirements(path):  # type: (str) -> None
    """Check whether a directory contains a requirements.txt file.

    Args:
        path (str): Path to the directory to check for the requirements.txt file.

    Returns:
        (bool): Whether the directory contains a requirements.txt file.
    """
    return os.path.exists(os.path.join(path, "requirements.txt"))


def prepare(path, name):  # type: (str, str) -> None
    """Prepare a Python script (or module) to be imported as a module.
    If the script does not contain a setup.py file, it creates a minimal setup.

    Args:
        path (str): Path to directory with the script or module.
        name (str): Name of the script or module.
    """
    setup_path = os.path.join(path, "setup.py")
    if not os.path.exists(setup_path):
        data = textwrap.dedent(
            """
        from setuptools import setup
        setup(packages=[''],
              name="%s",
              version='1.0.0',
              include_package_data=True)
        """
            % name
        )

        logger.info("Module %s does not provide a setup.py. \nGenerating setup.py" % name)

        files.write_file(setup_path, data)

        data = textwrap.dedent(
            """
        [wheel]
        universal = 1
        """
        )

        logger.info("Generating setup.cfg")

        files.write_file(os.path.join(path, "setup.cfg"), data)

        data = textwrap.dedent(
            """
        recursive-include . *
        recursive-exclude . __pycache__*
        recursive-exclude . *.pyc
        recursive-exclude . *.pyo
        """
        )

        logger.info("Generating MANIFEST.in")

        files.write_file(os.path.join(path, "MANIFEST.in"), data)


def install(path, capture_error=False):  # type: (str, bool) -> None
    """Install a Python module in the executing Python environment.

    Args:
        path (str):  Real path location of the Python module.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    cmd = "%s -m pip install . " % process.python_executable()

    if has_requirements(path):
        cmd += "-r requirements.txt"

    logger.info("Installing module with the following command:\n%s", cmd)

    process.check_error(
        shlex.split(cmd), errors.InstallModuleError, cwd=path, capture_error=capture_error
    )


def install_requirements(path, capture_error=False):  # type: (str, bool) -> None
    """Install dependencies from requirements.txt in the executing Python environment.

    Args:
        path (str):  Real path location of the requirements.txt file.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    cmd = "{} -m pip install -r requirements.txt".format(process.python_executable())

    logger.info("Installing dependencies from requirements.txt:\n{}".format(cmd))

    process.check_error(
        shlex.split(cmd), errors.InstallRequirementsError, cwd=path, capture_error=capture_error
    )


def import_module(uri, name=DEFAULT_MODULE_NAME):  # type: (str, str) -> module
    """Download, prepare and install a compressed tar file from S3 or provided directory as a
    module.
    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file (if provided), transforms it as a module, and
    installs it.

    Args:
        name (str): Name of the script or module.
        uri (str): The location of the module.

    Returns:
        (module): The imported module.
    """
    files.download_and_extract(uri, environment.code_dir)

    prepare(environment.code_dir, name)
    install(environment.code_dir)
    try:
        module = importlib.import_module(name)
        six.moves.reload_module(module)  # pylint: disable=too-many-function-args

        return module
    except Exception as e:  # pylint: disable=broad-except
        six.reraise(errors.ImportModuleError, errors.ImportModuleError(e), sys.exc_info()[2])
