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
import os
import shlex
import sys
import textwrap
import warnings

import six

from sagemaker_containers import _env, _errors, _files, _logging, _process

logger = _logging.get_logger()

DEFAULT_MODULE_NAME = 'default_user_module_name'


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


def has_requirements(path):  # type: (str) -> None
    return os.path.exists(os.path.join(path, 'requirements.txt'))


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

        logger.info('Module %s does not provide a setup.py. \nGenerating setup.py' % name)

        _files.write_file(setup_path, data)

        data = textwrap.dedent("""
        [wheel]
        universal = 1
        """)

        logger.info('Generating setup.cfg')

        _files.write_file(os.path.join(path, 'setup.cfg'), data)

        data = textwrap.dedent("""
        recursive-include . *
        recursive-exclude . __pycache__*
        recursive-exclude . *.pyc
        recursive-exclude . *.pyo
        """)

        logger.info('Generating MANIFEST.in')

        _files.write_file(os.path.join(path, 'MANIFEST.in'), data)


def install(path, capture_error=False):  # type: (str, bool) -> None
    """Install a Python module in the executing Python environment.
    Args:
        path (str):  Real path location of the Python module.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    cmd = '%s -m pip install -U . ' % _process.python_executable()

    if has_requirements(path):
        cmd += '-r requirements.txt'

    logger.info('Installing module with the following command:\n%s', cmd)

    _process.check_error(shlex.split(cmd), _errors.InstallModuleError, cwd=path, capture_error=capture_error)


def run(module_name, args=None, env_vars=None, wait=True, capture_error=False):
    # type: (str, list, dict, bool, bool) -> Popen
    """Run Python module as a script.

    Search sys.path for the named module and execute its contents as the __main__ module.

    Since the argument is a module name, you must not give a file extension (.py). The module name should be a valid
    absolute Python module name, but the implementation may not always enforce this (e.g. it may allow you to use a name
    that includes a hyphen).

    Package names (including namespace packages) are also permitted. When a package name is supplied instead of a
    normal module, the interpreter will execute <pkg>.__main__ as the main module. This behaviour is deliberately
    similar to the handling of directories and zipfiles that are passed to the interpreter as the script argument.

    Note This option cannot be used with built-in modules and extension modules written in C, since they do not have
    Python module files. However, it can still be used for precompiled modules, even if the original source file is
    not available. If this option is given, the first element of sys.argv will be the full path to the module file (
    while the module file is being located, the first element will be set to "-m"). As with the -c option,
    the current directory will be added to the start of sys.path.

    You can find more information at https://docs.python.org/3/using/cmdline.html#cmdoption-m

    Example:

        >>>import sagemaker_containers
        >>>from sagemaker_containers.beta.framework import mapping, modules

        >>>env = sagemaker_containers.training_env()
        {'channel-input-dirs': {'training': '/opt/ml/input/training'}, 'model_dir': '/opt/ml/model', ...}


        >>>hyperparameters = env.hyperparameters
        {'batch-size': 128, 'model_dir': '/opt/ml/model'}

        >>>args = mapping.to_cmd_args(hyperparameters)
        ['--batch-size', '128', '--model_dir', '/opt/ml/model']

        >>>env_vars = mapping.to_env_vars()
        ['SAGEMAKER_CHANNELS':'training', 'SAGEMAKER_CHANNEL_TRAINING':'/opt/ml/input/training',
        'MODEL_DIR':'/opt/ml/model', ...}

        >>>modules.run('user_script', args, env_vars)
        SAGEMAKER_CHANNELS=training SAGEMAKER_CHANNEL_TRAINING=/opt/ml/input/training \
        SAGEMAKER_MODEL_DIR=/opt/ml/model python -m user_script --batch-size 128 --model_dir /opt/ml/model

    Args:
        module_name (str): module name in the same format required by python -m <module-name> cli command.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    args = args or []
    env_vars = env_vars or {}

    cmd = [_process.python_executable(), '-m', module_name] + args

    _logging.log_script_invocation(cmd, env_vars)

    if wait:
        return _process.check_error(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)

    else:
        return _process.create(cmd, _errors.ExecuteUserScriptError, capture_error=capture_error)


def import_module(uri, name=DEFAULT_MODULE_NAME, cache=None):  # type: (str, str, bool) -> module
    """Download, prepare and install a compressed tar file from S3 or provided directory as a module.
    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, if provided, and transforms it as a module, and installs it.
    Args:
        name (str): name of the script or module.
        uri (str): the location of the module.
        cache (bool): default True. It will not download and install the module again if it is already installed.
    Returns:
        (module): the imported module
    """
    _warning_cache_deprecation(cache)
    _files.download_and_extract(uri, name, _env.code_dir)

    prepare(_env.code_dir, name)
    install(_env.code_dir)
    try:
        module = importlib.import_module(name)
        six.moves.reload_module(module)

        return module
    except Exception as e:
        six.reraise(_errors.ImportModuleError, _errors.ImportModuleError(e), sys.exc_info()[2])


def run_module(uri, args, env_vars=None, name=DEFAULT_MODULE_NAME, cache=None, wait=True, capture_error=False):
    # type: (str, list, dict, str, bool, bool, bool) -> Popen
    """Download, prepare and executes a compressed tar file from S3 or provided directory as a module.

    SageMaker Python SDK saves the user provided scripts as compressed tar files in S3
    https://github.com/aws/sagemaker-python-sdk.
    This function downloads this compressed file, transforms it as a module, and executes it.
    Args:
        uri (str): the location of the module.
        args (list):  A list of program arguments.
        env_vars (dict): A map containing the environment variables to be written.
        name (str): name of the script or module.
        cache (bool): If True it will avoid downloading the module again, if already installed.
        wait (bool): If True run_module will wait for the user module to exit and check the exit code,
                     otherwise it will launch the user module with subprocess and return the process object.
    """
    _warning_cache_deprecation(cache)
    env_vars = env_vars or {}
    env_vars = env_vars.copy()

    _files.download_and_extract(uri, name, _env.code_dir)

    prepare(_env.code_dir, name)
    install(_env.code_dir)

    _env.write_env_vars(env_vars)

    return run(name, args, env_vars, wait, capture_error)


def _warning_cache_deprecation(cache):
    if cache is not None:
        msg = 'the cache parameter is unnecessary anymore. Cache is always set to True'
        warnings.warn(msg, DeprecationWarning)
