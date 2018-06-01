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
import importlib
import os
import traceback

import sagemaker_containers
from sagemaker_containers import _errors, _files, _logging

logger = _logging.get_logger()

SUCCESS_CODE = 0
DEFAULT_FAILURE_CODE = 1


def _exit_processes(exit_code):  # type:
    """Exit main thread and child processes.

    For more information:
        https://docs.python.org/2/library/os.html#process-management
        https://docs.python.org/3/library/os.html#process-management

    Args:
        exit_code (int): exit code
    """
    os._exit(exit_code)


def train():
    try:
        # TODO: iquintero - add error handling for ImportError to let the user know
        # if the framework module is not defined.
        env = sagemaker_containers.training_env()

        framework_name, entry_point_name = env.framework_module.split(':')

        framework = importlib.import_module(framework_name)

        # the logger is configured after importing the framework library, allowing the framework to
        # configure logging at import time.
        _logging.configure_logger(env.log_level)

        logger.info('Imported framework %s', framework_name)

        entry_point = getattr(framework, entry_point_name)

        entry_point()

        logger.info('Reporting training SUCCESS')
        _files.write_success_file()
        _exit_processes(SUCCESS_CODE)

    except _errors.ClientError as e:

        failure_message = str(e)
        _files.write_failure_file(failure_message)

        logger.error(failure_message)
        _exit_processes(DEFAULT_FAILURE_CODE)
    except Exception as e:
        failure_msg = 'framework error: \n%s\n%s' % (traceback.format_exc(), str(e))

        _files.write_failure_file(failure_msg)
        logger.error('Reporting training FAILURE')

        logger.error(failure_msg)

        exit_code = getattr(e, 'errno', DEFAULT_FAILURE_CODE)
        _exit_processes(exit_code)
