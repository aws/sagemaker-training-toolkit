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
"""This module contains the train function, which is the main function
responsible for running training in the container.
"""
from __future__ import absolute_import

import importlib
import os
import sys
import traceback

from sagemaker_training import (
    entry_point,
    environment,
    errors,
    files,
    intermediate_output,
    logging_config,
    params,
    runner,
    SM_TRAINING_COMPILER_PATHS,
)

logger = logging_config.get_logger()

SUCCESS_CODE = 0
DEFAULT_FAILURE_CODE = 1


def _get_valid_failure_exit_code(exit_code):
    try:
        valid_exit_code = int(exit_code)
    except ValueError:
        valid_exit_code = DEFAULT_FAILURE_CODE

    return valid_exit_code


def _exit_processes(exit_code):  # type: (int) -> None
    """Exit main thread and child processes.

    For more information:
        https://docs.python.org/2/library/os.html#process-management
        https://docs.python.org/3/library/os.html#process-management

    Args:
        exit_code (int): exit code
    """
    if exit_code != 0:
        logger.error(f"Encountered exit_code {exit_code}")
    sys.exit(exit_code)


def train():
    """The main function responsible for running training in the container."""
    intermediate_sync = None
    exit_code = SUCCESS_CODE
    try:
        env = environment.Environment()

        region = os.environ.get("AWS_REGION", os.environ.get(params.REGION_NAME_ENV))
        s3_endpoint_url = os.environ.get(params.S3_ENDPOINT_URL, None)
        intermediate_sync = intermediate_output.start_sync(
            env.sagemaker_s3_output(), region, endpoint_url=s3_endpoint_url
        )

        if env.framework_module:
            framework_name, entry_point_name = env.framework_module.split(":")

            framework = importlib.import_module(framework_name)

            # the logger is configured after importing the framework library, allowing
            # the framework to configure logging at import time.
            logging_config.configure_logger(env.log_level)
            logger.info("Imported framework %s", framework_name)
            entrypoint = getattr(framework, entry_point_name)
            entrypoint()
        else:
            logging_config.configure_logger(env.log_level)

            mpi_enabled = env.additional_framework_parameters.get(params.MPI_ENABLED)
            runner_type = (
                runner.RunnerType.MPI
                if mpi_enabled and (env.current_instance_group in env.distribution_instance_groups)
                else runner.RunnerType.Process
            )

            entry_point.run(
                env.module_dir,
                env.user_entry_point,
                env.to_cmd_args(),
                env.to_env_vars(),
                runner_type=runner_type,
            )
        logger.info("Reporting training SUCCESS")

        files.write_success_file()
    except errors.ClientError as e:

        failure_msg = str(e)
        files.write_failure_file(failure_msg)
        logger.error("Reporting training FAILURE")

        logger.error(failure_msg)

        if intermediate_sync:
            intermediate_sync.join()

        exit_code = DEFAULT_FAILURE_CODE
    except Exception as e:  # pylint: disable=broad-except
        if any(path in traceback.format_exc() for path in SM_TRAINING_COMPILER_PATHS):
            failure_msg = "SMTrainingCompiler Error: \n%s\n%s" % (traceback.format_exc(), str(e))
        else:
            failure_msg = "Framework Error: \n%s\n%s" % (traceback.format_exc(), str(e))
        files.write_failure_file(failure_msg)
        logger.error("Reporting training FAILURE")

        logger.error(failure_msg)

        error_number = getattr(e, "errno", DEFAULT_FAILURE_CODE)
        exit_code = _get_valid_failure_exit_code(error_number)
    finally:
        if intermediate_sync:
            intermediate_sync.join()
        _exit_processes(exit_code)
