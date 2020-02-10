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
"""Placeholder docstring"""
from __future__ import absolute_import

import json
import logging

import sagemaker_training


def get_logger():
    """Returns a logger with the name 'sagemaker-training-toolkit',
    creating it if necessary.
    """
    return logging.getLogger("sagemaker-training-toolkit")


def configure_logger(level, log_format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"):
    # type: (int, str) -> None
    """Set logger configuration.

    Args:
        level (int): Logger level
        log_format (str): Logger format
    """
    logging.basicConfig(format=log_format, level=level)

    if level >= logging.INFO:
        logging.getLogger("boto3").setLevel(logging.INFO)
        logging.getLogger("s3transfer").setLevel(logging.INFO)
        logging.getLogger("botocore").setLevel(logging.WARN)


def log_script_invocation(cmd, env_vars, logger=None):
    """Placeholder docstring"""
    logger = logger or get_logger()

    prefix = "\n".join(["%s=%s" % (key, value) for key, value in env_vars.items()])
    env = sagemaker_training.training_env()
    message = """Invoking user script

Training Env:

%s

Environment variables:

%s

Invoking script with the following command:

%s

""" % (
        json.dumps(dict(env), indent=4),
        prefix,
        " ".join(cmd),
    )
    logger.info(message)
