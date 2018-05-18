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
from functools import wraps
import importlib
import os
import sys
import traceback

import six

from sagemaker_containers import env, errors, functions


SUCCESS_CODE = 0
DEFAULT_FAILURE_CODE = 1


def report_training_status(train_func):
    @wraps(train_func)
    def train_and_report(*args, **kwargs):
        training_env = env.TrainingEnv()
        exit_code = SUCCESS_CODE

        try:
            wrapped_train = functions.error_wrapper(train_func, errors.ClientError)
            wrapped_train(*args, **kwargs)
            training_env.write_success_file()
        except Exception as e:
            sub_e = e.args[0]
            exit_code = DEFAULT_FAILURE_CODE if not hasattr(sub_e, 'errno') or sub_e.errno is None else sub_e.errno
            failure_msg = 'Exception caught in training: {}\n{}\n'.format(e, traceback.format_exc())
            training_env.write_failure_file(failure_msg)
            six.reraise(*sys.exc_info())
        finally:
            # This may apply to child process after fork(), so we use os._exit instead of sys.exit
            # https://docs.python.org/2/library/os.html#process-management
            # https://docs.python.org/3/library/os.html#process-management
            os._exit(exit_code)

    return train_and_report


def train():
    training_env = env.TrainingEnv()

    # TODO: iquintero - add error handling for ImportError to let the user know
    # if the framework module is not defined.
    framework_name, entry_point = training_env.framework_module.split(':')
    framework = importlib.import_module(framework_name)
    entry = getattr(framework, entry_point)
    entry()
