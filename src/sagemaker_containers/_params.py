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

SAGEMAKER_PREFIX = 'sagemaker_'  # type: str
CURRENT_HOST_ENV = 'CURRENT_HOST'  # type: str
USER_PROGRAM_PARAM = 'sagemaker_program'  # type: str
USER_PROGRAM_ENV = USER_PROGRAM_PARAM.upper()  # type: str
TRAINING_JOB_ENV = 'TRAINING_JOB_NAME'  # type: str
SUBMIT_DIR_PARAM = 'sagemaker_submit_directory'  # type: str
SUBMIT_DIR_ENV = SUBMIT_DIR_PARAM.upper()  # type: str
ENABLE_METRICS_PARAM = 'sagemaker_enable_cloudwatch_metrics'  # type: str
ENABLE_METRICS_ENV = ENABLE_METRICS_PARAM.upper()  # type: str
LOG_LEVEL_PARAM = 'sagemaker_container_log_level'  # type: str
LOG_LEVEL_ENV = LOG_LEVEL_PARAM.upper()  # type: str
JOB_NAME_PARAM = 'sagemaker_job_name'  # type: str
JOB_NAME_ENV = JOB_NAME_PARAM.upper()  # type: str
TUNING_METRIC_PARAM = '_tuning_objective_metric'  # type: str
DEFAULT_MODULE_NAME_PARAM = 'default_user_module_name'  # type: str
REGION_NAME_PARAM = 'sagemaker_region'  # type: str
REGION_NAME_ENV = REGION_NAME_PARAM.upper()  # type: str
MODEL_SERVER_WORKERS_ENV = 'SAGEMAKER_MODEL_SERVER_WORKERS'  # type: str
MODEL_SERVER_TIMEOUT_ENV = 'SAGEMAKER_MODEL_SERVER_TIMEOUT'  # type: str
USE_NGINX_ENV = 'SAGEMAKER_USE_NGINX'  # type: str
FRAMEWORK_SERVING_MODULE_ENV = 'SAGEMAKER_SERVING_MODULE'  # type: str
FRAMEWORK_TRAINING_MODULE_ENV = 'SAGEMAKER_TRAINING_MODULE'  # type: str
SAGEMAKER_HYPERPARAMETERS = (
    USER_PROGRAM_PARAM, SUBMIT_DIR_PARAM, ENABLE_METRICS_PARAM, REGION_NAME_PARAM, LOG_LEVEL_PARAM, JOB_NAME_PARAM,
    DEFAULT_MODULE_NAME_PARAM, TUNING_METRIC_PARAM)  # type: set
