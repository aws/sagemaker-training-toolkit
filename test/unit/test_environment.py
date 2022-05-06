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
import itertools
import json
import logging
import os
import socket

from mock import Mock, patch
import pytest
import six

from sagemaker_training import environment, params
import test

builtins_open = "__builtin__.open" if six.PY2 else "builtins.open"

RESOURCE_CONFIG = dict(current_host="algo-1", hosts=["algo-1", "algo-2", "algo-3"])

INPUT_DATA_CONFIG = {
    "train": {
        "ContentType": "trainingContentType",
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
    "validation": {
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
}

USER_HYPERPARAMETERS = {"batch_size": 32, "learning_rate": 0.001, "hosts": ["algo-1", "algo-2"]}

SAGEMAKER_HYPERPARAMETERS = {
    "sagemaker_region": "us-west-2",
    "default_user_module_name": "net",
    "sagemaker_job_name": "sagemaker-training-job",
    "sagemaker_program": "main.py",
    "sagemaker_submit_directory": "imagenet",
    "sagemaker_enable_cloudwatch_metrics": True,
    "sagemaker_container_log_level": logging.WARNING,
    "_tuning_objective_metric": "loss:3.4",
    "sagemaker_parameter_server_num": 2,
    "sagemaker_s3_output": "s3://bucket",
}

ALL_HYPERPARAMETERS = dict(
    itertools.chain(USER_HYPERPARAMETERS.items(), SAGEMAKER_HYPERPARAMETERS.items())
)


def test_read_hyperparameters():
    test.write_json(ALL_HYPERPARAMETERS, environment.hyperparameters_file_dir)

    assert environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_value_serialized_hyperparameters():
    serialized_hps = {k: json.dumps(v) for k, v in ALL_HYPERPARAMETERS.items()}
    test.write_json(serialized_hps, environment.hyperparameters_file_dir)

    assert environment.read_hyperparameters() == ALL_HYPERPARAMETERS


def test_read_value_serialized_and_non_value_serialized_hyperparameters():
    hyperparameters = {k: json.dumps(v) for k, v in SAGEMAKER_HYPERPARAMETERS.items()}

    hyperparameters.update(USER_HYPERPARAMETERS)

    test.write_json(hyperparameters, environment.hyperparameters_file_dir)

    assert environment.read_hyperparameters() == ALL_HYPERPARAMETERS


@patch("sagemaker_training.environment._read_json", lambda x: {"a": 1})
@patch("json.loads")
def test_read_exception(loads):
    loads.side_effect = ValueError("Unable to read.")

    assert environment.read_hyperparameters() == {"a": 1}


def test_resource_config():
    test.write_json(RESOURCE_CONFIG, environment.resource_config_file_dir)

    assert environment.read_resource_config() == RESOURCE_CONFIG


def test_input_data_config():
    test.write_json(INPUT_DATA_CONFIG, environment.input_data_config_file_dir)

    assert environment.read_input_data_config() == INPUT_DATA_CONFIG


def test_channel_input_dirs():
    input_data_path = environment._input_data_dir
    assert environment.channel_path("evaluation") == os.path.join(input_data_path, "evaluation")
    assert environment.channel_path("training") == os.path.join(input_data_path, "training")


@patch("subprocess.check_output", lambda s: b"GPU 0\nGPU 1")
def test_gpu_count_in_gpu_instance():
    assert environment.num_gpus() == 2


@patch("subprocess.check_output", side_effect=OSError())
def test_gpu_count_in_cpu_instance(check_output):
    assert environment.num_gpus() == 0


@patch("multiprocessing.cpu_count", lambda: 2)
def test_cpu_count():
    assert environment.num_cpus() == 2


@pytest.fixture(name="training_env")
def create_training_env():
    with patch(
        "sagemaker_training.environment.read_resource_config", lambda: RESOURCE_CONFIG
    ), patch(
        "sagemaker_training.environment.read_input_data_config", lambda: INPUT_DATA_CONFIG
    ), patch(
        "sagemaker_training.environment.read_hyperparameters", lambda: ALL_HYPERPARAMETERS
    ), patch(
        "sagemaker_training.environment.num_cpus", lambda: 8
    ), patch(
        "sagemaker_training.environment.num_gpus", lambda: 4
    ):
        session_mock = Mock()
        session_mock.region_name = "us-west-2"
        old_environ = os.environ.copy()
        os.environ[params.TRAINING_JOB_ENV] = "training-job-42"

        yield environment.Environment()

        os.environ = old_environ


def test_create_training_env_without_training_files_and_directories_should_not_fail():
    training_env = environment.Environment()
    hostname = socket.gethostname()
    assert training_env.current_host == hostname
    assert training_env.hosts == [hostname]


def test_env():
    assert environment.input_dir.endswith("/opt/ml/input")
    assert environment.input_config_dir.endswith("/opt/ml/input/config")
    assert environment.model_dir.endswith("/opt/ml/model")
    assert environment.output_dir.endswith("/opt/ml/output")


def test_training_env(training_env):
    assert training_env.num_gpus == 4
    assert training_env.num_cpus == 8
    assert training_env.input_dir.endswith("/opt/ml/input")
    assert training_env.input_config_dir.endswith("/opt/ml/input/config")
    assert training_env.model_dir.endswith("/opt/ml/model")
    assert training_env.output_dir.endswith("/opt/ml/output")
    assert training_env.hyperparameters == USER_HYPERPARAMETERS
    assert training_env.resource_config == RESOURCE_CONFIG
    assert training_env.input_data_config == INPUT_DATA_CONFIG
    assert training_env.output_data_dir.endswith("/opt/ml/output/data")
    assert training_env.hosts == RESOURCE_CONFIG["hosts"]
    assert training_env.channel_input_dirs["train"].endswith("/opt/ml/input/data/train")
    assert training_env.channel_input_dirs["validation"].endswith("/opt/ml/input/data/validation")
    assert training_env.current_host == RESOURCE_CONFIG["current_host"]
    assert training_env.module_name == "main"
    assert training_env.user_entry_point == "main.py"
    assert training_env.module_dir == "imagenet"
    assert training_env.log_level == logging.WARNING
    assert training_env.network_interface_name == "eth0"
    assert training_env.job_name == "training-job-42"
    assert training_env.additional_framework_parameters == {"sagemaker_parameter_server_num": 2}


def test_env_mapping_properties(training_env):
    assert set(training_env.properties()) == {
        "additional_framework_parameters",
        "channel_input_dirs",
        "current_host",
        "framework_module",
        "hosts",
        "hyperparameters",
        "input_config_dir",
        "input_data_config",
        "input_dir",
        "log_level",
        "model_dir",
        "module_dir",
        "module_name",
        "network_interface_name",
        "num_cpus",
        "num_gpus",
        "output_data_dir",
        "output_dir",
        "resource_config",
        "user_entry_point",
        "job_name",
        "output_intermediate_dir",
        "is_master",
        "master_hostname",
    }


@patch("sagemaker_training.environment.num_cpus", lambda: 8)
@patch("sagemaker_training.environment.num_gpus", lambda: 4)
def test_env_dictionary():
    session_mock = Mock()
    session_mock.region_name = "us-west-2"
    os.environ[params.USER_PROGRAM_ENV] = "my_app.py"
    os.environ[params.LOG_LEVEL_ENV] = "20"
    test_env = environment.Environment()

    assert len(test_env) == len(test_env.properties())

    assert test_env["module_name"] == "my_app"
    assert test_env["log_level"] == logging.INFO


@pytest.mark.parametrize("sagemaker_program", ["program.py", "program"])
def test_env_module_name(sagemaker_program):
    session_mock = Mock()
    session_mock.region_name = "us-west-2"
    os.environ[params.USER_PROGRAM_ENV] = sagemaker_program
    module_name = environment.Environment().module_name

    del os.environ[params.USER_PROGRAM_ENV]

    assert module_name == "program"
