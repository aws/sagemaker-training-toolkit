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
import errno
import os

from mock import MagicMock, Mock, patch

from sagemaker_containers import _errors, _runner, _trainer


class TrainingEnv(Mock):
    framework_module = 'my_framework:entry_point'
    log_level = 20

    def sagemaker_s3_output(self):
        return 's3://bucket'


class TrainingEnvNoIntermediate(Mock):
    framework_module = 'my_framework:entry_point'
    log_level = 20

    def sagemaker_s3_output(self):
        return None


class ScriptTrainingEnv(TrainingEnv):
    framework_module = None

    def sagemaker_s3_output(self):
        return 's3://bucket'


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
def test_train(import_module):
    framework = Mock()
    import_module.return_value = framework
    _trainer.train()

    import_module.assert_called_with('my_framework')
    framework.entry_point.assert_called()


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_with_success(_exit, import_module):
    def success():
        pass

    framework = Mock(entry_point=success)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.SUCCESS_CODE)


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_fails(_exit, import_module):

    def fail():
        raise OSError(os.errno.ENOENT, 'No such file or directory')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(errno.ENOENT)


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_fails_with_no_error_number(_exit, import_module):

    def fail():
        raise Exception('No errno defined.')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.DEFAULT_FAILURE_CODE)


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_fails_with_invalid_error_number(_exit, import_module):
    class InvalidErrorNumberException(Exception):
        def __init__(self, *args, **kwargs):  # real signature unknown
            self.errno = 'invalid'

    def fail():
        raise InvalidErrorNumberException('No such file or directory')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.DEFAULT_FAILURE_CODE)


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('importlib.import_module')
@patch('sagemaker_containers.training_env', TrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_with_client_error(_exit, import_module):

    def fail():
        raise _errors.ClientError(os.errno.ENOENT, 'No such file or directory')

    framework = Mock(entry_point=fail)
    import_module.return_value = framework

    _trainer.train()

    _exit.assert_called_with(_trainer.DEFAULT_FAILURE_CODE)


@patch('inotify_simple.INotify', MagicMock())
@patch('boto3.client', MagicMock())
@patch('sagemaker_containers.entry_point.run')
@patch('sagemaker_containers.training_env', new_callable=ScriptTrainingEnv)
@patch('sagemaker_containers._trainer._exit_processes')
def test_train_script(_exit, training_env, run):
    _trainer.train()

    env = training_env()
    run.assert_called_with(env.module_dir, env.user_entry_point, env.to_cmd_args(),
                           env.to_env_vars(), runner=_runner.RunnerType.MPI)

    _exit.assert_called_with(_trainer.SUCCESS_CODE)


@patch('importlib.import_module')
@patch('sagemaker_containers._intermediate_output.start_sync')
@patch('sagemaker_containers.training_env', TrainingEnvNoIntermediate)
def test_train_no_intermediate(start_intermediate_folder_sync, import_module):
    framework = Mock()
    import_module.return_value = framework
    _trainer.train()

    import_module.assert_called_with('my_framework')
    framework.entry_point.assert_called()
    start_intermediate_folder_sync.asser_not_called()
