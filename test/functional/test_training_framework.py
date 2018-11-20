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

import errno
import importlib
import os
import shlex
import subprocess

import numpy as np
import pytest

import sagemaker_containers
from sagemaker_containers.beta.framework import entry_point, env, errors, functions, modules, trainer
import test
from test import fake_ml_framework

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def erase_user_module():
    yield
    try:
        cmd = shlex.split('pip uninstall -y --quiet %s' % modules.DEFAULT_MODULE_NAME)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        pass


USER_SCRIPT = """
import os
import test.fake_ml_framework as fake_ml
import numpy as np

def train(channel_input_dirs, hyperparameters):
    data = np.load(os.path.join(channel_input_dirs['training'], hyperparameters['training_data_file']))
    x_train = data['features']
    y_train = data['labels']
    optimizer = hyperparameters['optimizer']

    model = fake_ml.Model(optimizer=optimizer)

    model.fit(x=x_train, y=y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

    return model
"""

USER_SCRIPT_WITH_SAVE = """
import os
import test.fake_ml_framework as fake_ml
import numpy as np

def train(channel_input_dirs, hyperparameters):
    data = np.load(os.path.join(channel_input_dirs['training'], hyperparameters['training_data_file']))
    x_train = data['features']
    y_train = data['labels']
    optimizer = hyperparameters['optimizer']

    model = fake_ml.Model(optimizer=optimizer)

    model.fit(x=x_train, y=y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

    return model

def save(model, model_dir):
    model.save(os.path.join(model_dir, 'saved_model'))
"""

USER_SCRIPT_WITH_EXCEPTION = """
import os

def train(channel_input_dirs, hyperparameters):
    raise OSError(os.errno.ENOENT, 'No such file or directory')
"""

USER_MODE_SCRIPT = """
import argparse
import os
import test.fake_ml_framework as fake_ml
import numpy as np

parser = argparse.ArgumentParser()

# Data and model checkpoints directories
parser.add_argument('--training_data_file', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model_dir', type=str)

args = parser.parse_args()

data = np.load(os.path.join(os.environ['SM_CHANNEL_TRAINING'], args.training_data_file))
x_train = data['features']
y_train = data['labels']

model = fake_ml.Model(loss='elastic', optimizer='SGD')

model.fit(x=x_train, y=y_train, epochs=args.epochs, batch_size=args.batch_size)

model_file = os.path.join(os.environ['SM_MODEL_DIR'], 'saved_model')
model.save(model_file)
"""

BASH_SCRIPT = '#!/usr/bin/env python\n%s' % USER_MODE_SCRIPT

PARAMETER_SERVER_SCRIPT = """
from time import sleep

while True:
    print('Running parameter server')
    sleep(1)
"""

setup_file = test.File('setup.py', """
from setuptools import setup
setup(packages=[''],
      name="user_script",
      version='1.0.0',
      include_package_data=True)
""")


def framework_training_fn():
    training_env = sagemaker_containers.training_env()

    mod = modules.import_module(training_env.module_dir, training_env.module_name, False)

    model = mod.train(**functions.matching_args(mod.train, training_env))

    if model:
        if hasattr(mod, 'save'):
            mod.save(model, training_env.model_dir)
        else:
            model_file = os.path.join(training_env.model_dir, 'saved_model')
            model.save(model_file)


@pytest.mark.parametrize('user_script', [USER_SCRIPT_WITH_SAVE, USER_SCRIPT_WITH_SAVE])
def test_training_framework(user_script):
    with pytest.raises(ImportError):
        importlib.import_module(modules.DEFAULT_MODULE_NAME)

    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    file = test.File(name='user_script.py', data=user_script)
    module = test.UserModule(file).add_file(setup_file)

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program='user_script.py', epochs=10,
                           batch_size=64, optimizer='Adam')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    assert execute_an_wrap_exit(framework_training_fn) == trainer.SUCCESS_CODE

    model_path = os.path.join(env.model_dir, 'saved_model')
    model = fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.optimizer == 'Adam'


@pytest.mark.parametrize('user_script, sagemaker_program', [
    [USER_MODE_SCRIPT, 'user_script.py'],
    [BASH_SCRIPT, 'bash_script']
])
def test_trainer_report_success(user_script, sagemaker_program):

    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name=sagemaker_program, data=user_script))

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program=sagemaker_program, epochs=10,
                           batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    assert execute_an_wrap_exit(trainer.train) == trainer.SUCCESS_CODE

    model_path = os.path.join(env.model_dir, 'saved_model')

    model = fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.optimizer == 'SGD'
    assert os.path.exists(os.path.join(env.output_dir, 'success'))


def test_trainer_report_failure():
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=USER_SCRIPT_WITH_EXCEPTION)).add_file(setup_file)

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program='user_script.py', epochs=10,
                           batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    os.environ['SAGEMAKER_TRAINING_MODULE'] = 'test.functional.simple_framework:train'

    assert execute_an_wrap_exit(trainer.train) == errno.ENOENT

    failure_file = os.path.join(env.output_dir, 'failure')
    assert os.path.exists(failure_file)

    message = failure_message()

    assert message.startswith('framework error:')
    assert 'No such file or directory' in message


def framework_training_with_script_mode_fn():
    training_env = sagemaker_containers.training_env()

    entry_point.run(training_env.module_dir, training_env.user_entry_point, training_env.to_cmd_args(),
                    training_env.to_env_vars())


def framework_training_with_run_modules_fn():
    training_env = sagemaker_containers.training_env()

    modules.run_module(training_env.module_dir, training_env.to_cmd_args(),
                       training_env.to_env_vars(), training_env.module_name)


def test_parameter_server():
    module = test.UserModule(test.File(name='user_script.py', data=PARAMETER_SERVER_SCRIPT))
    hyperparameters = dict(sagemaker_program='user_script.py')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[test.Channel.create(name='training')])
    training_env = sagemaker_containers.training_env()
    process = entry_point.run(training_env.module_dir, training_env.user_entry_point,
                              training_env.to_cmd_args(), training_env.to_env_vars(), wait=False)
    # confirm the ps process is still hanging
    assert process.poll() is None
    process.kill()


@pytest.mark.parametrize('user_script, training_fn', [
    [USER_MODE_SCRIPT, framework_training_with_script_mode_fn],
    [USER_MODE_SCRIPT, framework_training_with_run_modules_fn]])
def test_script_mode(user_script, training_fn):
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=user_script))

    hyperparameters = dict(training_data_file=os.path.join(channel.path, 'training_data.npz'),
                           sagemaker_program='user_script.py', epochs=10, batch_size=64, model_dir=env.model_dir)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    assert execute_an_wrap_exit(training_fn) == trainer.SUCCESS_CODE

    model_path = os.path.join(env.model_dir, 'saved_model')

    model = fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.loss == 'elastic'
    assert model.optimizer == 'SGD'


@pytest.mark.parametrize('user_script, training_fn', [
    [USER_MODE_SCRIPT, framework_training_with_script_mode_fn],
    [USER_MODE_SCRIPT, framework_training_with_run_modules_fn]])
def test_script_mode_local_directory(user_script, training_fn, tmpdir):
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    tmp_code_dir = str(tmpdir)

    module = test.UserModule(test.File(name='user_script.py', data=user_script))
    module.create_tmp_dir_with_files(tmp_code_dir)

    hyperparameters = dict(training_data_file=os.path.join(channel.path, 'training_data.npz'),
                           sagemaker_program='user_script.py', sagemaker_submit_directory=tmp_code_dir,
                           epochs=10, batch_size=64, model_dir=env.model_dir)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel], local=True)

    assert execute_an_wrap_exit(training_fn) == trainer.SUCCESS_CODE

    model_path = os.path.join(env.model_dir, 'saved_model')

    model = fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.loss == 'elastic'
    assert model.optimizer == 'SGD'


USER_MODE_SCRIPT_WITH_ERROR = """
if __name__ == '__main__':
    42/0
"""


@pytest.mark.parametrize('training_fn', [
    framework_training_with_script_mode_fn,
    framework_training_with_run_modules_fn])
def test_script_mode_client_error(training_fn):
    channel = test.Channel.create(name='training')

    module = test.UserModule(test.File(name='user_script.py', data=USER_MODE_SCRIPT_WITH_ERROR))

    hyperparameters = dict(sagemaker_program='user_script.py')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    with pytest.raises(errors.ExecuteUserScriptError) as e:
        training_fn()

    message = str(e.value)
    assert 'ExecuteUserScriptError' in message


@pytest.mark.parametrize('training_fn', [
    framework_training_with_script_mode_fn,
    framework_training_with_run_modules_fn])
def test_script_mode_client_import_error(training_fn):
    channel = test.Channel.create(name='training')

    requirements_file = test.File('requirements.txt', '42/0')

    user_script = test.File(name='user_script', data='42/0')
    module = test.UserModule(user_script).add_file(setup_file).add_file(requirements_file).upload()

    hyperparameters = dict(sagemaker_program='user_script')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    with pytest.raises(errors.InstallModuleError) as e:
        training_fn()

    message = str(e.value)
    assert 'InstallModuleError:' in message


def failure_message():
    with open(os.path.join(env.output_dir, 'failure')) as f:
        return f.read()


def execute_an_wrap_exit(fn):
    try:
        fn()
        return trainer.SUCCESS_CODE
    except ValueError as e:
        return int(str(e))
