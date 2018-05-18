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

import importlib
from multiprocessing import Process
import os
import shlex
import subprocess

import numpy as np
import pytest

from sagemaker_containers import env, functions, modules, trainer
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


def framework_training_fn():
    training_env = env.TrainingEnv()

    mod = modules.download_and_import(training_env.module_dir, training_env.module_name, False)

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

    module = test.UserModule(test.File(name='user_script.py', data=user_script))

    hyperparameters = dict(training_data_file='training_data.npz',
                           sagemaker_program='user_script.py',
                           epochs=10, batch_size=64, optimizer='Adam')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    framework_training_fn()

    model_path = os.path.join(env.TrainingEnv().model_dir, 'saved_model')
    print(model_path)

    model = test.fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.optimizer == 'Adam'


@pytest.mark.parametrize('user_script', [USER_SCRIPT, USER_SCRIPT_WITH_SAVE])
def test_trainer_report_success(user_script):
    with pytest.raises(ImportError):
        importlib.import_module(modules.DEFAULT_MODULE_NAME)

    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=user_script))

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program='user_script.py',
                           epochs=10, batch_size=64, optimizer='SGD')

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    os.environ['SAGEMAKER_TRAINING_MODULE'] = 'test.functional.simple_framework:train'

    p = Process(target=trainer.train)
    p.start()
    p.join()
    assert p.exitcode == 0

    model_path = os.path.join(env.TrainingEnv().model_dir, 'saved_model')
    print(model_path)

    model = fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.optimizer == 'SGD'
    assert os.path.exists(os.path.join(env.TrainingEnv().output_dir, 'success'))


def test_trainer_report_failure():
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=USER_SCRIPT_WITH_EXCEPTION))

    hyperparameters = dict(training_data_file='training_data.npz', sagemaker_program='user_script.py',
                           epochs=10, batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    os.environ['SAGEMAKER_TRAINING_MODULE'] = 'test.functional.simple_framework:train'

    p = Process(target=trainer.train)
    p.start()
    p.join()
    assert p.exitcode == os.errno.ENOENT

    failure_file = os.path.join(env.TrainingEnv().output_dir, 'failure')
    assert os.path.exists(failure_file)
    with open(failure_file, 'r') as f:
        assert f.read().startswith('Exception caught in training:')
