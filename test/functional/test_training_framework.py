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

import os
import shlex
import subprocess

import numpy as np
import pytest

from sagemaker_containers import env, functions, modules
import test

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

    model = fake_ml.Model(optimizer='SGD')

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

    model = fake_ml.Model(loss='categorical_crossentropy')

    model.fit(x=x_train, y=y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'])

    return model

def save(model, model_dir):
    model.save(os.path.join(model_dir, 'saved_model'))
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


def test_training_framework_with_save():
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=USER_SCRIPT_WITH_SAVE))

    hyperparameters = dict(training_data_file='training_data.npz',
                           sagemaker_program='user_script.py',
                           epochs=10, batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    framework_training_fn()

    model_path = os.path.join(env.TrainingEnv().model_dir, 'saved_model')
    print(model_path)

    model = test.fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.loss == 'categorical_crossentropy'


def test_training_framework_without_save():
    channel = test.Channel.create(name='training')

    features = [1, 2, 3, 4]
    labels = [0, 1, 0, 1]
    np.savez(os.path.join(channel.path, 'training_data'), features=features, labels=labels)

    module = test.UserModule(test.File(name='user_script.py', data=USER_SCRIPT))

    hyperparameters = dict(training_data_file='training_data.npz',
                           sagemaker_program='user_script.py',
                           epochs=10, batch_size=64)

    test.prepare(user_module=module, hyperparameters=hyperparameters, channels=[channel])

    framework_training_fn()

    model_path = os.path.join(env.TrainingEnv().model_dir, 'saved_model')
    print(model_path)

    model = test.fake_ml_framework.Model.load(model_path)

    assert model.epochs == 10
    assert model.batch_size == 64
    assert model.optimizer == 'SGD'
