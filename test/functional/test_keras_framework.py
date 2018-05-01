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

import logging
import os

import numpy as np

import sagemaker_containers as smc

dir_path = os.path.dirname(os.path.realpath(__file__))

RESOURCE_CONFIG = dict(current_host='algo-1', hosts=['algo-1'])

INPUT_DATA_CONFIG = {'training': {'ContentType': 'trainingContentType',
                                  'TrainingInputMode': 'File',
                                  'S3DistributionType': 'FullyReplicated',
                                  'RecordWrapperType': 'None'}}

HYPERPARAMETERS = dict(training_data_file='training_data.npz', sagemaker_region='us-west-2',
                       default_user_module_name='net',
                       sagemaker_job_name='sagemaker-training-job', sagemaker_enable_cloudwatch_metrics=True,
                       sagemaker_container_log_level=logging.WARNING, sagemaker_program='user_script.py')

USER_SCRIPT = """
import os

import keras
import numpy as np

def train(channel_input_dirs, hyperparameters):
    data = np.load(os.path.join(channel_input_dirs['training'], hyperparameters['training_data_file']))
    x_train = data['features']
    y_train = keras.utils.to_categorical(data['labels'], 10)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation='softmax', input_dim=1))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=1)

    return model
"""


def keras_framework_training_fn():
    env = smc.Environment.create()

    mod = smc.modules.download_and_import(env.module_dir, env.module_name)

    model = mod.train(**smc.functions.matching_args(mod.train, env))

    if model:
        if hasattr(mod, 'save'):
            mod.save(model, env.model_dir)
        else:
            model_file = os.path.join(env.model_dir, 'saved_model')
            model.save(model_file)

    return model


def test_keras_framework(create_channel, create_training):
    create_training(script_name='user_script.py', script=USER_SCRIPT, hyperparameters=HYPERPARAMETERS,
                    resource_config=RESOURCE_CONFIG, input_data_config=INPUT_DATA_CONFIG)

    features = np.random.random((10, 1))
    labels = np.zeros((10, 1))

    create_channel(channel='training', file_name='training_data.npz', data=dict(features=features, labels=labels))

    model = keras_framework_training_fn()

    assert model.trainable
