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

import numpy as np

from sagemaker_containers import _content_types, _encoders, _env, _status_codes, _transformer, _worker
import test
from test import fake_ml_framework


def predict_fn(data, model):
    return model.predict(data)


def model_fn(model_dir):
    return fake_ml_framework.Model.load(os.path.join(model_dir, 'fake_ml_model'))


def test_transformer_implementation():
    test.create_resource_config()
    test.create_input_data_config()
    test.create_hyperparameters_config({'sagemaker_program': 'user_script.py'})

    model_path = os.path.join(_env.model_dir, 'fake_ml_model')
    fake_ml_framework.Model(weights=[6, 9, 42]).save(model_path)

    transform = _transformer.Transformer(model_fn=model_fn, predict_fn=predict_fn)

    transform.initialize()

    with _worker.Worker(transform_fn=transform.transform,
                        module_name='fake_ml_model').test_client() as client:
        payload = [6, 9, 42.]
        response = post(client, payload, _content_types.NPY, _content_types.JSON)

        assert response.status_code == _status_codes.OK

        assert response.get_data(as_text=True) == '[36.0, 81.0, 1764.0]'

        response = post(client, payload, _content_types.JSON, _content_types.CSV)

        assert response.status_code == _status_codes.OK
        assert response.get_data(as_text=True) == '36.0\n81.0\n1764.0\n'

        response = post(client, payload, _content_types.CSV, _content_types.NPY)

        assert response.status_code == _status_codes.OK
        response_data = _encoders.npy_to_numpy(response.get_data())

        np.testing.assert_array_almost_equal(response_data, np.asarray([36., 81., 1764.]))


def post(client, payload, content_type, accept):
    return client.post(path='/invocations', headers={'accept': accept},
                       data=_encoders.encode(payload, content_type), content_type=content_type)
