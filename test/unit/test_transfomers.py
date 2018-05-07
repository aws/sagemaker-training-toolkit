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
from mock import patch
import pytest

from sagemaker_containers import content_types, env, transformers, worker
import test


class TestTransformer(transformers.BaseTransformer):
    pass


transformer = TestTransformer()


@patch('sagemaker_containers.encoders.DefaultDecoder.decode')
def test_transformer_input_fn(loads):
    assert transformer.input_fn(42, content_types.JSON)

    loads.assert_called_with(42, content_types.JSON)


@patch('sagemaker_containers.encoders.DefaultEncoder.encode', lambda self, prediction, accept: prediction ** 2)
def test_transformer_output_fn():
    response = transformer.output_fn(2, content_types.CSV)
    assert response.response == 4
    assert response.headers['accept'] == content_types.CSV


@patch.object(transformer, 'output_fn')
@patch.object(transformer, 'predict_fn')
@patch.object(transformer, 'input_fn')
def test_transformer_transform_fn(input_fn, predict_fn, output_fn):
    assert transformer.transform_fn('model', 'input-data', 'content-type', 'accept')

    input_fn.assert_called_with(content_type='content-type', input_data='input-data')
    predict_fn.assert_called_with(data=input_fn(), model='model')
    output_fn.assert_called_with(accept='accept', prediction=predict_fn())


def test_model_fn():
    with pytest.raises(NotImplementedError):
        transformer.model_fn('model_dir')


def test_predict_fn():
    with pytest.raises(NotImplementedError):
        transformer.predict_fn('model', 'data')


@patch.object(transformer, 'model_fn', lambda model_dir: {'my-model': model_dir})
def test_initialize():
    transformer.initialize()

    assert transformer._model == {'my-model': env.ServingEnv().model_dir}


request = test.request(data='42')


@patch('sagemaker_containers.worker.Request', lambda: request)
def test_transform_backwards_compatibility():
    def new_transform(model, content, content_type, accept):
        return worker.Response(response=[content], accept=accept)

    with patch.object(transformer, 'transform_fn', new_transform):
        result = transformer.transform()

        assert result.response == ['42']
        assert result.headers['accept'] == 'application/json'


@patch('sagemaker_containers.worker.Request', lambda: request)
def test_transform():
    def new_transform(model, content, content_type, accept):
        return ['42'], 'application/json'

    with patch.object(transformer, 'transform_fn', new_transform):
        result = transformer.transform()

        assert result.response == ['42']
        assert result.headers['accept'] == 'application/json'
