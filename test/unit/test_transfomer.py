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
from mock import MagicMock, patch
import pytest

from sagemaker_containers import content_types, env, errors, status_codes, transformer
import test


@patch('sagemaker_containers.encoders.decode')
def test_default_input_fn(loads):
    assert transformer.default_input_fn(42, content_types.JSON)

    loads.assert_called_with(42, content_types.JSON)


@patch('sagemaker_containers.encoders.encode', lambda prediction, accept: prediction ** 2)
def test_default_output_fn():
    response = transformer.default_output_fn(2, content_types.CSV)
    assert response.response == 4
    assert response.headers['accept'] == content_types.CSV


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        transformer.default_model_fn('model_dir')


def test_predict_fn():
    with pytest.raises(NotImplementedError):
        transformer.default_predict_fn('data', 'model')


request = test.request(data='42', content_type=content_types.JSON)


def test_transformer_initialize_with_default_model_fn():
    with pytest.raises(NotImplementedError):
        transformer.Transformer().initialize()


error_from_fn = ValueError('Failed')


def fn_with_error(*args, **kwargs):
    raise error_from_fn


def test_transformer_initialize_with_client_error():
    with pytest.raises(errors.ClientError) as e:
        transformer.Transformer(model_fn=fn_with_error).initialize()

    assert e.value.args[0] == error_from_fn


@pytest.mark.parametrize('input_fn, predict_fn, output_fn', [
    (fn_with_error, MagicMock(), MagicMock()),
    (MagicMock(), fn_with_error, MagicMock()),
    (MagicMock(), MagicMock(), fn_with_error),
])
@patch('sagemaker_containers.worker.Request', lambda: request)
def test_transformer_transform_with_client_error(input_fn, predict_fn, output_fn):
    with pytest.raises(errors.ClientError) as e:
        transform = transformer.Transformer(model_fn=MagicMock(), input_fn=input_fn,
                                            predict_fn=predict_fn, output_fn=output_fn)

        transform.transform()
    assert e.value.args[0] == error_from_fn


@patch('sagemaker_containers.worker.Request', lambda: request)
def test_transformer_with_default_predict_fn():
    with pytest.raises(NotImplementedError):
        transformer.Transformer().transform()


def test_initialize():
    model_fn = MagicMock()

    transformer.Transformer(model_fn=model_fn).initialize()

    model_fn.assert_called_with(env.MODEL_PATH)


@patch('sagemaker_containers.worker.Request', lambda: request)
@patch('sagemaker_containers.worker.Response', autospec=True)
def test_transformer_transform(response):
    model_fn, input_fn, predict_fn = (MagicMock(), MagicMock(), MagicMock())
    output_fn = MagicMock(return_value=response)

    transform = transformer.Transformer(model_fn=model_fn, input_fn=input_fn,
                                        predict_fn=predict_fn, output_fn=output_fn)

    transform.initialize()
    assert transform.transform() == response

    input_fn.assert_called_with(request.content, request.content_type)
    predict_fn.assert_called_with(input_fn(), model_fn())
    output_fn.assert_called_with(predict_fn(), request.accept)


@patch('sagemaker_containers.worker.Request', lambda: request)
def test_transformer_transform_backwards_compatibility():
    model_fn, input_fn, predict_fn, output_fn = (MagicMock(), MagicMock(), MagicMock(), MagicMock(return_value=(0, 1)))

    transform = transformer.Transformer(model_fn=model_fn, input_fn=input_fn,
                                        predict_fn=predict_fn, output_fn=output_fn)

    transform.initialize()

    assert transform.transform().status_code == status_codes.OK

    input_fn.assert_called_with(request.content, request.content_type)
    predict_fn.assert_called_with(input_fn(), model_fn())
    output_fn.assert_called_with(predict_fn(), request.accept)
