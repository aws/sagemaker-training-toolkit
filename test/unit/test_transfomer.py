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
import json

from mock import MagicMock, patch
import pytest
from six.moves import http_client

from sagemaker_containers import _content_types, _env, _errors, _transformer
import test


@patch('sagemaker_containers._encoders.decode')
def test_default_input_fn(loads):
    assert _transformer.default_input_fn(42, _content_types.JSON)

    loads.assert_called_with(42, _content_types.JSON)


@patch('sagemaker_containers._encoders.encode', lambda prediction, accept: prediction ** 2)
def test_default_output_fn():
    response = _transformer.default_output_fn(2, _content_types.CSV)
    assert response.response == 4
    assert response.headers['accept'] == _content_types.CSV


def test_default_model_fn():
    with pytest.raises(NotImplementedError):
        _transformer.default_model_fn('model_dir')


def test_predict_fn():
    with pytest.raises(NotImplementedError):
        _transformer.default_predict_fn('data', 'model')


request = test.request(data='42',
                       headers={'ContentType': _content_types.JSON})


def test_transformer_initialize_with_default_model_fn():
    with pytest.raises(NotImplementedError):
        _transformer.Transformer().initialize()


error_from_fn = ValueError('Failed')


def fn_with_error(*args, **kwargs):
    raise error_from_fn


def test_transformer_initialize_with_client_error():
    with pytest.raises(_errors.ClientError) as e:
        _transformer.Transformer(model_fn=fn_with_error).initialize()
    assert e.value.args[0] == error_from_fn


@pytest.mark.parametrize('input_fn, predict_fn, output_fn', [
    (fn_with_error, MagicMock(), MagicMock()),
    (MagicMock(), fn_with_error, MagicMock()),
    (MagicMock(), MagicMock(), fn_with_error),
])
@patch('sagemaker_containers._worker.Request', lambda: request)
def test_transformer_transform_with_client_error(input_fn, predict_fn, output_fn):
    with pytest.raises(_errors.ClientError) as e:
        transform = _transformer.Transformer(model_fn=MagicMock(), input_fn=input_fn,
                                             predict_fn=predict_fn, output_fn=output_fn)

        transform.transform()
    assert e.value.args[0] == error_from_fn


def test_transformer_transform_with_unsupported_content_type():
    bad_request = test.request(data=None, headers={'ContentType': 'fake/content-type'})
    with patch('sagemaker_containers._worker.Request', lambda: bad_request):
        response = _transformer.Transformer().transform()

    assert response.status_code == http_client.UNSUPPORTED_MEDIA_TYPE

    response_body = json.loads(response.response[0].decode('utf-8'))
    assert response_body['error'] == 'UnsupportedFormatError'
    assert bad_request.content_type in response_body['error-message']


def test_transformer_transform_with_unsupported_accept_type():
    def empty_fn(*args):
        pass

    bad_request = test.request(data=None, headers={'Accept': 'fake/content-type'})
    with patch('sagemaker_containers._worker.Request', lambda: bad_request):
        t = _transformer.Transformer(model_fn=empty_fn, input_fn=empty_fn, predict_fn=empty_fn)
        response = t.transform()

    assert response.status_code == http_client.NOT_ACCEPTABLE

    response_body = json.loads(response.response[0].decode('utf-8'))
    assert response_body['error'] == 'UnsupportedFormatError'
    assert bad_request.accept in response_body['error-message']


@patch('sagemaker_containers._worker.Request', lambda: request)
def test_transformer_with_default_predict_fn():
    with pytest.raises(NotImplementedError):
        _transformer.Transformer().transform()


def test_initialize():
    model_fn = MagicMock()

    _transformer.Transformer(model_fn=model_fn).initialize()

    model_fn.assert_called_with(_env.model_dir)


@patch('sagemaker_containers._worker.Request',
       lambda: MagicMock(content='42',
                         content_type=_content_types.JSON,
                         accept=_content_types.NPY))
def test_transformer_transform():
    model_fn, input_fn, predict_fn = (MagicMock(), MagicMock(), MagicMock())
    output_fn = MagicMock(return_value='response')

    transform = _transformer.Transformer(model_fn=model_fn, input_fn=input_fn,
                                         predict_fn=predict_fn, output_fn=output_fn)

    transform.initialize()
    assert transform.transform() == 'response'

    input_fn.assert_called_with('42', _content_types.JSON)
    predict_fn.assert_called_with(input_fn(), model_fn())
    output_fn.assert_called_with(predict_fn(), _content_types.NPY)


@patch('sagemaker_containers._worker.Request',
       lambda: MagicMock(content='13',
                         content_type=_content_types.CSV,
                         accept=_content_types.ANY))
def test_transformer_transform_backwards_compatibility():
    model_fn, input_fn, predict_fn, output_fn = (MagicMock(), MagicMock(), MagicMock(), MagicMock(return_value=(0, 1)))

    transform = _transformer.Transformer(model_fn=model_fn, input_fn=input_fn,
                                         predict_fn=predict_fn, output_fn=output_fn)

    transform.initialize()

    assert transform.transform().status_code == http_client.OK

    input_fn.assert_called_with('13', _content_types.CSV)
    predict_fn.assert_called_with(input_fn(), model_fn())
    output_fn.assert_called_with(predict_fn(), _content_types.ANY)


@patch('sagemaker_containers._worker.Request',
       lambda: MagicMock(content='13',
                         content_type=_content_types.CSV,
                         accept=_content_types.ANY))
def test_transformer_with_custom_transform_fn():
    model = MagicMock()

    def model_fn(model_dir):
        return model

    transform_fn = MagicMock()

    transform = _transformer.Transformer(model_fn=model_fn, transform_fn=transform_fn)
    transform.initialize()
    transform.transform()

    transform_fn.assert_called_with(model, '13',
                                    _content_types.CSV,
                                    _content_types.ANY)


def test_transformer_too_many_custom_methods():
    with pytest.raises(ValueError) as e:
        _transformer.Transformer(input_fn=MagicMock(), predict_fn=MagicMock(),
                                 output_fn=MagicMock(), transform_fn=MagicMock())

    assert 'Cannot use transform_fn implementation with input_fn, predict_fn, and/or output_fn' in str(e)
