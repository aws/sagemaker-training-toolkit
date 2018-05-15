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
from mock import Mock, patch
import numpy as np
import pytest

from sagemaker_containers import content_types, encoders


@patch('numpy.load', lambda x: 'loaded %s' % x)
@patch('sagemaker_containers.encoders.BytesIO', lambda x: 'byte io %s' % x)
def test_npy_to_numpy():
    assert encoders.npy_to_numpy(42) == 'loaded byte io 42'


@patch('numpy.save', autospec=True)
@patch('sagemaker_containers.encoders.BytesIO', autospec=True)
def array_to_npy(bytes_io, save):
    encoders.array_to_npy(42)

    bytes_io.return_value.getvalue.assert_called()
    save.assert_called_with(bytes_io(), 42)


@patch('json.loads', lambda x: 'loaded %s' % x)
def test_json_to_numpy():
    assert encoders.json_to_numpy(42) == 'loaded 42'


def test_array_to_json():
    assert encoders.array_to_json(42) == '42'

    assert encoders.array_to_json(np.asarray([42])) == '[42]'

    with pytest.raises(TypeError):
        encoders.array_to_json(lambda x: 3)


@patch('numpy.genfromtxt', autospec=True)
@patch('sagemaker_containers.encoders.StringIO', autospec=True)
def test_csv_to_numpy(string_io, genfromtxt):
    encoders.csv_to_numpy('42')

    string_io.assert_called_with('42')
    genfromtxt.assert_called_with(string_io(), dtype=np.float32, delimiter=',')


@patch('numpy.savetxt', autospec=True)
@patch('sagemaker_containers.encoders.StringIO', autospec=True)
def test_array_to_csv(string_io, savetxt):
    encoders.array_to_csv(42)

    string_io.return_value.getvalue.assert_called()
    savetxt.assert_called_with(string_io(), 42, delimiter=',', fmt='%s')


@pytest.mark.parametrize(
    'content_type', [content_types.JSON, content_types.CSV, content_types.NPY]
)
def test_encode(content_type):
    encoder = Mock()
    with patch.dict(encoders._encoders_map, {content_type: encoder}, clear=True):
        encoders.encode(42, content_type)

        encoder.assert_called_once_with(42)


def test_encode_error():
    with pytest.raises(encoders.UnsupportedFormatError):
        encoders.encode(42, content_types.OCTET_STREAM)


def test_decode_error():
    with pytest.raises(encoders.UnsupportedFormatError):
        encoders.decode(42, content_types.OCTET_STREAM)


@pytest.mark.parametrize(
    'content_type', [content_types.JSON, content_types.CSV, content_types.NPY]
)
def test_decode(content_type):
    decoder = Mock()
    with patch.dict(encoders._decoders_map, {content_type: decoder}, clear=True):
        encoders.decode(42, content_type)

        decoder.assert_called_once_with(42)
