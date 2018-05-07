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
import numpy as np
import pytest
from six import StringIO

from sagemaker_containers import content_types, encoders


@patch('numpy.load', lambda x: 'loaded %s' % x)
@patch('sagemaker_containers.encoders.BytesIO', lambda x: 'byte io %s' % x)
def test_npy_decode():
    assert encoders.NpyDecoder().decode(42) == 'loaded byte io 42'


@patch('numpy.save', autospec=True)
@patch('sagemaker_containers.encoders.BytesIO', autospec=True)
def test_npy_encode(bytes_io, save):
    encoders.NpyEncoder().encode(42)

    bytes_io.return_value.getvalue.assert_called()
    save.assert_called_with(bytes_io(), 42)


@patch('json.loads', lambda x: 'loaded %s' % x)
def test_json_decode():
    assert encoders.JsonDecoder().decode(42) == 'loaded 42'


def test_json_encode():
    assert encoders.JsonEncoder().encode(42) == '42'

    assert encoders.JsonEncoder().encode(np.asarray([42])) == '[42]'

    assert encoders.JsonEncoder().encode(StringIO('42')) == '"42"'

    with pytest.raises(TypeError):
        encoders.JsonEncoder().encode(lambda x: 3)


@patch('numpy.genfromtxt', autospec=True)
@patch('sagemaker_containers.encoders.StringIO', autospec=True)
def test_csv_decode(string_io, genfromtxt):
    encoders.CsvDecoder.decode('42')

    string_io.assert_called_with('42')
    genfromtxt.assert_called_with(string_io(), dtype=np.float32, delimiter=',')


@patch('numpy.savetxt', autospec=True)
@patch('sagemaker_containers.encoders.StringIO', autospec=True)
def test_csv_encode(string_io, savetxt):
    encoders.CsvEncoder().encode(42)

    string_io.return_value.getvalue.assert_called()
    savetxt.assert_called_with(string_io(), 42, delimiter=',', fmt='%s')


@pytest.mark.parametrize('target, content_type', [
    ('sagemaker_containers.encoders.JsonEncoder.encode', content_types.JSON),
    ('sagemaker_containers.encoders.CsvEncoder.encode', content_types.CSV),
    ('sagemaker_containers.encoders.NpyEncoder.encode', content_types.NPY)
])
def test_default_encoder(target, content_type):
    with patch(target) as encoder_obj:
        encoders.DefaultEncoder().encode(42, content_type)

        encoder_obj.assert_called_once_with(42)


def test_default_encoder_error():
    with pytest.raises(encoders.UnsupportedFormatError):
        encoders.DefaultEncoder().encode(42, content_types.OCTET_STREAM)


def test_default_decoder_error():
    with pytest.raises(encoders.UnsupportedFormatError):
        encoders.DefaultDecoder().decode(42, content_types.OCTET_STREAM)


@pytest.mark.parametrize('target, content_type', [
    ('sagemaker_containers.encoders.JsonDecoder.decode', content_types.JSON),
    ('sagemaker_containers.encoders.CsvDecoder.decode', content_types.CSV),
    ('sagemaker_containers.encoders.NpyDecoder.decode', content_types.NPY)
])
def test_default_decoder(target, content_type):
    with patch(target) as decoder:
        encoders.DefaultDecoder().decode(42, content_type)

        decoder.assert_called_once_with(42)
