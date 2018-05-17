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
from six import BytesIO

from sagemaker_containers import content_types, encoders


@pytest.mark.parametrize('target', ([42, 6, 9], [42., 6., 9.], ['42', '6', '9'], [u'42', u'6', u'9'], {42: {'6': 9.}}))
def test_npy_to_numpy(target):
    buffer = BytesIO()
    np.save(buffer, target)
    input_data = buffer.getvalue()

    actual = encoders.npy_to_numpy(input_data)

    np.testing.assert_equal(actual, np.array(target))


@pytest.mark.parametrize('target', ([42, 6, 9], [42., 6., 9.], ['42', '6', '9'], [u'42', u'6', u'9'], {42: {'6': 9.}}))
def test_array_to_npy(target):
    input_data = np.array(target)

    actual = encoders.array_to_npy(input_data)

    np.testing.assert_equal(np.load(BytesIO(actual)), np.array(target))

    actual = encoders.array_to_npy(target)

    np.testing.assert_equal(np.load(BytesIO(actual)), np.array(target))


@pytest.mark.parametrize(
    'target, expected', [('[42, 6, 9]', np.array([42, 6, 9])),
                         ('[42.0, 6.0, 9.0]', np.array([42., 6., 9.])),
                         ('["42", "6", "9"]', np.array(['42', '6', '9'])),
                         (u'["42", "6", "9"]', np.array([u'42', u'6', u'9']))]
)
def test_json_to_numpy(target, expected):
    actual = encoders.json_to_numpy(target)
    np.testing.assert_equal(actual, expected)

    np.testing.assert_equal(encoders.json_to_numpy(target, dtype=int), expected.astype(int))

    np.testing.assert_equal(encoders.json_to_numpy(target, dtype=float), expected.astype(float))


@pytest.mark.parametrize(
    'target, expected', [([42, 6, 9], '[42, 6, 9]'),
                         ([42., 6., 9.], '[42.0, 6.0, 9.0]'),
                         (['42', '6', '9'], '["42", "6", "9"]'),
                         ({42: {'6': 9.}}, '{"42": {"6": 9.0}}')]
)
def test_array_to_json(target, expected):
    actual = encoders.array_to_json(target)
    np.testing.assert_equal(actual, expected)

    actual = encoders.array_to_json(np.array(target))
    np.testing.assert_equal(actual, expected)


def test_array_to_json_exception():
    with pytest.raises(TypeError):
        encoders.array_to_json(lambda x: 3)


@pytest.mark.parametrize(
    'target, expected', [('42\n6\n9\n', np.array([42, 6, 9])),
                         ('42.0\n6.0\n9.0\n', np.array([42., 6., 9.])),
                         ('42\n6\n9\n', np.array([42, 6, 9]))]
)
def test_csv_to_numpy(target, expected):
    actual = encoders.csv_to_numpy(target)
    np.testing.assert_equal(actual, expected)

    actual = encoders.csv_to_numpy(target)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    'target, expected', [([42, 6, 9], '42\n6\n9\n'),
                         ([42., 6., 9.], '42.0\n6.0\n9.0\n'),
                         (['42', '6', '9'], '42\n6\n9\n')])
def test_array_to_csv(target, expected):
    actual = encoders.array_to_csv(target)
    np.testing.assert_equal(actual, expected)

    actual = encoders.array_to_csv(np.array(target))
    np.testing.assert_equal(actual, expected)


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
