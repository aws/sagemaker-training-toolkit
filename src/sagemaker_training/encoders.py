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
"""This module contains utilities to encode and decode different content types."""
from __future__ import absolute_import

import csv
import io
import json

import numpy as np
from scipy.sparse import issparse
from six import BytesIO, StringIO

from sagemaker_training import content_types, errors
from sagemaker_training.recordio import (
    _write_numpy_to_dense_tensor,
    _write_spmatrix_to_sparse_tensor,
)


def array_to_npy(array_like):
    """Convert an array-like object to the NPY format.

    To understand what an array-like object is, please see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): Array-like object to be converted to NPY.

    Returns:
        (obj): NPY array.
    """
    buffer = BytesIO()
    np.save(buffer, array_like)
    return buffer.getvalue()


def npy_to_numpy(npy_array):
    """Convert an NPY array into numpy.

    Args:
        npy_array (npy array): NPY array to be converted.
    Returns:
        (np.array): Converted numpy array.
    """
    stream = BytesIO(npy_array)
    return np.load(stream, allow_pickle=True)


def array_to_json(array_like):
    """Convert an array-like object to JSON.

    To understand what an array-like object is, please see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): Array-like object to be
                                                           converted to JSON.

    Returns:
        (str): Object serialized to JSON.
    """

    def default(_array_like):
        if hasattr(_array_like, "tolist"):
            return _array_like.tolist()
        return json.JSONEncoder().default(_array_like)

    return json.dumps(array_like, default=default)


def json_to_numpy(string_like, dtype=None):
    """Convert a JSON object to a numpy array.

        Args:
            string_like (str): JSON string.
            dtype (dtype, optional):  Data type of the resulting array. If None,
                                      the dtypes will be determined by the
                                      contents of each column, individually.
                                      This argument can only be used to
                                      'upcast' the array.  For downcasting,
                                      use the .astype(t) method.
        Returns:
            (np.array): Numpy array.
        """
    data = json.loads(string_like)
    return np.array(data, dtype=dtype)


def csv_to_numpy(string_like, dtype=None):
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None, the
                                  dtypes will be determined by the contents of
                                  each column, individually. This argument can
                                  only be used to 'upcast' the array.  For
                                  downcasting, use the .astype(t) method.
    Returns:
        (np.array): Numpy array.
    """
    try:
        stream = StringIO(string_like)
        reader = csv.reader(stream, delimiter=",", quotechar='"', doublequote=True, strict=True)
        array = np.array([row for row in reader]).squeeze()
        array = array.astype(dtype)
    except ValueError as e:
        if dtype is not None:
            raise errors.ClientError(
                "Error while writing numpy array: {}. dtype is: {}".format(e, dtype)
            )
    except Exception as e:
        raise errors.ClientError("Error while decoding csv: {}".format(e))
    return array


def array_to_csv(array_like):
    """Convert an array like object to CSV.

    To understand what an array-like object is, please see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): Array-like object to be converted to CSV.

    Returns:
        (str): Object serialized to CSV.
    """
    array = np.array(array_like)
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))  # pylint: disable=unsubscriptable-object

    try:
        stream = StringIO()
        writer = csv.writer(
            stream, lineterminator="\n", delimiter=",", quotechar='"', doublequote=True, strict=True
        )
        writer.writerows(array)
        return stream.getvalue()
    except csv.Error as e:
        raise errors.ClientError("Error while encoding csv: {}".format(e))


def array_to_recordio_protobuf(array_like, labels=None):
    """Convert an array like object to recordio-protobuf format.

    To understand what an array-like object is, please see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

     Args:
        array_like (np.array or scipy.sparse.csr_matrix): Array-like object to be
                                                          converted to recordio-protobuf.
        labels (np.array or scipy.sparse.csr_matrix): Array-like object representing
                                                      the labels to be encoded.

    Returns:
        buffer: Bytes buffer recordio-protobuf.
    """

    if len(array_like.shape) == 1:
        array_like = array_like.reshape(1, array_like.shape[0])
    assert len(array_like.shape) == 2, "Expecting a 1 or 2 dimensional array"

    buffer = io.BytesIO()

    if issparse(array_like):
        _write_spmatrix_to_sparse_tensor(buffer, array_like, labels)
    else:
        _write_numpy_to_dense_tensor(buffer, array_like, labels)
    buffer.seek(0)
    return buffer.getvalue()


encoders_map = {
    content_types.NPY: array_to_npy,
    content_types.CSV: array_to_csv,
    content_types.JSON: array_to_json,
}
_decoders_map = {
    content_types.NPY: npy_to_numpy,
    content_types.CSV: csv_to_numpy,
    content_types.JSON: json_to_numpy,
}


def decode(obj, content_type):
    """Decode an object of one of the default content types to a numpy array.

    Args:
        obj (object): Object to be decoded.
        content_type (str): Content type to be used.

    Returns:
        np.array: Decoded object.
    """
    try:
        decoder = _decoders_map[content_type]
        return decoder(obj)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)


def encode(array_like, content_type):
    """Encode an array-like object in a specific content_type to a numpy array.

    To understand what an array-like object is, please see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): Array-like object to be
            converted to numpy.
        content_type (str): Content type to be used.

    Returns:
        (np.array): Object converted as numpy array.
    """
    try:
        encoder = encoders_map[content_type]
        return encoder(array_like)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)
