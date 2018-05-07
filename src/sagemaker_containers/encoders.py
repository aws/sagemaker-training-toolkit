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
from __future__ import absolute_import

import json
import textwrap

import numpy as np
from six import BytesIO, StringIO

from sagemaker_containers import content_types


class NpyEncoder(object):
    content_type = content_types.NPY

    @staticmethod
    def encode(np_array):  # type: (np.array) -> object
        """Encode an numpy array into memory.

        Args:
            np_array (np.array): np array object to be serialized.

        Returns:
            (obj): serialized np array.
        """
        buffer = BytesIO()
        np.save(buffer, np_array)
        return buffer.getvalue()


class NpyDecoder(object):
    content_type = content_types.NPY

    @staticmethod
    def decode(obj):  # type: (object) -> np.array
        """Decode arrays or pickled objects from memory into numpy arrays.

        Args:
            obj (object): object to be deserialized.

        Returns:
            (np.array): deserialized numpy array.
        """
        stream = BytesIO(obj)
        return np.load(stream)


class JsonEncoder(json.JSONEncoder):
    content_type = content_types.JSON

    def default(self, o):
        if hasattr(o, 'read'):
            return o.read()

        if hasattr(o, 'tolist'):
            return o.tolist()

        return super(JsonEncoder, self).default(o)


class JsonDecoder(object):
    content_type = content_types.JSON

    @staticmethod
    def decode(string):  # type: (object) -> object
        """Decode a JSON string in a Python object.

        Args:
            string (str): JSON string to be decoded.

        Returns:
            (object): deserialized Python object.
        """
        return json.loads(string)


class CsvDecoder(object):
    content_type = content_types.CSV

    @staticmethod
    def decode(string):  # type: (str) -> object
        """Decode a CSV string in a Python object.

        Args:
            string (str): CSV string to be decoded

        Returns:
            (object): decoded Python object.
        """
        stream = StringIO(string)
        return np.genfromtxt(stream, dtype=np.float32, delimiter=',')


class CsvEncoder(object):
    content_type = content_types.CSV

    @staticmethod
    def encode(obj):  # type: (object) -> str
        """Encode python objects, streams, numpy arrays into CSV.

        Args:
            obj (object): object to be encoded.

        Returns:
            str:  decoded object.
        """
        stream = StringIO()
        np.savetxt(stream, obj, delimiter=',', fmt='%s')
        return stream.getvalue()


DEFAULT_ENCODERS = frozenset([JsonEncoder(), CsvEncoder(), NpyEncoder()])
DEFAULT_DECODERS = frozenset([JsonDecoder(), CsvDecoder(), NpyDecoder()])


class DefaultEncoder(object):
    def __init__(self, encoders=None):
        encoders = encoders or DEFAULT_ENCODERS

        self._encoders_map = {encoder.content_type: encoder for encoder in encoders}

    def encode(self, obj, content_type):
        """Encode an object to a one of the default content types.

        Args:
            obj (object): to be encoded.
            content_type (str): content type to be used.

        Returns:
            object: encoded object.
        """
        try:
            return self._encoders_map[content_type].encode(obj)
        except KeyError:
            raise UnsupportedFormatError(content_type)


default_encoder = DefaultEncoder()


class DefaultDecoder(object):
    def __init__(self, decoders=None):
        decoders = decoders or DEFAULT_DECODERS

        self._decoders_map = {decoder.content_type: decoder for decoder in decoders}

    def decode(self, obj, content_type):
        """Decode an object to a one of the default content types.

        Args:
            obj (object): to be decoded.
            content_type (str): content type to be used.

        Returns:
            object: decoded object.
        """
        try:
            return self._decoders_map[content_type].decode(obj)
        except KeyError:
            raise UnsupportedFormatError(content_type)


default_decoder = DefaultDecoder()


class UnsupportedFormatError(Exception):
    def __init__(self, content_type, **kwargs):
        super(Exception, self).__init__(**kwargs)
        self.message = textwrap.dedent(
            """Content type %s is not supported by this framework.

               Please implement input_fn to to deserialize the request data or an output_fn to serialize the
               response. For more information: https://github.com/aws/sagemaker-python-sdk#input-processing"""
            % content_type)
