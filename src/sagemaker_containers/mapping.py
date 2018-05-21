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

import collections
import itertools

import six

SplitResultSpec = collections.namedtuple('SplitResultSpec', 'included excluded')


def to_cmd_args(mapping):  # type: (dict) -> list
    """Transform a dictionary in a list of cmd arguments.

    Example:

        >>>args = mapping.to_cmd_args({'model_dir': '/opt/ml/model', 'batch_size': 25})
        >>>
        >>>print(args)
        ['--model_dir', '/opt/ml/model', '--batch_size', 25]

    Args:
        mapping (dict[str, object]): A Python mapping.

    Returns:
        (list): List of cmd arguments
    """

    sorted_keys = sorted(mapping.keys())

    def arg_name(obj):
        string = decode(obj)
        if not string:
            return u''
        if len(string) > 1:
            return u'--%s' % string
        else:
            return u'-%s' % string

    arg_names = [arg_name(argument) for argument in sorted_keys]

    def arg_value(value):
        if hasattr(value, 'items'):
            map_items = ['%s=%s' % (k, v) for k, v in sorted(value.items())]
            return ','.join(map_items)
        return decode(value)

    arg_values = [arg_value(mapping[key]) for key in sorted_keys]

    items = zip(arg_names, arg_values)

    return [item for item in itertools.chain.from_iterable(items)]


def decode(object):
    if six.PY3 and isinstance(object, six.binary_type):
        return object.decode('latin1')
    elif six.PY3:
        return str(object)
    elif isinstance(object, six.text_type):
        return object
    else:
        return str(object).decode('utf-8')


def split_by_criteria(dictionary, keys=None, prefix=None):  # type: (dict, set or list or tuple) -> SplitResultSpec
    """Split a dictionary in two by the provided keys.

    Args:
        dictionary (dict[str, object]): A Python dictionary
        keys (sequence [str]): A sequence of keys which will be added the split criteria
        prefix (str): A prefix which will be added the split criteria

    Returns:
        `SplitResultSpec` : A collections.namedtuple with the following attributes:

            * Args:
                included (dict[str, object]: A dictionary with the keys included in the criteria.
                excluded (dict[str, object]: A dictionary with the keys not included in the criteria.
    """
    keys = keys or []
    keys = set(keys)

    included_items = {k: dictionary[k] for k in dictionary.keys() if k in keys or (prefix and k.startswith(prefix))}
    excluded_items = {k: dictionary[k] for k in dictionary.keys() if k not in included_items}

    return SplitResultSpec(included=included_items, excluded=excluded_items)


class MappingMixin(collections.Mapping):
    def properties(self):  # type: () -> list
        """
            Returns:
                (list[str]) List of public properties
        """

        _type = type(self)
        return [_property for _property in dir(_type) if self._is_property(_property)]

    def _is_property(self, _property):
        return isinstance(getattr(type(self), _property), property)

    def __getitem__(self, k):
        if not self._is_property(k):
            raise KeyError('Trying to access non property %s' % k)
        return getattr(self, k)

    def __len__(self):
        return len(self.properties())

    def __iter__(self):
        items = {_property: getattr(self, _property) for _property in self.properties()}
        return iter(items)

    def __str__(self):
        return str(dict(self))
