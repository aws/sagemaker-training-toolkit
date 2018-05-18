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

    def dasherize(string):
        if not string:
            return ''
        if len(string) > 1:
            return '--%s' % string
        return '-%s' % string

    arg_names = [dasherize(argument) for argument in mapping.keys()]

    def to_str(value):
        if hasattr(value, 'items'):
            return ','.join(['%s=%s' % (str(k), v) for k, v in value.items()])
        return str(value)

    arg_values = [to_str(value) for value in mapping.values()]

    items = zip(arg_names, arg_values)

    return list(itertools.chain.from_iterable(items))


def split_by_criteria(dictionary, keys):  # type: (dict, set or list or tuple) -> SplitResultSpec
    """Split a dictionary in two by the provided keys.

    Args:
        dictionary (dict[str, object]): A Python dictionary
        keys (sequence [str]): A sequence of keys which will be the split criteria

    Returns:
        `SplitResultSpec` : A collections.namedtuple with the following attributes:

            * Args:
                included (dict[str, object]: A dictionary with the keys included in the criteria.
                excluded (dict[str, object]: A dictionary with the keys not included in the criteria.
    """
    keys = set(keys)
    included_items = {k: dictionary[k] for k in dictionary.keys() if k in keys}
    excluded_items = {k: dictionary[k] for k in dictionary.keys() if k not in keys}

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
