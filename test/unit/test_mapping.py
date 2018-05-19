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

import pytest
import six

from sagemaker_containers import mapping


@pytest.mark.parametrize('dictionary, keys, expected', [
    ({}, (), ({}, {})),
    ({'x': 1, 'y': 2}, 'x', ({'x': 1}, {'y': 2})),
    ({'x': 1, 'y': 2}, (), ({}, {'x': 1, 'y': 2})),
    ({'x': 1, 'y': 2}, ('x', 'y'), ({'x': 1, 'y': 2}, {}))
])
def test_split_by_criteria(dictionary, keys, expected):
    assert mapping.split_by_criteria(dictionary, keys) == expected


class ProcessEnvironment(mapping.MappingMixin):
    @property
    def a(self):
        return 1

    @property
    def b(self):
        return 2

    def d(self):
        return 23

    def __init__(self):
        self.c = 3


def test_mapping_mixin():
    p = ProcessEnvironment()

    assert p['a'] == 1
    assert len(p) == 2
    assert p['b'] == 2
    assert str(p) in ("{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}")


@pytest.mark.parametrize('property, error, msg', [
    ('c', AttributeError, "type object 'ProcessEnvironment' has no attribute 'c'"),
    ('d', KeyError, 'Trying to access non property d'),
    ('non_existent_field', AttributeError,
     "type object 'ProcessEnvironment' has no attribute 'non_existent_field'")
])
def test_mapping_throws_exception_trying_to_access_non_properties(property, error, msg):
    with pytest.raises(error) as e:
        ProcessEnvironment()[property]

    assert str(e.value.args[0]) == msg


@pytest.mark.parametrize(
    'target, expected',
    [({'da-sh': '1', 'un_der': '2', 'un-sh': '3', 'da_der': '2'},
      ['--da-sh', '1', '--un_der', '2', '--un-sh', '3', '--da_der', '2']),

     ({},
      []),

     ({'': ''},
      ['', '']),

     ({'unicode': u'¡ø'},
      ['--unicode', '¡ø']), #'--bytes', "b'2'" if six.PY3 else '2', '--floats', '4.0', '--int', '2']),

     ({'U': u'1', 'b': b'2', 'T': '', '': '42'},
      ['-U', '1', '-b', "b'2'" if six.PY3 else '2', '-T', '', '', '42']),

     ({'nested': ['1', ['2', '3', [['6']]]]},
      ['--nested', "['1', ['2', '3', [['6']]]]"]),

     ({'map': {'a': [1, 3, 4]}, 'channel_dirs': {'train': 'foo', 'eval': 'bar'}},
      ['--map', "a=[1, 3, 4]", '--channel_dirs', 'train=foo,eval=bar']),

     ({'truthy': True, 'falsy': False},
      ['--truthy', 'True', '--falsy', 'False'])

     ])
def test_to_cmd_args(target, expected):
    actual = mapping.to_cmd_args(target)

    assert sorted(actual) == sorted(expected)
