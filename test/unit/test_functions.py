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

import inspect

import pytest as pytest

from sagemaker_containers import _functions


@pytest.mark.parametrize('fn, expected', [
    (lambda: None, inspect.ArgSpec([], None, None, None)),
    (lambda x, y='y': None, inspect.ArgSpec(['x', 'y'], None, None, ('y',))),
    (lambda *args: None, inspect.ArgSpec([], 'args', None, None)),
    (lambda **kwargs: None, inspect.ArgSpec([], None, 'kwargs', None)),
    (lambda x, y, *args, **kwargs: None, inspect.ArgSpec(['x', 'y'], 'args', 'kwargs', None))
])
def test_getargspec(fn, expected):
    assert _functions.getargspec(fn) == expected


@pytest.mark.parametrize('fn, env, expected', [
    (lambda: None, {}, {}),
    (lambda x, y='y': None, dict(x='x', y=None, t=3), dict(x='x', y=None)),
    (lambda not_in_env_arg: None, dict(x='x', y=None, t=3), {}),
    (lambda *args: None, dict(x='x', y=None, t=3), {}),
    (lambda *arguments, **keywords: None, dict(x='x', y=None, t=3), dict(x='x', y=None, t=3)),
    (lambda **kwargs: None, dict(x='x', y=None, t=3), dict(x='x', y=None, t=3))
])
def test_matching_args(fn, env, expected):
    assert _functions.matching_args(fn, env) == expected


def test_error_wrapper():
    assert _functions.error_wrapper(lambda x: x * 10, NotImplementedError)(3) == 30


def test_error_wrapper_exception():
    with pytest.raises(NotImplementedError) as e:
        _functions.error_wrapper(lambda x: x, NotImplementedError)(2, 3)
    assert type(e.value.args[0]) == TypeError
