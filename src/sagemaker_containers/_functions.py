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
import sys

import six

from sagemaker_containers import _mapping


def matching_args(fn, dictionary):  # type: (function, _mapping.Mapping) -> dict
    """Given a function fn and a dict dictionary, returns the function arguments that match the dict keys.

    Example:

        def train(channel_dirs, model_dir): pass

        dictionary = {'channel_dirs': {}, 'model_dir': '/opt/ml/model', 'other_args': None}

        args = functions.matching_args(train, dictionary) # {'channel_dirs': {}, 'model_dir': '/opt/ml/model'}

        train(**args)
    Args:
        fn (function): a function
        dictionary (dict): the dictionary with the keys

    Returns:
        (dict) a dictionary with only matching arguments.
    """
    arg_spec = getargspec(fn)

    if arg_spec.keywords:
        return dictionary

    return _mapping.split_by_criteria(dictionary, arg_spec.args).included


def getargspec(fn):  # type: (function) -> inspect.ArgSpec
    """Get the names and default values of a function's arguments.

    Args:
        fn (function): a function

    Returns:
        `inspect.ArgSpec`:  A collections.namedtuple with the following attributes:

            * Args:
                args (list): a list of the argument names (it may contain nested lists).
                varargs (str): name of the * argument or None.
                keywords (str): names of the ** argument or None.
                defaults (tuple): an n-tuple of the default values of the last n arguments.
    """
    if six.PY2:
        return inspect.getargspec(fn)
    elif six.PY3:
        full_arg_spec = inspect.getfullargspec(fn)
        return inspect.ArgSpec(full_arg_spec.args, full_arg_spec.varargs, full_arg_spec.varkw, full_arg_spec.defaults)


def error_wrapper(fn, error_class):  # type: (function or None, Exception) -> ...
    """Wraps function fn in a try catch block that re-raises error_class.

    Args:
        fn (function): function to wrapped
        error_class (Exception): Error class to be re-raised

    Returns:
        (object): fn wrapped in a try catch.
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            six.reraise(error_class, error_class(e), sys.exc_info()[2])

    return wrapper
