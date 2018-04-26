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

import sagemaker_containers as smc


@pytest.mark.parametrize('dictionary, keys, expected', [
    ({}, (), ({}, {})),
    ({'x': 1, 'y': 2}, 'x', ({'x': 1}, {'y': 2})),
    ({'x': 1, 'y': 2}, (), ({}, {'x': 1, 'y': 2})),
    ({'x': 1, 'y': 2}, ('x', 'y'), ({'x': 1, 'y': 2}, {}))
])
def test_split_by_criteria(dictionary, keys, expected):
    assert smc.collections.split_by_criteria(dictionary, keys) == expected
