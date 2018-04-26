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


def split_by_criteria(dictionary, keys):  # type: (dict, set) -> (dict, dict)
    """Split a dictionary in two by the provided keys.

    Args:
        dictionary (dict[str, object]): A Python dictionary
        keys (set[str]): Set of keys which will be the split criteria

    Returns:
        criteria (dict[string, object]), not_criteria (dict[string, object]): the result of the split criteria.
    """
    dict_matching_criteria = {k: dictionary[k] for k in dictionary.keys() if k in keys}
    dict_not_matching_criteria = {k: dictionary[k] for k in dictionary.keys() if k not in keys}

    return dict_matching_criteria, dict_not_matching_criteria
