# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import test


class Model(object):
    x = None
    y = None

    def __init__(self, **kwargs):
        self.parameters = kwargs

    def fit(self, x, y, **kwargs):
        self.parameters.update(kwargs)
        self.parameters['x'] = x.tolist()
        self.parameters['y'] = y.tolist()

    def save(self, model_dir):
        test.write_json(self.parameters, model_dir)
