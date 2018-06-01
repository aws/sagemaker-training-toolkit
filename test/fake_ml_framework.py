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
import numpy as np

from sagemaker_containers import _files
import test


class Model(object):
    def __init__(self, weights=None, bias=1, loss=None, optimizer=None, epochs=None, batch_size=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.weights = weights
        self.bias = bias

    def fit(self, x, y, epochs=None, batch_size=None):
        self.weights = (y / x + self.bias).tolist()
        self.epochs = epochs
        self.batch_size = batch_size

    def save(self, model_dir):
        test.write_json(self.__dict__, model_dir)

    @classmethod
    def load(cls, model_dir):
        clazz = cls()
        clazz.__dict__ = _files.read_json(model_dir)
        return clazz

    def predict(self, data):
        return np.asarray(self.weights) * np.asarray(data)
