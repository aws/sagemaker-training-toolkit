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

import textwrap

from sagemaker_containers import encoders, env, worker


class BaseTransformer(object):
    """A Transformer is a proxy between the worker and the framework transformation functions.

    The BaseTransformer implements the following functions:

        functions required by worker.Worker:

            initialize, transform

        functions required by the frameworks:

            input_fn, output_fn, transform_fn

    Examples:
    >>>import os

    >>>from sagemaker_containers import env, modules, transformers
    >>>import Keras

    >>>serving_env = env.ServingEnv()

    >>>class KerasTransformer(transformers.BaseTransformer):
    >>>     def predict_fn(self, model, data):
    >>>         return model.predict(data)
    >>>
    >>>     def model_fn(self, model_dir):
    >>>         return Keras.models.load_model(os.path.join(model_dir, 'minimlmodel'))

    >>>transformer = KerasTransformer()

    >>>mod = modules.download_and_import(serving_env.module_dir, serving_env.module_name)

    >>>transformer.load_user_fns(mod)
    """

    def __init__(self):
        self._model = None
        self._call = None

    def model_fn(self, model_dir):
        """Function responsible to load the model.
            For more information about model loading https://github.com/aws/sagemaker-python-sdk#model-loading.

        Args:
            model_dir (str): The directory where model files are stored.

        Returns:
            (obj) the loaded model.
        """
        raise NotImplementedError(textwrap.dedent("""
        Please provide a model_fn implementation.
        See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
        """))

    @staticmethod
    def input_fn(input_data, content_type):
        """Takes request data and de-serializes the data into an object for prediction.

            When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
            the model server receives two pieces of information:

                - The request Content-Type, for example "application/json"
                - The request data content, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.

            The input_fn is responsible to take the request data and pre-process it before prediction.

        Args:
            input_data (obj): the request data content.
            content_type (str): the request Content-Type.

        Returns:
            (obj): data ready for prediction.
        """
        return encoders.default_decoder.decode(input_data, content_type)

    def predict_fn(self, model, data):
        """Function responsible for model predictions.

        Args:
            model (obj): model loaded by model_fn
            data: de-serializes data returned by input_fn

         Returns:
             (obj): data ready for prediction.
        """
        raise NotImplementedError()

    @staticmethod
    def output_fn(prediction, accept):
        """Function responsible to serialize the prediction for the response.

        Args:
            prediction (obj): prediction returned by predict_fn .
            accept (str): accept content-type expected by the client.

        Returns:
            (worker.Response): a Flask response object with the following args:

                * Args:
                    response: the serialized data to return
                    accept: the content-type that the data was transformed to.
        """
        return worker.Response(encoders.default_encoder.encode(prediction, accept), accept)

    def transform_fn(self, model, input_data, content_type, accept):
        """Function responsible for input processing, prediction, and output processing.

        Args:
            model (obj): model loaded by model_fn
            input_data (obj): the request data content.
            content_type (str): the request Content-Type.
            accept: the content-type that the data was transformed to.

        Returns:
            (worker.Response): a Flask response object with the following args:

                * Args:
                    response: the serialized data to return
                    accept: the content-type that the data was transformed to.
        """
        data = self.input_fn(input_data=input_data, content_type=content_type)
        prediction = self.predict_fn(model=model, data=data)
        return self.output_fn(prediction=prediction, accept=accept)

    def initialize(self):  # type: () -> None
        """Execute any initialization necessary to start making predictions with the Transformer.

        The default implementation is used to load the model.

        This function is called by sagemaker_containers.worker.Worker, before starting the Flask application.
        The gunicorn server forks multiple workers, executing multiple Flask applications in parallel.
        This function will be called once per each worker.

        It does not have return type or arguments.
        """
        self._model = self.model_fn(model_dir=env.ServingEnv().model_dir)

    def transform(self):  # type: () -> worker.Response
        """Responsible to make predictions against the model.

        Returns:
            (worker.Response): a Flask response object with the following args:

                * Args:
                    response: the serialized data to return
                    accept: the content-type that the data was transformed to.
        """
        request = worker.Request()

        result = self.transform_fn(self._model, request.content, request.content_type, request.accept)

        if not isinstance(result, worker.Response):
            # transforms tuple in Response for backwards compatibility
            return worker.Response(response=result[0], accept=result[1])

        return result
