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

import json
import textwrap
import traceback

from six.moves import http_client

from sagemaker_containers import _encoders, _env, _errors, _functions, _worker


def default_model_fn(model_dir):
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


def default_input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.

        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.

        The input_fn is responsible to take the request data and pre-process it before prediction.

    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.

    Returns:
        (obj): data ready for prediction.
    """
    return _encoders.decode(input_data, content_type)


def default_predict_fn(data, model):
    """Function responsible for model predictions.

    Args:
        model (obj): model loaded by model_fn
        data: de-serializes data returned by input_fn

     Returns:
         (obj): data ready for prediction.
    """
    raise NotImplementedError(textwrap.dedent("""
    Please provide a predict_fn implementation.
    See documentation for predict_fn at https://github.com/aws/sagemaker-python-sdk
    """))


def default_output_fn(prediction, accept):
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
    return _worker.Response(_encoders.encode(prediction, accept), accept)


class Transformer(object):
    """The Transformer is a proxy between the worker and the framework transformation functions.

    It implements the default framework functions for serving.

    Examples:
    >>>import os
    >>>from sagemaker_containers import _env, _modules, _transformer
    >>>import Keras
    >>>ServingEnv = _env.ServingEnv()
    >>>
    >>>def predict_fn(model, data):
    >>>     return model.predict(data)
    >>>
    >>>def model_fn(model_dir):
    >>>     return Keras.models.load_model(os.path.join(model_dir, 'minimlmodel'))
    >>>
    >>>transformer = _transformer.Transformer(predict_fn=predict_fn, model_fn=model_fn)
    >>>
    >>>mod = _modules.download_and_import(ServingEnv.module_dir, ServingEnv.module_name)
    >>>transformer.load_user_fns(mod)
    """

    def __init__(self, model_fn=None, input_fn=None, predict_fn=None, output_fn=None,
                 transform_fn=None, error_class=_errors.ClientError):
        """Default constructor. Wraps the any non default framework function in an error class to isolate
        framework from user errors.

        Args:
            model_fn (fn): Function responsible to load the model.
            input_fn (fn): Takes request data and de-serializes the data into an object for prediction.
            predict_fn (fn): Function responsible for model predictions.
            output_fn (fn): Function responsible to serialize the prediction for the response.
            transform_fn (fn): Function responsible for taking input data and returning a prediction
                as a serialized response. This function takes the place of ``input_fn``,
                ``predict_fn``, and ``output_fn``.
            error_class (Exception): Error class used to separate framework and user errors.
        """
        self._model = None
        self._model_fn = _functions.error_wrapper(model_fn, error_class) if model_fn else default_model_fn

        if transform_fn and (input_fn or predict_fn or output_fn):
            raise ValueError('Cannot use transform_fn implementation with input_fn, predict_fn, and/or output_fn')

        if transform_fn is not None:
            self._transform_fn = _functions.error_wrapper(transform_fn, error_class)
        else:
            self._transform_fn = self._default_transform_fn

        self._input_fn = _functions.error_wrapper(input_fn, error_class) if input_fn else default_input_fn
        self._predict_fn = _functions.error_wrapper(predict_fn, error_class) if predict_fn else default_predict_fn
        self._output_fn = _functions.error_wrapper(output_fn, error_class) if output_fn else default_output_fn
        self._error_class = error_class

    def initialize(self):  # type: () -> None
        """Execute any initialization necessary to start making predictions with the Transformer.
        The default implementation is used to load the model.
        This function is called by sagemaker_containers.beta.framework.worker.Worker,
        before starting the Flask application.
        The gunicorn server forks multiple workers, executing multiple Flask applications in parallel.
        This function will be called once per each worker.
        It does not have return type or arguments.
        """
        self._model = self._model_fn(_env.model_dir)

    def transform(self):  # type: () -> _worker.Response
        """Take a request with input data, deserialize it, make a prediction, and return a
        serialized response.

        Returns:
            sagemaker_containers.beta.framework.worker.Response: a Flask response object with
                the following args:

                * response: the serialized data to return
                * accept: the content type that the data was serialized into
        """
        request = _worker.Request()
        result = self._transform_fn(self._model, request.content, request.content_type, request.accept)

        if isinstance(result, tuple):
            # transforms tuple in Response for backwards compatibility
            return _worker.Response(response=result[0], accept=result[1])

        return result

    def _default_transform_fn(self, model, content, content_type, accept):
        """Make predictions against the model and return a serialized response.

        This serves as the default implementation of transform_fn, used when the user has not
        implemented one themselves.

        Args:
            model (obj): model loaded by model_fn.
            content: request content.
            content_type (str): the request Content-Type.
            accept (str): accept content-type expected by the client.

        Returns:
            sagemaker_containers.beta.framework.worker.Response or tuple:
                the serialized response data and its content type, either as a Response object or
                a tuple of the form (response_data, content_type)
        """
        try:
            data = self._input_fn(content, content_type)
        except _errors.UnsupportedFormatError as e:
            return self._error_response(e, http_client.UNSUPPORTED_MEDIA_TYPE)

        prediction = self._predict_fn(data, model)

        try:
            result = self._output_fn(prediction, accept)
        except _errors.UnsupportedFormatError as e:
            return self._error_response(e, http_client.NOT_ACCEPTABLE)

        return result

    def _error_response(self, error, status_code):
        body = json.dumps({'error': error.__class__.__name__,
                           'error-message': str(error),
                           'stack-trace': traceback.format_exc()})
        return _worker.Response(response=body, status=status_code)
