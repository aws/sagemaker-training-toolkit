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

import flask
from six.moves import http_client

from sagemaker_containers import _content_types, _env, _logging, _mapping

env = _env.ServingEnv()


def default_healthcheck_fn():  # type: () -> Response
    """Ping is default health-check handler. Returns 200 with no content.

    During a new serving container startup, Amazon SageMaker starts sending periodic GET requests to the /ping endpoint
    to ensure that the container is ready for predictions.

    The simplest requirement on the container is to respond with an HTTP 200 status code and an empty body. This
    indicates to Amazon SageMaker that the container is ready to accept inference requests at the /invocations endpoint.

    If the container does not begin to consistently respond with 200s during the first 30 seconds after startup,
    the CreateEndPoint and UpdateEndpoint APIs will fail.

    While the minimum bar is for the container to return a static 200, a container developer can use this functionality
    to perform deeper checks. The request timeout on /ping attempts is 2 seconds.

    More information on how health-check works can be found here:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests

    Returns:
        (flask.Response): with status code 200
    """
    return Response(status=http_client.OK)


class Worker(flask.Flask):
    """Flask application that receives predictions from a Transformer ready for inferences."""

    def __init__(self, transform_fn, initialize_fn=None, module_name=None, healthcheck_fn=None):
        """Creates and Flask application from a transformer.

        Args:
            transform_fn (function): responsible to make predictions against the model. Follows the signature:

                * Returns:
                    `sagemaker_containers.transformers.TransformSpec`: named tuple with prediction data.


            initialize_fn (function, optional): this function is called when the Flask application starts.
                It doest not have return type or arguments.

            healthcheck_fn (function, optional): function that will be used for healthcheck calls when the containers
                starts, if not specified, it will use ping as the default healthcheck call. Signature:

                * Returns:
                    `flask.app.Response`: response object with new healthcheck response.

            module_name (str): the module name which implements the worker. If not specified, it will use
                                    sagemaker_containers.ServingEnv().module_name as the default module name.
        """
        super(Worker, self).__init__(module_name or env.module_name)

        # the logger is configured after importing the framework library, allowing the framework to
        # configure logging at import time.
        _logging.configure_logger(env.log_level)

        if initialize_fn:
            self.before_first_request(initialize_fn)

        self.add_url_rule(rule='/invocations', endpoint='invocations', view_func=transform_fn, methods=["POST"])
        self.add_url_rule(rule='/ping', endpoint='ping', view_func=healthcheck_fn or default_healthcheck_fn)

        self.request_class = Request


class Response(flask.Response):
    default_mimetype = _content_types.JSON

    def __init__(self, response=None, accept=None, status=http_client.OK, headers=None,
                 mimetype=None, direct_passthrough=False):
        headers = headers or {}
        headers['accept'] = accept
        super(Response, self).__init__(response, status, headers, mimetype, accept, direct_passthrough)


class Request(flask.Request, _mapping.MappingMixin):
    """The Request object used to read request data.

    Example:

    POST /invocations
    Content-Type: 'application/json'.
    Accept: 'application/json'.

    42

    >>> from sagemaker_containers import _env

    >>> request = Request()
    >>> data = request.data

    >>> print(str(request))

    {'content_length': '2', 'content_type': 'application/json', 'data': '42', 'accept': 'application/json', ... }


    """
    default_mimetype = _content_types.JSON

    def __init__(self, environ=None, serving_env=None):  # type: (dict, _env.ServingEnv) -> None
        super(Request, self).__init__(environ=environ or flask.request.environ)

        serving_env = serving_env or env

        self._default_accept = serving_env.default_accept

    @property
    def content_type(self):  # type: () -> str
        """The request's content-type.

        Returns:
            (str): The value, if any, of the header 'ContentType' (used by some AWS services) and 'Content-Type'.
                    Otherwise, returns 'application/json' as default.
        """
        # todo(mvsusp): consider a better default content-type
        return self.headers.get('ContentType') or self.headers.get('Content-Type') or _content_types.JSON

    @property
    def accept(self):  # type: () -> str
        """The content-type for the response to the client.

        Returns:
            (str): The value of the header 'Accept' or the user-supplied SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT
                    environment variable.
        """
        accept = self.headers.get('Accept')

        if not accept or accept == _content_types.ANY:
            return self._default_accept
        else:
            return accept

    @property
    def content(self):  # type: () -> object
        """The request incoming data.

        It automatic decodes from utf-8

        Returns:
            (obj): incoming data
        """
        as_text = self.content_type in _content_types.UTF8_TYPES

        return self.get_data(as_text=as_text)
