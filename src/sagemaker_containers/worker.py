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

import collections

from flask import Flask, Response

from sagemaker_containers import env, status_codes

serving_env = env.ServingEnv()


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
    return Response(status=status_codes.OK)


def run(transform_fn, initialize_fn=None, healthcheck_fn=None, module_name=None):
    # type: (function, function or None, str or None) -> Flask
    """Creates and Flask application from a transformer.

    Args:
        transform_fn (function): responsible to make predictions against the model. Follows the signature:

            * Returns:
                `sagemaker_containers.worker.TransformSpec`: named tuple with prediction data.


        initialize_fn (function, optional): this function is called when the Flask application starts.
            It doest not have return type or arguments.

        healthcheck_fn (function, optional): function that will be used for healthcheck calls when the containers
            starts, if not specified, it will use ping as the default healthcheck call. Signature:

            * Returns:
                `flask.app.Response`: response object with new healthcheck response.

        module_name (str): the module name which implements the worker. If not specified, it will use
                                sagemaker_containers.ServingEnv().module_name as the default module name.

    Returns:
        (Flask): an instance of Flask ready for inferences.
    """
    app = Flask(import_name=module_name or serving_env.module_name)

    if initialize_fn:
        initialize_fn()

    def invocations_fn():
        transform_spec = transform_fn()

        return Response(response=transform_spec.prediction,
                        status=status_codes.OK,
                        mimetype=transform_spec.accept)

    app.add_url_rule(rule='/invocations', endpoint='invocations', view_func=invocations_fn, methods=["POST"])
    app.add_url_rule(rule='/ping', endpoint='ping', view_func=healthcheck_fn or default_healthcheck_fn)

    return app


TransformSpec = collections.namedtuple('TransformSpec', 'prediction accept')
