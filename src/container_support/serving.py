#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import json
import logging
import os
import pkg_resources
import shutil
import signal
import subprocess
import sys

from flask import Flask, request, Response

import container_support as cs

logger = logging.getLogger(__name__)

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
OCTET_STREAM_CONTENT_TYPE = "application/octet-stream"
ANY_CONTENT_TYPE = '*/*'
UTF8_CONTENT_TYPES = [JSON_CONTENT_TYPE, CSV_CONTENT_TYPE]


class Server(object):
    """A simple web service wrapper for custom inference code.
    """

    def __init__(self, name, transformer):
        """ Initialize the web service instance.

        :param name: the name of the service
        :param transformer: a function that transforms incoming request data to
                            an outgoing inference response.
        """
        self.transformer = transformer
        self.app = self._build_flask_app(name)
        self.log = self.app.logger

    @classmethod
    def from_env(cls):
        cs.configure_logging()
        logger.info("creating Server instance")
        env = cs.HostingEnvironment()

        user_module = env.import_user_module() if env.user_script_name else None

        framework = cs.ContainerEnvironment.load_framework()
        transformer = framework.transformer(user_module)

        server = Server("model server", transformer)
        logger.info("returning initialized server")
        return server

    @classmethod
    def start(cls):
        """Prepare the container for model serving, configure and launch the model server stack.
        """

        logger.info("reading config")
        env = cs.HostingEnvironment()
        env.start_metrics_if_enabled()

        if env.user_script_name:
            Server._download_user_module(env)
            env.pip_install_requirements()

        logger.info("importing user module")
        logger.info('loading framework-specific dependencies')
        framework = cs.ContainerEnvironment.load_framework()
        framework.load_dependencies()

        nginx_pid = 0
        gunicorn_bind_address = '0.0.0.0:8080'
        if env.use_nginx:
            logger.info("starting nginx")
            nginx_conf = pkg_resources.resource_filename('container_support', 'etc/nginx.conf')
            subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
            subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
            gunicorn_bind_address = 'unix:/tmp/gunicorn.sock'
            nginx_pid = subprocess.Popen(['nginx', '-c', nginx_conf]).pid

        logger.info("starting gunicorn")
        gunicorn_pid = subprocess.Popen(["gunicorn",
                                         "--timeout", str(env.model_server_timeout),
                                         "-k", "gevent",
                                         "-b", gunicorn_bind_address,
                                         "--worker-connections", str(1000 * env.model_server_workers),
                                         "-w", str(env.model_server_workers),
                                         "container_support.wsgi:app"]).pid

        signal.signal(signal.SIGTERM, lambda a, b: Server._sigterm_handler(nginx_pid, gunicorn_pid))

        children = set([nginx_pid, gunicorn_pid]) if nginx_pid else gunicorn_pid
        logger.info("inference server started. waiting on processes: %s" % children)

        while True:
            pid, _ = os.wait()
            if pid in children:
                break

        Server._sigterm_handler(nginx_pid, gunicorn_pid)

    @classmethod
    @cs.retry(stop_max_delay=1000 * 60 * 10,
              wait_exponential_multiplier=100,
              wait_exponential_max=60000)
    def _download_user_module(cls, env):
        Server._download_user_module_internal(env)

    @classmethod
    def _download_user_module_internal(cls, env):
        path = os.path.join(env.code_dir, env.user_script_name)
        if os.path.exists(path):
            return

        try:
            env.download_user_module()
        except:  # noqa
            try:
                shutil.rmtree(env.code_dir)
            except OSError:
                pass
            raise

    @staticmethod
    def _sigterm_handler(nginx_pid, gunicorn_pid):
        logger.info("stopping inference server")

        if nginx_pid:
            try:
                os.kill(nginx_pid, signal.SIGQUIT)
            except OSError:
                pass

        try:
            os.kill(gunicorn_pid, signal.SIGTERM)
        except OSError:
            pass

        sys.exit(0)

    def _build_flask_app(self, name):
        """ Construct the Flask app that will handle requests.

        :param name: the name of the service
        :return: a Flask app ready to handle requests
        """
        app = Flask(name)
        app.add_url_rule('/ping', 'healthcheck', self._healthcheck)
        app.add_url_rule('/invocations', 'invoke', self._invoke, methods=["POST"])
        app.register_error_handler(Exception, self._default_error_handler)
        return app

    def _invoke(self):
        """Handles requests by delegating to the transformer function.

        :return: 200 response, with transformer result in body.
        """

        # Accepting both ContentType and Content-Type headers. ContentType because Coral and Content-Type because,
        # well, it is just the html standard
        input_content_type = request.headers.get('ContentType', request.headers.get('Content-Type', JSON_CONTENT_TYPE))
        requested_output_content_type = request.headers.get('Accept', JSON_CONTENT_TYPE)

        # utf-8 decoding is automatic in Flask if the Content-Type is valid. But that does not happens always.
        content = request.get_data().decode('utf-8') if input_content_type in UTF8_CONTENT_TYPES else request.get_data()

        try:
            response_data, output_content_type = \
                self.transformer.transform(content, input_content_type, requested_output_content_type)
            # OK
            ret_status = 200
        except Exception as e:
            ret_status, response_data = self._handle_invoke_exception(e)
            output_content_type = JSON_CONTENT_TYPE

        return Response(response=response_data,
                        status=ret_status,
                        mimetype=output_content_type)

    def _handle_invoke_exception(self, e):
        data = json.dumps(str(e))
        if isinstance(e, UnsupportedContentTypeError):
            # Unsupported Media Type
            return 415, data
        elif isinstance(e, UnsupportedAcceptTypeError):
            # Not Acceptable
            return 406, data
        elif isinstance(e, UnsupportedInputShapeError):
            # Precondition Failed
            return 412, data
        else:
            self.log.exception(e)
            raise e

    @staticmethod
    def _healthcheck():
        """Default healthcheck handler. Returns 200 status with no content. Note that the
        `InvokeEndpoint API`_ contract requires that the service only returns 200 when
        it is ready to start serving requests.

        :return: 200 response if the serer is ready to handle requests.
        """
        return '', 200

    def _default_error_handler(self, exception):
        """ Default error handler. Returns 500 status with no content.

        :param exception: the exception that triggered the error
        :return: 500 response
        """

        self.log.error(exception)
        return '', 500


class Transformer(object):
    """A ``Transformer`` encapsulates the function(s) responsible for parsing incoming request data,
    passing it through a prediction function, and converting the result into something
    that can be returned as the body of an HTTP response.
    """

    def __init__(self, transform_fn=lambda x, y, z: (x, z)):
        self.transform_fn = transform_fn

    def transform(self, data, input_content_type, output_content_type):
        """Transforms input data into a prediction result. The input data must
        be in a format compatible with the configured ``input_fn``. The output format
        will be determined by the ``output_fn``.

        :param data: input data
        :param input_content_type: content type of input specified in request header
        :param output_content_type: requested content type of output specified in request header
        :return: the transformed result
        """
        return self.transform_fn(data, input_content_type, output_content_type)


class UnsupportedContentTypeError(Exception):
    def __init__(self, *args, **kwargs):
        self.message = 'Requested unsupported ContentType: ' + args[0]
        super(Exception, self).__init__(self.message, *args, **kwargs)


class UnsupportedAcceptTypeError(Exception):
    def __init__(self, *args, **kwargs):
        self.message = 'Requested unsupported ContentType in Accept: ' + args[0]
        super(Exception, self).__init__(self.message, *args, **kwargs)


class UnsupportedInputShapeError(Exception):
    def __init__(self, *args, **kwargs):
        self.message = 'Model can have only 1 input data, but it has: ' + str(args[0])
        super(Exception, self).__init__(self.message, *args, **kwargs)
