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

import signal
import subprocess
import sys

import pkg_resources

import sagemaker_containers
from sagemaker_containers import _env

UNIX_SOCKET_BIND = 'unix:/tmp/gunicorn.sock'
HTTP_BIND = '0.0.0.0:8080'


def add_terminate_signal(process):
    def terminate(signal_number, stack_frame):
        process.terminate()

    signal.signal(signal.SIGTERM, terminate)


def start(module_app):

    env = _env.ServingEnv()
    gunicorn_bind_address = HTTP_BIND

    nginx = None

    if env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        nginx_config_file = pkg_resources.resource_filename(sagemaker_containers.__name__, '/etc/nginx.conf')
        nginx = subprocess.Popen(['nginx', '-c', nginx_config_file])

        add_terminate_signal(nginx)

    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(env.model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', gunicorn_bind_address,
                                 '--worker-connections', str(1000 * env.model_server_workers),
                                 '-w', str(env.model_server_workers),
                                 '--log-level', 'info',
                                 module_app])

    add_terminate_signal(gunicorn)

    while True:
        if nginx and nginx.poll():
            nginx.terminate()
            break
        elif gunicorn.poll():
            gunicorn.terminate()
            break

    sys.exit(0)
