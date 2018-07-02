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

import os
import signal
import subprocess

import pkg_resources

import sagemaker_containers
from sagemaker_containers import _env

UNIX_SOCKET_BIND = 'unix:/tmp/gunicorn.sock'
HTTP_BIND = '0.0.0.0:8080'


def _add_sigterm_handler(nginx, gunicorn):
    def _terminate(signo, frame):
        if nginx:
            try:
                os.kill(nginx.pid, signal.SIGQUIT)
            except OSError:
                pass

        try:
            os.kill(gunicorn.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)


def start(module_app):
    env = _env.ServingEnv()
    gunicorn_bind_address = HTTP_BIND

    nginx = None

    if env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        nginx_config_file = pkg_resources.resource_filename(sagemaker_containers.__name__,
                                                            '/etc/nginx.conf')
        nginx = subprocess.Popen(['nginx', '-c', nginx_config_file])

    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(env.model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', gunicorn_bind_address,
                                 '--worker-connections', str(1000 * env.model_server_workers),
                                 '-w', str(env.model_server_workers),
                                 '--log-level', 'info',
                                 module_app])

    _add_sigterm_handler(nginx, gunicorn)

    # wait for child processes. if either exit, so do we.
    pids = {c.pid for c in [nginx, gunicorn] if c}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
