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
"""Placeholder docstring"""
from __future__ import absolute_import

import os
import re
import signal
import subprocess
import sys

import pkg_resources

import sagemaker_training
from sagemaker_training import env, files, logging_config, modules

logger = logging_config.get_logger()

UNIX_SOCKET_BIND = "unix:/tmp/gunicorn.sock"

nginx_config_file = os.path.join("/etc", "sagemaker-nginx.conf")
nginx_config_template_file = pkg_resources.resource_filename(
    sagemaker_training.__name__, "/etc/nginx.conf.template"
)


def _create_nginx_config(serving_env):
    """Placeholder docstring"""
    template = files.read_file(nginx_config_template_file)

    pattern = re.compile(r"%(\w+)%")
    template_values = {
        "NGINX_HTTP_PORT": serving_env.http_port,
        "NGINX_PROXY_READ_TIMEOUT": str(serving_env.model_server_timeout),
    }

    config = pattern.sub(lambda x: template_values[x.group(1)], template)

    logger.info("nginx config: \n%s\n", config)

    files.write_file(nginx_config_file, config)


def _add_sigterm_handler(nginx, gunicorn):
    """Placeholder docstring"""

    def _terminate(signo, frame):  # pylint: disable=unused-argument
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
    """Placeholder docstring"""
    serving_env = env.ServingEnv()
    gunicorn_bind_address = "0.0.0.0:{}".format(serving_env.http_port)

    nginx = None

    if serving_env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        _create_nginx_config(serving_env)
        nginx = subprocess.Popen(["nginx", "-c", nginx_config_file])

    # Install user module before starting GUnicorn
    if serving_env.module_name:
        modules.import_module(serving_env.module_dir, serving_env.module_name)

    pythonpath = ",".join(sys.path + [env.code_dir])

    gunicorn = subprocess.Popen(
        [
            "gunicorn",
            "--timeout",
            str(serving_env.model_server_timeout),
            "-k",
            "gevent",
            "--pythonpath",
            pythonpath,
            "-b",
            gunicorn_bind_address,
            "--worker-connections",
            str(1000 * serving_env.model_server_workers),
            "-w",
            str(serving_env.model_server_workers),
            "--log-level",
            "info",
            module_app,
        ]
    )

    _add_sigterm_handler(nginx, gunicorn)

    # wait for child processes. if either exit, so do we.
    pids = {c.pid for c in [nginx, gunicorn] if c}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break


def next_safe_port(port_range, after=None):
    """Placeholder docstring"""
    first_and_last_port = port_range.split("-")
    first_safe_port = int(first_and_last_port[0])
    last_safe_port = int(first_and_last_port[1])
    safe_port = first_safe_port
    if after:
        safe_port = int(after) + 1
        if safe_port < first_safe_port or safe_port > last_safe_port:
            raise ValueError(
                "{} is outside of the acceptable port range for SageMaker: {}".format(
                    safe_port, port_range
                )
            )

    return str(safe_port)
