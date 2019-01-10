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
import argparse
import inspect
import logging
import os
import subprocess
import time
from typing import Any, List, Tuple  # noqa ignore=F401 imported but unused

import paramiko
import psutil

import gethostname
from sagemaker_containers import _logging, _process, _timeout

logger = _logging.get_logger()
logging.getLogger("paramiko").setLevel(logging.INFO)


class WorkerRunner(_process.ProcessRunner):
    """Runner responsible for preparing MPI distributed training and waiting for MPI
     master execution to finish.
    """

    def __init__(self, user_entry_point, args, env_vars, master_hostname):
        super(WorkerRunner, self).__init__(user_entry_point, args, env_vars)
        self._master_hostname = str(master_hostname)

    def run(self, wait=True, capture_error=False):  # type: (bool, bool) -> None
        """The WorkerRunner proceeds as following:

        - wait for the MPI Master to create its SSH daemon
        - start its SSH daemon
        - monitor the MPI orted process and wait it to finish the MPI execution
        """
        logger.info('Starting MPI run as worker node.')
        if wait:
            logger.info('Waiting for MPI Master to create SSH daemon.')
            self._wait_master_to_start()
        logger.info('MPI Master online, creating SSH daemon.')

        _start_sshd_daemon()

        if wait:
            logger.info('Waiting for MPI process to finish.')
            _wait_orted_process_to_finish()
            time.sleep(30)
        logger.info('MPI process finished.')

    def _wait_master_to_start(self):  # type: () -> None
        while not _can_connect(self._master_hostname):
            time.sleep(1)

    def _wait_master_to_finish(self):  # type: () -> None
        while _can_connect(self._master_hostname):
            time.sleep(30)


def _wait_orted_process_to_finish():  # type: () -> None
    orted = _orted_process()
    psutil.wait_procs(orted)


def _orted_process():
    """Waits maximum of 5 minutes for orted process to start"""
    for i in range(5 * 60):
        procs = [p for p in psutil.process_iter(attrs=['name']) if p.info['name'] == 'orted']

        if procs:
            return procs
        time.sleep(1)


class MasterRunner(_process.ProcessRunner):
    """Responsible to prepare MPI distributed training and syncronize work with the Workers.
    """

    def __init__(self, user_entry_point, args, env_vars, master_hostname, hosts, process_per_host,
                 custom_mpi_options, network_interface_name, interval=1,
                 timeout_in_seconds=60 * 60):

        super(MasterRunner, self).__init__(user_entry_point, args, env_vars)

        self._master_hostname = master_hostname
        self._hosts = hosts
        self._process_per_host = process_per_host
        self._custom_mpi_options = custom_mpi_options
        self._network_interface_name = network_interface_name
        self._interval = interval
        self.timeout_in_seconds = timeout_in_seconds

    def _setup(self):  # type: () -> None
        logger.info('Starting MPI run as worker node.')
        logger.info('Creating SSH daemon.')
        _start_sshd_daemon()

        self._wait_for_workers()

    def _wait_for_workers(self):  # type: () -> None
        logger.info('Waiting for MPI workers to establish their SSH connections')

        workers = [host for host in self._hosts if host != self._master_hostname]
        with _timeout.timeout(seconds=self.timeout_in_seconds):
            for host in workers:
                while not _can_connect(host):
                    time.sleep(self._interval)
                logger.info('Worker %s available for communication', host)

    def _create_command(self):  # type: () -> List[str, Any]
        num_hosts = len(self._hosts)
        num_processes = self._process_per_host * num_hosts

        # By default, use one process per GPU, or one process per node (if training with CPU).
        if self._process_per_host == 1:
            host_list = self._hosts
        else:
            host_list = ['%s:%s' % (host, self._process_per_host) for host in self._hosts]

        msg = 'Env Hosts: %s Hosts: %s process_per_hosts: %s num_processes: %s'
        logger.info(msg, self._hosts, host_list, self._process_per_host, num_processes)

        overridden_known_options, additional_options = _parse_custom_mpi_options(
            self._custom_mpi_options)

        logger.info("Network interface name: %s" % self._network_interface_name)

        # TODO(mvs): explain MPI setttings
        command = ['mpirun',
                   '--host', ','.join(host_list),
                   '-np', str(num_processes),

                   '--allow-run-as-root',
                   '--display-map',
                   '--tag-output',

                   '-mca', 'btl_tcp_if_include', self._network_interface_name,
                   '-mca', 'oob_tcp_if_include', self._network_interface_name,
                   '-mca', 'plm_rsh_no_tree_spawn', '1',
                   '-bind-to', 'socket', '-map-by', 'slot',
                   '-mca', 'pml', 'ob1', '-mca', 'btl', '^openib',
                   '-mca', 'orte_abort_on_non_zero_status', '1',

                   '-x', 'NCCL_MIN_NRINGS=4',
                   '-x', 'NCCL_SOCKET_IFNAME=%s' % self._network_interface_name,
                   '-x', 'NCCL_DEBUG=%s' % overridden_known_options.NCCL_DEBUG,
                   '-x', 'LD_LIBRARY_PATH',
                   '-x', 'PATH',
                   '-x', 'LD_PRELOAD=%s' % inspect.getfile(gethostname),

                   ]

        command.extend(additional_options)

        for credential in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']:
            if credential in os.environ:
                command.extend(['-x', credential])

        for name in self._env_vars:
            command.extend(['-x', name])

        command.extend(super(MasterRunner, self)._create_command())

        return command


_SSH_DAEMON_NOT_FOUND_ERROR_MESSAGE = """
SSH daemon not found, please install SSH to allow MPI to communicate different nodes in cluster.

You can install ssh by running following commands:
-------------------------------------------------

1. Install SSH via apt-get:

apt-get update && apt-get install -y --no-install-recommends openssh-server && mkdir -p /var/run/sshd

2. SSH login fix. Otherwise user is kicked off after login:
sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

3. Create SSH key to allow password less ssh between different docker instances:
mkdir -p /root/.ssh/ && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config
"""


def _start_sshd_daemon():  # type: () -> None
    sshd_executable = '/usr/sbin/sshd'

    if not os.path.exists(sshd_executable):
        raise RuntimeError(_SSH_DAEMON_NOT_FOUND_ERROR_MESSAGE)

    subprocess.Popen([sshd_executable, '-D'])


def _can_connect(host, port=22):  # type: (str, int) -> bool
    """Checks if the connection to provided ``host`` and ``port`` is possible or not.
       Args:
           host (str): Hostname for the host to check connection.
           port (int): Port name of the host to check connection on.
    """
    try:
        logger.debug('Testing connection to host %s', host)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host,
                       port=port)
        client.close()
        logger.info('Can connect to host %s', host)
        return True
    except Exception as e:
        logger.info('Cannot connect to host %s', host)

        logger.info('Connection failed with exception: \n %s', str(e))
        return False


def _parse_custom_mpi_options(custom_mpi_options):
    # type: (str) -> Tuple[argparse.Namespace, List[str]]
    """Parse custom MPI options provided by user. Known options default value will be overridden
    and unknown options would be identified separately."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--NCCL_DEBUG', default="INFO", type=str)

    return parser.parse_known_args(custom_mpi_options.split())
