# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Contains functionality related to SM Distributed Data Parallel Training."""
import inspect
import json
import logging
import os
import subprocess
import time

import paramiko

import gethostname
from sagemaker_training import environment, errors, logging_config, process, timeout

logger = logging_config.get_logger()
logging.getLogger("paramiko").setLevel(logging.INFO)


class SMDataParallelRunner(process.ProcessRunner):
    """Prepare SMDataParallel-based distributed training.

    This includes setup of smddprun command via MPI and synchronizing work
    with the worker nodes.
    """

    def __init__(
        self,
        user_entry_point,
        args,
        env_vars,
        master_hostname,
        hosts,
        network_interface_name,
        interval=1,
        timeout_in_seconds=60 * 60,
    ):
        """Initialize a SMDataParallelRunner.

        SMDataParallelRunner is responsible for preparing distributed
        training with MPI and synchronizing work among the Workers.

        Args:
            user_entry_point (str): The name of the user entry point.
            args ([str]): A list of arguments to include when executing the entry point.
            env_vars (Dict[str, str]): A dictionary of environment variables.
            master_hostname (str): The master hostname.
            hosts ([str]): A list of hosts.
            network_interface_name (str): The network interface name.
            interval (int or float): The interval at which to check the connection in seconds.
                Defaults to 1 second.
            timeout_in_seconds (int): The number of seconds to wait for workers. Defaults to
                3600 seconds (ie. 1 hour).
        """

        super(SMDataParallelRunner, self).__init__(user_entry_point, args, env_vars)

        self._master_hostname = master_hostname
        self._hosts = hosts
        self._network_interface_name = network_interface_name
        self._interval = interval
        self.timeout_in_seconds = timeout_in_seconds

    def _setup(self):  # type: () -> None
        logger.info("Starting MPI run as worker node.")
        logger.info("Creating SSH daemon.")
        _start_sshd_daemon()

        self._wait_for_workers()

    def _wait_for_workers(self):  # type: () -> None
        logger.info("Waiting for MPI workers to establish their SSH connections")

        workers = [host for host in self._hosts if host != self._master_hostname]
        with timeout.timeout(seconds=self.timeout_in_seconds):
            for host in workers:
                while not _can_connect(host):
                    time.sleep(self._interval)
                logger.info("Worker %s available for communication", host)

    def _get_mpirun_command(
        self,
        num_hosts,
        host_list,
        smdataparallel_flag,
        num_processes,
        smdataparallel_server_addr=None,
        smdataparallel_server_port=None,
    ):
        """Fetch mpi command for SMDataParallel

        TODO: add flag for toggling NCCL_DEBUG
        """
        mpirun_command = [
            "mpirun",
            "--host",
            ",".join(host_list),
            "-np",
            str(num_processes),
            "--allow-run-as-root",
            "--tag-output",
            "--oversubscribe",
            "-mca",
            "btl_tcp_if_include",
            self._network_interface_name,
            "-mca",
            "oob_tcp_if_include",
            self._network_interface_name,
            "-mca",
            "plm_rsh_no_tree_spawn",
            "1",
            "-mca",
            "pml",
            "ob1",
            "-mca",
            "btl",
            "^openib",
            "-mca",
            "orte_abort_on_non_zero_status",
            "1",
            "-mca",
            "plm_rsh_num_concurrent",
            str(num_hosts),
            "-x",
            "NCCL_SOCKET_IFNAME=%s" % self._network_interface_name,
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            "-x",
            smdataparallel_flag,
            "-x",
            "FI_PROVIDER=sockets",
            "-x",
            "RDMAV_FORK_SAFE=1",
            "-x",
            "LD_PRELOAD=%s" % inspect.getfile(gethostname),
        ]

        if smdataparallel_server_addr and smdataparallel_server_port:
            # in case of multi-node [distributed] training, smdataparallel_server_addr,
            # smdataparallel_server_port and interconnect_bandwidth will need to be set

            instance_type = self._get_instance_type()

            mpirun_command.extend(
                [
                    "-x",
                    "SMDATAPARALLEL_SERVER_ADDR={}".format(smdataparallel_server_addr),
                    "-x",
                    "SMDATAPARALLEL_SERVER_PORT={}".format(smdataparallel_server_port),
                    "-x",
                    "SAGEMAKER_INSTANCE_TYPE={}".format(instance_type),
                ]
            )

        smddprun_command = ["smddprun"]
        mpirun_command.extend(smddprun_command)
        return mpirun_command

    def _get_instance_type(self):
        """Get instance type"""
        sm_training_env = json.loads(self._env_vars.get("SM_TRAINING_ENV"))
        instance_type = sm_training_env.get("additional_framework_parameters").get(
            "sagemaker_instance_type"
        )
        logger.info("instance type: %s" % instance_type)
        return instance_type

    def _create_command(self):
        """Create mpi-based smddprun command.

        Based on the number of hosts, smddprun command differs.
        Single-node: SMDATAPARALLEL_USE_SINGLENODE flag set to 1
        Multi-node: SMDATAPARALLEL_USE_HOMOGENEOUS flag set to 1
        """
        host_list = self._hosts
        num_hosts = len(self._hosts)
        # SMDATAPARALLEL expects instances to have 8 GPUs
        # each GPU will run 1 process
        num_processes_per_host = 8
        num_processes = num_hosts * num_processes_per_host
        logger.info("Network interface name: %s" % self._network_interface_name)
        logger.info("Host: %s" % self._hosts)
        if num_hosts > 1:
            # multi-node; use homogeneous
            # homogeneous mode uses 16 processes per host; 8 server; 8 worker
            smdataparallel_server_addr = self._master_hostname
            smdataparallel_server_port = 7592
            host_list = ["{}:{}".format(host, num_processes_per_host) for host in self._hosts]
            smdataparallel_flag = "SMDATAPARALLEL_USE_HOMOGENEOUS=1"
            command = self._get_mpirun_command(
                num_hosts,
                host_list,
                smdataparallel_flag,
                num_processes,
                smdataparallel_server_addr,
                smdataparallel_server_port,
            )
        else:
            # single-node
            smdataparallel_flag = "SMDATAPARALLEL_USE_SINGLENODE=1"
            command = self._get_mpirun_command(
                num_hosts, host_list, smdataparallel_flag, num_processes
            )

        return command

    def _python_command(self):
        """Use mpi4py to force processes to abort if an uncaught exception occurs.

        https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#mpi-process-hangs-after-an-unhandled-python-exception
        """
        return super(SMDataParallelRunner, self)._python_command() + ["-m", "mpi4py"]

    def run(self, wait=True, capture_error=False):
        """Run the process.

        Args:
            wait (bool): A boolean indicating whether to wait and check for errors.
                Defaults to True.
            capture_error (bool): A boolean indicating whether to direct stderr to a stream
                that can later be read. Defaults to False.

        Returns:
            process (subprocess.Popen): The spawned process.
        """
        self._setup()

        cmd = self._create_command()
        cmd.extend(super(SMDataParallelRunner, self)._create_command())
        logging_config.log_script_invocation(cmd, self._env_vars)
        if wait:
            process_spawned = process.check_error(
                cmd,
                errors.ExecuteUserScriptError,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        else:
            process_spawned = process.create(
                cmd,
                errors.ExecuteUserScriptError,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        self._tear_down()
        return process_spawned


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
    sshd_executable = "/usr/sbin/sshd"

    if not os.path.exists(sshd_executable):
        raise RuntimeError(_SSH_DAEMON_NOT_FOUND_ERROR_MESSAGE)

    subprocess.Popen([sshd_executable, "-D"])


def _can_connect(host, port=22):
    # type: (str, int) -> bool
    """Check if the connection to provided ``host`` and ``port`` is possible.

       Args:
           host (str): Hostname for the host to check connection.
           port (int): Port name of the host to check connection on.
    """
    try:
        logger.debug("Testing connection to host %s at port %s", host, port)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port)
        logger.info("Can connect to host %s at port %s", host, port)
        return True
    except Exception:  # pylint: disable=broad-except
        logger.info("Cannot connect to host %s at port %s", host, port)
        logger.exception("Connection failed")
        return False
    finally:
        client.close()
        logger.info("Connection closed")
