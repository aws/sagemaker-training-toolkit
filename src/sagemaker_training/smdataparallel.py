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
import argparse
from inspect import getfile, isclass
import json
import logging
import os
import subprocess
import time

import paramiko

import gethostname
from sagemaker_training import (
    environment,
    errors,
    logging_config,
    process,
    SM_EFA_NCCL_INSTANCES,
    SM_EFA_RDMA_INSTANCES,
    timeout,
)


logger = logging_config.get_logger()
logging.getLogger("paramiko").setLevel(logging.INFO)

DEFAULT_ERROR_CLASS = errors.ExecuteUserScriptError


def get_dataparallel_exception_classes():
    """Get ddp exception classes"""
    exception_classes = []
    try:
        from smdistributed.dataparallel import exceptions

        # list of exceptions SMDDP wants training toolkit to catch and log
        exception_classes += [ex for ex in dir(exceptions) if isclass(getattr(exceptions, ex))]
    # relaxed exception type in case of custom exceptions thrown during import
    except Exception:  # pylint: disable=broad-except
        logger.info(
            "smdistributed.dataparallel not found or "
            "using an older version without custom exceptions."
            "SM training toolkit will track user script error only"
        )
    if not exception_classes:
        exception_classes = [DEFAULT_ERROR_CLASS]
    return exception_classes


MPI_FINISHED_STATUS_FILE = "/tmp/done"


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
        processes_per_host,
        master_hostname,
        hosts,
        custom_mpi_options,
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
            custom_mpi_options (str): A string of custom MPI options to be parsed.
            network_interface_name (str): The network interface name.
            interval (int or float): The interval at which to check the connection in seconds.
                Defaults to 1 second.
            timeout_in_seconds (int): The number of seconds to wait for workers. Defaults to
                3600 seconds (ie. 1 hour).
        """

        super(SMDataParallelRunner, self).__init__(
            user_entry_point, args, env_vars, processes_per_host
        )

        self._master_hostname = master_hostname
        self._hosts = hosts
        self._processes_per_host = processes_per_host
        self._custom_mpi_options = custom_mpi_options
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
        try:
            with timeout.timeout(seconds=self.timeout_in_seconds):
                for host in workers:
                    while not _can_connect(host):
                        time.sleep(self._interval)
                    logger.info("Worker %s available for communication", host)
        except timeout.TimeoutError:
            logger.exception(
                "Connection between the hosts couldn't established. Aborting the training."
            )
            raise

    def _get_mpirun_command(
        self,
        num_hosts,
        host_list,
        smdataparallel_flag,
        num_processes,
        smdataparallel_server_addr=None,
        smdataparallel_server_port=None,
    ):
        """Fetch mpi command for SMDataParallel"""
        overridden_known_options, additional_options = _parse_custom_mpi_options(
            self._custom_mpi_options
        )

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
            "btl_vader_single_copy_mechanism",
            "none",
            "-mca",
            "plm_rsh_num_concurrent",
            str(num_hosts),
            "-x",
            "NCCL_SOCKET_IFNAME=%s" % self._network_interface_name,
            "-x",
            "NCCL_DEBUG=%s" % overridden_known_options.NCCL_DEBUG,
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            "-x",
            smdataparallel_flag,
            "-x",
            "FI_PROVIDER=efa",
            "-x",
            "RDMAV_FORK_SAFE=1",
            "-x",
            "LD_PRELOAD=%s" % getfile(gethostname),
        ]

        mpirun_command.extend(additional_options)

        instance_type = self._get_instance_type()
        # EFA settings
        if instance_type in SM_EFA_NCCL_INSTANCES:
            # Use simple protocol to handle the out-of-order data delivery from EFA
            mpirun_command.extend(["-x", "NCCL_PROTO=simple"])

        if instance_type in SM_EFA_RDMA_INSTANCES:
            # Use EFA's RDMA functionality for one-sided and two-sided transfer
            mpirun_command.extend(["-x", "FI_EFA_USE_DEVICE_RDMA=1"])

        if smdataparallel_server_addr and smdataparallel_server_port:
            # in case of multi-node [distributed] training, smdataparallel_server_addr,
            # smdataparallel_server_port and interconnect_bandwidth will need to be set

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
        if not instance_type:
            # Heterogeneous mode
            instance_type = sm_training_env.get("current_instance_type", None)
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
        num_processes = self._processes_per_host * num_hosts

        logger.info("Network interface name: %s" % self._network_interface_name)
        logger.info("Host: %s" % self._hosts)
        if num_hosts > 1:
            # multi-node; use homogeneous
            # homogeneous mode uses 16 processes per host; 8 server; 8 worker
            smdataparallel_server_addr = self._master_hostname
            smdataparallel_server_port = 7592
            host_list = ["{}:{}".format(host, self._processes_per_host) for host in self._hosts]
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

        msg = "Env Hosts: %s Hosts: %s process_per_hosts: %s num_processes: %s"
        logger.info(msg, self._hosts, host_list, self._processes_per_host, num_processes)

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

        exception_classes = []
        exception_classes += process.get_debugger_exception_classes()
        exception_classes += get_dataparallel_exception_classes()

        # remove potential duplication
        exception_classes = list(set(exception_classes))
        if wait:
            process_spawned = process.check_error(
                cmd,
                exception_classes,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        else:
            process_spawned = process.create(
                cmd,
                exception_classes,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )

        logger.info("Begin writing status file from leader node to worker nodes")
        # Write status file to all nodes
        status_file = MPI_FINISHED_STATUS_FILE + "." + self._master_hostname
        for host in self._hosts:
            if host != self._master_hostname:
                status = _write_status_file(host, status_file)
                retry_count = 5 if not status else 0
                while not status:
                    if retry_count == 0:
                        break
                    logger.info(f"Retry creating status file onto {host}")
                    retry_count -= 1
                    time.sleep(1)
                    status = _write_status_file(host, status_file)

                if not status:
                    logger.info(f"Failed to create status file onto {host}")

        time.sleep(30)
        logger.info("Finished writing status file from leader node to worker nodes")

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
        logger.info("Cannot connect to host %s at port %s. Retrying...", host, port)
        return False
    finally:
        client.close()
        logger.info("Connection closed")


def _write_status_file(host, status_file):
    try:
        logger.info(f"Start writing mpirun finished status to {host}")
        output = subprocess.run(
            ["ssh", str(host), "touch", f"{status_file}"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"output from subprocess run {output}")
        logger.info("Finished writing status file")
        return True
    except subprocess.CalledProcessError:
        logger.info(f"Cannot connect to {host}")
        return False


def _parse_custom_mpi_options(custom_mpi_options):
    """Parse custom MPI options provided by user. Known options default value will be overridden
    and unknown options will be identified separately."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--NCCL_DEBUG", default="INFO", type=str)

    return parser.parse_known_args(custom_mpi_options.split())
