# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import asyncio
import inspect
import os

from mock import ANY, MagicMock, patch
import pytest

import gethostname
from sagemaker_training import environment, mpi


def does_not_connect():
    raise ValueError("cannot connect")


def connect():
    pass


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class MockSSHClient(MagicMock):
    def __init__(self, *args, **kw):
        super(MockSSHClient, self).__init__(*args, **kw)
        self.connect = MagicMock(side_effect=[does_not_connect, connect, does_not_connect])


@patch("sagemaker_training.mpi._write_env_vars_to_file")
@patch("os.path.exists")
@patch("time.sleep")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("psutil.wait_procs")
@patch("psutil.process_iter")
@patch("paramiko.AutoAddPolicy")
@patch("subprocess.Popen")
def test_mpi_worker_run(
    popen, policy, process_iter, wait_procs, ssh_client, sleep, path_exists, write_env_vars
):

    process = MagicMock(info={"name": "orted"})
    process_iter.side_effect = lambda attrs: [process]

    worker = mpi.WorkerRunner(
        user_entry_point="train.sh",
        args=["-v", "--lr", "35"],
        env_vars={"LD_CONFIG_PATH": "/etc/ld"},
        processes_per_host="1",
        master_hostname="algo-1",
    )

    worker.run()

    write_env_vars.assert_called_once()

    ssh_client().load_system_host_keys.assert_called()
    ssh_client().set_missing_host_key_policy.assert_called_with(policy())
    ssh_client().connect.assert_called_with("algo-1", port=22)
    ssh_client().close.assert_called()
    wait_procs.assert_called_with([process])

    popen.assert_called_with(["/usr/sbin/sshd", "-D"])
    path_exists.assert_called_with("/usr/sbin/sshd")


@patch("sagemaker_training.mpi._write_env_vars_to_file")
@patch("os.path.exists")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("subprocess.Popen")
def test_mpi_worker_run_no_wait(popen, ssh_client, path_exists, write_env_vars):
    worker = mpi.WorkerRunner(
        user_entry_point="train.sh",
        args=["-v", "--lr", "35"],
        env_vars={"LD_CONFIG_PATH": "/etc/ld"},
        processes_per_host=1,
        master_hostname="algo-1",
    )

    worker.run(wait=False)

    write_env_vars.assert_called_once()

    ssh_client.assert_not_called()

    popen.assert_called_with(["/usr/sbin/sshd", "-D"])
    path_exists.assert_called_with("/usr/sbin/sshd")


@patch("asyncio.gather", new_callable=AsyncMock)
@patch("os.path.exists")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("asyncio.create_subprocess_shell")
@patch("sagemaker_training.environment.Environment")
def test_mpi_master_run(
    training_env, async_shell, policy, ssh_client, path_exists, async_gather, event_loop
):

    with patch.dict(os.environ, clear=True):
        os.environ["AWS_ACCESS_KEY_ID"] = "ABCD"
        master = mpi.MasterRunner(
            user_entry_point="train.sh",
            args=["-v", "--lr", "35"],
            env_vars={"LD_CONFIG_PATH": "/etc/ld"},
            processes_per_host=2,
            master_hostname="algo-1",
            hosts=["algo-1", "algo-2"],
            custom_mpi_options="-v --lr 35",
            network_interface_name="ethw3",
        )
        process = master.run(wait=False)

        ssh_client().load_system_host_keys.assert_called()
        ssh_client().set_missing_host_key_policy.assert_called_with(policy())
        ssh_client().connect.assert_called_with("algo-2", port=22)
        ssh_client().close.assert_called()
        cmd = [
            "mpirun",
            "--host",
            "algo-1:2,algo-2:2",
            "-np",
            "4",
            "--allow-run-as-root",
            "--display-map",
            "--tag-output",
            "-mca",
            "btl_tcp_if_include",
            "ethw3",
            "-mca",
            "oob_tcp_if_include",
            "ethw3",
            "-mca",
            "plm_rsh_no_tree_spawn",
            "1",
            "-bind-to",
            "none",
            "-map-by",
            "slot",
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
            "-x",
            "NCCL_MIN_NRINGS=4",
            "-x",
            "NCCL_SOCKET_IFNAME=ethw3",
            "-x",
            "NCCL_DEBUG=INFO",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            "-x",
            "LD_PRELOAD=%s" % inspect.getfile(gethostname),
            "-v",
            "--lr",
            "35",
            "-x",
            "AWS_ACCESS_KEY_ID",
            "-x",
            "LD_CONFIG_PATH",
            "/bin/sh",
            "-c",
            "./train.sh -v --lr 35",
        ]
        extended_cmd = " ".join(cmd)
        async_shell.assert_called_with(
            extended_cmd,
            env=ANY,
            cwd=environment.code_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
        )
        async_shell.assert_called_once()
        async_gather.assert_called_once()
        assert process == async_shell.return_value
        path_exists.assert_called_with("/usr/sbin/sshd")


@patch("asyncio.gather", new_callable=AsyncMock)
@patch("os.path.exists")
@patch("sagemaker_training.process.python_executable", return_value="usr/bin/python3")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("asyncio.create_subprocess_shell")
@patch("sagemaker_training.environment.Environment")
def test_mpi_master_run_python(
    training_env,
    async_shell,
    policy,
    ssh_client,
    python_executable,
    path_exists,
    async_gather,
    event_loop,
):

    with patch.dict(os.environ, clear=True):

        master = mpi.MasterRunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={"LD_CONFIG_PATH": "/etc/ld"},
            master_hostname="algo-1",
            hosts=["algo-1", "algo-2"],
            processes_per_host=2,
            custom_mpi_options="-v --lr 35",
            network_interface_name="ethw3",
        )

        process = master.run(wait=False)

        ssh_client().load_system_host_keys.assert_called()
        ssh_client().set_missing_host_key_policy.assert_called_with(policy())
        ssh_client().connect.assert_called_with("algo-2", port=22)
        ssh_client().close.assert_called()
        cmd = [
            "mpirun",
            "--host",
            "algo-1:2,algo-2:2",
            "-np",
            "4",
            "--allow-run-as-root",
            "--display-map",
            "--tag-output",
            "-mca",
            "btl_tcp_if_include",
            "ethw3",
            "-mca",
            "oob_tcp_if_include",
            "ethw3",
            "-mca",
            "plm_rsh_no_tree_spawn",
            "1",
            "-bind-to",
            "none",
            "-map-by",
            "slot",
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
            "-x",
            "NCCL_MIN_NRINGS=4",
            "-x",
            "NCCL_SOCKET_IFNAME=ethw3",
            "-x",
            "NCCL_DEBUG=INFO",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            "-x",
            "LD_PRELOAD=%s" % inspect.getfile(gethostname),
            "-v",
            "--lr",
            "35",
            "-x",
            "LD_CONFIG_PATH",
            "usr/bin/python3",
            "-m",
            "mpi4py",
            "train.py",
            "-v",
            "--lr",
            "35",
        ]
        async_shell.assert_called_with(
            " ".join(cmd),
            cwd=environment.code_dir,
            env=ANY,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
        )
        async_shell.assert_called_once()
        async_gather.assert_called_once()
        assert process == async_shell.return_value
        path_exists.assert_called_with("/usr/sbin/sshd")


@patch("sagemaker_training.logging_config.log_script_invocation")
def test_connection(log):
    with pytest.raises(Exception):
        mpi._can_connect("test_host")
        log.assert_called_with("Cannot connect to host test_host")
        log.assert_called_with("Connection failed with exceptio: ")
