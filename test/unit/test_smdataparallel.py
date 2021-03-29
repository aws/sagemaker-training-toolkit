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
from __future__ import absolute_import

import inspect
import os

from mock import ANY, patch

import gethostname
from sagemaker_training import environment, smdataparallel
from test.unit.test_mpi import MockSSHClient


@patch("os.path.exists")
@patch("sagemaker_training.process.python_executable", return_value="usr/bin/python3")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("subprocess.Popen")
@patch("sagemaker_training.environment.Environment")
def test_smdataparallel_run_multi_node_python(
    training_env, popen, policy, ssh_client, python_executable, path_exists
):
    with patch.dict(os.environ, clear=True):
        hosts = ["algo-1", "algo-2"]
        master_hostname = hosts[0]
        num_hosts = len(hosts)
        num_processes_per_host = 8
        num_processes = num_processes_per_host * num_hosts
        host_list = ["{}:{}".format(host, num_processes_per_host) for host in hosts]
        network_interface_name = "ethw3"
        smdataparallel_server_addr = master_hostname
        smdataparallel_server_port = 7592
        smdataparallel_flag = "SMDATAPARALLEL_USE_HOMOGENEOUS=1"

        smdataparallel_runner = smdataparallel.SMDataParallelRunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={
                "SM_TRAINING_ENV": '{"additional_framework_parameters":{"sagemaker_instance_type":"ml.p3.16xlarge"}}'
            },
            master_hostname=master_hostname,
            hosts=hosts,
            network_interface_name=network_interface_name,
        )

        process = smdataparallel_runner.run(wait=False)

        ssh_client().load_system_host_keys.assert_called()
        ssh_client().set_missing_host_key_policy.assert_called_with(policy())
        ssh_client().connect.assert_called_with("algo-2", port=22)
        ssh_client().close.assert_called()

        popen.assert_called_with(
            [
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
                network_interface_name,
                "-mca",
                "oob_tcp_if_include",
                network_interface_name,
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
                "NCCL_SOCKET_IFNAME=%s" % network_interface_name,
                "-x",
                "NCCL_DEBUG=INFO",
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
                "LD_PRELOAD=%s" % inspect.getfile(gethostname),
                "-x",
                "SMDATAPARALLEL_SERVER_ADDR=%s" % smdataparallel_server_addr,
                "-x",
                "SMDATAPARALLEL_SERVER_PORT=%s" % str(smdataparallel_server_port),
                "-x",
                "SAGEMAKER_INSTANCE_TYPE=ml.p3.16xlarge",
                "smddprun",
                "usr/bin/python3",
                "-m",
                "mpi4py",
                "train.py",
                "-v",
                "--lr",
                "35",
            ],
            cwd=environment.code_dir,
            env=ANY,
            stderr=None,
        )

        assert process == popen()
        path_exists.assert_called_with("/usr/sbin/sshd")


@patch("os.path.exists")
@patch("sagemaker_training.process.python_executable", return_value="usr/bin/python3")
@patch("paramiko.SSHClient", new_callable=MockSSHClient)
@patch("paramiko.AutoAddPolicy")
@patch("subprocess.Popen")
@patch("sagemaker_training.environment.Environment")
def test_smdataparallel_run_single_node_python(
    training_env, popen, policy, ssh_client, python_executable, path_exists
):
    with patch.dict(os.environ, clear=True):
        hosts = ["algo-1"]
        master_hostname = hosts[0]
        num_hosts = len(hosts)
        num_processes_per_host = 8
        num_processes = num_processes_per_host * num_hosts
        host_list = hosts
        network_interface_name = "ethw3"
        smdataparallel_flag = "SMDATAPARALLEL_USE_SINGLENODE=1"

        smdataparallel_runner = smdataparallel.SMDataParallelRunner(
            user_entry_point="train.py",
            args=["-v", "--lr", "35"],
            env_vars={},
            master_hostname=master_hostname,
            hosts=hosts,
            network_interface_name=network_interface_name,
        )

        process = smdataparallel_runner.run(wait=False)

        popen.assert_called_with(
            [
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
                network_interface_name,
                "-mca",
                "oob_tcp_if_include",
                network_interface_name,
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
                "NCCL_SOCKET_IFNAME=%s" % network_interface_name,
                "-x",
                "NCCL_DEBUG=INFO",
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
                "LD_PRELOAD=%s" % inspect.getfile(gethostname),
                "smddprun",
                "usr/bin/python3",
                "-m",
                "mpi4py",
                "train.py",
                "-v",
                "--lr",
                "35",
            ],
            cwd=environment.code_dir,
            env=ANY,
            stderr=None,
        )

        assert process == popen()
        path_exists.assert_called_with("/usr/sbin/sshd")
