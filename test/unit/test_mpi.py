# # Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the 'License'). You
# # may not use this file except in compliance with the License. A copy of
# # the License is located at
# #
# #     http://aws.amazon.com/apache2.0/
# #
# # or in the 'license' file accompanying this file. This file is
# # distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# # ANY KIND, either express or implied. See the License for the specific
# # language governing permissions and limitations under the License.
from __future__ import absolute_import

import inspect
import os

from mock import ANY, MagicMock, patch

import gethostname
from sagemaker_containers import _env, _mpi


def does_not_connect():
    raise ValueError('cannot connect')


def connect():
    pass


class MockSSHClient(MagicMock):

    def __init__(self, *args, **kw):
        super(MockSSHClient, self).__init__(*args, **kw)
        self.connect = MagicMock(side_effect=[does_not_connect,
                                              connect,
                                              does_not_connect])


@patch('paramiko.SSHClient', new_callable=MockSSHClient)
@patch('psutil.wait_procs')
@patch('psutil.process_iter')
@patch('paramiko.AutoAddPolicy')
@patch('subprocess.Popen')
def test_mpi_worker_run(popen, policy, process_iter, wait_procs, ssh_client):

    process = MagicMock(info={'name': 'orted'})
    process_iter.side_effect = lambda attrs: [process]

    worker = _mpi.WorkerRunner(user_entry_point='train.sh',
                               args=['-v', '--lr', '35'],
                               env_vars={'LD_CONFIG_PATH': '/etc/ld'},
                               master_hostname='algo-1')

    worker.run()

    ssh_client().load_system_host_keys.assert_called()
    ssh_client().set_missing_host_key_policy.assert_called_with(policy())
    ssh_client().connect.assert_called_with('algo-1', port=22)
    ssh_client().close.assert_called()
    wait_procs.assert_called_with([process])

    popen.assert_called_with(['/usr/sbin/sshd', '-D'])


@patch('paramiko.SSHClient', new_callable=MockSSHClient)
@patch('subprocess.Popen')
def test_mpi_worker_run_no_wait(popen, ssh_client):
    worker = _mpi.WorkerRunner(user_entry_point='train.sh',
                               args=['-v', '--lr', '35'],
                               env_vars={'LD_CONFIG_PATH': '/etc/ld'},
                               master_hostname='algo-1')

    worker.run(wait=False)

    ssh_client.assert_not_called()

    popen.assert_called_with(['/usr/sbin/sshd', '-D'])


@patch('paramiko.SSHClient', new_callable=MockSSHClient)
@patch('paramiko.AutoAddPolicy')
@patch('subprocess.Popen')
@patch('sagemaker_containers.training_env')
def test_mpi_master_run(training_env, popen, policy, ssh_client):
    with patch.dict(os.environ, clear=True):

        master = _mpi.MasterRunner(user_entry_point='train.sh',
                                   args=['-v', '--lr', '35'],
                                   env_vars={'LD_CONFIG_PATH': '/etc/ld'},
                                   master_hostname='algo-1',
                                   hosts=['algo-1', 'algo-2'],
                                   process_per_host=2,
                                   custom_mpi_options='-v --lr 35',
                                   network_interface_name='ethw3')

        process = master.run(wait=False)

        ssh_client().load_system_host_keys.assert_called()
        ssh_client().set_missing_host_key_policy.assert_called_with(policy())
        ssh_client().connect.assert_called_with('algo-2', port=22)
        ssh_client().close.assert_called()

        popen.assert_called_with([
            'mpirun',
            '--host', 'algo-1:2,algo-2:2',
            '-np', '4', '--allow-run-as-root',
            '--display-map',
            '--tag-output',
            '-mca', 'btl_tcp_if_include', 'ethw3',
            '-mca', 'oob_tcp_if_include', 'ethw3',
            '-mca', 'plm_rsh_no_tree_spawn', '1',
            '-bind-to', 'socket', '-map-by', 'slot',
            '-mca', 'pml', 'ob1',
            '-mca', 'btl', '^openib',
            '-mca', 'orte_abort_on_non_zero_status', '1',
            '-x', 'NCCL_MIN_NRINGS=4',
            '-x', 'NCCL_SOCKET_IFNAME=ethw3',
            '-x', 'NCCL_DEBUG=INFO',
            '-x', 'LD_LIBRARY_PATH',
            '-x', 'PATH',
            '-x', 'LD_PRELOAD=%s' % inspect.getfile(gethostname),
            '-v', '--lr', '35', '-x', 'LD_CONFIG_PATH', '/bin/sh', '-c', './train.sh -v --lr 35'],
            cwd=_env.code_dir,
            env=ANY, stderr=None)

        assert process == popen()
