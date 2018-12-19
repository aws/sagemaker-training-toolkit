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
import json
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = {'rank': rank, 'size': size}
data = comm.gather(data, root=0)
if rank == 0:
    assert data == [{'rank': 0, 'size': 2},
                    {'rank': 1, 'size': 2}]

    model = os.path.join(os.environ['SM_MODEL_DIR'], 'result.json')
    with open(model, 'w+') as f:
        json.dump(data, f)
else:
    assert data is None
