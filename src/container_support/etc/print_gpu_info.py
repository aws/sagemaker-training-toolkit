#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

#!/usr/bin/env python
import subprocess

try:
    args = ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=utilization.gpu"]
    msg = subprocess.check_output(args)

    gpus = msg.decode("utf-8").strip().split('\n')

    msgs = ["gpu-{}={}".format(idx, val) for idx, val in enumerate(gpus)]

    fields = ','.join(msgs)
    print('gpu_utilization {}'.format(fields))
except:
    pass
