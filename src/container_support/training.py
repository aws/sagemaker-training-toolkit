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

import container_support as cs
import logging
import os
import traceback
from container_support import TrainingEnvironment

logger = logging.getLogger(__name__)


class Trainer(object):
    @classmethod
    def start(cls):
        base_dir = None
        exit_code = 0
        cs.configure_logging()
        logger.info("Training starting")
        try:
            env = TrainingEnvironment()
            env.start_metrics_if_enabled()
            base_dir = env.base_dir

            fw = TrainingEnvironment.load_framework()
            fw.train()
            env.write_success_file()
        except Exception as e:
            trc = traceback.format_exc()
            message = 'uncaught exception during training: {}\n{}\n'.format(e, trc)
            logger.error(message)
            TrainingEnvironment.write_failure_file(message, base_dir)
            exit_code = 1 if not hasattr(e, 'errno') else e.errno
            raise e
        finally:
            # Since threads in Python cannot be stopped, this is the only way to stop the application
            # https://stackoverflow.com/questions/9591350/what-is-difference-between-sys-exit0-and-os-exit0
            os._exit(exit_code)
