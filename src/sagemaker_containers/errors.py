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
from __future__ import absolute_import

import six


class ClientError(Exception):
    pass


class _CalledProcessError(Exception):
    """This exception is raised when a process run by check_call() or
    check_output() returns a non-zero exit status.

    Attributes:
      cmd, return_code, output
    """

    def __init__(self, cmd, return_code=None, output=None):
        self.return_code = return_code
        self.cmd = cmd
        self.output = output

    def __str__(self):
        # transforms a byte string (b'') in unicode
        error = self.output.decode('latin1') if six.PY3 else self.output
        message = '%s:\nCommand "%s"\n%s' % (type(self).__name__, self.cmd, error)
        return message.strip()


class InstallModuleError(_CalledProcessError):
    pass


class ImportModuleError(ClientError):
    pass


class ExecuteUserScriptError(_CalledProcessError):
    pass


class ChannelDoesNotExistException(Exception):
    def __init__(self, channel_name):
        super(ChannelDoesNotExistException, self).__init__('Channel %s is not a valid channel' % channel_name)
