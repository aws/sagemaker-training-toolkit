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

import textwrap

import six


class ClientError(Exception):
    pass


class _CalledProcessError(ClientError):
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
        if six.PY3 and self.output:
            error_msg = '\n%s' % self.output.decode('latin1')
        elif self.output:
            error_msg = '\n%s' % self.output
        else:
            error_msg = ''

        message = '%s:\nCommand "%s"%s' % (type(self).__name__, self.cmd, error_msg)
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


class UnsupportedFormatError(Exception):
    def __init__(self, content_type, **kwargs):
        self.message = textwrap.dedent(
            """Content type %s is not supported by this framework.

            Please implement input_fn to to deserialize the request data or an output_fn to
            serialize the response. For more information, see the SageMaker Python SDK README."""
            % content_type)
        super(Exception, self).__init__(self.message, **kwargs)
