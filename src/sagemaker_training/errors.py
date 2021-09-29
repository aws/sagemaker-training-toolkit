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
"""This module contains custom exceptions."""
from __future__ import absolute_import

import textwrap

import six


class ClientError(Exception):
    """Error class used to separate framework and user errors."""


class _CalledProcessError(ClientError):
    """This exception is raised when a process run by check_call() or
    check_output() returns a non-zero exit status.

    Attributes:
      cmd, return_code, output
    """

    def __init__(self, cmd, return_code=None, output=None, info=None):
        self.return_code = str(return_code)
        self.cmd = cmd
        self.output = output
        self.extra_info = info
        super(_CalledProcessError, self).__init__()

    def __str__(self):
        if six.PY3 and self.output:
            # error_msg = "%s" % self.output.decode("latin1")
            if isinstance(self.output, bytes):
                error_msg = "%s" % self.output.decode("utf-8")
            else:
                error_msg = "%s" % self.output
        elif self.output:
            error_msg = "%s" % self.output
        else:
            error_msg = ""
        if self.extra_info is None:
            message = '%s:\nExitCode %s\nErrorMessage "%s"\nCommand "%s"' % (
                type(self).__name__,
                self.return_code,
                error_msg,
                self.cmd,
            )
        else:
            message = '%s:\nExitCode %s\nErrorMessage "%s"\nExtraInfo "%s"\nCommand "%s"' % (
                type(self).__name__,
                self.return_code,
                error_msg,
                self.extra_info,
                self.cmd,
            )
        return message.strip()


class InstallModuleError(_CalledProcessError):
    """Error class indicating a module failed to install."""


class InstallRequirementsError(_CalledProcessError):
    """Error class indicating a module failed to install."""


class ImportModuleError(ClientError):
    """Error class indicating a module failed to import."""


class ExecuteUserScriptError(_CalledProcessError):
    """Error class indicating a user script failed to execute."""


class ChannelDoesNotExistError(Exception):
    """Error class indicating a channel does not exist."""

    def __init__(self, channel_name):
        super(ChannelDoesNotExistError, self).__init__(
            "Channel %s is not a valid channel" % channel_name
        )


class UnsupportedFormatError(Exception):
    """Error class indicating a content type is not supported by the current framework."""

    def __init__(self, content_type, **kwargs):
        self.message = textwrap.dedent(
            """Content type %s is not supported by this framework.

            Please implement input_fn to to deserialize the request data or an output_fn to
            serialize the response. For more information, see the SageMaker Python SDK README."""
            % content_type
        )
        super(UnsupportedFormatError, self).__init__(self.message, **kwargs)
