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
"""This module contains custom timeout functionality."""
from __future__ import absolute_import

from contextlib import contextmanager
import signal


class TimeoutError(Exception):  # pylint: disable=redefined-builtin
    """Override the Python 3 TimeoutError built-in exception.

    This builtin is being overridden for the purpose of compatibility with Python 2,
    since TimeoutError is not a built-in exception in Python 2.
    """


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.

    Usage:
    with timeout(seconds=5):
        my_slow_function(...)

    Args:
        seconds (int): The time limit, in seconds.
        minutes (int): The time limit, in minutes.
        hours (int): The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError("timed out after {} seconds".format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)
