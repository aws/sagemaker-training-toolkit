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
"""This module contains functionality to spawn new processes, check for errors,
and execute the user entry point within a process.
"""
from __future__ import absolute_import

import asyncio
from asyncio.subprocess import PIPE
from inspect import getmembers
from inspect import isclass
import os
import re
import subprocess
import sys

import six

from sagemaker_training import (
    _entry_point_type,
    _MPI_ERRORS_,
    _PYTHON_ERRORS_,
    environment,
    errors,
    logging_config,
)

logger = logging_config.get_logger()

# Default limit of the stream is 2 ** 16 KB, we can increase it to 128KB in subproc call
_DEFAULT_BUF_SIZE = 1024 * 64
DEFAULT_ERROR_CLASS = errors.ExecuteUserScriptError


def get_debugger_exception_classes():
    """Set exception classes"""
    exception_classes = []
    if os.environ.get("USE_SMDEBUG") == "0":
        logger.info("Sagemaker Debugger is not enabled, debugger exceptions will not be imported.")
    else:
        try:
            from smdebug import exceptions

            # list of exceptions debugger wants training toolkit to catch and log
            exception_classes += [ex for ex in dir(exceptions) if isclass(getattr(exceptions, ex))]
        except ImportError:
            logger.info("Exceptions not imported for SageMaker Debugger as it is not installed.")

    if not exception_classes:
        exception_classes = [DEFAULT_ERROR_CLASS]
    return exception_classes


def get_tensorflow_exception_classes():
    """TensorFlow exception classes are reused by XLA. XLA is present in SageMaker Training Compiler
    enabled TensorFlow and PyTorch DLCs."""
    exception_classes = []
    try:
        from tensorflow.python.framework import errors_impl

        # list of exceptions from TensorFlow that sagemaker-training-toolkit to catch and log
        exception_classes += [name for name, obj in getmembers(errors_impl) if isclass(obj)]
        # adding XlaRuntimeError as a str (process.watch can handle str) as there is
        # no proper import of module tensorflow/compiler/xla/python/xla_client.py available.
        exception_classes += ["XlaRuntimeError"]
    except ImportError:
        logger.info("Exceptions not imported for SageMaker TF as Tensorflow is not installed.")

    if not exception_classes:
        exception_classes = [DEFAULT_ERROR_CLASS]
    return exception_classes


def process_error_classes(error_classes):
    """Process error classes and return a list of string.
    Input could be class, string, or None

    Args:
        error_classes (list): List of error classes

    Returns:
        error_classes: processed classes
    """
    if not error_classes:
        return []
    if not isinstance(error_classes, list):
        error_classes = [error_classes]
    return [error.__name__ if isclass(error) else error for error in error_classes]


async def watch(stream, proc_per_host, error_classes=None):
    """Process the stdout and stderr streams on the fly.
    Decode the output lines
    Remove new line characters (if any)
    Prepend tags for easier search on CloudWatch
    Look for errors in the stderr

    Args:
        stream: asyncio subprocess PIPE
        proc_per_host (int): Number of processes per each host
        error_classes (list): List of exception classes to watch and raise

    Returns:
        output: Filtered stderr
    """
    error_classes = process_error_classes(error_classes)
    output = []
    buf_size = _DEFAULT_BUF_SIZE
    start = False
    while True:
        if stream is None:
            break
        lines = await stream.read(buf_size)
        if not lines or lines == "":
            break

        # If `lines` contains non-utf-8 characters, replace them with �
        lines = lines.decode("utf-8", "replace").strip().split("\n")
        for line in lines:
            err_line = line
            if "<stdout>" in line:
                line = re.sub(
                    r"\[(\d),(\d+)\]<stdout>",
                    lambda x: (
                        f"[{x[1]},mpirank:{x[2]},algo-{(int(x[2])//proc_per_host)+1}]<stdout>"
                    ),
                    line,
                )
            elif "<stderr>" in line:
                line = re.sub(
                    r"\[(\d),(\d+)\]<stderr>",
                    lambda x: (
                        f"[{x[1]},mpirank:{x[2]},algo-{(int(x[2])//proc_per_host)+1}]<stderr>"
                    ),
                    line,
                )
            print(line)
            # log only if necessary, remove node and rank id for de-duplication
            err_line = re.sub(r"\[(\d),(\d)\]<stderr>", "", err_line)
            # in case error piped to stdout
            err_line = re.sub(r"\[(\d),(\d)\]<stdout>", "", err_line)

            if start:
                if err_line not in output:
                    output.append(err_line.strip(" :\n") + "\n")
            else:
                if any(
                    str(err) in err_line
                    for err in (
                        _PYTHON_ERRORS_ + _MPI_ERRORS_ + error_classes
                        if isinstance(error_classes, list)
                        else [error_classes]
                    )
                ):
                    # start logging error message if target exceptions found
                    start = True
                    output.append(err_line.strip(" :\n") + "\n")
    return " ".join(output)


async def run_async(cmd, processes_per_host, env, cwd, stderr, error_classes=None, **kwargs):
    """Method responsible for launching asyncio subprocess shell
    Use asyncio gather to collect processed stdout and stderr

    Args:
        cmd (list): The command to be run
        processes_per_host (int): Number of processes per host
        env: os.environ
        cwd (str): The location from which to run the command (default: None).
            If None, this defaults to the ``code_dir`` of the environment.
        error_classes (list): List of exception classes to watch and raise
        **kwargs: Extra arguments that are passed to the asyncio create subprocess constructor.

    Returns:
        return_code: Launched Process's return code
        output: Processed [stdout, stderr]
        asyncio.subprocess.Process: The asyncio process for the given command.

    Raises:
        ExecuteUserScriptError: If there is an exception raised when creating the process.
    """
    cmd = " ".join(cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd, env=env, cwd=cwd, stdout=PIPE, stderr=stderr, **kwargs
    )

    output = await asyncio.gather(
        watch(proc.stdout, processes_per_host, error_classes=error_classes),
        watch(proc.stderr, processes_per_host, error_classes=error_classes),
    )
    logger.info("Waiting for the process to finish and give a return code.")
    return_code = await proc.wait()
    logger.info(f"Done waiting for a return code. Received {return_code} from exiting process.")
    return return_code, output, proc


def create(
    cmd,
    error_classes,
    processes_per_host,
    cwd=None,
    env=None,
    capture_error=False,
    **kwargs,
):
    """Spawn a process with asyncio for the given command.

    Args:
        cmd (list): The command to be run.
        error_classes (list): List of exception classes to watch and raise.
        cwd (str): The location from which to run the command (default: None).
            If None, this defaults to the ``code_dir`` of the environment.
        env: os.environ
        capture_error (bool): Whether or not to direct stderr to a stream
            that can later be read (default: False).
        **kwargs: Extra arguments that are passed to the asyncio create subprocess constructor.

    Returns:
        asyncio.subprocess.Process: The asyncio process for the given command.

    Raises:
        ExecuteUserScriptError: If there is an exception raised when creating the process.
    """
    try:
        stderr = PIPE if capture_error else None
        rc, output, proc = asyncio.run(
            run_async(
                cmd,
                processes_per_host,
                env=env or os.environ,
                cwd=cwd or environment.code_dir,
                stderr=stderr,
                error_classes=error_classes,
                **kwargs,
            )
        )
        return rc, output, proc
    except Exception as e:  # pylint: disable=broad-except
        six.reraise(
            DEFAULT_ERROR_CLASS,
            DEFAULT_ERROR_CLASS(e),
            sys.exc_info()[2],
        )


def check_error(cmd, error_classes, processes_per_host, cwd=None, capture_error=True, **kwargs):
    """Run a commmand, raising an exception if there is an error.

    Args:
        cmd ([str]): The command to be run.
        error_classes (list): List of exception classes to watch and raise.
        processes_per_host (int): Number of processes per host
        capture_error (bool): Whether or not to include stderr in
            the exception message (default: True). In either case,
            stderr is streamed to the process's output.
        **kwargs: Extra arguments that are passed to the subprocess.Popen constructor.

    Returns:
        subprocess.Popen: The process for the given command.

    Raises:
        ExecuteUserScriptError: If there is an exception raised when creating the process.
    """
    error_classes = process_error_classes(error_classes)
    if capture_error:
        return_code, output, process = create(
            cmd,
            error_classes,
            processes_per_host,
            env=os.environ,
            cwd=cwd or environment.code_dir,
            capture_error=True,
            **kwargs,
        )
        stderr = " ".join(output)
        # remove duplicate while preserve order
        stderr = "\n".join(list(dict.fromkeys(stderr.split("\n")))).strip()
    else:
        stderr = None
        # remove extra quotes for subprocess.Popen
        cmd[-1] = cmd[-1].strip('"')
        process = subprocess.Popen(
            cmd,
            env=os.environ,
            cwd=cwd or environment.code_dir,
            stderr=stderr,
            **kwargs,
        )
        return_code = process.wait()
    if return_code:
        extra_info = None
        if return_code == 137:
            extra_info = "OutOfMemory: Process killed by SIGKILL (signal 9)"

        # throw internal error classes first
        internal_errors = [err for err in dir(errors) if isclass(getattr(errors, err))]
        error_class = next(
            (name for name in error_classes if name in internal_errors), "ExecuteUserScriptError"
        )
        error_class = getattr(errors, error_class)

        # only replace ExecuteUserScriptError with custom library errors
        if stderr and error_class == DEFAULT_ERROR_CLASS:
            # find the first target error in stderr
            error_name = next((str(name) for name in error_classes if str(name) in stderr), False)
            if error_name:
                error_class = type(
                    error_name,
                    (errors._CalledProcessError,),  # pylint: disable=protected-access
                    {},
                )

        raise error_class(
            cmd=" ".join(cmd) if isinstance(cmd, list) else cmd,
            return_code=return_code,
            output=stderr,
            info=extra_info,
        )

    return process


def python_executable():
    """Return the real path for the Python executable, if it exists.
    Return RuntimeError otherwise.

    Returns:
        (str): The real path of the current Python executable.
    """
    if not sys.executable:
        raise RuntimeError("Failed to retrieve the real path for the Python executable binary")
    return sys.executable


class ProcessRunner(object):
    """Responsible for executing the user entry point within a process."""

    def __init__(self, user_entry_point, args, env_vars, processes_per_host):
        """Initialize a ProcessRunner, which is responsible for executing the user
        entry point within a process.

        Args:
            user_entry_point (str): The name of the user entry point.
            args ([str]): A list of arguments to include when executing the entry point.
            env_vars (dict(str,str)): A dictionary of environment variables.
        """
        self._user_entry_point = user_entry_point
        self._args = args
        self._env_vars = env_vars
        self._processes_per_host = processes_per_host

    def _create_command(self):
        entrypoint_type = _entry_point_type.get(environment.code_dir, self._user_entry_point)

        if entrypoint_type is _entry_point_type.PYTHON_PACKAGE:
            entry_module = self._user_entry_point.replace(".py", "")
            return self._python_command() + ["-m", entry_module] + self._args
        elif entrypoint_type is _entry_point_type.PYTHON_PROGRAM:
            return self._python_command() + [self._user_entry_point] + self._args
        else:
            args = [
                six.moves.shlex_quote(arg)  # pylint: disable=too-many-function-args
                for arg in self._args
            ]
            return [
                "/bin/sh",
                "-c",
                '"./%s %s"' % (self._user_entry_point, " ".join(args)),
            ]

    def _python_command(self):  # pylint: disable=no-self-use
        return [python_executable()]

    def _setup(self):
        pass

    def _tear_down(self):
        pass

    def run(self, wait=True, capture_error=False):
        """Run the process.

        Args:
            wait (bool): A boolean indicating whether to wait and check for errors.
                Defaults to True.
            capture_error (bool): A boolean indicating whether to direct stderr to a stream
                that can later be read. Defaults to False.

        Returns:
            process (subprocess.Popen): The spawned process.
        """
        self._setup()

        cmd = self._create_command()

        logging_config.log_script_invocation(cmd, self._env_vars)

        exception_classes = []
        exception_classes += get_debugger_exception_classes()
        exception_classes += get_tensorflow_exception_classes()
        if wait:
            process = check_error(
                cmd,
                exception_classes,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )
        else:
            _, _, process = create(
                cmd,
                exception_classes,
                self._processes_per_host,
                capture_error=capture_error,
                cwd=environment.code_dir,
            )

        self._tear_down()
        return process
