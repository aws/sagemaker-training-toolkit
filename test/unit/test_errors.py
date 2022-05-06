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

from sagemaker_training import errors


def test_install_module_error():
    error = errors.InstallModuleError(["python", "-m", "42"], return_code=42)
    assert (
        str(error) == 'InstallModuleError:\nExitCode 42\nErrorMessage ""\nCommand'
        " \"['python', '-m', '42']\""
    )


def test_execute_user_script_error():
    error = errors.ExecuteUserScriptError(["python", "-m", "42"], return_code=42)

    assert (
        str(error) == 'ExecuteUserScriptError:\nExitCode 42\nErrorMessage ""\nCommand'
        " \"['python', '-m', '42']\""
    )


def test_install_module_error_with_output():
    error = errors.InstallModuleError(["python", "-m", "42"], return_code=42, output="42")

    assert (
        str(error) == 'InstallModuleError:\nExitCode 42\nErrorMessage "42"\nCommand'
        " \"['python', '-m', '42']\""
    )


def test_execute_user_script_error_with_output():
    error = errors.ExecuteUserScriptError(["python", "-m", "42"], return_code=137, output=b"42")

    assert (
        str(error) == 'ExecuteUserScriptError:\nExitCode 137\nErrorMessage "42"\nCommand'
        " \"['python', '-m', '42']\""
    )


def test_execute_user_script_error_with_output_and_info():
    error = errors.ExecuteUserScriptError(
        ["python", "-m", "42"], return_code=137, output="42", info="SIGKILL"
    )

    assert (
        str(error) == "ExecuteUserScriptError:\nExitCode 137\nErrorMessage"
        " \"42\"\nExtraInfo \"SIGKILL\"\nCommand \"['python', '-m', '42']\""
    )
