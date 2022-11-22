# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This file is executed when the sagemaker_training package is imported."""
from __future__ import absolute_import

# list of errors: To show user error message on the SM Training job page
# [x for x in dir(__builtins__) if 'Error' in x or 'Exception' in x]
_PYTHON_ERRORS_ = [
    "BaseException",
    "Exception",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "EOFError",
    "EnvironmentError",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "IOError",
    "ImportError",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PermissionError",
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "RuntimeError",
    "SyntaxError",
    "SystemError",
    "TabError",
    "TimeoutError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "ValueError",
    "ZeroDivisionError",
    "Invalid requirement",
    "ResourceExhaustedError",
    "OutOfRangeError",
    "InvalidArgumentError",
]

_MPI_ERRORS_ = ["mpirun.real", "ORTE"]

SM_EFA_NCCL_INSTANCES = [
    "ml.g4dn.8xlarge",
    "ml.g4dn.12xlarge",
    "ml.g5.48xlarge",
    "ml.p3dn.24xlarge",
    "ml.p4d.24xlarge",
    "ml.trn1.32xlarge",
]

SM_TRAINING_COMPILER_PATHS = [
    "tensorflow/compiler/xla",
    "tensorflow/compiler/tf2xla",
    "tensorflow/python/compiler/xla",
    "torch_xla/",
]
