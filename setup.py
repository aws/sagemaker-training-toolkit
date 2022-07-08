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
from __future__ import absolute_import

from glob import glob
import os
import sys

import setuptools


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def read_version():
    return read("VERSION").strip()


packages = setuptools.find_packages(where="src", exclude=("test",))

required_packages = [
    "numpy",
    "boto3",
    "six",
    "pip",
    "retrying>=1.3.3",
    "gevent",
    "inotify_simple==1.2.1",
    "werkzeug>=0.15.5",
    "paramiko>=2.4.2",
    "psutil>=5.6.7",
    "protobuf>=3.9.2,<3.20",
    "scipy>=1.2.2",
]

# enum is introduced in Python 3.4. Installing enum back port
if sys.version_info < (3, 4):
    required_packages.append("enum34 >= 1.1.6")

gethostname = setuptools.Extension(
    "gethostname",
    sources=["src/sagemaker_training/c/gethostname.c", "src/sagemaker_training/c/jsmn.c"],
    include_dirs=["src/sagemaker_training/c"],
    extra_compile_args=["-Wall", "-shared", "-export-dynamic", "-ldl"],
)

setuptools.setup(
    name="sagemaker_training",
    version=read_version(),
    description="Open source library for creating containers to run on Amazon SageMaker.",
    packages=packages,
    package_dir={"sagemaker_training": "src/sagemaker_training"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    ext_modules=[gethostname],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-training-toolkit/",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=required_packages,
    extras_require={
        "test": [
            "tox==3.13.1",
            "pytest==4.4.1",
            "pytest-cov",
            "mock",
            "sagemaker[local]<2",
            "black==22.3.0 ; python_version >= '3.7'",
        ]
    },
    entry_points={"console_scripts": ["train=sagemaker_training.cli.train:main"]},
)
