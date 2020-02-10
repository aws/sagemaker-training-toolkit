# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
packages.append("sagemaker_containers.etc")

required_packages = [
    "numpy",
    "boto3",
    "six",
    "pip",
    "flask==1.1.1",
    "gunicorn",
    "typing",
    "retrying==1.3.3",
    "gevent",
    "inotify_simple",
    "werkzeug==0.15.5",
    "paramiko==2.4.2",
    "psutil==5.4.8",
    "protobuf>=3.1",
    "scipy>=1.2.2",
]

# enum is introduced in Python 3.4. Installing enum back port
if sys.version_info < (3, 4):
    required_packages.append("enum34 >= 1.1.6")

gethostname = setuptools.Extension(
    "gethostname",
    sources=["src/sagemaker_containers/c/gethostname.c", "src/sagemaker_containers/c/jsmn.c"],
    include_dirs=["src/sagemaker_containers/c"],
    extra_compile_args=["-Wall", "-shared", "-export-dynamic", "-ldl"],
)

setuptools.setup(
    name="sagemaker_containers",
    version=read_version(),
    description="Open source library for creating containers to run on Amazon SageMaker.",
    packages=packages,
    package_dir={
        "sagemaker_containers": "src/sagemaker_containers",
        "sagemaker_containers.etc": "etc",
    },
    package_data={"sagemaker_containers.etc": ["*"]},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    ext_modules=[gethostname],
    long_description=read("README.rst"),
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-containers/",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=required_packages,
    extras_require={
        "test": [
            "tox==3.13.1",
            "pytest==4.4.1",
            "pytest-cov",
            "mock",
            "sagemaker[local]>=1.16.2",
            "black==19.3b0 ; python_version >= '3.6'",
        ]
    },
    entry_points={
        "console_scripts": [
            "serve=sagemaker_containers.cli.serve:main",
            "train=sagemaker_containers.cli.train:main",
        ]
    },
)
