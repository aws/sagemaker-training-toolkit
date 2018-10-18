import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


PKG_NAME = 'container_support'

setup(
    name='sagemaker_container_support',
    version='1.0.9',
    description='Open source library for creating containers to run on Amazon SageMaker.',

    packages=[PKG_NAME],
    package_dir={PKG_NAME: 'src/container_support'},
    package_data={PKG_NAME: ['etc/*']},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    scripts=['bin/entry.py'],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-container-support/',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    install_requires=['Flask >=0.12.2, <1', 'boto3 >=1.6.18, <2', 'six >=1.11.0, <2',
                      'gunicorn >=19.7.1, <20', 'gevent >=1.2.2, <2'],
    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock']
    },
)
