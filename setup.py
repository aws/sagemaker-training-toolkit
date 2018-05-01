from glob import glob
import os

from setuptools import find_packages, setup


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


setup(
    name='sagemaker_containers',
    version='2.0.0',
    description='Open source library for creating containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('src/*.py')],
    long_description=read('README.md'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-containers/',
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
    install_requires=['boto3', 'six', 'pip'],

    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'mock', 'sagemaker', 'keras==2.1.6', 'tensorflow==1.7.0',
                 'numpy']
    }
)
