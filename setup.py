from glob import glob
import os

from setuptools import find_packages, setup


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


packages = find_packages(where='src', exclude=('test',))
packages.append('sagemaker_containers.etc')

setup(
    name='sagemaker_containers',
    version='2.2.4',
    description='Open source library for creating containers to run on Amazon SageMaker.',

    packages=packages,
    package_dir={
        'sagemaker_containers': 'src/sagemaker_containers',
        'sagemaker_containers.etc': 'etc'
    },
    package_data={'sagemaker_containers.etc': ['*']},
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
    install_requires=['boto3', 'six', 'pip', 'flask', 'gunicorn', 'gevent', 'werkzeug'],

    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'mock', 'sagemaker', 'numpy']
    },

    entry_points={
          'console_scripts': ['serve=sagemaker_containers.cli.serve:main',
                              'train=sagemaker_containers.cli.train:main'],
    }
)
