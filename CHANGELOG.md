# Changelog

## v5.1.0 (2025-08-08)

### Features

 * Add support for Ultraserver jobs

### Bug Fixes and Other Changes

 * formatting
 * compile args not working on macOS

## v5.0.0 (2025-06-04)

### Breaking Changes

 * updating protobuf version to 5.28.1

## v5.0.0

### Breaking Changes
  **Upgraded `protobuf` to v5.28.1**

  * This upgrade introduces a breaking change due to `protobuf` v5.28.1 dropping support for Python versions earlier than 3.8.

  * Downstream packages that depend on `sagemaker-training-toolkit` for builds or tests may need to:
    - Regenerate their protobuf code with the updated version.
    - Update related testing/build dependencies to maintain compatibility.

## v4.9.0 (2025-02-11)

### Features

 * Add Code Owners file

## v4.8.4 (2025-02-03)

### Bug Fixes and Other Changes

 * account for possible race condition when creating /opt/ml/code

## v4.8.3 (2024-12-09)

### Bug Fixes and Other Changes

 * resolve failing unit test
 * avoid parsing stderr as JSON

## v4.8.2 (2024-12-06)

### Bug Fixes and Other Changes

 * temporarily hardcode neuron cores for trn2

## v4.8.1 (2024-09-09)

### Bug Fixes and Other Changes

 * Added p5 as a supported NCCL instance

## v4.8.0 (2024-08-14)

### Features

 * Add support for py39 and py310

### Bug Fixes and Other Changes

 * typo in the run unit tests command
 * run unit tests in sequence order for release process as well to prevent coverage conflicting issues
 * chore: removing unnecessary logging information

## v4.7.4 (2023-10-31)

### Bug Fixes and Other Changes

 * update the boto deps to use latest boto

## v4.7.3 (2023-10-23)

### Bug Fixes and Other Changes

 * bypass DNS check for studio local exec

## v4.7.2 (2023-10-19)

### Bug Fixes and Other Changes

 * use smddprun only if it is installed

## v4.7.1 (2023-10-17)

### Bug Fixes and Other Changes

 * Add NCCL_PROTO=simple environment variable to handle the out-of-order data delivery from EFA
 * toolkit build failure

## v4.7.0 (2023-08-08)

### Features

 * support codeartifact for installing requirements.txt packages

## v4.6.1 (2023-06-19)

### Bug Fixes and Other Changes

 * removed unused import statment
 * forgot to run black on torch_distributed.py after updating my comments from last commit
 * Modified my comment on line 98-103 in torch_distrbuted.py to comply with formatting standard.
 * Revert "Ran black on entire sagemaker-trianing-toolkit directory"
 * Ran black on entire sagemaker-trianing-toolkit directory
 * Ran Black (python formatter) on the files with my code updates (torch_distributed.py and test_torch_distributed.py)
 * Added test for neuron_parallel_compile in test_torch_distributed.py
 * Updated comment syntax based on feedback in pull request as well as added full example of the neuron_parallel_compile command as it would appear in the command line
 * added unit test for neuron_parallel_compile code change
 * Updated torch_distributed.py

## v4.6.0 (2023-06-15)

### Features

 * add smddp exception classes in mpi distribution

## v4.5.0 (2023-04-26)

### Features

 * add NCCL_PROTO, NCCL_ALGO environments for modelparallel jobs

## v4.4.10 (2023-04-10)

### Bug Fixes and Other Changes

 * unpin sagemaker version as the credential issue fixed

## v4.4.9 (2023-04-05)

### Bug Fixes and Other Changes

 * increase worker waiting time for ORTE proc

## v4.4.8 (2023-03-09)

### Bug Fixes and Other Changes

 * upagrade protobuf version for tensorflow 2.12

## v4.4.7 (2023-03-02)

### Bug Fixes and Other Changes

 * Revert SMDDP collectives feature from smdataparallel runner

## v4.4.6 (2023-02-22)

## v4.4.5 (2023-01-24)

## v4.4.4 (2023-01-23)

### Bug Fixes and Other Changes

 * Update libraries for SMDDP collectives validation

## v4.4.3 (2023-01-18)

### Bug Fixes and Other Changes

 * Upgrade protobuf to prevent conflicts with smdebugger.

## v4.4.2 (2023-01-16)

## v4.4.1 (2022-12-13)

### Bug Fixes and Other Changes

 * Add support for p4de instances, update when FI_EFA_USE_DEVICE_RDMA flag is set to only p4d{e} instances.

## v4.4.0 (2022-12-06)

### Features

 * integrate SMDDP collectives into smdataparallel runner

## v4.3.2 (2022-11-29)

### Bug Fixes and Other Changes

 * add general exception to filter

## v4.3.1 (2022-10-27)

### Bug Fixes and Other Changes

 * integrate upcoming dataparallel change to modelparallel
 * add unit tests for torchrun launcher and collections package deprecationWarning

## v4.3.0 (2022-10-20)

### Features

 * Add torch_distributed support for Trainium instances in SageMaker

## v4.2.10 (2022-10-17)

### Bug Fixes and Other Changes

 * * feature: Add neuron cores support (#21)

## v4.2.9 (2022-09-26)

### Bug Fixes and Other Changes

 * Add SageMaker Debugger exceptions

## v4.2.8 (2022-09-12)

## v4.2.7 (2022-09-10)

### Bug Fixes and Other Changes

 * improve worker node wait logic and update EFA flags

## v4.2.6 (2022-08-18)

### Bug Fixes and Other Changes

 * Enable PT XLA distributed training on homogeneous clusters

## v4.2.5 (2022-08-17)

### Bug Fixes and Other Changes

 * relax exception type

## v4.2.4 (2022-08-15)

## v4.2.3 (2022-08-11)

### Bug Fixes and Other Changes

 * update num_processes_per_host for smdataparallel runner

## v4.2.2 (2022-08-10)

### Bug Fixes and Other Changes

 * Removed version hardcoding for sagemaker test dependency
 * update distribution_instance_group for pytorch ddp
 * specify flake8 config explicitly

## v4.2.1 (2022-07-29)

### Bug Fixes and Other Changes

 * handle utf-8 decoding exceptions while processing stdout and stderr streams

## v4.2.0 (2022-07-08)

### Features

 * Heterogeneous cluster changes

## v4.1.6 (2022-06-28)

### Bug Fixes and Other Changes

 * update: protobuf version to overlap with TF requirements

## v4.1.5 (2022-06-22)

### Bug Fixes and Other Changes

 * Fix none exception class issue for mpi

## v4.1.4 (2022-06-10)

### Bug Fixes and Other Changes

 * Use framework provided error class and stack trace as error message

## v4.1.3 (2022-06-03)

## v4.1.2 (2022-05-25)

### Bug Fixes and Other Changes

 * fix flaky issue with incorrect rc being given

## v4.1.1 (2022-04-27)

### Bug Fixes and Other Changes

 * missing args when shell script is used

## v4.1.0 (2022-04-05)

### Features

 * add back FI_EFA_USE_DEVICE_RDMA=1 flag, revert 2936f22

## v4.0.1 (2022-01-29)

## v4.0.0 (2021-10-08)

### Breaking Changes

 * Add py38, dropped py36 and py2 support. Bump pypi to 4.0.0 (changes from PR #108)

## v3.9.3 ~ 4.0.0 (2021-10-07)

## Breaking Changes

 * Added `py38`, Removed `py36` and `py27` support

### Bug Fixes and Other Changes

 * Use asyncio to read stdout and stderr streams in realtime
 * Fix delayed logging issues
 * Convey user informative message if process gets OOM Killed
 * Filter out stderr to look for error messages and report
 * Report Exit code on training job failures
 * Prepend tags to MPI logs to enable easy filtering in CloudWatch
 * All the changes are from PR #108

### Documentation Changes

 * Update SM doc urls
 * Update Amazon Licensing
 ### Testing and Release Infrastructure

 * Install libssl1.1 and openssl packages in Dockerfiles
 * Added `asyncio` package
 * Updated tests to use `asyncio` package

## v3.9.2 (2021-04-27)

### Bug Fixes and Other Changes

 * Reverted -x FI_EFA_USE_DEVICE_RDMA=1 to fix a crash on PyTorch Dataloaders for Distributed training

## v3.9.1 (2021-04-13)

### Bug Fixes and Other Changes

 * [smdataparallel] better messages to establish the SSH connection between workers

## v3.9.0 (2021-04-07)

### Features

 * smdataparallel enable EFA RDMA flag

## v3.8.0 (2021-04-05)

### Features

 * smdataparallel custom mpi options support

## v3.7.5 (2021-03-30)

## v3.7.4 (2021-03-29)

### Bug Fixes and Other Changes

 * Update Dockerfile to accomomdate Rust dependency.

## v3.7.3 (2021-02-02)

### Bug Fixes and Other Changes

 * set btl_vader_single_copy_mechanism to none to avoid Read -1 Warning messages

## v3.7.2 (2020-12-18)

### Bug Fixes and Other Changes

 * set btl_vader_single_copy_mechanism to none

## v3.7.1 (2020-12-17)

### Bug Fixes and Other Changes

 * decode binary stderr string before dumping it out

## v3.7.0 (2020-12-09)

### Features

 * add data parallelism support (#3)

### Bug Fixes and Other Changes

 * update tox to use sagemaker 2.18.0 for tests
 * use format in place of f-strings and use comment style type annotations

## v3.6.4 (2020-12-08)

### Bug Fixes and Other Changes

 * workaround to print stderr when capturing

### Testing and Release Infrastructure

 * use ECR-hosted image for ubuntu:16.04

## v3.6.3.post0 (2020-11-11)

### Documentation Changes

 * fix typo in ENVIRONMENT_VARIABLES.md

## v3.6.3 (2020-10-26)

### Bug Fixes and Other Changes

 * propagate log level to aws services

## v3.6.2 (2020-08-04)

### Bug Fixes and Other Changes

 * check for script entry point even if setup.py is present

## v3.6.1.post1 (2020-08-03)

### Testing and Release Infrastructure

 * pin sagemaker<2 in test dependencies

## v3.6.1.post0 (2020-07-23)

### Documentation Changes

 * remove unofficially-supported environment variable

## v3.6.1 (2020-07-10)

### Bug Fixes and Other Changes

 * use '-bind-to none' flag to improve performance.

## v3.6.0 (2020-06-29)

### Features

 * persist env vars in /etc/environment for MPI processes

## v3.5.2.post0 (2020-06-29)

### Testing and Release Infrastructure

 * clarify feature request issue template

## v3.5.2 (2020-06-03)

### Bug Fixes and Other Changes

 * run Python script entry point as script and install from requirements.txt

## v3.5.1.post0 (2020-05-14)

### Documentation Changes

 * clean up README usage examples

## v3.5.1 (2020-05-11)

### Bug Fixes and Other Changes

 * Remove typing

## v3.5.0.post0 (2020-04-29)

### Testing and Release Infrastructure

 * Test against Python 3.7 in PR builds

## v3.5.0 (2020-04-27)

### Features

 * Add Python 3.7 support

## v3.4.2 (2020-04-21)

### Bug Fixes and Other Changes

 * Remove unused config files

### Documentation Changes

 * clean up README and other documentation

## v3.4.1 (2020-04-20)

### Bug Fixes and Other Changes

 * Remove etc directory

### Testing and Release Infrastructure

 * Add requirements.txt integration test in dummy container

## v3.4.0 (2020-04-16)

### Deprecations and Removals

 * Remove modules.download_and_install

### Bug Fixes and Other Changes

 * Refactor env
 * Refactor entry_point

### Documentation Changes

 * Update and add docstrings

### Testing and Release Infrastructure

 * Update GitHub issue and pull request templates

## v3.3.2 (2020-04-08)

### Bug Fixes and Other Changes

 * Refactor modules and entry_point (first pass)

## v3.3.1 (2020-04-06)

### Bug Fixes and Other Changes

 * Revert "change: stream stderr even when capture_error is True"
 * Use shlex.quote to construct bash command
 * Relax dependencies version requirements
 * Extract module to correct location in download_and_install
 * Upgrade psutil

### Testing and Release Infrastructure

 * Fix cleanup with requirements.txt functional tests
 * create __init__.py file for Python 2 import of protobuf during tests (#260)
 * Mark intermediate_output functional tests as xfail if not run on Linux

## v3.3.0 (2020-02-25)

### Deprecations and Removals

 * Remove serving CLI entry point

### Bug Fixes and Other Changes

 * Pin inotify-simple version

## v3.2.0 (2020-02-17)

### Deprecations and Removals

 * Remove legacy serving stack

### Features

 * Support specifying S3 endpoint URL

### Bug Fixes and Other Changes

 * Fix memory leak in gethostname and adapt len semantics to Posix

## v3.1.0 (2020-02-13)

### Deprecations and Removals

 * Remove beta directory

## v3.0.0 (2020-02-11)

### Breaking Changes

 * rename package from sagemaker_containers to sagemaker_training_toolkit

### Bug Fixes and Other Changes

 * modify download_and_install to work with local tarball
 * change scipy version pin to lower bound

## v2.6.2 (2019-12-18)

### Bug fixes and other changes

 * Add `scipy` to requried packages

## v2.6.1 (2019-11-30)

### Bug fixes and other changes

 * bug-fix: array_to_recordio_protobuf should return byte buffer instead of Stream
 * bug-fix: Typo in the execution-parameters routing rule

## v2.6.0 (2019-11-25)

### Features

 * adding support for execution_parameters endpoint for serving

## v2.5.12 (2019-11-15)

### Bug fixes and other changes

 * Adding support for encoding to recordio

## v2.5.11 (2019-10-29)

### Bug fixes and other changes

 * stream stderr even when capture_error is True

## v2.5.10 (2019-10-24)

### Bug fixes and other changes

 * use built-in csv library in csv encoding/decoding for correct quoted string handling.

## v2.5.9 (2019-09-25)

### Bug fixes and other changes

 * Patch os.path.exists for sshd

## v2.5.8 (2019-09-24)

### Bug fixes and other changes

 * Mark gethostname tests as xfail if run locally

## v2.5.7 (2019-09-23)

### Bug fixes and other changes

 * Add Pylint to development process

## v2.5.6 (2019-09-19)

### Bug fixes and other changes

 * Use copy when installing user module from local path
 * Integrate black into development process

## v2.5.5 (2019-07-31)

### Bug fixes and other changes

 * Update setup.py

## v2.5.4 (2019-07-30)

### Bug fixes and other changes

 * install user module before GUnicorn starts
 * include /opt/ml/code to GUnicorn PYTHONPATH

## v2.5.3 (2019-07-22)

### Bug fixes and other changes

 * ensure exit code is an int

## v2.5.2 (2019-07-18)

### Bug fixes and other changes

 * pin flake and werkzeug versions
 * add GPU default for MPI processes per host

### Documentation changes

 * fix env var in readme

## v2.5.1 (2019-06-27)

### Bug fixes and other changes

 * Added execution-parameters to nginx.conf.template

## v2.5.0 (2019-06-24)

### Features

 * entrypoint run waits for hostname resolution

## v2.4.10.post0 (2019-05-29)

### Documentation changes

 * fix path for training script location

## v2.4.10 (2019-05-20)

### Bug fixes and other changes

 * Detailed documentation for SageMaker Containers - training
 * download_and_extract local tar file

## v2.4.9 (2019-05-08)

### Bug fixes and other changes

 * add test for network isolation mode training
 * remove unnecessary name argument from download and extract function

## v2.4.8 (2019-05-02)

### Bug fixes and other changes

 * use mpi4py in MPI command for Python executables

## v2.4.7 (2019-04-30)

### Bug fixes and other changes

 * allow MPI options to be passed through entry_point.run

## v2.4.6.post0 (2019-04-24)

### Documentation changes

 * add commit message format to CONTRIBUTING.md and PR template

## v2.4.6 (2019-04-23)

### Bug fixes and other changes

 * update for automated releases

## v2.4.5

* bug-fix: use specified args, entry point, and env vars when creating a runner

## v2.4.4.post2

* doc-fix: Convert README to RST
* doc-fix: Update README with newer frameworks using SageMaker Containers

## v2.4.4.post1

* Specify ``long_description_content_type`` in setup

## v2.4.4

* bug-fix: correctly set NGINX_PROXY_READ_TIMEOUT to match model_sever_timeout.
* enhancement: remove numpy version restriction.

## v2.4.3

* bug-fix: Fix recursive directory navigation in intermediate output.

## v2.4.2

* bug-fix: Rename libchangehostname to gethostname to match POSIX function name

## v2.4.1

* feature: C extension reads hostname from resourceconfig instead of env var.

## v2.4.0

* feature: Generic OpenMPI support
* bug-fix: Fix response content_type handling

## v2.3.5

* bug-fix: Accept header ANY ('*/*') fallback to default accept
* feature: Add intermediate output to S3 during training
* bug-fix: reintroduce ``_modules.s3_download`` and ``_modules.download_and_install`` for backward compatibility

## v2.3.4

* feature: add capture_error flag to process.check_error and process.create and to all functions that runs process: modules.run, modules.run_module, and entry_point.run

## v2.3.3

* bug-fix: reintroduce _modules.prepare to import_module

## v2.3.2

* bug-fix: reintroduce _modules.prepare for backwards compatibility

## v2.3.1

* [breaking change] remove ``_modules.prepare`` and ``_modules.download_and_install``
* [breaking change] move ``_modules.s3_download`` to ``_files.s3_download``
* feature: support for Bash commands and Python scripts

## v2.3.0

* feature: Allow for dynamic nginx.conf creation
* feature: Provide support for additional environment variables. (http_port, safe_port_range and accept)

## v2.2.7

* feature: Making pip install less noisy
* bug-fix: Stream stderr instead of capturing it when running user script

## v2.2.6

* feature: Make it optional for run_module method to wait for the subprocess to exit
* feature: Allow additional sagemaker hyperparameters to be stored in TrainingEnv

## v2.2.5

* feature: Transformer: support user-supplied ``transform_fn``

## v2.2.4

* bug-fix: remove request size limit correctly

## v2.2.3

* enhancement: remove request size limit

## v2.2.2

* bug-fix: Fix choosing region for S3 client

## v2.2.1

* bug-fix: Use regional endpoint for S3 clients

## v2.2.0

* [breaking change] Remove ``status_codes`` module and use ``six.moves.http_client`` instead
* [breaking change] Move ``UnsupportedFormatError`` from ``encoders`` module to ``errors`` module
* Return 4XX status codes for ``UnsupportedFormatError`` from default input/output handlers

## v2.1.0

* Allow for local modules to work with AWS SageMaker framework containers.
* Support for training outside of AWS SageMaker Training.

## v2.0.4

* Fix output_data_dir to reference an existing directory.
* Fix error message.
* Make pip install verbose.

## v2.0.3

* Fix error class for user script errors.
* Adding Readme.

## v2.0.2

* Improve logging
* Support for hyperparameters with JSON serialized and non serialized keys altogether
* Training Environment transforms to env vars
* Created beta framework entrypoint
* Filter SageMaker provided hyperparameters and user provided hyperparameters
* Script mode
* Cache module installation
* Support to requirements.txt
* Decoder/Encoder support for numpy, JSON, and CSV

## v1.0.4

* bug: Configuration: Change module names to string in __all__
* bug: Environment: handle hyperparameter injected by tuning jobs

## v1.0.3

* bug: Training: Move processing of requirements file out to the specific container.

## v1.0.2

* feature: TrainingEnvironment: read new environment variable for job name

## v1.0.1

* feature: Documentation: add descriptive README

## v1.0.0

* Initial commit
