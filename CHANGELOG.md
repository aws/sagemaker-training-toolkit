# Changelog

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
