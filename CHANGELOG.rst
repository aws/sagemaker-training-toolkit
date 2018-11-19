=========
CHANGELOG
=========

2.3.1
=====

* feature: support for Bash commands and Python scripts

2.3.0
=====

* feature: Allow for dynamic nginx.conf creation
* feature: Provide support for additional environment variables. (http_port, safe_port_range and accept)

2.2.7
=====

* feature: Making pip install less noisy
* bug-fix: Stream stderr instead of capturing it when running user script

2.2.6
=====

* feature: Make it optional for run_module method to wait for the subprocess to exit
* feature: Allow additional sagemaker hyperparameters to be stored in TrainingEnv

2.2.5
=====

* feature: Transformer: support user-supplied ``transform_fn``

2.2.4
=====

* bug-fix: remove request size limit correctly

2.2.3
=====

* enhancement: remove request size limit

2.2.2
=====

* bug-fix: Fix choosing region for S3 client

2.2.1
=====

* bug-fix: Use regional endpoint for S3 clients

2.2.0
=====

* [breaking change] Remove ``status_codes`` module and use ``six.moves.http_client`` instead
* [breaking change] Move ``UnsupportedFormatError`` from ``encoders`` module to ``errors`` module
* Return 4XX status codes for ``UnsupportedFormatError`` from default input/output handlers

2.1.0
=====

* Allow for local modules to work with AWS SageMaker framework containers.
* Support for training outside of AWS SageMaker Training.

2.0.4
=====

* Fix output_data_dir to reference an existing directory.
* Fix error message.
* Make pip install verbose.

2.0.3
=====

* Fix error class for user script errors.
* Adding Readme.

2.0.2
=====

* Improve logging
* Support for hyperparameters with JSON serialized and non serialized keys altogether
* Training Environment transforms to env vars
* Created beta framework entrypoint
* Filter SageMaker provided hyperparameters and user provided hyperparameters
* Script mode
* Cache module installation
* Support to requirements.txt
* Decoder/Encoder support for numpy, JSON, and CSV

1.0.4
=====

* bug: Configuration: Change module names to string in __all__
* bug: Environment: handle hyperparameter injected by tuning jobs

1.0.3
=====

* bug: Training: Move processing of requirements file out to the specific container.

1.0.2
=====

* feature: TrainingEnvironment: read new environment variable for job name

1.0.1
=====

* feature: Documentation: add descriptive README

1.0.0
=====

* Initial commit
