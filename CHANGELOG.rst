=========
CHANGELOG
=========

1.1.2
=====

* bug-fix: Only load hosting environment variables once when each Gunicorn worker starts

1.1.1
=====

* bug-fix: Accept header ANY ('*/*') fallback to default accept

1.1.0
=====

* feature: Allow for dynamic nginx.conf creation
* feature: Provide support for additional environment variables. (http_port, safe_port_range and accept)

1.0.9
=====

* bug-fix: remove request size limit correctly

1.0.8
=====

* enhancement: remove request size limit

1.0.7
=====

* bug: Pip dependencies should be installed when the server start
* bug-fix: Utils: Fix region-choosing when making S3 client

1.0.6
=====

* enhancement: Utils: Specify region name in boto call

1.0.5
=====

* bug-fix: Serving: Change error message retrieval to use str(e)

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
