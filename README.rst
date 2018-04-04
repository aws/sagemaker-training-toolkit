===========================
SageMaker Container Support
===========================

This package makes it easier to build docker images which run on Amazon SageMaker. For example, the official SageMaker `TensorFlow <https://github.com/aws/sagemaker-tensorflow-containers>`_ and `MXNet <https://github.com/aws/sagemaker-mxnet-containers>`_ images use this package. The requirements for SageMaker-compatible images are documented here: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html

Features provided:

- Integration with `SageMaker Python SDK  <https://github.com/aws/sagemaker-python-sdk>`_ Estimators, including:

  - Downloading user-provided python code
  - Deserializing hyperparameters (which preserves their python types)

- ``bin/entry.py``, which acts as the docker entrypoint required by SageMaker
- Reading in the metadata files provided to the container during training, as described here: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
- nginx + Gunicorn HTTP server for serving inference requests which complies with the interface described here: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html

License
-------

This library is licensed under the Apache 2.0 License.
