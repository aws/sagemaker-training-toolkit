#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
import tarfile

import boto3
from six.moves.urllib.parse import urlparse

from container_support import ContainerEnvironment


def parse_s3_url(url):
    """ Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip('/')


def download_s3_resource(source, target):
    """ Downloads the s3 object source and stores in a new file with path target.
    """
    print("Downloading {} to {}".format(source, target))
    region = os.environ.get('AWS_REGION',
                            os.environ.get(ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME.upper()))
    s3 = boto3.resource('s3', region_name=region)

    script_bucket_name, script_key_name = parse_s3_url(source)
    script_bucket = s3.Bucket(script_bucket_name)
    script_bucket.download_file(script_key_name, target)

    return target


def untar_directory(tar_file_path, extract_dir_path):
    with open(tar_file_path, 'rb') as f:
        with tarfile.open(mode='r:gz', fileobj=f) as t:
            t.extractall(path=extract_dir_path)
