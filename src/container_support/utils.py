import tarfile

import boto3
from six.moves.urllib.parse import urlparse


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
    s3 = boto3.resource('s3')

    script_bucket_name, script_key_name = parse_s3_url(source)
    script_bucket = s3.Bucket(script_bucket_name)
    script_bucket.download_file(script_key_name, target)

    return target


def untar_directory(tar_file_path, extract_dir_path):
    with open(tar_file_path, 'rb') as f:
        with tarfile.open(mode='r:gz', fileobj=f) as t:
            t.extractall(path=extract_dir_path)
