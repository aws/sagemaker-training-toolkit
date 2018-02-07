import pytest
from mock import patch, call

import container_support as cs


def test_parse_s3_url_invalid():
    with pytest.raises(ValueError):
        cs.parse_s3_url("nots3://blah/blah")


def test_parse_s3_url():
    assert ("bucket", "key") == cs.parse_s3_url("s3://bucket/key")


def test_parse_s3_url_no_key():
    assert ("bucket", "") == cs.parse_s3_url("s3://bucket/")


def test_download_s3():
    with patch('boto3.resource') as patched:
        assert cs.download_s3_resource("s3://bucket/key", "target") == "target"
        assert [call('s3'),
                call().Bucket('bucket'),
                call().Bucket().download_file('key', 'target')] == patched.mock_calls


def test_untar_directory():
    with patch('container_support.utils.open', create=True) as mocked_open, patch('tarfile.open') as mocked_tarfile:
        cs.untar_directory('a/b/c', 'd/e/f')
        assert call('a/b/c', 'rb') in mocked_open.mock_calls
        assert call().__enter__().extractall(path='d/e/f') in mocked_tarfile.mock_calls
