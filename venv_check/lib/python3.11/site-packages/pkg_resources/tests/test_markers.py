from unittest import mock

from pkg_resources import evaluate_marker


@mock.patch('platform.python_version', return_value='2.7.10')
def test_ordering(python_version_mock):
    assert evaluate_marker("python_full_version > '2.7.3'") is True
