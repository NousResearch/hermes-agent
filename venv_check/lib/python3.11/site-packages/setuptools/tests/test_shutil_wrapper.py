import stat
import sys
from unittest.mock import Mock

from setuptools import _shutil


def test_rmtree_readonly(monkeypatch, tmp_path):
    """Verify onerr works as expected"""

    tmp_dir = tmp_path / "with_readonly"
    tmp_dir.mkdir()
    some_file = tmp_dir.joinpath("file.txt")
    some_file.touch()
    some_file.chmod(stat.S_IREAD)

    expected_count = 1 if sys.platform.startswith("win") else 0
    chmod_fn = Mock(wraps=_shutil.attempt_chmod_verbose)
    monkeypatch.setattr(_shutil, "attempt_chmod_verbose", chmod_fn)

    _shutil.rmtree(tmp_dir)
    assert chmod_fn.call_count == expected_count
    assert not tmp_dir.is_dir()
