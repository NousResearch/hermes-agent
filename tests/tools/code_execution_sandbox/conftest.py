"""Shared fixtures for the confinement tests.

``child_env`` is imported here rather than in a test module: pytest registers fixtures
found in a conftest, and importing it next to a test function that takes ``child_env`` as
a parameter trips lint rule F811.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from tests.tools._child_env_fixtures import child_env


@pytest.fixture
def short_dir():
    """A directory short enough for an AF_UNIX socket.

    pytest's own ``tmp_path`` lands under a long root in CI, and an RPC socket inside it
    dies with ENAMETOOLONG before the test can assert anything.
    """
    path = Path(tempfile.mkdtemp(prefix="kb_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
