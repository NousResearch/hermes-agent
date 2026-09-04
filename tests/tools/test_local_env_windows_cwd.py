"""LocalEnvironment retains a native Windows cwd after each Git Bash call."""

import ntpath
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows",
    reason="exercises Git Bash's native pwd output",
)


@pytest.fixture
def temp_tree():
    root = Path(tempfile.mkdtemp(prefix="hermes-cwd-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_temp_and_space_path_remain_native_after_cd(temp_tree):
    target = temp_tree / "spaced dir"
    target.mkdir()
    env = LocalEnvironment(cwd=str(temp_tree), timeout=30)
    try:
        result = env.execute(f"cd '{target.as_posix()}'", timeout=30)

        assert result["returncode"] == 0
        assert env.cwd == ntpath.normpath(str(target))
        assert Path(env.cwd).is_dir()
    finally:
        env.cleanup()


def test_drive_form_probe_is_canonicalized_for_non_c_drive():
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = r"C:\old"
    env._cwd_marker = "__HERMES_CWD_TEST__"
    result = {
        "output": "ok\n__HERMES_CWD_TEST__D:/Work Space/demo__HERMES_CWD_TEST__\n"
    }

    with patch("tools.environments.local.os.path.isdir", return_value=True):
        env._extract_cwd_from_output(result)

    assert env.cwd == r"D:\Work Space\demo"
    assert result["output"] == "ok"
