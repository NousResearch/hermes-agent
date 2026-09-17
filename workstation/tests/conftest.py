"""Workstation contracts always use an ephemeral personal installation."""
import pytest
import atexit
import os
import shutil
import tempfile

# Fixtures run after collection: isolate import-time logger/store side effects too.
_session_home = tempfile.mkdtemp(prefix="hermes-workstation-tests-")
os.environ["HERMES_HOME"] = os.path.join(_session_home, "hermes")
os.environ["HERMES_WORKSTATION_HOME"] = os.path.join(_session_home, "workstation")
os.environ["HERMES_KANBAN_DB"] = os.path.join(_session_home, "hermes", "kanban.db")
os.environ["HERMES_TEST_ISOLATION"] = _session_home
atexit.register(shutil.rmtree, _session_home, True)


@pytest.fixture(autouse=True)
def isolated_workstation_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_WORKSTATION_HOME", str(tmp_path / "workstation"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "hermes" / "kanban.db"))
