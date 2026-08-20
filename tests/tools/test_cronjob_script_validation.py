"""Tests for _validate_cron_script_path in tools/cronjob_tools.py.

Validates two fixes:
  1. Absolute paths within scripts_dir are accepted (write_file returns
     absolute paths, so the LLM naturally passes them to the cron tool)
  2. Script existence is checked at creation time (not 60s later at the
     next scheduler tick)

Both fixes align _validate_cron_script_path with the resolution logic
already used by _run_job_script in cron/scheduler.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestAbsolutePathNormalization:
    """Absolute paths within scripts_dir should be accepted, not rejected."""

    def test_absolute_path_within_scripts_dir_accepted(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "monitor.py").write_text("print('ok')")

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(str(scripts_dir / "monitor.py"))
        assert err is None, f"Absolute path within scripts_dir should be accepted, got: {err}"

    def test_absolute_path_outside_scripts_dir_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("/etc/passwd")
        assert err is not None, "Absolute path outside scripts_dir should be rejected"

    def test_absolute_path_nonexistent_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(str(scripts_dir / "nope.py"))
        assert err is not None
        assert "not found" in err.lower()

    def test_create_with_absolute_path_succeeds(self, cron_env, monkeypatch):
        """The natural LLM flow: write_file returns absolute path, pass it to cronjob."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        script_path = scripts_dir / "a-share-test.py"
        script_path.write_text("print('ok')")

        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                prompt="Analyze A-shares",
                script=str(script_path),
            )
        )
        assert result["success"] is True


class TestExistenceValidation:
    """Script must exist at creation time — fail fast, not at scheduler tick."""

    def test_nonexistent_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("does_not_exist.py")
        assert err is not None, "Non-existent script should be rejected"
        assert "not found" in err.lower()

    def test_existing_script_accepted(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "monitor.py").write_text("print('ok')")

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("monitor.py")
        assert err is None, f"Existing script should be accepted, got: {err}"

    def test_nonexistent_script_in_subdirectory_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("checks/check_sumitaro.py")
        assert err is not None

    def test_existing_script_in_subdirectory_accepted(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "checks").mkdir()
        (scripts_dir / "checks" / "check_sumitaro.py").write_text("print('ok')")

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("checks/check_sumitaro.py")
        assert err is None, f"Existing script in subdirectory should be accepted, got: {err}"

    def test_create_with_nonexistent_script_fails(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                prompt="Monitor things",
                script="check_sumitaro.py",
            )
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_create_with_existing_script_succeeds(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "monitor.py").write_text("print('ok')")

        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                prompt="Monitor things",
                script="monitor.py",
            )
        )
        assert result["success"] is True
        assert result["job"]["script"] == "monitor.py"


class TestIsFileCheck:
    """Directories and non-files must be rejected even if they exist."""

    def test_directory_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "mydir").mkdir()

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("mydir")
        assert err is not None
        assert "not a file" in err.lower()

    def test_directory_absolute_path_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        scripts_dir = cron_env / "scripts"
        (scripts_dir / "mydir").mkdir()

        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(str(scripts_dir / "mydir"))
        assert err is not None
        assert "not a file" in err.lower()


class TestSecurityBoundary:
    """Traversal and escape attempts must still be blocked."""

    def test_traversal_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("../../etc/passwd")
        assert err is not None

    def test_tilde_nonexistent_user_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("~nonexistent_user_xyz/foo.py")
        assert err is not None

    def test_empty_script_still_allowed(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import _validate_cron_script_path

        assert _validate_cron_script_path(None) is None
        assert _validate_cron_script_path("") is None
        assert _validate_cron_script_path("  ") is None
