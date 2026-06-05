"""Tests for cron job script-path validation.

A script-based (no_agent) cron job stores a *relative* script name that the
host-side runner resolves under ``HERMES_HOME/scripts/``.  Historically the
job could be created pointing at a script that did not exist there (the agent
had written it into its sandbox workspace instead), so the failure only
surfaced at run time as ``Script not found``.  These tests pin the fail-fast
behaviour at the create/edit boundary and the actionable error message.
"""

import os
from unittest.mock import patch

from tools.cronjob_tools import _validate_cron_script_path


class TestCronScriptExistence:
    def test_rejects_script_that_does_not_exist(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            err = _validate_cron_script_path("aaw2_monitor.py")

        assert err is not None
        # The message must name the script and tell the agent exactly where to
        # place it (the sandbox-visible scripts dir).
        assert "aaw2_monitor.py" in err
        assert "~/.hermes/scripts/" in err

    def test_accepts_script_that_exists(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        scripts_dir = hermes_home / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "aaw2_monitor.py").write_text("print('hi')\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            err = _validate_cron_script_path("aaw2_monitor.py")

        assert err is None

    def test_empty_script_is_allowed(self, tmp_path):
        # Empty / None means "clear the field" — always OK, no file required.
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            assert _validate_cron_script_path("") is None
            assert _validate_cron_script_path(None) is None

    def test_absolute_path_still_rejected(self, tmp_path):
        # Existence check must not weaken the existing path-safety guard.
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            err = _validate_cron_script_path("/etc/passwd")
        assert err is not None
        assert "relative" in err.lower()
