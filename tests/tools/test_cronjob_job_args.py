"""Tests for tools/cronjob_job_args.py::_validate_cron_script_path (issue #105761).

Regression coverage: the validator used to accept a script path that pointed
at a file that doesn't exist, only failing later at every cron fire with a
generic "Script not found" error from cron/scheduler_script.py. It also
hardcoded "~/.hermes/scripts/" in its messages even though resolution goes
through get_hermes_home(), which is per-profile.
"""

from hermes_constants import get_hermes_home
from tools.cronjob_job_args import _validate_cron_script_path


class TestValidateCronScriptPath:
    def test_missing_script_file_is_rejected_at_creation_time(self):
        error = _validate_cron_script_path("does_not_exist.sh")
        assert error is not None
        assert "not found" in error.lower()

    def test_existing_script_file_passes(self):
        scripts_dir = get_hermes_home() / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "real.sh").write_text("#!/bin/sh\necho hi\n")
        assert _validate_cron_script_path("real.sh") is None

    def test_missing_file_error_names_the_resolved_scripts_dir(self):
        # The resolved dir must appear literally so the message stays correct
        # under profiles, where get_hermes_home() is not the global ~/.hermes.
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("missing.py")
        assert str(scripts_dir) in error

    def test_absolute_path_error_names_the_resolved_scripts_dir_not_global_literal(self):
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("/etc/passwd")
        assert str(scripts_dir) in error
