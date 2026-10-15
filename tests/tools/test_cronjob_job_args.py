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


class TestScriptArgumentsValidation:
    """#20300 / #43: an argument-bearing ``script`` value validates at CREATE time.

    ``_validate_cron_script_path`` used to resolve the whole value as one filename,
    so ``"job.py expire"`` was rejected as a missing file (or, where validation was
    skipped, stored and failed on every fire). Only the path part should be
    validated; the arguments belong to the script.
    """

    def test_script_with_arguments_validates_when_the_script_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "job.py").write_text("print('hi')\n")

        from tools.cronjob_job_args import _validate_cron_script_path

        assert _validate_cron_script_path("job.py expire") is None

    def test_missing_script_with_arguments_reports_the_path_not_the_argv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "scripts").mkdir()

        from tools.cronjob_job_args import _validate_cron_script_path

        err = _validate_cron_script_path("nope.py expire")
        assert err is not None
        assert "Script file not found" in err
        # Must name the real path, so the operator is not sent hunting for a file
        # literally called "nope.py expire".
        assert "nope.py" in err

    def test_traversal_is_still_blocked_with_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "scripts").mkdir()

        from tools.cronjob_job_args import _validate_cron_script_path

        err = _validate_cron_script_path("../../etc/passwd read")
        assert err is not None
        assert "traversal" in err or "escapes" in err

    def test_absolute_path_still_blocked_with_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "scripts").mkdir()

        from tools.cronjob_job_args import _validate_cron_script_path

        err = _validate_cron_script_path("/usr/bin/env python")
        assert err is not None
        assert "relative" in err

    def test_plain_filename_still_validates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "job.py").write_text("print('hi')\n")

        from tools.cronjob_job_args import _validate_cron_script_path

        assert _validate_cron_script_path("job.py") is None
        assert _validate_cron_script_path("") is None
