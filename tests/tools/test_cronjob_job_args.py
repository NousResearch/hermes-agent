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


class TestOriginFromEnv:
    """The cron origin must capture parent_chat_id for thread sessions so a
    deleted thread can fall back to its parent channel at delivery time."""

    def test_captures_parent_chat_id_for_thread_origin(self):
        from tools.cronjob_job_args import _origin_from_env
        from gateway.session_context import clear_session_vars, set_session_vars

        tokens = set_session_vars(
            platform="discord",
            chat_id="thread-456",
            chat_name="#general / reminder thread",
            thread_id="thread-456",
            parent_chat_id="parent-channel-123",
        )
        try:
            origin = _origin_from_env()
        finally:
            clear_session_vars(tokens)

        assert origin["parent_chat_id"] == "parent-channel-123"
        assert origin["platform"] == "discord"
        assert origin["thread_id"] == "thread-456"

    def test_parent_chat_id_omitted_when_absent(self):
        from tools.cronjob_job_args import _origin_from_env
        from gateway.session_context import clear_session_vars, set_session_vars

        tokens = set_session_vars(platform="telegram", chat_id="999")
        try:
            origin = _origin_from_env()
        finally:
            clear_session_vars(tokens)

        assert "parent_chat_id" not in origin

    def test_captures_chat_type_for_dm_origin(self):
        from tools.cronjob_job_args import _origin_from_env
        from gateway.session_context import clear_session_vars, set_session_vars

        tokens = set_session_vars(platform="telegram", chat_id="user-1", chat_type="dm")
        try:
            origin = _origin_from_env()
        finally:
            clear_session_vars(tokens)

        assert origin["chat_type"] == "dm"

    def test_chat_type_omitted_when_absent(self):
        from tools.cronjob_job_args import _origin_from_env
        from gateway.session_context import clear_session_vars, set_session_vars

        tokens = set_session_vars(platform="telegram", chat_id="999")
        try:
            origin = _origin_from_env()
        finally:
            clear_session_vars(tokens)

        assert "chat_type" not in origin
