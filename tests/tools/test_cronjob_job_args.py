"""Tests for tools/cronjob_job_args.py validators (issues #105761, #135942).

Regression coverage: the script-path validator used to accept a path that pointed
at a file that doesn't exist, only failing later at every cron fire with a
generic "Script not found" error from cron/scheduler_script.py. It also
hardcoded "~/.hermes/scripts/" in its messages even though resolution goes
through get_hermes_home(), which is per-profile.

The deliver-target validator (below) is new: an explicit ``platform:target``
element used to be parsed only when the job fired, so a malformed target
surfaced as delivery_failed after the job's work had already run (#135942).
"""

from unittest.mock import patch

from hermes_constants import get_hermes_home
from tools.cronjob_job_args import (
    _validate_cron_script_path,
    _validate_platform_deliver_targets,
)


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


class TestValidatePlatformDeliverTargets:
    """Explicit ``platform:target`` deliver elements must resolve through the send
    path's resolver at create/update time, not first at fire time (#135942)."""

    def test_unresolvable_target_is_rejected_upfront(self):
        # "not-a-chat" is neither an explicit Telegram id/@username nor a channel-directory
        # entry, so the strict (model-in-the-loop) resolver used by send_message rejects it.
        with (
            patch("tools.send_message_tool.prepare_send_message_platforms"),
            patch("gateway.channel_directory.resolve_channel_name", return_value=None),
        ):
            error = _validate_platform_deliver_targets("telegram:not-a-chat")
        assert error is not None
        assert "invalid deliver target" in error
        assert "telegram:not-a-chat" in error

    def test_explicit_numeric_target_passes(self):
        # A numeric Telegram chat id with a topic is explicit syntax; no directory
        # or network lookup is involved.
        with patch("tools.send_message_tool.prepare_send_message_platforms"):
            assert _validate_platform_deliver_targets("telegram:-1001234567890:17585") is None

    def test_routing_tokens_and_bare_platforms_bypass_the_resolver(self):
        # local/origin/all/bot-chat[:profile] and bare platform names (home-channel
        # delivery) follow their own fire-time rules; the resolver must not see them.
        with (
            patch("tools.send_message_tool.prepare_send_message_platforms"),
            patch(
                "tools.send_message_targets.resolve_send_target",
                side_effect=AssertionError("resolver must not be consulted"),
            ),
        ):
            assert _validate_platform_deliver_targets(
                "local,origin,all,bot-chat,bot-chat:worker,telegram") is None
