"""Tests for approvals.noninteractive_mode — fail-closed for headless contexts."""

from unittest.mock import patch as mock_patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    _get_noninteractive_approval_mode,
    check_dangerous_command,
)


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")


@pytest.fixture
def headless(monkeypatch):
    """No interactive CLI, no gateway, no cron — the fail-open branch."""
    for var in (
        "HERMES_INTERACTIVE",
        "HERMES_GATEWAY_SESSION",
        "HERMES_CRON_SESSION",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(var, raising=False)


def _config(approvals):
    return mock_patch("hermes_cli.config.load_config", return_value={"approvals": approvals})


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------

class TestNoninteractiveApprovalModeParsing:
    def test_default_is_approve(self):
        """Absent config keeps today's behaviour — operators opt in to deny."""
        with _config({}):
            assert _get_noninteractive_approval_mode() == "approve"

    def test_explicit_deny(self):
        with _config({"noninteractive_mode": "deny"}):
            assert _get_noninteractive_approval_mode() == "deny"

    def test_explicit_approve(self):
        with _config({"noninteractive_mode": "approve"}):
            assert _get_noninteractive_approval_mode() == "approve"

    @pytest.mark.parametrize("value", ["DENY", " deny ", "block", "closed", "no"])
    def test_deny_synonyms_and_whitespace(self, value):
        with _config({"noninteractive_mode": value}):
            assert _get_noninteractive_approval_mode() == "deny"

    def test_unrecognised_value_falls_back_to_approve(self):
        """An unknown value must not silently start blocking a live deployment."""
        with _config({"noninteractive_mode": "maybe"}):
            assert _get_noninteractive_approval_mode() == "approve"

    def test_unreadable_config_falls_back_to_approve(self):
        with mock_patch("hermes_cli.config.load_config", side_effect=OSError("boom")):
            assert _get_noninteractive_approval_mode() == "approve"


# ---------------------------------------------------------------------------
# effect on check_dangerous_command
# ---------------------------------------------------------------------------

class TestNoninteractiveGate:
    def test_deny_blocks_a_dangerous_command(self, headless):
        with _config({"noninteractive_mode": "deny"}):
            result = check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is False
        assert "noninteractive_mode" in result["message"]

    def test_approve_preserves_existing_behaviour(self, headless):
        with _config({"noninteractive_mode": "approve"}):
            result = check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is True

    def test_default_config_preserves_existing_behaviour(self, headless):
        with _config({}):
            result = check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is True

    def test_harmless_command_is_unaffected(self, headless):
        with _config({"noninteractive_mode": "deny"}):
            result = check_dangerous_command("git status", "local")
        assert result["approved"] is True

    def test_cron_still_governed_by_cron_mode(self, headless, monkeypatch):
        """noninteractive_mode must not override the cron branch."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        with _config({"noninteractive_mode": "deny", "cron_mode": "approve"}):
            result = check_dangerous_command("git push --force origin main", "local")
        assert result["approved"] is True


# ---------------------------------------------------------------------------
# the consolidated guard — the path terminal execution actually takes
# ---------------------------------------------------------------------------

class TestConsolidatedGuard:
    """`terminal_tool` routes through `check_all_command_guards`, not
    `check_dangerous_command`. A policy that only covers the latter has no
    effect on real terminal calls."""

    def test_deny_blocks_through_the_consolidated_guard(self, headless):
        from tools.approval import check_all_command_guards
        with _config({"noninteractive_mode": "deny"}):
            result = check_all_command_guards("git reset --hard HEAD~3", "local")
        assert result["approved"] is False
        assert "noninteractive_mode" in result["message"]

    def test_approve_preserves_existing_behaviour_through_the_guard(self, headless):
        from tools.approval import check_all_command_guards
        with _config({"noninteractive_mode": "approve"}):
            result = check_all_command_guards("git reset --hard HEAD~3", "local")
        assert result["approved"] is True

    def test_default_preserves_existing_behaviour_through_the_guard(self, headless):
        from tools.approval import check_all_command_guards
        with _config({}):
            result = check_all_command_guards("git reset --hard HEAD~3", "local")
        assert result["approved"] is True

    def test_harmless_command_unaffected_through_the_guard(self, headless):
        from tools.approval import check_all_command_guards
        with _config({"noninteractive_mode": "deny"}):
            result = check_all_command_guards("git status", "local")
        assert result["approved"] is True

    def test_cron_still_governed_by_cron_mode_through_the_guard(self, headless, monkeypatch):
        from tools.approval import check_all_command_guards
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        with _config({"noninteractive_mode": "deny", "cron_mode": "approve"}):
            result = check_all_command_guards("git reset --hard HEAD~3", "local")
        assert result["approved"] is True


# ---------------------------------------------------------------------------
# the real DEFAULT_CONFIG / deep-merge contract, not a mocked load_config
# ---------------------------------------------------------------------------

def _write_hermes_config(monkeypatch, tmp_path, config):
    import yaml
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """load_config memoises; each case must read its own file."""
    import hermes_cli.config as cfg
    for attr in ("_CONFIG_CACHE", "_config_cache"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, None)
    yield


class TestDefaultConfigContract:
    def test_key_has_a_canonical_default(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["approvals"]["noninteractive_mode"] == "approve"

    def test_absent_from_user_config_deep_merges_to_approve(self, monkeypatch, tmp_path):
        """A user file that sets only `mode` must still resolve the new key."""
        _write_hermes_config(monkeypatch, tmp_path, {"approvals": {"mode": "manual"}})
        assert _get_noninteractive_approval_mode() == "approve"

    def test_user_config_overrides_the_default(self, monkeypatch, tmp_path):
        _write_hermes_config(
            monkeypatch, tmp_path, {"approvals": {"noninteractive_mode": "deny"}}
        )
        assert _get_noninteractive_approval_mode() == "deny"

    def test_sibling_keys_survive_the_merge(self, monkeypatch, tmp_path):
        """Setting the new key must not drop cron_mode's default."""
        from hermes_cli.config import load_config

        _write_hermes_config(
            monkeypatch, tmp_path, {"approvals": {"noninteractive_mode": "deny"}}
        )
        approvals = load_config()["approvals"]
        assert approvals["noninteractive_mode"] == "deny"
        assert approvals["cron_mode"] == "deny"
        assert approvals["timeout"] == 300
