"""The four approvals.* host power keys are read ONLY from the managed config layer
(/etc/hermes/config.yaml or HERMES_MANAGED_DIR), never from the user's config.yaml.

The agent's terminal runs as the user that owns ~/.hermes/config.yaml, so it can write that file in
ways no command filter can enumerate (variables, eval, sed, python, ...). Reading the keys from the
root-owned managed layer closes that whole class: whatever the agent writes into its own config, a
power command stays on the hardline floor."""

import time

import pytest

from hermes_cli import managed_scope
from tools import approval_context
from tools.approval import check_all_command_guards, clear_session, disable_session_yolo
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.approval_detection import (
    _MAX_SEPARATOR_FREE_COMMAND_CHARS, _command_parser_limit_exceeded, detect_dangerous_command,
    detect_hardline_command, detect_power_state_command)

_ALL_ON = {"allow_reboot": True, "allow_shutdown": True,
           "reboot_without_approval": True, "shutdown_without_approval": True}


@pytest.fixture
def managed_dir(tmp_path, monkeypatch):
    d = tmp_path / "managed"
    d.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(d))
    managed_scope.invalidate_managed_cache()

    def write(approvals: dict | None):
        if approvals is None:
            (d / "config.yaml").write_text("{}\n", encoding="utf-8")
        else:
            lines = ["approvals:"] + [f"  {k}: {str(v).lower()}" for k, v in approvals.items()]
            (d / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        managed_scope.invalidate_managed_cache()
    yield write
    managed_scope.invalidate_managed_cache()


def test_user_config_power_keys_are_ignored(monkeypatch, managed_dir):
    managed_dir(None)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()
    assert approval_context._get_power_no_approval_categories() == frozenset()


def test_ignored_user_keys_are_logged_once(monkeypatch, managed_dir, caplog):
    import logging
    managed_dir(None)
    monkeypatch.setattr(approval_context, "_warned_ignored_power_keys", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"allow_reboot": True})
    with caplog.at_level(logging.WARNING, logger="tools.approval"):
        approval_context._get_hardline_opt_ins()
        approval_context._get_hardline_opt_ins()
    hits = [r for r in caplog.records if "is ignored" in r.getMessage()]
    assert len(hits) == 1 and "approvals.allow_reboot" in hits[0].getMessage()


def test_managed_config_enables_the_categories(monkeypatch, managed_dir):
    managed_dir({"allow_reboot": True, "reboot_without_approval": True})
    monkeypatch.setattr(approval_context, "_get_approval_config", dict)
    assert approval_context._get_hardline_opt_ins() == frozenset({"reboot"})
    assert approval_context._get_power_no_approval_categories() == frozenset({"reboot"})


def test_managed_false_wins_over_user_true(monkeypatch, managed_dir):
    managed_dir({"allow_reboot": False})
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_no_managed_scope_fails_closed(monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    monkeypatch.setattr(managed_scope, "load_managed_config", dict)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_managed_read_error_fails_closed(monkeypatch):
    def boom():
        raise OSError("unreadable")
    monkeypatch.setattr(managed_scope, "load_managed_config", boom)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_agent_write_to_user_config_does_not_lift_the_floor(tmp_path, monkeypatch, managed_dir):
    """End to end: the guard chain reads a real user config.yaml with all four keys true (as an
    injected agent could write it by any route) and still hard-blocks reboot with no prompt."""
    from hermes_cli import config as hconfig
    managed_dir(None)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "approvals:\n  mode: manual\n" + "".join(f"  {k}: true\n" for k in _ALL_ON), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hconfig, "get_config_path", lambda: home / "config.yaml")
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    token = set_current_session_key("power_managed_source_test")
    disable_session_yolo("power_managed_source_test")
    clear_session("power_managed_source_test")
    asked = []
    try:
        result = check_all_command_guards(
            "sudo reboot", "local", approval_callback=lambda *a, **k: asked.append(a) or "once")
    finally:
        clear_session("power_managed_source_test")
        reset_current_session_key(token)
    assert result["approved"] is False and result.get("hardline") is True
    assert asked == []


# ---- Guard latency just under the separator-free parser cap ------------------------------------

@pytest.mark.parametrize("filler", [" ", "\t"])
def test_detection_is_fast_just_under_the_parser_cap(filler):
    """4095 separator-free chars is the largest input the floor still parses (4096+ short-circuits on
    the parser limit, so a longer probe proves nothing). The _CMDPOS prefix used to backtrack
    quadratically here: ~13 s hardline + ~12 s dangerous per command."""
    command = filler * (_MAX_SEPARATOR_FREE_COMMAND_CHARS - 2) + "q"
    assert len(command) == _MAX_SEPARATOR_FREE_COMMAND_CHARS - 1
    assert not _command_parser_limit_exceeded(command)
    start = time.monotonic()
    detect_hardline_command(command)
    detect_dangerous_command(command)
    detect_power_state_command(command)
    assert time.monotonic() - start < 2.0
