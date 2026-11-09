"""The four approvals.* host power keys are read ONLY from the root-owned /etc/hermes/config.yaml,
never from the user's config.yaml and never from a HERMES_MANAGED_DIR override.

The agent's terminal runs as the user that owns ~/.hermes/config.yaml and ~/.hermes/.env, so it can
write either file in ways no command filter can enumerate (variables, eval, sed, python, ...), and a
.env line can point HERMES_MANAGED_DIR at an agent-written directory. Reading the keys from a fixed,
ownership-checked path closes that whole class: whatever the agent writes, a power command stays on
the hardline floor."""

import logging
import os
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


def _yaml(approvals: dict | None) -> str:
    if approvals is None:
        return "{}\n"
    return "approvals:\n" + "".join(f"  {k}: {str(v).lower()}\n" for k, v in approvals.items())


@pytest.fixture
def root_config(tmp_path, monkeypatch):
    """A stand-in for /etc/hermes/config.yaml: a private dir + file, with the 'root' uid check
    pointed at the test user (the real check requires uid 0)."""
    d = tmp_path / "etc-hermes"
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)
    path = d / "config.yaml"
    monkeypatch.setattr(approval_context, "_POWER_MANAGED_CONFIG", str(path))
    monkeypatch.setattr(approval_context, "_POWER_MANAGED_OWNER_UID", os.getuid())
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)

    def write(approvals: dict | None, mode: int = 0o644):
        path.write_text(_yaml(approvals), encoding="utf-8")
        os.chmod(path, mode)
        return path
    return write


def test_user_config_power_keys_are_ignored(monkeypatch, root_config):
    root_config(None)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()
    assert approval_context._get_power_no_approval_categories() == frozenset()


def test_ignored_user_keys_are_logged_once(monkeypatch, root_config, caplog):
    root_config(None)
    monkeypatch.setattr(approval_context, "_warned_ignored_power_keys", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"allow_reboot": True})
    with caplog.at_level(logging.WARNING, logger="tools.approval"):
        approval_context._get_hardline_opt_ins()
        approval_context._get_hardline_opt_ins()
    hits = [r for r in caplog.records if "is ignored" in r.getMessage()]
    assert len(hits) == 1 and "approvals.allow_reboot" in hits[0].getMessage()


def test_root_config_enables_the_categories(monkeypatch, root_config):
    root_config({"allow_reboot": True, "reboot_without_approval": True})
    monkeypatch.setattr(approval_context, "_get_approval_config", dict)
    assert approval_context._get_hardline_opt_ins() == frozenset({"reboot"})
    assert approval_context._get_power_no_approval_categories() == frozenset({"reboot"})


def test_root_false_wins_over_user_true(monkeypatch, root_config):
    root_config({"allow_reboot": False})
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_missing_root_config_fails_closed(monkeypatch, root_config):
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: dict(_ALL_ON))
    assert approval_context._get_hardline_opt_ins() == frozenset()


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666])
def test_group_or_other_writable_file_is_ignored(monkeypatch, root_config, mode):
    root_config(dict(_ALL_ON), mode=mode)
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_writable_directory_is_ignored(monkeypatch, root_config):
    path = root_config(dict(_ALL_ON))
    os.chmod(path.parent, 0o777)
    try:
        assert approval_context._get_hardline_opt_ins() == frozenset()
    finally:
        os.chmod(path.parent, 0o755)


def test_symlinked_file_is_ignored(tmp_path, monkeypatch, root_config):
    path = root_config(None)
    target = tmp_path / "agent-written.yaml"
    target.write_text(_yaml(_ALL_ON), encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_wrong_owner_is_ignored(monkeypatch, root_config):
    root_config(dict(_ALL_ON))
    monkeypatch.setattr(approval_context, "_POWER_MANAGED_OWNER_UID", os.getuid() + 1)
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_hermes_managed_dir_is_not_a_source(tmp_path, monkeypatch, root_config):
    """A ~/.hermes/.env line can set HERMES_MANAGED_DIR (loaded into os.environ at start and on
    reload). Even when the generic managed layer follows it, the power keys must not."""
    root_config(None)
    agent_dir = tmp_path / "agent-managed"
    agent_dir.mkdir()
    (agent_dir / "config.yaml").write_text(_yaml(_ALL_ON), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(agent_dir))
    managed_scope.invalidate_managed_cache()
    try:
        assert managed_scope.load_managed_config().get("approvals", {}).get("allow_reboot") is True
        assert approval_context._get_hardline_opt_ins() == frozenset()
        assert approval_context._get_power_no_approval_categories() == frozenset()
    finally:
        managed_scope.invalidate_managed_cache()


def test_read_error_fails_closed(monkeypatch, root_config):
    def boom():
        raise OSError("unreadable")
    monkeypatch.setattr(approval_context, "_root_owned_power_approvals", boom)
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_malformed_root_config_fails_closed(monkeypatch, root_config):
    path = root_config(None)
    path.write_text("approvals: [allow_reboot\n", encoding="utf-8")
    assert approval_context._get_hardline_opt_ins() == frozenset()


def test_agent_write_to_user_config_does_not_lift_the_floor(tmp_path, monkeypatch, root_config):
    """End to end: the guard chain reads a real user config.yaml with all four keys true (as an
    injected agent could write it by any route) and still hard-blocks reboot with no prompt."""
    from hermes_cli import config as hconfig
    root_config(None)
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
    quadratically here: ~13 s hardline + ~12 s dangerous per command (now well under 0.5 s)."""
    command = filler * (_MAX_SEPARATOR_FREE_COMMAND_CHARS - 2) + "q"
    assert len(command) == _MAX_SEPARATOR_FREE_COMMAND_CHARS - 1
    assert not _command_parser_limit_exceeded(command)
    start = time.monotonic()
    detect_hardline_command(command)
    detect_dangerous_command(command)
    detect_power_state_command(command)
    assert time.monotonic() - start < 3.0
