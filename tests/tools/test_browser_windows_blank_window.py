"""Regression tests for Windows blank-window guards during local browser automation (#64867)."""

from __future__ import annotations

import pytest

from tools import browser_tool as bt


def _reset_headed_cache():
    bt._cached_headed_mode = None
    bt._headed_mode_resolved = False


@pytest.fixture(autouse=True)
def _clear_headed_env(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_HEADED", raising=False)
    monkeypatch.delenv("AGENT_BROWSER_ARGS", raising=False)
    _reset_headed_cache()
    yield
    _reset_headed_cache()


def test_append_agent_browser_arg_merges_and_dedupes():
    env: dict = {"AGENT_BROWSER_ARGS": "--no-sandbox,--disable-dev-shm-usage"}
    bt._append_agent_browser_arg(env, "--window-position=-2400,-2400")
    assert env["AGENT_BROWSER_ARGS"] == (
        "--no-sandbox,--disable-dev-shm-usage,--window-position=-2400,-2400"
    )
    bt._append_agent_browser_arg(env, "--window-position=-100,-100")
    # Same flag key already present — do not duplicate.
    assert env["AGENT_BROWSER_ARGS"] == (
        "--no-sandbox,--disable-dev-shm-usage,--window-position=-2400,-2400"
    )


def test_local_visibility_guards_force_headed_false(monkeypatch):
    monkeypatch.setattr(bt.os, "name", "posix")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {},
    )
    env: dict = {}
    flags = bt._apply_local_browser_visibility_guards(env, local_chromium=True)
    assert flags == ["--headed", "false"]
    assert env["AGENT_BROWSER_HEADED"] == "false"
    assert "AGENT_BROWSER_ARGS" not in env


def test_local_visibility_guards_park_window_on_windows(monkeypatch):
    monkeypatch.setattr(bt.os, "name", "nt")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {},
    )
    env: dict = {}
    flags = bt._apply_local_browser_visibility_guards(env, local_chromium=True)
    assert flags == ["--headed", "false"]
    assert env["AGENT_BROWSER_HEADED"] == "false"
    assert "--window-position=-2400,-2400" in env["AGENT_BROWSER_ARGS"]


def test_local_visibility_guards_respect_explicit_headed(monkeypatch):
    monkeypatch.setattr(bt.os, "name", "nt")
    monkeypatch.setenv("AGENT_BROWSER_HEADED", "1")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {},
    )
    env = {"AGENT_BROWSER_HEADED": "1"}
    flags = bt._apply_local_browser_visibility_guards(env, local_chromium=True)
    assert flags == []
    assert env["AGENT_BROWSER_HEADED"] == "1"
    assert "AGENT_BROWSER_ARGS" not in env


def test_local_visibility_guards_respect_config_headed(monkeypatch):
    """browser.headed: true must not get a forced --headed false (#64867)."""
    monkeypatch.setattr(bt.os, "name", "nt")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"headed": True}},
    )
    env: dict = {}
    flags = bt._apply_local_browser_visibility_guards(env, local_chromium=True)
    assert flags == []
    assert env == {}


def test_local_visibility_guards_skip_cdp_backends(monkeypatch):
    monkeypatch.setattr(bt.os, "name", "nt")
    env: dict = {}
    flags = bt._apply_local_browser_visibility_guards(env, local_chromium=False)
    assert flags == []
    assert env == {}


def test_run_browser_command_inserts_headed_false_for_local(monkeypatch, tmp_path):
    """End-to-end: local Popen argv must include --headed false before --json."""
    captured: dict = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = list(args)
            captured["env"] = dict(kwargs.get("env") or {})
            self.returncode = 0

        def wait(self, timeout=None):
            # Caller closes the stdout fd before wait(); write JSON into the
            # temp file so the success path parses cleanly.
            for path in tmp_path.rglob("_stdout_open"):
                path.write_text('{"success": true}\n', encoding="utf-8")
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(bt.os, "name", "nt")
    monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
    monkeypatch.setattr(bt, "_get_session_info", lambda task_id: {"session_name": "h_test"})
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_write_owner_pid", lambda *a, **k: None)
    monkeypatch.setattr(bt, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
    monkeypatch.setattr(bt, "_merge_browser_path", lambda path: path or "")
    monkeypatch.setattr(bt, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(bt.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")

    result = bt._run_browser_command("task", "open", ["https://example.com"], timeout=5)
    assert result.get("success") is True
    args = captured["args"]
    assert "--json" in args
    headed_at = args.index("--headed")
    assert args[headed_at : headed_at + 2] == ["--headed", "false"]
    assert headed_at < args.index("--json")
    assert captured["env"].get("AGENT_BROWSER_HEADED") == "false"
    assert "--window-position=-2400,-2400" in captured["env"].get("AGENT_BROWSER_ARGS", "")


def test_run_browser_command_skips_headed_false_when_config_headed(
    monkeypatch, tmp_path
):
    """config browser.headed:true keeps --headed and never forces --headed false."""
    captured: dict = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = list(args)
            captured["env"] = dict(kwargs.get("env") or {})
            self.returncode = 0

        def wait(self, timeout=None):
            for path in tmp_path.rglob("_stdout_open"):
                path.write_text('{"success": true}\n', encoding="utf-8")
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(bt.os, "name", "nt")
    monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
    monkeypatch.setattr(bt, "_get_session_info", lambda task_id: {"session_name": "h_test"})
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_write_owner_pid", lambda *a, **k: None)
    monkeypatch.setattr(bt, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
    monkeypatch.setattr(bt, "_merge_browser_path", lambda path: path or "")
    monkeypatch.setattr(bt, "_is_headed_mode", lambda: True)
    monkeypatch.setattr(bt.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")

    result = bt._run_browser_command("task", "open", ["https://example.com"], timeout=5)
    assert result.get("success") is True
    args = captured["args"]
    assert "--headed" in args
    # Presence of the bare --headed opt-in, without a following false.
    headed_at = args.index("--headed")
    assert headed_at + 1 == len(args) or args[headed_at + 1] != "false"
    assert ["--headed", "false"] != args[headed_at : headed_at + 2]
    assert captured["env"].get("AGENT_BROWSER_HEADED") != "false"
    assert "--window-position=-2400,-2400" not in captured["env"].get(
        "AGENT_BROWSER_ARGS", ""
    )