import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.active_sessions import active_session_registry_snapshot
from tui_gateway import server


def test_session_create_rejects_at_active_session_limit(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")
    token = set_hermes_home_override(home)

    def _clear_server_sessions():
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()

    try:
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None
        _clear_server_sessions()
        monkeypatch.setattr(server, "_start_agent_build", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))

        first = server._methods["session.create"]("r1", {"cols": 80})
        assert "result" in first
        sid = first["result"]["session_id"]

        second = server._methods["session.create"]("r2", {"cols": 80})
        assert second["error"]["message"] == (
            "Hermes is at the active session limit (1/1). "
            "Try again when another session finishes."
        )
        assert list(server._sessions) == [sid]

        closed = server._methods["session.close"]("r3", {"session_id": sid})
        assert closed["result"]["closed"] is True
        assert active_session_registry_snapshot() == []

        third = server._methods["session.create"]("r4", {"cols": 80})
        assert "result" in third
    finally:
        _clear_server_sessions()
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None
        reset_hermes_home_override(token)


def test_session_context_uses_session_cwd(monkeypatch, tmp_path):
    """Desktop/TUI sessions must pin the agent cwd per session.

    The gateway process itself is often launched from apps/desktop in dev, so
    falling back to os.getcwd() makes agents answer from the desktop app folder
    even when the sidebar/session cwd is a real project.
    """
    from agent.runtime_cwd import resolve_agent_cwd

    sid = "cwd-sid"
    session_key = "cwd-key"
    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    server._sessions[sid] = {"session_key": session_key, "cwd": str(project)}
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context(session_key)
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)
        server._sessions.pop(sid, None)


def test_handoff_fail_marks_only_inflight_rows(monkeypatch):
    class DbContext:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self.db

        def __exit__(self, *_args):
            return False

    class FakeDb:
        def __init__(self, state):
            self.state = state
            self.failed_with = None

        def get_handoff_state(self, _key):
            return {"state": self.state, "platform": "telegram", "error": None}

        def fail_handoff(self, _key, error):
            self.failed_with = error
            self.state = "failed"

    sid = "rt-handoff"
    server._sessions[sid] = {"session_key": "stored-handoff"}
    try:
        pending = FakeDb("pending")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(pending))
        result = server._methods["handoff.fail"]("r1", {"session_id": sid, "error": "timed out"})
        assert result["result"] == {"failed": True, "state": "failed"}
        assert pending.failed_with == "timed out"

        completed = FakeDb("completed")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(completed))
        result = server._methods["handoff.fail"]("r2", {"session_id": sid, "error": "late timeout"})
        assert result["result"] == {"failed": False, "state": "completed"}
        assert completed.failed_with is None
    finally:
        server._sessions.pop(sid, None)


def test_session_context_explicit_cwd_for_ephemeral_task(monkeypatch, tmp_path):
    """Background/preview tasks use ephemeral ids absent from `_sessions`, so the
    parent workspace is passed explicitly; it must pin instead of clearing back
    to the gateway launch dir."""
    from agent.runtime_cwd import resolve_agent_cwd

    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context("bg_deadbe", cwd=str(project))
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)


def _write_profile_cfg(home: Path, cwd: str | None) -> Path:
    import yaml

    home.mkdir(parents=True, exist_ok=True)
    cfg = {"terminal": {"cwd": cwd}} if cwd is not None else {}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home


def test_profile_configured_cwd_reads_target_profile(tmp_path):
    """A profile's own terminal.cwd is read from its config.yaml."""
    project = tmp_path / "proj"
    project.mkdir()
    home = _write_profile_cfg(tmp_path / "home", str(project))
    assert server._profile_configured_cwd(home) == str(project)


def test_profile_configured_cwd_skips_placeholders_and_missing(tmp_path):
    """Placeholder values, missing config, and bad paths fall through to None."""
    assert server._profile_configured_cwd(None) is None
    assert server._profile_configured_cwd(tmp_path / "nope") is None
    for placeholder in (".", "auto", "cwd", ""):
        home = _write_profile_cfg(tmp_path / placeholder.strip("."), placeholder)
        assert server._profile_configured_cwd(home) is None
    home = _write_profile_cfg(tmp_path / "ghost", str(tmp_path / "does-not-exist"))
    assert server._profile_configured_cwd(home) is None


def test_completion_cwd_prefers_profile_over_stale_env(monkeypatch, tmp_path):
    """Issue #40334: a new session bound to another profile must use THAT
    profile's terminal.cwd, not the launch profile's stale TERMINAL_CWD."""
    profile_b = tmp_path / "ef-design"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-b", str(profile_b))
    stale = tmp_path / "mahjong"
    stale.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "ef-design"}) == str(profile_b)
    # No profile → unchanged fallback to the launch env var.
    assert server._completion_cwd({}) == str(stale)


def test_completion_cwd_explicit_cwd_wins_over_profile(monkeypatch, tmp_path):
    """An explicit client-provided cwd still beats the profile config."""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    profile_b = tmp_path / "configured"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-c", str(profile_b))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    result = server._completion_cwd({"cwd": str(explicit), "profile": "ef-design"})
    assert result == str(explicit)


def test_terminal_task_cwd_local_backend_uses_session_cwd(monkeypatch, tmp_path):
    """A local terminal backend must keep host-validated session cwd behaviour."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    assert server._terminal_task_cwd({"cwd": str(project)}) == str(project)


def test_terminal_task_cwd_ssh_uses_remote_path_unvalidated(monkeypatch):
    """SSH (non-local) backend: the configured remote cwd is used verbatim even
    though it does not exist on the local host. This is the jonbohz fix — host
    `isdir()` validation would otherwise discard the remote path and fall back
    to os.getcwd(), running commands against the wrong machine."""
    remote = "/home/jonboh/workspace/proj"  # does not exist on this host
    assert not os.path.isdir(remote)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", remote)

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_falls_back_to_config(monkeypatch):
    """When TERMINAL_CWD is unset, the SSH path reads terminal.cwd from config."""
    remote = "/home/jonboh/workspace/from-config"
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": remote}})

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_sentinel_cwd_falls_back_to_session(monkeypatch):
    """Sentinel/auto cwd values are not real remote paths, so the SSH branch
    must defer to the session cwd rather than registering a meaningless dir."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "auto")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": "."}})

    assert server._terminal_task_cwd({"cwd": "/host/session/dir"}) == "/host/session/dir"


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0.0001)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_write_json_serializes_concurrent_writes(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)

    threads = [
        threading.Thread(target=server.write_json, args=({"seq": i, "text": "x" * 24},))
        for i in range(8)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    lines = "".join(out.parts).splitlines()

    assert len(lines) == 8
    assert {json.loads(line)["seq"] for line in lines} == set(range(8))


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server, "_real_stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_write_json_drops_detached_ws_frames(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)
    server._sessions["detached-sid"] = {"transport": server._detached_ws_transport}
    try:
        assert server.write_json({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"session_id": "detached-sid", "type": "message.delta"},
        }) is False
        assert out.parts == []
    finally:
        server._sessions.pop("detached-sid", None)


def test_tui_verbose_tool_details_fail_closed_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    setattr(redact_module, "redact_sensitive_text", fail_redaction)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    assert server._redact_tui_verbose_text("api_key=secret") == ""
    assert server._tool_args_text({"api_key": "secret"}) == ""
    assert server._tool_result_text("token=secret") == ""


def test_tui_verbose_tool_details_are_capped_before_emit(monkeypatch):
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_CHARS", 12)
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_LINES", 2)

    capped = server._cap_tui_verbose_text("one\ntwo\nthree\nfour")

    assert capped.startswith("[showing verbose tail; omitted ")
    assert capped.endswith("three\nfour")
    assert "one" not in capped


def test_tui_verbose_default_cap_stays_small(monkeypatch):
    # Regression guard for #34095: the verbose tool text shipped to the TUI is
    # rendered into a persisted, expanded-by-default trail block for the whole
    # session. Raising this cap back toward the old 16KB re-introduces the Ink
    # render-tree blowup that silently OOM-killed the TUI. Keep it small.
    assert server._TUI_VERBOSE_TEXT_MAX_CHARS <= 2_000

    huge = "x" * 40_000
    capped = server._cap_tui_verbose_text(huge)

    assert len(capped) < 2_000
    assert capped.startswith("[showing verbose tail; omitted ")


def test_tui_verbose_tool_events_omit_details_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    setattr(redact_module, "redact_sensitive_text", fail_redaction)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "redaction-test",
        {"tool_progress_mode": "verbose", "tool_started_at": {}},
    )

    server._on_tool_start("redaction-test", "tool-1", "terminal", {"command": "pwd"})
    server._on_tool_complete("redaction-test", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert events[0][0] == "tool.start"
    assert events[1][0] == "tool.complete"
    assert "args_text" not in events[0][2]
    assert "result_text" not in events[1][2]


def test_dispatch_rejects_non_object_request():
    resp = server.dispatch([])

    assert resp == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32600, "message": "invalid request: expected an object"},
    }


def test_dispatch_rejects_non_object_params():
    resp = server.dispatch({"id": "1", "method": "session.create", "params": []})

    assert resp == {
        "jsonrpc": "2.0",
        "id": "1",
        "error": {"code": -32602, "message": "invalid params: expected an object"},
    }


def test_voice_toggle_returns_configured_record_key(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+o"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    # ``voice.toggle`` action=on mutates ``os.environ["HERMES_VOICE"]``
    # directly (CLI parity, runtime-only flag). Take monkeypatch
    # ownership of the var so the change is reverted at teardown and
    # later tests don't inherit a stale ON state (Copilot round-5
    # review on #19835).
    monkeypatch.setenv("HERMES_VOICE", "0")

    on_resp = server.dispatch(
        {"id": "voice-on", "method": "voice.toggle", "params": {"action": "on"}}
    )
    status_resp = server.dispatch(
        {"id": "voice-status", "method": "voice.toggle", "params": {"action": "status"}}
    )

    assert on_resp["result"]["record_key"] == "ctrl+o"
    assert status_resp["result"]["record_key"] == "ctrl+o"


def test_voice_toggle_handles_non_dict_voice_cfg(monkeypatch):
    """Round-3 Copilot review regression on #19835.

    ``_load_cfg()`` is raw ``yaml.safe_load()`` output — a hand-edited
    ``voice: true`` / ``voice: cmd+b`` / ``voice: null`` leaves ``voice``
    as a bool/str/None, not a dict. Previously ``.get("record_key")``
    on a non-dict broke every ``voice.toggle`` branch. Now it falls
    back to the documented default.
    """
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"]):
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        status_resp = server.dispatch(
            {
                "id": "voice-status",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for voice={bad!r}"

    # Round-4 follow-up: the YAML root itself may be a non-dict. A
    # hand-edit that collapses config.yaml to a scalar / list would
    # otherwise crash ``.get("voice")`` before the inner isinstance
    # guard gets a chance to run.
    for bad_root in (True, None, [], "ctrl+b", 42):
        monkeypatch.setattr(server, "_load_cfg", lambda r=bad_root: r)

        status_resp = server.dispatch(
            {
                "id": "voice-status-root",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for root={bad_root!r}"


def test_voice_record_start_handles_non_dict_voice_cfg(monkeypatch):
    """Round-7 Copilot review regression on #19835.

    The ``voice.record`` start path previously read
    ``_load_cfg().get("voice", {}).get(...)`` without any shape checks.
    When ``voice`` is a non-dict (bool/scalar/list) ``get`` raises
    AttributeError and the handler returns 5025 instead of falling
    back to the VAD defaults. Now it uses ``_voice_cfg_dict()`` and
    non-numeric silence values are coerced to the documented defaults.
    """
    captured: dict = {}

    def fake_start_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda: None
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"], {"silence_threshold": "loud"}):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        resp = server.dispatch(
            {
                "id": "voice-record",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert (
            "result" in resp
        ), f"voice.record raised for voice={bad!r}: {resp.get('error')}"
        assert resp["result"]["status"] == "recording"
        assert captured["silence_threshold"] == 200
        assert captured["silence_duration"] == 3.0
        assert captured["auto_restart"] is False

    # Round-12 Copilot review regression on #19835: ``bool`` is a subclass
    # of ``int``, so the naive ``isinstance(threshold, (int, float))``
    # guard would forward ``silence_threshold: true`` as ``1`` instead
    # of falling back to the documented 200 default.
    for bad_bool_cfg in (
        {"silence_threshold": True, "silence_duration": False},
        {"silence_threshold": False},
        {"silence_duration": True},
    ):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda c=bad_bool_cfg: {"voice": c})

        resp = server.dispatch(
            {
                "id": "voice-record-bool",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert "result" in resp, f"voice.record raised for bool cfg={bad_bool_cfg!r}"
        assert (
            captured["silence_threshold"] == 200
        ), f"bool silence_threshold leaked through for {bad_bool_cfg!r}"
        assert (
            captured["silence_duration"] == 3.0
        ), f"bool silence_duration leaked through for {bad_bool_cfg!r}"
        assert captured["auto_restart"] is False


def test_voice_record_stop_forces_transcription(monkeypatch):
    captured: dict = {}

    def fake_stop_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: None,
            stop_continuous=fake_stop_continuous,
        ),
    )

    resp = server.dispatch(
        {
            "id": "voice-record-stop",
            "method": "voice.record",
            "params": {"action": "stop"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert captured["force_transcribe"] is True


def test_voice_record_stop_updates_event_session_id(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: True,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(server, "_voice_event_sid", "old-session")

    resp = server.dispatch(
        {
            "id": "voice-record-stop-session",
            "method": "voice.record",
            "params": {"action": "stop", "session_id": "new-session"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert server._voice_event_sid == "new-session"


def test_voice_record_start_reports_busy_when_stop_is_in_progress(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: False,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})

    resp = server.dispatch(
        {
            "id": "voice-record-busy",
            "method": "voice.record",
            "params": {"action": "start"},
        }
    )

    assert resp["result"]["status"] == "busy"


def test_voice_toggle_tts_branch_also_carries_record_key(monkeypatch):
    """Round-2 Copilot review regression on #19835.

    The ``tts`` branch used to omit ``record_key`` from its response, so a
    TUI client would parse ``r.record_key ?? 'ctrl+b'`` and reset a
    custom binding to the default on every TTS toggle. Every branch of
    ``voice.toggle`` now carries the configured key so frontend state
    stays authoritative.
    """
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+space"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.delenv("HERMES_VOICE_TTS", raising=False)

    tts_resp = server.dispatch(
        {"id": "voice-tts", "method": "voice.toggle", "params": {"action": "tts"}}
    )

    assert tts_resp["result"]["record_key"] == "ctrl+space"
    assert tts_resp["result"]["tts"] is True


def test_load_enabled_toolsets_prefers_tui_env(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, terminal, ,memory")

    assert server._load_enabled_toolsets() == ["web", "terminal", "memory"]


def test_load_enabled_toolsets_filters_invalid_tui_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    assert server._load_enabled_toolsets() == ["web"]
    assert "nope" in capsys.readouterr().err


def test_load_enabled_toolsets_accepts_plugin_env_after_discovery(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "plugin_demo")

    import toolsets

    discovered = {"ready": False}
    original_validate = toolsets.validate_toolset

    def fake_validate(name):
        return name == "plugin_demo" and discovered["ready"] or original_validate(name)

    monkeypatch.setattr(toolsets, "validate_toolset", fake_validate)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=lambda: discovered.update({"ready": True})
        ),
    )

    assert server._load_enabled_toolsets() == ["plugin_demo"]


def test_load_enabled_toolsets_rejects_disabled_mcp_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "mcp-off")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    # Sorted: ["kanban", "memory"]. `kanban` is auto-recovered by
    # _get_platform_tools because it's a non-configurable platform toolset
    # whose tools live in hermes-cli's universe (see toolsets.py).
    assert server._load_enabled_toolsets() == ["kanban", "memory"]
    err = capsys.readouterr().err
    assert "ignoring disabled MCP servers" in err
    assert "mcp-off" in err
    assert "using configured CLI toolsets" in err


def test_load_enabled_toolsets_falls_back_when_tui_env_invalid(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    assert server._load_enabled_toolsets() == ["kanban", "memory"]
    assert "using configured CLI toolsets" in capsys.readouterr().err


def test_load_enabled_toolsets_warns_when_config_fallback_fails(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() is None
    assert "could not be loaded" in capsys.readouterr().err


def test_load_enabled_toolsets_honors_builtin_env_if_config_fails(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web")

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() == ["web"]


def test_load_enabled_toolsets_all_env_means_all(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "all")

    assert server._load_enabled_toolsets() is None


def test_load_enabled_toolsets_all_env_warns_about_ignored_extra_entries(
    monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "all,nope")

    assert server._load_enabled_toolsets() is None
    assert "ignoring additional entries: nope" in capsys.readouterr().err


def test_load_enabled_toolsets_reports_disabled_mcp_separately(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web,mcp-off,nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )

    assert server._load_enabled_toolsets() == ["web"]
    err = capsys.readouterr().err
    assert "ignoring unknown HERMES_TUI_TOOLSETS entries: nope" in err
    assert "ignoring disabled MCP servers" in err
    assert "mcp-off" in err


def test_history_to_messages_preserves_tool_calls_for_resume_display():
    history = [
        {"role": "user", "content": "first prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search_files",
                        "arguments": json.dumps({"pattern": "resume"}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second prompt"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "first prompt"},
        {"context": "resume", "name": "search_files", "role": "tool"},
        {"role": "assistant", "text": "first answer"},
        {"role": "user", "text": "second prompt"},
    ]


def test_history_to_messages_renders_multimodal_content():
    # bb/gui preserves image URLs in the resume payload so the desktop
    # renderer's extractEmbeddedImages can pull them back out and display
    # the actual image instead of a placeholder. This also keeps the
    # resume payload in sync with the cached message.
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
        {"role": "assistant", "content": "saw it"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "look here\ndata:image/png;base64,abc"},
        {"role": "assistant", "text": "saw it"},
    ]


def test_session_resume_uses_parent_lineage_for_display(monkeypatch):
    captured = {}

    class FakeDB:
        def get_session(self, target):
            return {"id": target}

        def reopen_session(self, target):
            captured["reopened"] = target

        def get_messages_as_conversation(self, target, include_ancestors=False):
            captured.setdefault("history_calls", []).append((target, include_ancestors))
            return (
                [
                    {"role": "user", "content": "root prompt"},
                    {"role": "assistant", "content": "root answer"},
                ]
                if include_ancestors
                else [{"role": "user", "content": "tip prompt"}]
            )

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *args, **kwargs: types.SimpleNamespace(model="test"),
    )
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda agent, *a: {"model": "test", "tools": {}, "skills": {}},
    )
    monkeypatch.setattr(
        server, "_init_session", lambda sid, key, agent, history, cols=80: None
    )

    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "tip"}}
    )

    assert resp["result"]["messages"] == [
        {"role": "user", "text": "root prompt"},
        {"role": "assistant", "text": "root answer"},
    ]
    assert captured["history_calls"] == [("tip", False), ("tip", True)]


def test_session_resume_passes_stored_runtime_to_agent(monkeypatch):
    captured = {}

    class FakeDB:
        def get_session(self, target):
            return {
                "id": target,
                "model": "gpt-5.4",
                "billing_provider": "openai-codex",
                "model_config": '{"reasoning_config":{"enabled":true,"effort":"high"},"service_tier":"priority","base_url":"https://custom.example/v1","api_mode":"chat_completions"}',
            }

        def reopen_session(self, target):
            pass

        def get_messages_as_conversation(self, target, include_ancestors=False):
            return [{"role": "user", "content": "hello"}]

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model="gpt-5.4", provider="openai-codex")

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent, *a: {"model": agent.model, "provider": agent.provider})

    def fake_init_session(sid, key, agent, history, cols=80):
        server._sessions[sid] = {"agent": agent, "session_key": key}

    monkeypatch.setattr(server, "_init_session", fake_init_session)

    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "stored-session"}}
    )

    assert resp["result"]["info"] == {"model": "gpt-5.4", "provider": "openai-codex"}
    assert captured["model_override"] == {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
    }
    assert captured["provider_override"] == "openai-codex"
    assert captured["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    assert captured["service_tier_override"] == "priority"
    runtime_sid = resp["result"]["session_id"]
    assert server._sessions[runtime_sid]["model_override"] == captured["model_override"]


def test_persist_live_session_runtime_preserves_resume_metadata(monkeypatch):
    updates = {}

    class FakeDB:
        def get_session(self, session_id):
            assert session_id == "stored-session"
            return {"model_config": '{"_branched_from":"root"}'}

        def update_session_meta(self, session_id, model_config_json, model=None):
            updates["meta"] = (session_id, json.loads(model_config_json), model)

    agent = types.SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        base_url="https://custom.example/v1",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier="priority",
        _session_db=FakeDB(),
    )

    server._persist_live_session_runtime({"agent": agent, "session_key": "stored-session"})

    assert "model" not in updates
    assert updates["meta"] == (
        "stored-session",
        {
            "_branched_from": "root",
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://custom.example/v1",
            "api_mode": "chat_completions",
            "reasoning_config": {"enabled": True, "effort": "high"},
            "service_tier": "priority",
        },
        "gpt-5.4",
    )


def test_status_callback_emits_kind_and_text():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("tui_gateway.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )


def test_resolve_model_uses_inference_model_env(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", " anthropic/claude-sonnet-4.6\n")

    assert server._resolve_model() == "anthropic/claude-sonnet-4.6"


def test_resolve_model_strips_config_model(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"default": " nous/hermes-test "}}
    )

    assert server._resolve_model() == "nous/hermes-test"


def _sync_test_session(**extra):
    session = {
        "agent": types.SimpleNamespace(model="old/model"),
        "session_key": "session-key",
    }
    session.update(extra)
    return session


def _patch_config_model(monkeypatch, model, provider=""):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    cfg_model = {"default": model}
    if provider:
        cfg_model["provider"] = provider
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": cfg_model})


def test_config_sync_switches_unpinned_session(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", "nous"))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append((sid, raw, kw)),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == [
        (
            "sid",
            "new/model --provider nous",
            {"confirm_expensive_model": True, "pin_session_override": False},
        )
    ]
    assert session["config_model_seen"] == ("new/model", "nous")


def test_config_sync_skips_session_pinned_by_model_command(monkeypatch):
    _patch_config_model(monkeypatch, "new/model")
    session = _sync_test_session(
        config_model_seen=("old/model", ""),
        model_override={"model": "pinned/model"},
    )
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("pinned session must not be switched"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_noop_when_config_unchanged(monkeypatch):
    _patch_config_model(monkeypatch, "old/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("unchanged config must not switch"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_failure_emits_error_once_per_edit(monkeypatch):
    _patch_config_model(monkeypatch, "broken/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))

    def boom(*a, **k):
        raise ValueError("no such model")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    emits = []
    monkeypatch.setattr(
        server, "_emit", lambda ev, sid, payload: emits.append((ev, payload))
    )

    server._sync_agent_model_with_config("sid", session)
    server._sync_agent_model_with_config("sid", session)

    assert len(emits) == 1
    assert emits[0][0] == "error"
    assert "broken/model" in emits[0][1]["message"]


def test_startup_runtime_uses_tui_provider_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "nous")
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)

    assert server._resolve_startup_runtime() == ("nous/hermes-test", "nous")


def test_startup_runtime_does_not_treat_inference_provider_as_explicit(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "nous")
    monkeypatch.setattr(
        "hermes_cli.models.detect_static_provider_for_model",
        lambda model, provider: None,
    )

    assert server._resolve_startup_runtime() == ("nous/hermes-test", None)


def test_startup_runtime_detects_provider_for_model_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})

    def fake_detect(model, current_provider):
        assert model == "sonnet"
        assert current_provider == "auto"
        return "anthropic", "anthropic/claude-sonnet-4.6"

    monkeypatch.setattr(
        "hermes_cli.models.detect_static_provider_for_model", fake_detect
    )

    assert server._resolve_startup_runtime() == (
        "anthropic/claude-sonnet-4.6",
        "anthropic",
    )


def test_load_fallback_model_merges_chain_providers_first(monkeypatch):
    # Parity with HermesCLI / gateway: fallback_providers stays first and keeps
    # its order, with any distinct legacy fallback_model entry merged in after
    # (deduped on provider/model/base_url).
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "fallback_model": {"provider": "legacy", "model": "legacy-model"},
            "fallback_providers": fallback_chain,
        },
    )

    assert server._load_fallback_model() == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "legacy", "model": "legacy-model"},
    ]


def test_make_agent_passes_configured_fallback_chain(monkeypatch):
    captured = {}
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
    ]

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"default": "gpt-5.5", "provider": "openai-codex"},
            "fallback_providers": fallback_chain,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "token",
            "api_mode": "codex_responses",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = server._make_agent("sid", "session-key")

    assert agent.model == "gpt-5.5"
    assert captured["fallback_model"] == fallback_chain
    assert captured["platform"] == "tui"


def test_background_agent_kwargs_preserves_full_fallback_chain(monkeypatch):
    chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        _fallback_chain=chain,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == chain


def test_background_agent_kwargs_preserves_empty_fallback_chain(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
    )
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "max_turns": 25,
            "fallback_providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.5"},
            ],
        },
    )
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == []


def test_startup_runtime_resolves_short_alias_without_network(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network lookup should not run")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert provider == "anthropic"
    assert model.startswith("claude-sonnet")


def test_startup_runtime_does_not_call_network_detector(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network detector called")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert model
    assert provider in {None, "anthropic"}


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def test_session_close_commits_memory_and_fires_finalize_hook(monkeypatch):
    calls = {"hooks": []}

    agent = types.SimpleNamespace(session_id="session-key")
    agent.commit_memory_session = lambda history: calls.setdefault("history", history)
    server._sessions["sid"] = _session(
        agent=agent, history=[{"role": "user", "content": "hello"}]
    )
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id: calls["hooks"].append((event, session_id)),
    )

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.close", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["closed"] is True
        assert calls["history"] == [{"role": "user", "content": "hello"}]
        assert ("on_session_finalize", "session-key") in calls["hooks"]
    finally:
        server._sessions.pop("sid", None)


def test_ws_orphan_reap_closes_worker_when_session_stays_detached(monkeypatch):
    """A detached WS session past its grace window has its slash_worker closed.

    Regression for #38591 fallout: every dashboard refresh spawned a fresh
    session + _SlashWorker but never reaped the previous one, leaking one
    python subprocess per refresh.
    """
    closed = {"worker": False}

    class _FakeWorker:
        def close(self):
            closed["worker"] = True

    server._sessions["orphan-sid"] = _session(
        transport=server._detached_ws_transport,
        slash_worker=_FakeWorker(),
        running=False,
    )
    # Run the reap body synchronously (no real timer/grace) to assert behaviour.
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    try:
        # Directly invoke the orphaned-check + teardown the timer would run.
        assert server._ws_session_is_orphaned(server._sessions["orphan-sid"]) is True
        session = server._sessions.pop("orphan-sid")
        server._teardown_session(session)
        assert closed["worker"] is True
    finally:
        server._sessions.pop("orphan-sid", None)


def test_finalize_session_closes_slash_worker(monkeypatch):
    """_finalize_session closes the slash_worker subprocess itself.

    Regression for #38095: the worker cleanup used to live only in the
    callers (_teardown_session / _shutdown_sessions), so any code path that
    finalized a session without going through them leaked the worker. Folding
    close() into the single _finalized-guarded chokepoint makes the cleanup
    defense-in-depth and idempotent.
    """
    closed = {"count": 0}

    class _FakeWorker:
        def close(self):
            closed["count"] += 1

    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    session = _session(slash_worker=_FakeWorker())

    server._finalize_session(session)
    assert closed["count"] == 1
    assert session.get("_finalized") is True

    # Idempotent: a second finalize (or a follow-up teardown) must not
    # re-close the worker — the _finalized guard short-circuits.
    server._finalize_session(session)
    server._teardown_session(session)
    assert closed["count"] == 1


def test_ws_orphan_reap_spares_reattached_session(monkeypatch):
    """A session that rebinds a live transport is NOT considered orphaned."""

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    # Reattached: transport is a live (non-stdio) transport.
    reattached = _session(transport=_LiveTransport(), running=False)
    assert server._ws_session_is_orphaned(reattached) is False

    # Mid-turn sessions are also spared even if detached.
    mid_turn = _session(transport=server._detached_ws_transport, running=True)
    assert server._ws_session_is_orphaned(mid_turn) is False

    # Already finalized sessions are spared (idempotency).
    done = _session(
        transport=server._detached_ws_transport,
        running=False,
        _finalized=True,
    )
    assert server._ws_session_is_orphaned(done) is False


def test_ws_orphan_reap_disabled_when_grace_zero(monkeypatch):
    """Grace=0 disables the reaper entirely (pre-fix park-forever behaviour)."""
    fired = {"timer": False}

    class _Timer:
        def __init__(self, *a, **k):
            fired["timer"] = True

        def start(self):
            pass

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    server._schedule_ws_orphan_reap("any-sid")
    assert fired["timer"] is False


def test_init_session_fires_reset_hook(monkeypatch):
    hooks = []

    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id: hooks.append((event, session_id)),
    )

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    sid = "sid"
    try:
        server._init_session(
            sid,
            "session-key",
            types.SimpleNamespace(model="x"),
            history=[],
            cols=80,
        )
        assert ("on_session_reset", "session-key") in hooks
    finally:
        server._sessions.pop(sid, None)


def test_session_title_queues_when_db_row_not_ready(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return None

        def get_session(self, _key):
            return None

        def set_session_title(self, _key, _title):
            return False

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "queued title"},
            }
        )

        assert set_resp["result"]["pending"] is True
        assert set_resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] == "queued title"

        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_notification_event_routing_by_session_key(monkeypatch):
    """Background-process events surface only in the session that owns them."""
    mine = _session(session_key="mine")
    other = _session(session_key="other")
    monkeypatch.setattr(server, "_sessions", {"a": mine, "b": other})

    # My own event → handle it.
    assert server._notification_event_belongs_elsewhere(mine, {"session_key": "mine"}) is False
    # Global/system event with no owner → handle it.
    assert server._notification_event_belongs_elsewhere(mine, {"session_key": ""}) is False
    assert server._notification_event_belongs_elsewhere(mine, {}) is False
    # Owned by another *live* session → defer to that session's poller.
    assert server._notification_event_belongs_elsewhere(mine, {"session_key": "other"}) is True
    # Owner is gone (not in _sessions) → handle as fallback so it isn't lost.
    assert server._notification_event_belongs_elsewhere(mine, {"session_key": "ghost"}) is False


def test_session_create_does_not_persist_empty_row(monkeypatch):
    """session.create must NOT eagerly write a DB row.

    Every TUI/desktop launch opens a session here just to paint the composer;
    eagerly creating a row left an empty "Untitled" session behind for every
    launch the user never typed into. The row is created lazily on first prompt.
    """
    created = []

    class _FakeDB:
        def create_session(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(
        server.threading,
        "Timer",
        lambda *a, **k: types.SimpleNamespace(daemon=False, start=lambda: None),
    )

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    try:
        assert resp["result"]["stored_session_id"]
        assert created == [], "session.create should not persist an empty DB row"
    finally:
        server._sessions.pop(sid, None)


def test_ensure_session_db_row_persists_explicit_cwd(monkeypatch, tmp_path):
    """An explicitly chosen workspace is persisted as the session cwd."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, cwd=None):
            created.append({"key": key, "source": source, "model": model, "cwd": cwd})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path), "explicit_cwd": True})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "cwd": str(tmp_path)}
    ]


def test_ensure_session_db_row_defaults_to_no_workspace(monkeypatch, tmp_path):
    """Without an explicit workspace, cwd is left null so the session groups
    under "No workspace" rather than the gateway's launch directory."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, cwd=None):
            created.append({"key": key, "source": source, "model": model, "cwd": cwd})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path)})

    assert created == [{"key": "k1", "source": "tui", "model": "test-model", "cwd": None}]


def test_session_title_clears_pending_after_persist(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "old"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, title):
            self.title = title
            return True

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "fresh"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_does_not_queue_noop_when_row_exists(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "same title"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, _title):
            # Simulate sqlite UPDATE rowcount==0 for no-op update.
            return False

    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "same title"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "same title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_falls_back_to_pending_when_db_read_throws(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            raise RuntimeError("db temporarily locked")

    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_persist_for_pending_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = ""

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_pending_even_when_db_has_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "auto title"

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_rejects_empty_title_with_specific_error_code(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "   "},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4021
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_maps_valueerror_to_user_error(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            return {"id": _key}

        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "dup"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4022
        assert "already in use" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_errors_when_row_lookup_fails_after_noop(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            raise RuntimeError("row lookup failed")

        def set_session_title(self, _key, _title):
            return False

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 5007
        assert "row lookup failed" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_create_drops_pending_title_on_valueerror(monkeypatch):
    """When set_session_title raises ValueError during post-message title flush,
    pending_title should be dropped (non-retryable). Updated for post-#18370
    lazy session creation where title is applied post-first-message.
    """

    class _Agent:
        session_id = "test-session"
        model = "x"
        provider = "openrouter"
        base_url = ""
        api_key = ""
        _cached_system_prompt = ""

        def run_conversation(self, prompt, **kw):
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _FakeDB:
        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    agent = _Agent()
    session = {
        "agent": agent,
        "session_key": "test-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "pending_title": "duplicate title",
    }

    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **kw: None
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
        )
        assert session["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_config_set_yolo_toggles_session_scope():
    from tools.approval import clear_session, is_session_yolo_enabled

    server._sessions["sid"] = _session()
    try:
        resp_on = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_on["result"]["value"] == "1"
        assert is_session_yolo_enabled("session-key") is True

        resp_off = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_off["result"]["value"] == "0"
        assert is_session_yolo_enabled("session-key") is False
    finally:
        clear_session("session-key")
        server._sessions.clear()


def test_config_set_yolo_global_scope_writes_approvals_mode(tmp_path, monkeypatch):
    """Shift+click the desktop zap -> scope="global" flips persistent approvals.mode."""
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp_on = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_on["result"]["value"] == "1"
    assert resp_on["result"]["scope"] == "global"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"

    resp_off = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_off["result"]["value"] == "0"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "manual"


def test_config_set_yolo_global_scope_honors_explicit_value(tmp_path, monkeypatch):
    """An explicit value pins global approvals.mode regardless of prior state."""
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"

    # Setting it on again is idempotent — stays off.
    resp_again = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp_again["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text())["approvals"]["mode"] == "off"


def test_config_set_fast_updates_live_agent_and_config(monkeypatch):
    writes = []
    emits = []
    agent = types.SimpleNamespace(
        model="openai/gpt-5.4",
        request_overrides={"foo": "bar", "speed": "slow"},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id: {"service_tier": "priority"},
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert agent.request_overrides == {
            "foo": "bar",
            "service_tier": "priority",
        }
        assert ("agent.service_tier", "fast") in writes
        assert ("session.info", "sid", {"model": "x"}) in emits

        resp_normal = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "normal"},
            }
        )
        assert resp_normal["result"]["value"] == "normal"
        assert agent.service_tier is None
        assert agent.request_overrides == {"foo": "bar"}
        assert ("agent.service_tier", "normal") in writes
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_status_is_non_mutating(monkeypatch):
    writes = []
    emits = []
    agent = types.SimpleNamespace(service_tier="priority")
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "status"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert writes == []
        assert emits == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_unsupported_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="unsupported-model",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id: None,
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert "not available" in resp["error"]["message"]
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_missing_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert "without a selected model" in resp["error"]["message"]
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_busy_get_and_set(monkeypatch):
    writes = []

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"display": {"busy_input_mode": "steer"}},
    )
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )
    assert get_resp["result"]["value"] == "steer"

    set_resp = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "busy", "value": "interrupt"},
        }
    )
    assert set_resp["result"]["value"] == "interrupt"
    assert ("display.busy_input_mode", "interrupt") in writes


def test_config_set_yolo_process_scope_treats_false_like_env_as_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "false")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo"},
        }
    )

    assert resp["result"]["value"] == "1"
    assert os.environ.get("HERMES_YOLO_MODE") == "1"


def test_config_get_statusbar_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "statusbar"}}
    )

    assert resp["result"]["value"] == "top"


def test_config_get_busy_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )

    assert resp["result"]["value"] == "interrupt"


def test_config_set_statusbar_survives_non_dict_display(tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": "broken"}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "statusbar", "value": "bottom"},
        }
    )

    assert resp["result"]["value"] == "bottom"
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["tui_statusbar"] == "bottom"


def test_config_set_details_mode_pins_all_sections(tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"tools": "expanded", "activity": "hidden"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode", "value": "collapsed"},
        }
    )

    assert resp["result"] == {"key": "details_mode", "value": "collapsed"}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["details_mode"] == "collapsed"
    assert saved["display"]["sections"] == {
        "thinking": "collapsed",
        "tools": "collapsed",
        "subagents": "collapsed",
        "activity": "collapsed",
    }


def test_config_set_section_writes_per_section_override(tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": "hidden"},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": "hidden"}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["sections"] == {"activity": "hidden"}


def test_config_set_section_clears_override_on_empty_value(tmp_path, monkeypatch):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"activity": "hidden", "tools": "expanded"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": ""},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": ""}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["sections"] == {"tools": "expanded"}


def test_config_set_section_rejects_unknown_section_or_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    bad_section = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.bogus", "value": "hidden"},
        }
    )
    assert bad_section["error"]["code"] == 4002

    bad_mode = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "details_mode.tools", "value": "maximised"},
        }
    )
    assert bad_mode["error"]["code"] == 4002


def test_config_mouse_uses_documented_key_with_legacy_fallback(monkeypatch):
    cfg = {"display": {"tui_mouse": False}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_legacy = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_legacy["result"]["value"] == "off"

    set_toggle = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"key": "mouse"}}
    )
    # /mouse (no arg) toggles between 'all' and 'off'. Starting from
    # tui_mouse: False (→ 'off'), the toggle flips to 'all'.
    assert set_toggle["result"] == {"key": "mouse", "value": "all"}
    assert writes == [("display.mouse_tracking", "all")]

    cfg["display"] = {"mouse_tracking": 0, "tui_mouse": True}
    get_canonical = server.handle_request(
        {"id": "3", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_canonical["result"]["value"] == "off"

    cfg["display"] = {"mouse_tracking": None, "tui_mouse": False}
    get_null = server.handle_request(
        {"id": "4", "method": "config.get", "params": {"key": "mouse"}}
    )
    # mouse_tracking present-but-None defers neither to tui_mouse nor to
    # the legacy off bucket: it falls through to the 'all' default.
    assert get_null["result"]["value"] == "all"


def test_config_mouse_accepts_preset_strings_and_aliases(monkeypatch):
    cfg = {"display": {"mouse_tracking": "all"}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    # Direct preset.
    set_wheel = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "mouse", "value": "wheel"},
        }
    )
    assert set_wheel["result"] == {"key": "mouse", "value": "wheel"}
    assert writes[-1] == ("display.mouse_tracking", "wheel")

    # Alias for buttons.
    set_click = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "mouse", "value": "click"},
        }
    )
    assert set_click["result"] == {"key": "mouse", "value": "buttons"}
    assert writes[-1] == ("display.mouse_tracking", "buttons")

    # Unknown value → 4002.
    bad = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"key": "mouse", "value": "rainbows"},
        }
    )
    assert bad["error"]["code"] == 4002


def test_enable_gateway_prompts_sets_gateway_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    server._enable_gateway_prompts()

    assert server.os.environ["HERMES_GATEWAY_SESSION"] == "1"
    assert server.os.environ["HERMES_EXEC_ASK"] == "1"
    assert server.os.environ["HERMES_INTERACTIVE"] == "1"


def test_setup_status_reports_provider_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)

    resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})

    assert resp["result"]["provider_configured"] is False


def test_setup_runtime_check_rejects_empty_runtime_key(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": "openrouter",
            "api_key": "",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["provider"] == "openrouter"


def test_setup_runtime_check_allows_no_key_custom_runtime(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": "custom",
            "api_key": "no-key-required",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is True
    assert resp["result"]["provider"] == "custom"


def test_setup_runtime_check_rejects_implicit_bedrock_when_unconfigured(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": "bedrock",
            "api_key": "aws-sdk",
            "source": "iam-role",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["provider"] == "bedrock"


def test_complete_slash_drops_removed_provider_alias():
    # `/provider` was folded into a single `/model` command, so autocomplete
    # must no longer offer the dead alias...
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/pro"}}
    )

    assert not any(item["text"] == "provider" for item in resp["result"]["items"])

    # ...while `/model` stays the canonical command.
    resp_model = server.handle_request(
        {"id": "2", "method": "complete.slash", "params": {"text": "/mod"}}
    )

    assert any(item["text"] == "model" for item in resp_model["result"]["items"])


def test_complete_slash_returns_plain_string_fields():
    # prompt_toolkit hands us FormattedText (a list subclass) for
    # display/display_meta; the TUI's CompletionItem contract is plain
    # strings, and shipping the raw list trips Ink's row layout into
    # 1-char truncation of the next column (/goal → /goa).
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/g"}}
    )

    items = resp["result"]["items"]
    goal = next((it for it in items if it["text"] == "goal"), None)
    assert goal is not None
    assert isinstance(goal["display"], str), goal["display"]
    assert isinstance(goal["meta"], str), goal["meta"]
    assert goal["display"] == "/goal"
    for item in items:
        assert isinstance(item["display"], str), item
        assert isinstance(item["meta"], str), item


def test_complete_slash_includes_tui_details_command():
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/det"}}
    )

    assert any(item["text"] == "/details" for item in resp["result"]["items"])


def test_complete_slash_includes_tui_mouse_command():
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/mou"}}
    )

    assert any(item["text"] == "/mouse" for item in resp["result"]["items"])


def test_complete_slash_details_args():
    resp_root = server.handle_request(
        {"id": "0", "method": "complete.slash", "params": {"text": "/details"}}
    )
    resp_section = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/details t"}}
    )
    resp_mode = server.handle_request(
        {
            "id": "2",
            "method": "complete.slash",
            "params": {"text": "/details thinking e"},
        }
    )

    assert resp_root["result"]["replace_from"] == len("/details")
    assert any(item["text"] == " thinking" for item in resp_root["result"]["items"])
    assert any(item["text"] == "thinking" for item in resp_section["result"]["items"])
    assert any(item["text"] == "expanded" for item in resp_mode["result"]["items"])


def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "low"},
        }
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}

    resp_show = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "show"},
        }
    )
    assert resp_show["result"]["value"] == "show"
    assert server._sessions["sid"]["show_reasoning"] is True
    assert server._load_cfg()["display"]["sections"]["thinking"] == "expanded"

    resp_hide = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "hide"},
        }
    )
    assert resp_hide["result"]["value"] == "hide"
    assert server._sessions["sid"]["show_reasoning"] is False
    assert server._load_cfg()["display"]["sections"]["thinking"] == "hidden"


def test_config_set_verbose_updates_session_mode_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(verbose_logging=False)
    server._sessions["sid"] = _session(agent=agent)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "verbose", "value": "cycle"},
        }
    )

    assert resp["result"]["value"] == "verbose"
    assert server._sessions["sid"]["tool_progress_mode"] == "verbose"
    assert agent.verbose_logging is True



def test_config_set_model_waits_for_lazy_agent_before_switch(monkeypatch):
    """A model switch against a lazy-created live session must apply to the
    real agent, not just process env, before the prompt is dispatched.
    """

    agent_ready = threading.Event()
    agent = types.SimpleNamespace(model="old/model", provider="old-provider")
    session = _session(agent=agent)
    session["agent"] = None
    session["agent_ready"] = agent_ready
    server._sessions["sid"] = session
    calls = []

    def fake_start(sid, target):
        calls.append(("start", sid))
        target["agent"] = agent
        agent_ready.set()

    def fake_apply(sid, target, raw, **kwargs):
        calls.append(("apply", sid, target.get("agent"), raw))
        if target.get("agent") is not agent:
            raise AssertionError("model switch ran before lazy agent was ready")
        return {"value": "new/model", "warning": ""}

    monkeypatch.setattr(server, "_start_agent_build", fake_start)
    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "model", "value": "new/model"},
            }
        )

        assert resp["result"]["value"] == "new/model"
        assert calls == [("start", "sid"), ("apply", "sid", agent, "new/model")]
    finally:
        server._sessions.pop("sid", None)

def test_config_set_model_uses_live_switch_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["args"] = (sid, session["session_key"], raw)
        return {"value": "new/model", "warning": "catalog unreachable"}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "model", "value": "new/model"},
        }
    )

    assert resp["result"]["value"] == "new/model"
    assert resp["result"]["warning"] == "catalog unreachable"
    assert seen["args"] == ("sid", "session-key", "new/model")


def test_config_set_model_requires_confirmation_for_expensive_model(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"
        switched = False

        def switch_model(self, **_kwargs):
            self.switched = True

    result = types.SimpleNamespace(
        success=True,
        new_model="openai/gpt-5.5-pro",
        target_provider="openrouter",
        api_key="sk-or",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
        model_info=types.SimpleNamespace(
            has_cost_data=lambda: True,
            cost_input=25.0,
            cost_output=125.0,
        ),
    )

    agent = _Agent()
    server._sessions["sid"] = _session(agent=agent)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
            },
        }
    )

    assert resp["result"]["confirm_required"] is True
    assert "did you mean to select openai/gpt-5.5?" in resp["result"]["confirm_message"]
    assert agent.switched is False

    confirmed = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
                "confirm_expensive_model": True,
            },
        }
    )

    assert confirmed["result"]["confirm_required"] is False
    assert confirmed["result"]["value"] == "openai/gpt-5.5-pro"
    assert agent.switched is True


def test_config_set_model_global_persists(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-old"

        def switch_model(self, **kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="anthropic/claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-new",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    saved = {}

    def _switch_model(**kwargs):
        seen.update(kwargs)
        return result

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch_model)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg))

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "anthropic/claude-sonnet-4.6 --global",
            },
        }
    )

    assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
    assert seen["is_global"] is True
    assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
    assert saved["model"]["provider"] == "anthropic"
    assert saved["model"]["base_url"] == "https://api.anthropic.com"


def test_config_set_model_does_not_leak_inference_provider_env(monkeypatch):
    """A /model switch must NOT mutate process-global env vars. The desktop /
    dashboard tui_gateway backend hosts every same-profile session in one
    process; writing HERMES_INFERENCE_PROVIDER on a switch leaked the new
    provider into every other live session's next agent rebuild. The switch
    must instead record a per-session override and leave shared env untouched.

    (Was test_config_set_model_syncs_inference_provider_env, which asserted the
    leaky env-sync contract that caused the cross-session contamination bug.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        # Shared process env is UNCHANGED (the contamination vector is gone).
        assert os.environ["HERMES_INFERENCE_PROVIDER"] == "openrouter"
        # The switch was recorded as a per-session override instead.
        assert session["model_override"]["provider"] == "anthropic"
        assert session["model_override"]["model"] == "claude-sonnet-4.6"
    finally:
        server._sessions.clear()


def test_config_set_model_records_per_session_override_not_env(monkeypatch):
    """Regression for #16857 via the per-session override (not env vars):
    /model must record the user's explicit provider on the session so a later
    /new (which rebuilds via _make_agent honoring model_override) honours that
    choice — WITHOUT writing process-global env vars that would leak into
    sibling sessions.

    (Was test_config_set_model_syncs_tui_provider_unconditionally.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="deepseek-v4-pro",
        target_provider="custom:xuanji",
        api_key="sk-xuanji",
        base_url="https://xuanji.example/v1",
        api_mode="chat_completions",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "deepseek-v4-pro --provider custom:xuanji",
                },
            }
        )

        # No process-global env mutation.
        assert "HERMES_TUI_PROVIDER" not in os.environ
        assert "HERMES_INFERENCE_PROVIDER" not in os.environ
        # The user's explicit provider + resolved endpoint live on the session,
        # carried into the next /new rebuild by _make_agent.
        override = session["model_override"]
        assert override["provider"] == "custom:xuanji"
        assert override["model"] == "deepseek-v4-pro"
        assert override["base_url"] == "https://xuanji.example/v1"
        assert override["api_key"] == "sk-xuanji"
        assert override["api_mode"] == "chat_completions"
    finally:
        server._sessions.clear()


def test_config_set_model_switches_agent_without_touching_env(monkeypatch):
    """A /model switch mutates the target session's agent in place and records
    a per-session override; it does NOT write HERMES_MODEL / HERMES_TUI_PROVIDER
    etc. into the shared process environment.

    (Was test_config_set_model_syncs_tui_provider_env.)
    """

    class Agent:
        model = "gpt-5.3-codex"
        provider = "openai-codex"
        base_url = ""
        api_key = ""

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]

    agent = Agent()
    session = _session(agent=agent)
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "openai-codex")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    def fake_switch_model(**kwargs):
        return types.SimpleNamespace(
            success=True,
            new_model="anthropic/claude-sonnet-4.6",
            target_provider="anthropic",
            api_key="key",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            warning_message="",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
        # Agent switched in place...
        assert agent.model == "anthropic/claude-sonnet-4.6"
        assert agent.provider == "anthropic"
        # ...override recorded on the session...
        assert session["model_override"]["model"] == "anthropic/claude-sonnet-4.6"
        assert session["model_override"]["provider"] == "anthropic"
        # ...and the shared process env was NOT touched.
        assert os.environ["HERMES_TUI_PROVIDER"] == "openai-codex"
        assert "HERMES_MODEL" not in os.environ
        assert "HERMES_INFERENCE_MODEL" not in os.environ
    finally:
        server._sessions.clear()


def test_config_set_personality_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "personality", "value": "bogus"},
        }
    )

    assert "error" in resp
    assert "Unknown personality" in resp["error"]["message"]


def test_config_set_personality_preserves_history_and_returns_info(monkeypatch):
    agent = types.SimpleNamespace(
        ephemeral_system_prompt=None, _cached_system_prompt="old"
    )
    session = _session(
        agent=agent,
        history=[{"role": "user", "text": "hi"}],
        history_version=4,
    )
    emits = []

    server._sessions["sid"] = session
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    monkeypatch.setattr(
        server, "_session_info", lambda agent, *a: {"model": getattr(agent, "model", "?")}
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(server, "_write_config_key", lambda path, value: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "personality", "value": "helpful"},
        }
    )

    assert resp["result"]["history_reset"] is False
    assert resp["result"]["info"] == {"model": "?"}
    # History is preserved with a pivot marker appended
    assert len(session["history"]) == 2
    assert session["history"][0] == {"role": "user", "text": "hi"}
    assert session["history"][1]["role"] == "user"
    assert "personality" in session["history"][1]["content"].lower()
    assert "You are helpful." in session["history"][1]["content"]
    assert session["history_version"] == 5
    # Agent's system prompt was updated in-place; cached prompt untouched
    assert agent.ephemeral_system_prompt == "You are helpful."
    assert agent._cached_system_prompt == "old"
    assert ("session.info", "sid", {"model": "?"}) in emits


def test_session_compress_uses_compress_helper(monkeypatch):
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server,
        "_compress_session_history",
        lambda session, focus_topic=None, **_kw: (2, {"total": 42}),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})

    with patch("tui_gateway.server._emit") as emit:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )

    assert resp["result"]["removed"] == 2
    assert resp["result"]["usage"]["total"] == 42
    emit.assert_any_call("session.info", "sid", {"model": "x"})
    # Final status.update clears the pinned "compressing" indicator so the
    # status bar can revert to the neutral state when compaction finishes.
    emit.assert_any_call("status.update", "sid", {"kind": "status", "text": "ready"})


def test_session_compress_syncs_session_key_after_rotation(monkeypatch):
    """When AIAgent._compress_context rotates session_id (compression split),
    the gateway session_key must follow so subsequent approval routing,
    DB title/history lookups, and slash worker resume target the new
    continuation session — mirrors HermesCLI._manual_compress's
    session_id sync (cli.py).
    """
    agent = types.SimpleNamespace(session_id="rotated-id")
    server._sessions["sid"] = _session(agent=agent)
    server._sessions["sid"]["session_key"] = "old-key"
    server._sessions["sid"]["pending_title"] = "stale title"

    monkeypatch.setattr(
        server,
        "_compress_session_history",
        lambda session, focus_topic=None, **_kw: (2, {"total": 42}),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    restart_calls = []
    monkeypatch.setattr(
        server, "_restart_slash_worker", lambda sid, s: restart_calls.append(s)
    )

    try:
        with patch("tui_gateway.server._emit"):
            server.handle_request(
                {
                    "id": "1",
                    "method": "session.compress",
                    "params": {"session_id": "sid"},
                }
            )

        assert server._sessions["sid"]["session_key"] == "rotated-id"
        assert server._sessions["sid"]["pending_title"] is None
        assert len(restart_calls) == 1
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_sets_approval_session_key(monkeypatch):
    from tools.approval import get_current_session_key

    captured = {}

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            captured["session_key"] = get_current_session_key(default="")
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "ping"},
        }
    )

    assert resp["result"]["status"] == "streaming"
    assert captured["session_key"] == "session-key"


def test_prompt_submit_expands_context_refs(monkeypatch):
    captured = {}

    class _Agent:
        model = "test/model"
        base_url = ""
        api_key = ""

        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            captured["prompt"] = prompt
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    fake_ctx = types.ModuleType("agent.context_references")
    fake_ctx.preprocess_context_references = (
        lambda message, **kwargs: types.SimpleNamespace(
            blocked=False,
            message="expanded prompt",
            warnings=[],
            references=[],
            injected_tokens=0,
        )
    )
    fake_meta = types.ModuleType("agent.model_metadata")
    fake_meta.get_model_context_length = lambda *args, **kwargs: 100000

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setitem(sys.modules, "agent.context_references", fake_ctx)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", fake_meta)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "@diff"},
        }
    )

    assert captured["prompt"] == "expanded prompt"


def test_image_attach_appends_local_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._detect_file_drop = lambda raw: {
        "path": Path("/tmp/cat.png"),
        "is_image": True,
        "remainder": "",
    }
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: Path("/tmp/cat.png")

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach",
            "params": {"session_id": "sid", "path": "/tmp/cat.png"},
        }
    )

    assert resp["result"]["attached"] is True
    assert resp["result"]["name"] == "cat.png"
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_image_attach_accepts_unquoted_screenshot_path_with_spaces(monkeypatch):
    screenshot = Path("/tmp/Screenshot 2026-04-21 at 1.04.43 PM.png")
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._detect_file_drop = lambda raw: {
        "path": screenshot,
        "is_image": True,
        "remainder": "",
    }
    fake_cli._split_path_input = lambda raw: (
        "/tmp/Screenshot",
        "2026-04-21 at 1.04.43 PM.png",
    )
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach",
            "params": {"session_id": "sid", "path": str(screenshot)},
        }
    )

    assert resp["result"]["attached"] is True
    assert resp["result"]["path"] == str(screenshot)
    assert resp["result"]["remainder"] == ""
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_file_attach_uploads_remote_file_into_session_workspace(monkeypatch, tmp_path):
    """Remote case: client path doesn't exist on gateway → decode data_url bytes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "/Users/alice/Downloads/report.txt",
                    "name": "report.txt",
                    "data_url": "data:text/plain;base64,aGVsbG8gd29ybGQ=",
                },
            }
        )

        stored = workspace / ".hermes" / "desktop-attachments" / "report.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is True
        assert resp["result"]["path"] == str(stored)
        assert resp["result"]["ref_text"] == "@file:.hermes/desktop-attachments/report.txt"
        assert stored.read_text(encoding="utf-8") == "hello world"
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_copies_gateway_visible_file_outside_workspace(monkeypatch, tmp_path):
    """Local case: gateway can see the file but it's outside the workspace → copy in."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("outside workspace", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: source

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": str(source)},
            }
        )

        stored = workspace / ".hermes" / "desktop-attachments" / "outside.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is True
        assert resp["result"]["ref_text"] == "@file:.hermes/desktop-attachments/outside.txt"
        assert stored.read_text(encoding="utf-8") == "outside workspace"
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_uses_in_workspace_file_without_copying(monkeypatch, tmp_path):
    """Local case: file already inside the workspace → ref it directly, no copy."""
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    source = workspace / "data" / "exam.csv"
    source.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: source

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": str(source)},
            }
        )

        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is False
        assert resp["result"]["ref_text"] == "@file:data/exam.csv"
        # No copy: nothing staged under desktop-attachments.
        assert not (workspace / ".hermes" / "desktop-attachments").exists()
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_errors_when_unresolvable_and_no_bytes(monkeypatch, tmp_path):
    """Remote path not on gateway and no data_url → actionable error, not a stage."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": "/Users/alice/missing.txt"},
            }
        )

        assert "error" in resp
        assert "no data_url" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_quotes_ref_with_spaces(monkeypatch, tmp_path):
    """Staged names with spaces must be backtick-quoted so the @file: ref parses."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "name": "my exam schedule.csv",
                    "data_url": "data:text/csv;base64,YSxiCg==",
                },
            }
        )

        assert resp["result"]["attached"] is True
        assert resp["result"]["ref_text"] == "@file:`.hermes/desktop-attachments/my exam schedule.csv`"
    finally:
        server._sessions.pop("sid", None)


def test_commands_catalog_surfaces_quick_commands(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "quick_commands": {
                "build": {"type": "exec", "command": "npm run build"},
                "git": {"type": "alias", "target": "/shell git"},
                "notes": {
                    "type": "exec",
                    "command": "cat NOTES.md",
                    "description": "Open design notes",
                },
            }
        },
    )

    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    pairs = dict(resp["result"]["pairs"])
    assert "npm run build" in pairs["/build"]
    assert pairs["/git"].startswith("alias →")
    assert pairs["/notes"] == "Open design notes"

    user_cat = next(
        c for c in resp["result"]["categories"] if c["name"] == "User commands"
    )
    user_pairs = dict(user_cat["pairs"])
    assert set(user_pairs) == {"/build", "/git", "/notes"}

    assert resp["result"]["canon"]["/build"] == "/build"
    assert resp["result"]["canon"]["/notes"] == "/notes"


def test_commands_catalog_includes_tui_mouse_command():
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    pairs = dict(resp["result"]["pairs"])
    tui_cat = next(c for c in resp["result"]["categories"] if c["name"] == "TUI")
    tui_pairs = dict(tui_cat["pairs"])

    assert "/mouse" in pairs
    assert "/mouse" in tui_pairs


def test_commands_catalog_filters_gateway_only_commands_and_keeps_status_visible():
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    pairs = dict(resp["result"]["pairs"])
    canon = resp["result"]["canon"]

    assert "/status" in pairs
    assert canon["/status"] == "/status"

    assert "/topic" not in pairs
    assert "/approve" not in pairs
    assert "/deny" not in pairs
    assert "/sethome" not in pairs

    assert "/update" in pairs
    assert canon["/update"] == "/update"

    assert "/topic" not in canon
    assert "/approve" not in canon
    assert "/deny" not in canon
    assert "/set-home" not in canon


def test_session_status_reads_live_gateway_agent(monkeypatch):
    agent = types.SimpleNamespace(
        model="live-model",
        provider="live-provider",
        session_total_tokens=1234,
    )
    server._sessions["sid"] = _session(agent=agent, running=True)

    class _DB:
        def get_session(self, key):
            assert key == "session-key"
            return {
                "title": "Live TUI",
                "started_at": 1_700_000_000,
                "updated_at": 1_700_000_060,
            }

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    out = resp["result"]["output"]
    assert "Hermes TUI Status" in out
    assert "Session ID: session-key" in out
    assert "Title: Live TUI" in out
    assert "Model: live-model (live-provider)" in out
    assert "Tokens: 1,234" in out
    assert "Agent Running: Yes" in out


def test_skills_reload_runs_in_gateway_process(monkeypatch):
    import agent.skill_commands as skill_commands

    called = {}
    monkeypatch.setattr(
        skill_commands,
        "reload_skills",
        lambda: called.setdefault(
            "result",
            {
                "added": [{"name": "new-skill", "description": "demo"}],
                "removed": [],
                "total": 42,
            },
        ),
    )

    resp = server.handle_request({"id": "1", "method": "skills.reload", "params": {}})

    assert called["result"]["total"] == 42
    assert "new-skill" in resp["result"]["output"]
    assert "42 skill(s) available" in resp["result"]["output"]


def test_snapshot_restore_is_blocked_from_tui_worker():
    server._sessions["sid"] = _session()
    try:
        worker_resp = server.handle_request(
            {
                "id": "1",
                "method": "slash.exec",
                "params": {"command": "snapshot restore latest", "session_id": "sid"},
            }
        )
        dispatch_resp = server.handle_request(
            {
                "id": "2",
                "method": "command.dispatch",
                "params": {
                    "arg": "restore latest",
                    "name": "snapshot",
                    "session_id": "sid",
                },
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert worker_resp["error"]["code"] == 4018
    assert (
        "snapshot restore mutates live config/state" in worker_resp["error"]["message"]
    )
    assert dispatch_resp["result"]["type"] == "exec"
    assert (
        "/snapshot restore is blocked in the TUI" in dispatch_resp["result"]["output"]
    )


def test_command_dispatch_exec_nonzero_surfaces_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"boom": {"type": "exec", "command": "boom"}}},
    )
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr="failed"
        ),
    )

    resp = server.handle_request(
        {"id": "1", "method": "command.dispatch", "params": {"name": "boom"}}
    )

    assert "error" in resp
    assert "failed" in resp["error"]["message"]


def test_plugins_list_surfaces_loader_error(monkeypatch):
    with patch("hermes_cli.plugins.get_plugin_manager", side_effect=Exception("boom")):
        resp = server.handle_request(
            {"id": "1", "method": "plugins.list", "params": {}}
        )

    assert "error" in resp
    assert "boom" in resp["error"]["message"]


def test_complete_slash_surfaces_completer_error(monkeypatch):
    with patch(
        "hermes_cli.commands.SlashCommandCompleter",
        side_effect=Exception("no completer"),
    ):
        resp = server.handle_request(
            {"id": "1", "method": "complete.slash", "params": {"text": "/mo"}}
        )

    assert "error" in resp
    assert "no completer" in resp["error"]["message"]


def test_input_detect_drop_attaches_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: {
        "path": Path("/tmp/cat.png"),
        "is_image": True,
        "remainder": "",
    }

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "input.detect_drop",
            "params": {"session_id": "sid", "text": "/tmp/cat.png"},
        }
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["text"] == "[User attached image: cat.png]"


def test_input_detect_drop_path_with_spaces(tmp_path):
    """input.detect_drop correctly handles image paths containing spaces."""
    # Create a minimal PNG file with a space in its name
    img = tmp_path / "screenshot with spaces.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # valid PNG header

    server._sessions["sid"] = _session()

    resp = server.handle_request(
        {
            "id": "2",
            "method": "input.detect_drop",
            "params": {"session_id": "sid", "text": str(img)},
        }
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["path"] == str(img)
    assert resp["result"]["text"] == f"[User attached image: {img.name}]"
    # Verify attachment was recorded in the session
    assert len(server._sessions["sid"]["attached_images"]) == 1
    assert server._sessions["sid"]["attached_images"][0] == str(img)


def test_input_detect_drop_path_with_spaces_and_remainder(tmp_path):
    """input.detect_drop splits remainder when path contains spaces."""
    img = tmp_path / "photo with space.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"fakejpeg")  # minimal-ish JPEG header

    server._sessions["sid"] = _session()

    user_input = f"{img} describe this image"
    resp = server.handle_request(
        {
            "id": "3",
            "method": "input.detect_drop",
            "params": {"session_id": "sid", "text": user_input},
        }
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["path"] == str(img)
    # Remainder becomes the text sent to the model
    assert resp["result"]["text"] == "describe this image"
    assert server._sessions["sid"]["attached_images"][0] == str(img)


def test_rollback_restore_resolves_number_and_file_path():
    calls = {}

    class _Mgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}, {"hash": "bbb222"}]

        def restore(self, cwd, target, file_path=None):
            calls["args"] = (cwd, target, file_path)
            return {"success": True, "message": "done"}

    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=[]
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "rollback.restore",
            "params": {"session_id": "sid", "hash": "2", "file_path": "src/app.tsx"},
        }
    )

    assert resp["result"]["success"] is True
    assert calls["args"][1] == "bbb222"
    assert calls["args"][2] == "src/app.tsx"


# ── session.steer ────────────────────────────────────────────────────


def test_session_steer_calls_agent_steer_when_agent_supports_it():
    """The TUI RPC method must call agent.steer(text) and return a
    queued status without touching interrupt state.
    """
    calls = {}

    class _Agent:
        def steer(self, text):
            calls["steer_text"] = text
            return True

        def interrupt(self, *args, **kwargs):
            calls["interrupt_called"] = True

    server._sessions["sid"] = _session(agent=_Agent())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "also check auth.log"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "result" in resp, resp
    assert resp["result"]["status"] == "queued"
    assert resp["result"]["text"] == "also check auth.log"
    assert calls["steer_text"] == "also check auth.log"
    assert "interrupt_called" not in calls  # must NOT interrupt


def test_session_steer_rejects_empty_text():
    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(steer=lambda t: True)
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "   "},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4002


def test_session_steer_errors_when_agent_has_no_steer_method():
    server._sessions["sid"] = _session(agent=types.SimpleNamespace())  # no steer()
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4010


def test_session_info_includes_mcp_servers(monkeypatch):
    fake_status = [
        {"name": "github", "transport": "http", "tools": 12, "connected": True},
        {"name": "filesystem", "transport": "stdio", "tools": 4, "connected": True},
        {"name": "broken", "transport": "stdio", "tools": 0, "connected": False},
    ]
    fake_mod = types.ModuleType("tools.mcp_tool")
    fake_mod.get_mcp_status = lambda: fake_status
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mod)

    info = server._session_info(types.SimpleNamespace(tools=[], model="", provider="openai-codex"))

    assert info["provider"] == "openai-codex"
    assert info["mcp_servers"] == fake_status


# ---------------------------------------------------------------------------
# History-mutating commands must reject while session.running is True.
# Without these guards, prompt.submit's post-run history write either
# clobbers the mutation (version matches) or silently drops the agent's
# output (version mismatch) — both produce UI<->backend state desync.
# ---------------------------------------------------------------------------


def test_session_undo_rejects_while_running():
    """Fix for TUI silent-drop #1: /undo must not mutate history
    while the agent is mid-turn — would either clobber the undo or
    cause prompt.submit to silently drop the agent's response."""
    server._sessions["sid"] = _session(
        running=True,
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("error"), "session.undo should reject while running"
        assert resp["error"]["code"] == 4009
        assert "session busy" in resp["error"]["message"]
        # History must be unchanged
        assert len(server._sessions["sid"]["history"]) == 2
    finally:
        server._sessions.pop("sid", None)


def test_session_undo_allowed_when_idle():
    """Regression guard: when not running, /undo still works."""
    server._sessions["sid"] = _session(
        running=False,
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert resp["result"]["removed"] == 2
        assert server._sessions["sid"]["history"] == []
    finally:
        server._sessions.pop("sid", None)


def test_session_compress_rejects_while_running(monkeypatch):
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
        assert resp.get("error")
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)


def test_rollback_restore_rejects_full_history_while_running(monkeypatch):
    """Full-history rollback must reject; file-scoped rollback still allowed."""
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "rollback.restore",
                "params": {"session_id": "sid", "hash": "abc"},
            }
        )
        assert resp.get("error"), "full-history rollback should reject while running"
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_mismatch_surfaces_warning(monkeypatch):
    """Fix for TUI silent-drop #2: the defensive backstop at prompt.submit
    must attach a 'warning' to message.complete when history was
    mutated externally during the turn (instead of silently dropping
    the agent's output)."""
    # Agent bumps history_version itself mid-run to simulate an external
    # mutation slipping past the guards.
    session_ref = {"s": None}

    class _RacyAgent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            # Simulate: something external bumped history_version
            # while we were running.
            with session_ref["s"]["history_lock"]:
                session_ref["s"]["history_version"] += 1
            return {
                "final_response": "agent reply",
                "messages": [{"role": "assistant", "content": "agent reply"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_RacyAgent())
    session_ref["s"] = server._sessions["sid"]
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # History should NOT contain the agent's output (version mismatch)
        assert server._sessions["sid"]["history"] == []

        # message.complete must carry a 'warning' so the UI / operator
        # knows the output was not persisted.
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" in payload, (
            "message.complete must include a 'warning' field on "
            "history_version mismatch — otherwise the UI silently "
            "shows output that was never persisted"
        )
        assert (
            "not saved" in payload["warning"].lower()
            or "changed" in payload["warning"].lower()
        )
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_match_persists_normally(monkeypatch):
    """Regression guard: the backstop does not affect the happy path."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": "reply",
                "messages": [{"role": "assistant", "content": "reply"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result")

        # History was written
        assert server._sessions["sid"]["history"] == [
            {"role": "assistant", "content": "reply"}
        ]
        assert server._sessions["sid"]["history_version"] == 1

        # No warning should be attached
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" not in payload
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_can_truncate_before_user_ordinal(monkeypatch):
    """Desktop user-message edits should restart the turn from the edited user."""

    seen = {}

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "edited reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "edited reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    original_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
    ]
    server._sessions["sid"] = _session(agent=_Agent(), history=original_history)

    class _StubDb:
        def __init__(self):
            self.replaced = []

        def replace_messages(self, session_id, messages):
            self.replaced.append((session_id, list(messages)))

    stub_db = _StubDb()

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: stub_db)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edited second",
                    "truncate_before_user_ordinal": 1,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        assert seen["prompt"] == "edited second"
        assert seen["history"] == original_history[:2]
        assert server._sessions["sid"]["history"] == [
            *original_history[:2],
            {"role": "user", "content": "edited second"},
            {"role": "assistant", "content": "edited reply"},
        ]
        assert server._sessions["sid"]["history_version"] == 2
        assert stub_db.replaced == [("session-key", original_history[:2])]
    finally:
        server._sessions.pop("sid", None)


# ---------------------------------------------------------------------------
# session.interrupt must only cancel pending prompts owned by the calling
# session — it must not blast-resolve clarify/sudo/secret prompts on
# unrelated sessions sharing the same tui_gateway process.  Without
# session scoping the other sessions' prompts silently resolve to empty
# strings, unblocking their agent threads as if the user cancelled.
# ---------------------------------------------------------------------------


def test_interrupt_only_clears_own_session_pending():
    """session.interrupt on session A must NOT release pending prompts
    that belong to session B."""
    import types

    session_a = _session()
    session_a["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    session_b = _session()
    session_b["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    server._sessions["sid_a"] = session_a
    server._sessions["sid_b"] = session_b

    try:
        # Simulate pending prompts on both sessions (what _block creates
        # while a clarify/sudo/secret request is outstanding).
        ev_a = threading.Event()
        ev_b = threading.Event()
        server._pending["rid-a"] = ("sid_a", ev_a)
        server._pending["rid-b"] = ("sid_b", ev_b)
        server._answers.clear()

        # Interrupt session A.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.interrupt",
                "params": {"session_id": "sid_a"},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # Session A's pending must be released to empty.
        assert ev_a.is_set(), "sid_a pending Event should be set after interrupt"
        assert server._answers.get("rid-a") == ""

        # Session B's pending MUST remain untouched — no cross-session blast.
        assert not ev_b.is_set(), (
            "CRITICAL: session.interrupt on sid_a released a pending prompt "
            "belonging to sid_b — other sessions' clarify/sudo/secret "
            "prompts are being silently cancelled"
        )
        assert "rid-b" not in server._answers
    finally:
        server._sessions.pop("sid_a", None)
        server._sessions.pop("sid_b", None)
        server._pending.pop("rid-a", None)
        server._pending.pop("rid-b", None)
        server._answers.pop("rid-a", None)
        server._answers.pop("rid-b", None)


def test_interrupt_clears_multiple_own_pending():
    """When a single session has multiple pending prompts (uncommon but
    possible via nested tool calls), interrupt must release all of them."""
    import types

    sess = _session()
    sess["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    server._sessions["sid"] = sess

    try:
        ev1, ev2 = threading.Event(), threading.Event()
        server._pending["r1"] = ("sid", ev1)
        server._pending["r2"] = ("sid", ev2)

        resp = server.handle_request(
            {"id": "1", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )
        assert resp.get("result")
        assert ev1.is_set() and ev2.is_set()
        assert server._answers.get("r1") == "" and server._answers.get("r2") == ""
    finally:
        server._sessions.pop("sid", None)
        for key in ("r1", "r2"):
            server._pending.pop(key, None)
            server._answers.pop(key, None)


def test_clear_pending_without_sid_clears_all():
    """_clear_pending(None) is the shutdown path — must still release
    every pending prompt regardless of owning session."""
    ev1, ev2, ev3 = threading.Event(), threading.Event(), threading.Event()
    server._pending["a"] = ("sid_x", ev1)
    server._pending["b"] = ("sid_y", ev2)
    server._pending["c"] = ("sid_z", ev3)
    try:
        server._clear_pending(None)
        assert ev1.is_set() and ev2.is_set() and ev3.is_set()
    finally:
        for key in ("a", "b", "c"):
            server._pending.pop(key, None)
            server._answers.pop(key, None)


def test_respond_unpacks_sid_tuple_correctly():
    """After the (sid, Event) tuple change, _respond must still work."""
    ev = threading.Event()
    server._pending["rid-x"] = ("sid_x", ev)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "clarify.respond",
                "params": {"request_id": "rid-x", "answer": "the answer"},
            }
        )
        assert resp.get("result")
        assert ev.is_set()
        assert server._answers.get("rid-x") == "the answer"
    finally:
        server._pending.pop("rid-x", None)
        server._answers.pop("rid-x", None)


# ---------------------------------------------------------------------------
# /model switch and other agent-mutating commands must reject while the
# session is running.  agent.switch_model() mutates self.model, self.provider,
# self.base_url, self.client etc. in place — the worker thread running
# agent.run_conversation is reading those on every iteration.  Same class of
# bug as the session.undo / session.compress mid-run silent-drop; same fix
# pattern: reject with 4009 while running.
# ---------------------------------------------------------------------------


def test_config_set_model_rejects_while_running(monkeypatch):
    """/model via config.set must reject during an in-flight turn."""
    seen = {"called": False}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["called"] = True
        return {"value": raw, "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6",
                },
            }
        )
        assert resp.get("error")
        assert resp["error"]["code"] == 4009
        assert "session busy" in resp["error"]["message"]
        assert not seen["called"], (
            "_apply_model_switch was called mid-turn — would race with "
            "the worker thread reading agent.model / agent.client"
        )
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_allowed_when_idle(monkeypatch):
    """Regression guard: idle sessions can still switch models."""
    seen = {"called": False}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["called"] = True
        return {"value": "newmodel", "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    server._sessions["sid"] = _session(running=False)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "model", "value": "newmodel"},
            }
        )
        assert resp.get("result")
        assert resp["result"]["value"] == "newmodel"
        assert seen["called"]
    finally:
        server._sessions.pop("sid", None)


def test_mirror_slash_side_effects_rejects_mutating_commands_while_running(monkeypatch):
    """Slash worker passthrough (e.g. /model, /personality, /prompt,
    /compress) must reject during an in-flight turn.  Same race as
    config.set — mutates live agent state while run_conversation is
    reading it."""
    import types

    applied = {"model": False, "compress": False}

    def _fake_apply_model(sid, session, arg):
        applied["model"] = True
        return {"value": arg, "warning": ""}

    def _fake_compress(session, focus):
        applied["compress"] = True
        return (0, {})

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply_model)
    monkeypatch.setattr(server, "_compress_session_history", _fake_compress)

    session = _session(running=True)
    session["agent"] = types.SimpleNamespace(model="x")

    for cmd, expected_name in [
        ("/model new/model", "model"),
        ("/personality default", "personality"),
        ("/prompt", "prompt"),
        ("/compress", "compress"),
    ]:
        warning = server._mirror_slash_side_effects("sid", session, cmd)
        assert (
            "session busy" in warning
        ), f"{cmd} should have returned busy warning, got: {warning!r}"
        assert f"/{expected_name}" in warning

    # None of the mutating side-effect helpers should have fired.
    assert not applied["model"], "model switch fired despite running session"
    assert not applied["compress"], "compress fired despite running session"


def test_mirror_slash_side_effects_allowed_when_idle(monkeypatch):
    """Regression guard: idle session still runs the side effects."""
    import types

    applied = {"model": False}

    def _fake_apply_model(sid, session, arg):
        applied["model"] = True
        return {"value": arg, "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply_model)

    session = _session(running=False)
    session["agent"] = types.SimpleNamespace(model="x")

    warning = server._mirror_slash_side_effects("sid", session, "/model foo")
    # Should NOT contain "session busy" — the switch went through.
    assert "session busy" not in warning
    assert applied["model"]


def test_mirror_slash_compress_does_not_prelock_history(monkeypatch):
    """Regression guard: /compress side effect must not hold history_lock
    when calling _compress_session_history (the helper snapshots under
    the same non-reentrant lock internally)."""
    import types

    seen = {"compress": False, "sync": False}
    emitted = []

    def _fake_compress(session, focus_topic=None, **_kw):
        seen["compress"] = True
        assert not session["history_lock"].locked()
        return (0, {"total": 0})

    def _fake_sync(_sid, _session):
        seen["sync"] = True

    monkeypatch.setattr(server, "_compress_session_history", _fake_compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _fake_sync)
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    session = _session(running=False)
    session["agent"] = types.SimpleNamespace(model="x")

    warning = server._mirror_slash_side_effects("sid", session, "/compress")

    assert warning == ""
    assert seen["compress"]
    assert seen["sync"]
    assert ("session.info", "sid", {"model": "x"}) in emitted


# ---------------------------------------------------------------------------
# session.create / session.close race: fast /new churn must not orphan the
# slash_worker subprocess or the global approval-notify registration.
# ---------------------------------------------------------------------------


def test_session_create_close_race_does_not_orphan_worker(monkeypatch):
    """Regression guard: if session.close runs while session.create's
    _build thread is still constructing the agent, the build thread
    must detect the orphan and clean up the slash_worker + notify
    registration it's about to install.  Without the cleanup those
    resources leak — the subprocess stays alive until atexit and the
    notify callback lingers in the global registry."""
    import threading

    closed_workers: list[str] = []
    unregistered_keys: list[str] = []

    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key
            self._closed = False

        def close(self):
            self._closed = True
            closed_workers.append(self.key)

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

    # Make _build block until we release it — simulates slow agent init.
    # Also signal when _build actually reaches _make_agent so the test
    # can close the session at the right moment: session.create now
    # defers _start_agent_build behind a 50ms timer (see the
    # `_deferred_build` path in @method("session.create")), so closing
    # before the build thread has even started would skip the orphan
    # detection entirely and the test would race a non-event.
    build_started = threading.Event()
    release_build = threading.Event()
    build_entered = threading.Event()

    def _slow_make_agent(sid, key, session_id=None, session_db=None):
        build_started.set()
        build_entered.set()
        release_build.wait(timeout=3.0)
        return _FakeAgent()

    # Stub everything _build touches
    monkeypatch.setattr(server, "_make_agent", _slow_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(create_session=lambda *a, **kw: None),
    )
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    # Shim register/unregister to observe leaks
    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(
        _approval,
        "unregister_gateway_notify",
        lambda key: unregistered_keys.append(key),
    )
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    # Start: session.create spawns _build thread, returns synchronously
    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {"cols": 80},
        }
    )
    assert resp.get("result"), f"got error: {resp.get('error')}"
    sid = resp["result"]["session_id"]
    assert build_entered.wait(timeout=1.0), "deferred build did not start"

    # Wait until the (deferred) build thread has actually entered
    # _make_agent — otherwise session.close pops _sessions[sid] before
    # _build ever runs, _start_agent_build never calls _build, and we
    # never exercise the orphan-cleanup path.
    assert build_started.wait(timeout=2.0), "build thread never entered _make_agent"

    # Build thread is blocked in _slow_make_agent.  Close the session
    # NOW — this pops _sessions[sid] before _build can install the
    # worker/notify.
    close_resp = server.handle_request(
        {
            "id": "2",
            "method": "session.close",
            "params": {"session_id": sid},
        }
    )
    assert close_resp.get("result", {}).get("closed") is True

    # At this point session.close saw slash_worker=None (not yet
    # installed) so it didn't close anything.  Release the build thread
    # and let it finish — it should detect the orphan and clean up the
    # worker it just allocated + unregister the notify.
    release_build.set()

    # Give the build thread a moment to run through its finally.
    for _ in range(100):
        if closed_workers:
            break
        import time

        time.sleep(0.02)

    assert (
        len(closed_workers) == 1
    ), f"orphan worker was not cleaned up — closed_workers={closed_workers}"
    # Notify may be unregistered by both session.close (unconditional)
    # and the orphan-cleanup path; the key guarantee is that the build
    # thread does at least one unregister call (any prior close
    # already popped the callback; the duplicate is a no-op).
    assert len(unregistered_keys) >= 1, (
        f"orphan notify registration was not unregistered — "
        f"unregistered_keys={unregistered_keys}"
    )


def test_session_create_no_race_keeps_worker_alive(monkeypatch):
    """Regression guard: when session.close does NOT race, the build
    thread must install the worker + notify normally and leave them
    alone (no over-eager cleanup)."""
    closed_workers: list[str] = []
    unregistered_keys: list[str] = []

    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key

        def close(self):
            closed_workers.append(self.key)

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(create_session=lambda *a, **kw: None),
    )
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(
        _approval,
        "unregister_gateway_notify",
        lambda key: unregistered_keys.append(key),
    )
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {"cols": 80},
        }
    )
    sid = resp["result"]["session_id"]

    # Wait for the build to finish (ready event inside session dict).
    session = server._sessions[sid]
    session["agent_ready"].wait(timeout=2.0)

    # Build finished without a close race — nothing should have been
    # cleaned up by the orphan check.
    assert (
        closed_workers == []
    ), f"build thread closed its own worker despite no race: {closed_workers}"
    assert (
        unregistered_keys == []
    ), f"build thread unregistered its own notify despite no race: {unregistered_keys}"

    # Session should have the live worker installed.
    assert session.get("slash_worker") is not None

    # Cleanup
    server._sessions.pop(sid, None)


def test_get_db_degrades_cleanly_when_sessiondb_init_fails(monkeypatch):
    fake_mod = types.ModuleType("hermes_state")

    class _BrokenSessionDB:
        def __init__(self):
            raise RuntimeError("locking protocol")

    fake_mod.SessionDB = _BrokenSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)

    assert server._get_db() is None
    assert server._db_error == "locking protocol"


def test_session_create_continues_when_state_db_is_unavailable(monkeypatch):
    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key

        def close(self):
            return None

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

    emits = []

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emits.append(a))

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    session["agent_ready"].wait(timeout=2.0)

    assert session["agent_error"] is None
    assert session["agent"] is not None
    assert not any(args and args[0] == "error" for args in emits)

    server._sessions.pop(sid, None)


def test_session_create_lazy_info_reports_desktop_contract(monkeypatch):
    """The lazy session.create info payload must carry desktop_contract, else
    the desktop GUI reads it as undefined and falsely warns "Backend out of
    date" on every launch even against a current backend."""

    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **kw: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    info = resp["result"]["info"]

    assert info["desktop_contract"] == server.DESKTOP_BACKEND_CONTRACT

    server._sessions.pop(resp["result"]["session_id"], None)


def test_session_list_returns_clean_error_when_state_db_is_unavailable(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "locking protocol")

    resp = server.handle_request({"id": "1", "method": "session.list", "params": {}})

    assert "error" in resp
    assert "state.db unavailable: locking protocol" in resp["error"]["message"]


# --------------------------------------------------------------------------
# session.delete — TUI resume picker `d` key
# --------------------------------------------------------------------------


def test_session_delete_requires_session_id(monkeypatch):
    """Empty / missing session_id is a 4006 client error (no DB call)."""
    called: list[tuple] = []

    class _DB:
        def delete_session(self, *a, **kw):
            called.append((a, kw))
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request({"id": "1", "method": "session.delete", "params": {}})
    assert "error" in resp
    assert resp["error"]["code"] == 4006
    assert called == []


def test_session_delete_returns_db_unavailable_when_no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "locked")

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "abc"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert "state.db unavailable" in resp["error"]["message"]


def test_session_delete_refuses_active_session(monkeypatch):
    """Cannot delete a session currently bound to a live TUI session."""
    called: list[str] = []

    class _DB:
        def delete_session(self, sid, sessions_dir=None):
            called.append(sid)
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setitem(server._sessions, "live", {"session_key": "key-live"})
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.delete",
                "params": {"session_id": "key-live"},
            }
        )
    finally:
        server._sessions.pop("live", None)

    assert "error" in resp
    assert resp["error"]["code"] == 4023
    assert "active session" in resp["error"]["message"]
    assert called == [], "delete_session must not be called for active sessions"


def test_session_delete_fails_closed_when_active_snapshot_raises(monkeypatch):
    """Concurrent ``_sessions`` mutation from another RPC thread can raise
    ``RuntimeError: dictionary changed size during iteration``.  When the
    handler can't enumerate active sessions safely it must refuse the
    delete (fail closed) rather than fall through and allow it."""

    class _DB:
        def delete_session(self, *a, **kw):
            raise AssertionError("delete must not run when active snapshot fails")

    class _ExplodingDict:
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_sessions", _ExplodingDict())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert "enumerate active sessions" in resp["error"]["message"]


def test_session_delete_returns_4007_when_missing(monkeypatch):
    class _DB:
        def delete_session(self, sid, sessions_dir=None):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "ghost"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 4007


def test_session_delete_propagates_db_exception(monkeypatch):
    class _DB:
        def delete_session(self, sid, sessions_dir=None):
            raise RuntimeError("disk full")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert "disk full" in resp["error"]["message"]


def test_session_delete_success_returns_deleted_id(monkeypatch):
    """Happy path — DB delete succeeds, response carries the deleted id
    and the on-disk sessions dir is forwarded so transcript files get
    cleaned up alongside the row."""
    captured: dict = {}

    class _DB:
        def delete_session(self, sid, sessions_dir=None):
            captured["sid"] = sid
            captured["sessions_dir"] = sessions_dir
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "old-1"}}
    )

    assert "result" in resp, resp
    assert resp["result"] == {"deleted": "old-1"}
    assert captured["sid"] == "old-1"
    # sessions_dir must be forwarded so transcript files get cleaned up
    # too — not just the SQLite row.  The autouse _isolate_hermes_home
    # fixture pins HERMES_HOME to a temp dir; the handler should append
    # /sessions to it.
    assert captured["sessions_dir"] is not None
    assert str(captured["sessions_dir"]).endswith("sessions")


# --------------------------------------------------------------------------
# model.options — curated-list parity with `hermes model` and classic /model
# --------------------------------------------------------------------------


def test_model_options_does_not_overwrite_curated_models(monkeypatch):
    """The TUI model.options handler must surface the same curated model
    list as `hermes model` and the classic CLI /model picker.

    Regression: earlier versions of this handler unconditionally replaced
    each provider's curated ``models`` field with ``provider_model_ids()``
    (live /models catalog).  That pulled in hundreds of non-agentic models
    for providers like Nous whose /models endpoint returns image/video
    generators, rerankers, embeddings, and TTS models alongside chat models.
    """
    curated_providers = [
        {
            "slug": "nous",
            "name": "Nous",
            "models": ["moonshotai/kimi-k2.5", "anthropic/claude-opus-4.7"],
            "total_models": 30,
            "source": "built-in",
            "is_current": False,
            "is_user_defined": False,
        },
    ]

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"providers": {}, "custom_providers": []},
    )

    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=curated_providers,
    ) as listing:
        # If provider_model_ids gets called at all, the handler is still
        # overwriting curated with live — that's the regression we're
        # guarding against.
        with patch("hermes_cli.models.provider_model_ids") as live_fetch:
            resp = server._methods["model.options"](99, {"session_id": ""})

    assert "result" in resp, resp
    providers = resp["result"]["providers"]
    nous = next((p for p in providers if p.get("slug") == "nous"), None)
    assert nous is not None
    assert nous["models"] == [
        "moonshotai/kimi-k2.5",
        "anthropic/claude-opus-4.7",
    ]
    assert nous["total_models"] == 30
    # Handler must not consult the live catalog — curated is the truth.
    live_fetch.assert_not_called()
    # list_authenticated_providers is the single source.
    assert listing.call_count == 1


def test_model_options_propagates_list_exception(monkeypatch):
    """If list_authenticated_providers itself raises, surface as an RPC
    error rather than swallowing to a blank picker."""
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"providers": {}, "custom_providers": []},
    )
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        side_effect=RuntimeError("catalog blew up"),
    ):
        resp = server._methods["model.options"](77, {"session_id": ""})
    assert "error" in resp
    assert resp["error"]["code"] == 5033
    assert "catalog blew up" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# prompt.submit — auto-title
# ---------------------------------------------------------------------------


class _ImmediateThread:
    """Runs the target callable synchronously so assertions can follow."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def test_prompt_submit_auto_titles_session_on_complete(monkeypatch):
    """maybe_auto_title is called after a successful (complete) prompt."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": "Rome was founded in 753 BC.",
                "messages": [
                    {"role": "user", "content": "Tell me about Rome"},
                    {"role": "assistant", "content": "Rome was founded in 753 BC."},
                ],
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    with patch("agent.title_generator.maybe_auto_title") as mock_title:
        server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "Tell me about Rome"},
            }
        )

    mock_title.assert_called_once()
    args = mock_title.call_args.args
    assert args[1] == "session-key"
    assert args[2] == "Tell me about Rome"
    assert args[3] == "Rome was founded in 753 BC."


def test_prompt_submit_skips_auto_title_when_interrupted(monkeypatch):
    """maybe_auto_title must NOT be called when the agent was interrupted."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": "partial answer",
                "interrupted": True,
                "messages": [],
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    with patch("agent.title_generator.maybe_auto_title") as mock_title:
        server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "Tell me about Rome"},
            }
        )

    mock_title.assert_not_called()


def test_prompt_submit_skips_auto_title_when_response_empty(monkeypatch):
    """maybe_auto_title must NOT be called when the agent returns an empty reply."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": "",
                "messages": [],
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    with patch("agent.title_generator.maybe_auto_title") as mock_title:
        server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "Tell me about Rome"},
            }
        )

    mock_title.assert_not_called()


def test_prompt_submit_surfaces_backend_error_as_visible_text(monkeypatch):
    """When the backend fails with no visible response (e.g. invalid model slug
    → provider 4xx), the TUI must surface result['error'] as visible text
    instead of emitting a blank message.complete turn."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": None,
                "messages": [],
                "api_calls": 0,
                "completed": False,
                "failed": True,
                "error": "HTTP 400: invalid model id 'kimi-k2.6'",
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload or {})),
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        }
    )

    complete_events = [e for e in emitted if e[0] == "message.complete"]
    assert complete_events, "expected message.complete to be emitted"
    payload = complete_events[-1][2]
    assert payload.get("status") == "error"
    assert payload.get("text", "").startswith("Error:")
    assert "kimi-k2.6" in payload.get("text", "")


def test_prompt_submit_preserves_empty_response_without_error(monkeypatch):
    """An empty final_response with NO backend error must stay empty — do not
    synthesize an error string. Preserves the existing None/empty-sentinel
    semantics owned by downstream handlers."""

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None
        ):
            return {
                "final_response": None,
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload or {})),
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        }
    )

    complete_events = [e for e in emitted if e[0] == "message.complete"]
    assert complete_events, "expected message.complete to be emitted"
    payload = complete_events[-1][2]
    # Status stays "complete" because no error flag was set
    assert payload.get("status") == "complete"
    # Text stays empty — we did NOT fabricate an "Error:" string
    text = payload.get("text", "")
    assert text in {"", None}, f"expected empty text, got {text!r}"


# ── active live TUI sessions ─────────────────────────────────────────


def test_session_active_list_reports_live_sessions(monkeypatch):
    class _DB:
        def get_session_title(self, key):
            return {"key-a": "Research", "key-b": "Implement"}.get(key, "")

    previous_sessions = dict(server._sessions)
    server._sessions.clear()
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    server._sessions["sid-a"] = _session(
        agent=types.SimpleNamespace(model="model-a"),
        history=[{"role": "user", "content": "find docs"}],
        session_key="key-a",
        created_at=10.0,
        last_active=20.0,
    )
    server._sessions["sid-b"] = _session(
        agent=types.SimpleNamespace(model="model-b"),
        history=[{"role": "assistant", "content": "writing code"}],
        running=True,
        session_key="key-b",
        created_at=11.0,
        last_active=30.0,
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.active_list",
                "params": {"current_session_id": "sid-b"},
            }
        )
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)

    session_rows = resp["result"]["sessions"]
    assert [row["id"] for row in session_rows] == ["sid-a", "sid-b"]

    rows = {row["id"]: row for row in session_rows}
    assert rows["sid-a"] == {
        "current": False,
        "id": "sid-a",
        "last_active": 20.0,
        "message_count": 1,
        "model": "model-a",
        "preview": "find docs",
        "session_key": "key-a",
        "started_at": 10.0,
        "status": "idle",
        "title": "Research",
    }
    assert rows["sid-b"]["current"] is True
    assert rows["sid-b"]["status"] == "working"
    assert rows["sid-b"]["title"] == "Implement"
    assert rows["sid-b"]["preview"] == "writing code"


def test_session_active_list_excludes_finalized_sessions(monkeypatch):
    """#38950: a finalized-but-not-yet-popped session must not inflate the count.

    The WS grace-reap and idle reaper set ``_finalized`` inside
    ``_teardown_session`` before popping the entry from ``_sessions``. During
    that window ``session.active_list`` would otherwise still report the dead
    session, which is exactly the footer "N sessions" count that only ever grew
    until a gateway restart. A live session on the real stdio transport (the
    standalone ``hermes --tui`` case) must still be reported.
    """
    class _DB:
        def get_session_title(self, key):
            return {"key-live": "Live", "key-dead": "Dead"}.get(key, "")

    previous_sessions = dict(server._sessions)
    server._sessions.clear()
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    server._sessions["sid-live"] = _session(
        agent=types.SimpleNamespace(model="model-live"),
        history=[{"role": "user", "content": "still here"}],
        session_key="key-live",
        created_at=10.0,
        last_active=20.0,
    )
    dead = _session(
        agent=types.SimpleNamespace(model="model-dead"),
        history=[{"role": "user", "content": "gone"}],
        session_key="key-dead",
        created_at=11.0,
        last_active=21.0,
    )
    dead["_finalized"] = True
    server._sessions["sid-dead"] = dead
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.active_list",
                "params": {},
            }
        )
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)

    session_rows = resp["result"]["sessions"]
    assert [row["id"] for row in session_rows] == ["sid-live"]



def test_session_activate_returns_inflight_stream_before_completion(monkeypatch):
    """Switching into a still-running live session must hydrate partial output.

    The committed session history is only updated after run_conversation returns,
    so session.activate needs an explicit in-flight payload sourced from the
    backend stream callback.
    """
    started = threading.Event()
    release = threading.Event()
    done = threading.Event()

    class _Agent:
        model = "model-live"

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            assert prompt == "write a long answer"
            assert conversation_history == []
            stream_callback("partial ")
            stream_callback("answer")
            started.set()
            assert release.wait(2), "test timed out waiting to finish fake model turn"
            return {
                "final_response": "partial answer complete",
                "messages": [
                    {"role": "user", "content": "write a long answer"},
                    {"role": "assistant", "content": "partial answer complete"},
                ],
            }

    server._sessions["sid-live"] = _session(agent=_Agent())
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})

    def _emit(event, sid, payload=None):
        if event == "message.complete":
            done.set()

    monkeypatch.setattr(server, "_emit", _emit)

    try:
        submit = server.handle_request(
            {
                "id": "submit",
                "method": "prompt.submit",
                "params": {"session_id": "sid-live", "text": "write a long answer"},
            }
        )
        assert submit["result"]["status"] == "streaming"
        assert started.wait(2), "fake model did not stream before activation"

        resp = server.handle_request(
            {
                "id": "activate",
                "method": "session.activate",
                "params": {"session_id": "sid-live"},
            }
        )

        inflight = resp["result"].get("inflight")
        assert inflight == {
            "assistant": "partial answer",
            "streaming": True,
            "user": "write a long answer",
        }
        assert resp["result"]["messages"] == []

        release.set()
        assert done.wait(2), "fake model turn did not complete"
        completed = server.handle_request(
            {
                "id": "activate-done",
                "method": "session.activate",
                "params": {"session_id": "sid-live"},
            }
        )
        assert completed["result"].get("inflight") is None
        assert completed["result"]["messages"] == [
            {"role": "user", "text": "write a long answer"},
            {"role": "assistant", "text": "partial answer complete"},
        ]
    finally:
        release.set()
        done.wait(2)
        server._sessions.pop("sid-live", None)


def test_session_activate_switches_live_session_without_closing_siblings(monkeypatch):
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})
    server._sessions["sid-a"] = _session(
        agent=types.SimpleNamespace(model="model-a"),
        history=[{"role": "user", "content": "old"}],
        session_key="key-a",
    )
    server._sessions["sid-b"] = _session(
        agent=types.SimpleNamespace(model="model-b"),
        history=[
            {"role": "user", "content": "new prompt"},
            {"role": "assistant", "content": "new answer"},
        ],
        running=True,
        session_key="key-b",
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.activate", "params": {"session_id": "sid-b"}}
        )

        assert "sid-a" in server._sessions
        assert "sid-b" in server._sessions
        assert resp["result"]["session_id"] == "sid-b"
        assert resp["result"]["session_key"] == "key-b"
        assert resp["result"]["running"] is True
        assert resp["result"]["status"] == "working"
        assert resp["result"]["info"] == {"model": "model-b"}
        assert resp["result"]["messages"] == [
            {"role": "user", "text": "new prompt"},
            {"role": "assistant", "text": "new answer"},
        ]
    finally:
        server._sessions.pop("sid-a", None)
        server._sessions.pop("sid-b", None)


# ── session.most_recent ──────────────────────────────────────────────


def test_session_most_recent_returns_first_non_denied(monkeypatch):
    """Drops `tool` rows like session.list does, returns the first hit."""

    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200):
            return [
                {"id": "tool-1", "source": "tool", "title": "noise", "started_at": 100},
                {"id": "tui-1", "source": "tui", "title": "real", "started_at": 99},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] == "tui-1"
    assert resp["result"]["title"] == "real"
    assert resp["result"]["source"] == "tui"


def test_session_most_recent_returns_null_when_only_tool_rows(monkeypatch):
    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200):
            return [{"id": "tool-1", "source": "tool", "started_at": 1}]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] is None


def test_session_most_recent_folds_db_exception_into_null_result(monkeypatch):
    """Per contract, errors are folded into the null-result shape so
    callers don't have to special-case JSON-RPC error envelopes for
    'no answer' (Copilot review on #17130)."""

    class _BrokenDB:
        def list_sessions_rich(self, *, source=None, limit=200):
            raise RuntimeError("db locked")

    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert "error" not in resp
    assert resp["result"]["session_id"] is None


def test_session_most_recent_handles_db_unavailable(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] is None


# ── browser.manage ───────────────────────────────────────────────────


def _stub_urlopen(monkeypatch, *, ok: bool):
    """Patch urllib.request.urlopen used by browser.manage to short-circuit probes."""

    class _Resp:
        status = 200 if ok else 503

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(_url, timeout=2.0):  # noqa: ARG001 — match urllib signature
        if not ok:
            raise OSError("probe failed")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _opener)


def _stub_urlopen_capture(monkeypatch, *, ok: bool):
    urls: list[str] = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _opener(url, timeout=2.0):  # noqa: ARG001 — match urllib signature
        urls.append(url)
        if not ok:
            raise OSError("probe failed")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _opener)
    return urls


def test_browser_manage_status_reads_env_var(monkeypatch):
    """Status returns the env var verbatim (no network I/O)."""
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")

    resp = server.handle_request(
        {"id": "1", "method": "browser.manage", "params": {"action": "status"}}
    )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"


def test_browser_manage_status_falls_back_to_config_cdp_url(monkeypatch):
    """When env is unset, status surfaces ``browser.cdp_url`` from
    config.yaml so users see what the next tool call will read."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    fake_cfg = types.SimpleNamespace(
        read_raw_config=lambda: {"browser": {"cdp_url": "http://lan:9222"}}
    )
    with patch.dict(sys.modules, {"hermes_cli.config": fake_cfg}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "status"}}
        )

    assert resp["result"] == {"connected": True, "url": "http://lan:9222"}


def test_browser_manage_status_does_not_call_get_cdp_override(monkeypatch):
    """Regression guard for Copilot's "status must not block" review:
    status must NOT route through `_get_cdp_override`, which performs a
    `/json/version` HTTP probe with a multi-second timeout."""
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")

    fake = types.SimpleNamespace(
        _get_cdp_override=lambda: pytest.fail(  # noqa: PT015 — fail loudly if called
            "_get_cdp_override must not run on /browser status (network I/O)"
        )
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "status"}}
        )

    assert resp["result"]["connected"] is True


def test_browser_manage_connect_sets_env_and_cleans_twice(monkeypatch):
    """`/browser connect` must reach the live process: set env, reap browser
    sessions before AND after publishing the new URL.  The double-cleanup
    closes the supervisor swap window where ``_ensure_cdp_supervisor``
    could re-attach to the *old* CDP endpoint between steps."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    cleanup_calls: list[str] = []

    def _cleanup_all():
        cleanup_calls.append(os.environ.get("BROWSER_CDP_URL", ""))

    fake = types.SimpleNamespace(
        cleanup_all_browsers=_cleanup_all,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://127.0.0.1:9222"},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert resp["result"]["messages"] == ["Chromium-family browser is already listening on port 9222"]
    assert os.environ.get("BROWSER_CDP_URL") == "http://127.0.0.1:9222"
    # First cleanup runs against the OLD env (none here), second against the NEW.
    assert cleanup_calls == ["", "http://127.0.0.1:9222"]


def test_browser_manage_connect_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        urls = _stub_urlopen_capture(monkeypatch, ok=True)
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert resp["result"]["messages"] == ["Chromium-family browser is already listening on port 9222"]
    assert urls[0] == "http://127.0.0.1:9222/json/version"


def test_browser_manage_connect_default_local_reports_launch_hint(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda evt, sid, payload=None: emitted.append((evt, payload or {})),
    )
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        with (
            patch(
                "hermes_cli.browser_connect.try_launch_chrome_debug", return_value=False
            ),
            patch(
                "hermes_cli.browser_connect.get_chrome_debug_candidates",
                return_value=[],
            ),
        ):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {
                        "action": "connect",
                        "session_id": "sess-1",
                        "url": "http://localhost:9222",
                    },
                }
            )

    assert resp["result"]["connected"] is False
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert (
        resp["result"]["messages"][0]
        == "Chromium-family browser isn't running with remote debugging — attempting to launch..."
    )
    assert any(
        "No supported Chromium-family browser executable was found" in line
        for line in resp["result"]["messages"]
    )
    assert any(
        "--remote-debugging-port=9222" in line for line in resp["result"]["messages"]
    )
    assert "BROWSER_CDP_URL" not in os.environ
    progress = [p["message"] for evt, p in emitted if evt == "browser.progress"]
    assert progress == resp["result"]["messages"]


def test_browser_manage_connect_no_session_skips_progress_events(monkeypatch):
    """Without a session_id the TUI prints messages from the response;
    emitting ``browser.progress`` events would double-render. Gate the
    emit so callers without a session see the bundled list only."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda evt, sid, payload=None: emitted.append((evt, payload or {})),
    )
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        with (
            patch(
                "hermes_cli.browser_connect.try_launch_chrome_debug", return_value=False
            ),
            patch(
                "hermes_cli.browser_connect.get_chrome_debug_candidates",
                return_value=[],
            ),
        ):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": "http://localhost:9222"},
                }
            )

    assert resp["result"]["connected"] is False
    assert resp["result"]["messages"]  # bundled list still populated
    assert [evt for evt, _ in emitted if evt == "browser.progress"] == []


def test_browser_manage_connect_handles_null_url(monkeypatch):
    """Explicit ``{"url": null}`` (or empty string) must fall back to the
    default loopback URL instead of raising a TypeError that gets swallowed
    by the outer 5031 catch."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": None},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"


def test_browser_manage_connect_rejects_non_string_url(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": 9222},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "must be a string" in resp["error"]["message"]
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_default_local_retries_after_launch(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    attempts = {"n": 0}

    def _opener(_url, timeout=2.0):  # noqa: ARG001 — match urllib signature
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("not ready")
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _opener)
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with patch(
            "hermes_cli.browser_connect.try_launch_chrome_debug", return_value=True
        ):
            resp = server.handle_request(
                {"id": "1", "method": "browser.manage", "params": {"action": "connect"}}
            )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert resp["result"]["messages"] == [
        "Chromium-family browser isn't running with remote debugging — attempting to launch...",
        "Chromium-family browser launched and listening on port 9222",
    ]
    assert os.environ["BROWSER_CDP_URL"] == "http://127.0.0.1:9222"


def test_browser_manage_connect_rejects_unreachable_endpoint(monkeypatch):
    """An unreachable endpoint must NOT mutate the env or reap sessions."""
    monkeypatch.setenv("BROWSER_CDP_URL", "http://existing:9222")
    cleanup_calls: list[str] = []
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: cleanup_calls.append(
            os.environ.get("BROWSER_CDP_URL", "")
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=False)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://unreachable:9222"},
            }
        )

    assert "error" in resp
    # Env preserved; nothing reaped.
    assert os.environ["BROWSER_CDP_URL"] == "http://existing:9222"
    assert cleanup_calls == []


def test_browser_manage_connect_normalizes_bare_host_port(monkeypatch):
    """Persist a parsed `scheme://host:port` URL so `_get_cdp_override`
    can normalize it; storing a bare host:port would break subsequent
    tool calls (Copilot review on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "127.0.0.1:9222"},
            }
        )

    assert resp["result"]["connected"] is True
    # Bare host:port got promoted to a full URL with explicit scheme.
    assert resp["result"]["url"].startswith("http://")
    assert os.environ["BROWSER_CDP_URL"].startswith("http://")


def test_browser_manage_connect_strips_discovery_path(monkeypatch):
    """User-supplied discovery paths like `/json` or `/json/version`
    must collapse to bare `scheme://host:port`; otherwise
    ``_resolve_cdp_override`` will append ``/json/version`` again and
    produce a duplicate path (Copilot review round-2 on #17120)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        _stub_urlopen(monkeypatch, ok=True)
        resp = server.handle_request(
            {
                "id": "1",
                "method": "browser.manage",
                "params": {"action": "connect", "url": "http://127.0.0.1:9222/json"},
            }
        )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == "http://127.0.0.1:9222"
    assert os.environ["BROWSER_CDP_URL"] == "http://127.0.0.1:9222"


def test_browser_manage_connect_preserves_devtools_browser_endpoint(monkeypatch):
    """Concrete devtools websocket endpoints (e.g. Browserbase) must
    survive verbatim — we only collapse discovery-style paths."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://browserbase.example/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        # If urlopen is reached for a concrete ws endpoint, the test
        # would still pass because _stub_urlopen returned ok=True before;
        # patch it to assert-fail so we prove the HTTP probe is skipped.
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("urlopen called")
        ):
            with patch("socket.create_connection", return_value=_OkSocket()):
                resp = server.handle_request(
                    {
                        "id": "1",
                        "method": "browser.manage",
                        "params": {"action": "connect", "url": concrete},
                    }
                )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == concrete
    assert os.environ["BROWSER_CDP_URL"] == concrete


def test_browser_manage_connect_local_devtools_ws_preserves_path(monkeypatch):
    """Regression: ``ws://127.0.0.1:9222/devtools/browser/<id>`` is a real
    connectable endpoint; default-local normalization must not strip the
    ``/devtools/browser/...`` path or it breaks valid local CDP connects."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://127.0.0.1:9222/devtools/browser/abc123"

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with patch("socket.create_connection", return_value=_OkSocket()):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": concrete},
                }
            )

    assert resp["result"]["connected"] is True
    assert resp["result"]["url"] == concrete
    assert os.environ["BROWSER_CDP_URL"] == concrete


def test_browser_manage_connect_rejects_invalid_port(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": "http://localhost:abc"},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "invalid port" in resp["error"]["message"]
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_rejects_missing_host(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "browser.manage",
            "params": {"action": "connect", "url": "http://:9222"},
        }
    )

    assert resp["error"]["code"] == 4015
    assert "missing host" in resp["error"]["message"]
    assert "BROWSER_CDP_URL" not in os.environ


def test_browser_manage_connect_concrete_ws_skips_http_probe(monkeypatch):
    """Regression for round-2 Copilot review: a hosted CDP endpoint
    (no HTTP discovery) must connect via TCP-only reachability check.
    The HTTP probe used to reject these even though they're valid."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "wss://chrome.browserless.io/devtools/browser/sess-1"

    seen_targets: list[tuple[str, int]] = []

    class _OkSocket:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_create_connection(addr, timeout=None):
        seen_targets.append(addr)
        return _OkSocket()

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        # urlopen would 404/ECONNREFUSED on a real hosted CDP endpoint;
        # asserting it's never called proves the probe was skipped.
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("urlopen called")
        ):
            with patch("socket.create_connection", side_effect=_fake_create_connection):
                resp = server.handle_request(
                    {
                        "id": "1",
                        "method": "browser.manage",
                        "params": {"action": "connect", "url": concrete},
                    }
                )

    assert resp["result"] == {"connected": True, "url": concrete}
    # wss → port 443, host preserved verbatim.
    assert seen_targets == [("chrome.browserless.io", 443)]


def test_browser_manage_connect_concrete_ws_tcp_unreachable(monkeypatch):
    """If the TCP reachability check fails for a concrete ws endpoint,
    return a clear 5031 error — no fallback to the HTTP probe (which
    can never succeed for these URLs anyway)."""
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: None,
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    concrete = "ws://offline.example/devtools/browser/missing"

    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        with patch("socket.create_connection", side_effect=OSError("ECONNREFUSED")):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "browser.manage",
                    "params": {"action": "connect", "url": concrete},
                }
            )

    assert "error" in resp
    assert resp["error"]["code"] == 5031


def test_browser_manage_disconnect_drops_env_and_cleans(monkeypatch):
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    cleanup_count = {"n": 0}
    fake = types.SimpleNamespace(
        cleanup_all_browsers=lambda: cleanup_count.__setitem__(
            "n", cleanup_count["n"] + 1
        ),
        _get_cdp_override=lambda: os.environ.get("BROWSER_CDP_URL", ""),
    )
    with patch.dict(sys.modules, {"tools.browser_tool": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "browser.manage", "params": {"action": "disconnect"}}
        )

    assert resp["result"] == {"connected": False}
    assert "BROWSER_CDP_URL" not in os.environ
    # Two cleanups: once before env removal, once after, matching connect.
    assert cleanup_count["n"] == 2


# ── config.get indicator normalization ───────────────────────────────


def test_config_get_indicator_returns_known_value_verbatim(monkeypatch):
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": "emoji"}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "emoji"}


def test_config_get_indicator_normalizes_casing_and_whitespace(monkeypatch):
    """Hand-edited config.yaml stays consistent with what the TUI shows.

    Frontend's `normalizeIndicatorStyle` lowercases + trims, so config.get
    must do the same — otherwise `/indicator` prints 'EMOJI ' while the
    UI is actually rendering the kaomoji default."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": " EMOJI "}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "emoji"}


def test_config_get_indicator_falls_back_to_default_for_unknown(monkeypatch):
    """An unknown value in config.yaml falls back to the same default
    the frontend uses (`_INDICATOR_DEFAULT`)."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": "rainbow"}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "kaomoji"}


def test_config_get_indicator_falls_back_when_unset(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {}})
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "kaomoji"}


# ── config.set indicator validation ──────────────────────────────────


def test_config_set_indicator_accepts_known_value(monkeypatch):
    written: dict = {}
    monkeypatch.setattr(
        server,
        "_write_config_key",
        lambda k, v: written.update({k: v}),
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "indicator", "value": "EMOJI"},
        }
    )
    assert resp["result"] == {"key": "indicator", "value": "emoji"}
    assert written == {"display.tui_status_indicator": "emoji"}


def test_config_set_indicator_falsy_non_string_surfaces_in_error(monkeypatch):
    """`0` / `False` / `[]` are not valid styles, but the error message
    must still tell the user what they sent — `value or ""` would have
    erased them to a blank string."""
    monkeypatch.setattr(server, "_write_config_key", lambda *a, **k: None)

    for bad in (0, False, []):
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "indicator", "value": bad},
            }
        )
        assert "error" in resp
        msg = resp["error"]["message"]
        assert "unknown indicator" in msg
        # The exact repr varies; `0`/`False` stringify with content,
        # `[]` becomes an empty list — what matters is the diagnostic
        # is no longer just `unknown indicator: ` with nothing after.
        assert msg.split("; ")[0] != "unknown indicator: ''"


def test_config_set_indicator_none_keeps_blank_repr(monkeypatch):
    """`None` is the genuine 'no value' case — empty raw is acceptable."""
    monkeypatch.setattr(server, "_write_config_key", lambda *a, **k: None)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "indicator", "value": None},
        }
    )
    assert "error" in resp
    assert "unknown indicator: ''" in resp["error"]["message"]


# ── reload.env ───────────────────────────────────────────────────────


def test_reload_env_rpc_calls_hermes_cli_reload_env(monkeypatch):
    """reload.env mirrors classic CLI's `/reload` — re-reads ~/.hermes/.env
    into the gateway process and reports the count of vars updated."""
    calls = {"n": 0}

    def _fake_reload():
        calls["n"] += 1
        return 7

    fake = types.SimpleNamespace(reload_env=_fake_reload)
    with patch.dict(sys.modules, {"hermes_cli.config": fake}):
        resp = server.handle_request({"id": "1", "method": "reload.env", "params": {}})

    assert resp["result"] == {"updated": 7}
    assert calls["n"] == 1


def test_reload_env_rpc_surfaces_errors(monkeypatch):
    def _broken():
        raise RuntimeError("env path locked")

    fake = types.SimpleNamespace(reload_env=_broken)
    with patch.dict(sys.modules, {"hermes_cli.config": fake}):
        resp = server.handle_request({"id": "1", "method": "reload.env", "params": {}})

    assert "error" in resp
    assert "env path locked" in resp["error"]["message"]


# ── max_iterations config reading ─────────────────────────────────────


def _setup_make_agent_mocks(monkeypatch, cfg):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_resolve_startup_runtime", lambda: ("test-model", None)
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda: None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})


def test_make_agent_reads_nested_max_turns(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {"agent": {"max_turns": 200}})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 200


def test_make_agent_waits_for_shared_mcp_discovery(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {})
    waited = []

    from hermes_cli import mcp_startup

    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=0.75: waited.append(timeout),
    )

    with patch("run_agent.AIAgent"):
        server._make_agent("sid1", "key1")

    assert waited == [0.75]


def test_make_agent_nested_max_turns_takes_priority(monkeypatch):
    _setup_make_agent_mocks(
        monkeypatch, {"agent": {"max_turns": 500}, "max_turns": 100}
    )

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 500


def test_make_agent_defaults_to_90(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 90


def test_make_agent_uses_session_runtime_overrides(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {})
    resolved = {}

    def fake_resolve_runtime_provider(requested=None, target_model=None):
        resolved["requested"] = requested
        resolved["target_model"] = target_model
        return {
            "provider": requested,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent(
            "sid1",
            "key1",
            model_override="gpt-5.4",
            provider_override="openai-codex",
            reasoning_config_override={"enabled": True, "effort": "high"},
            service_tier_override="priority",
        )

    assert resolved == {"requested": "openai-codex", "target_model": "gpt-5.4"}
    assert mock_agent.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_agent.call_args.kwargs["provider"] == "openai-codex"
    assert mock_agent.call_args.kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert mock_agent.call_args.kwargs["service_tier"] == "priority"


def test_make_agent_handles_null_agent_config(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {"agent": None, "max_turns": 80})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 80


class _FakeAgentForBackground:
    base_url = None
    api_key = None
    provider = None
    api_mode = None
    acp_command = None
    acp_args = None
    model = "test-model"
    enabled_toolsets = None
    ephemeral_system_prompt = None
    providers_allowed = None
    providers_ignored = None
    providers_order = None
    provider_sort = None
    provider_require_parameters = False
    provider_data_collection = None
    reasoning_config = None
    service_tier = None
    request_overrides = {}
    _fallback_model = None


def test_background_agent_kwargs_reads_nested_max_turns(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"max_turns": 300}})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 300


def test_background_agent_kwargs_falls_back_to_root_max_turns(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 50})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 50


def test_background_agent_kwargs_defaults_to_25(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 25


def test_background_agent_kwargs_handles_null_agent_config(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": None, "max_turns": 40})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 40


def test_config_show_displays_nested_max_turns(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"agent": {"max_turns": 120}, "enabled_toolsets": [], "verbose": False},
    )
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    resp = server.handle_request({"id": "1", "method": "config.show", "params": {}})
    sections = resp["result"]["sections"]
    agent_rows = next(
        section["rows"] for section in sections if section["title"] == "Agent"
    )

    assert ["Max Turns", "120"] in agent_rows


def test_notification_poller_delivers_completion(monkeypatch):
    """Poller picks up completion events and triggers agent turns."""
    from tools.process_registry import process_registry

    turns = []
    emitted = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            turns.append(prompt)
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            self._target()

    sess = _session(agent=_Agent())
    server._sessions["sid_poll"] = sess
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    # Clear queue
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    process_registry._completion_consumed.discard("proc_poller_test")

    stop = threading.Event()

    # Put event on queue, then immediately signal stop so the poller
    # runs exactly one iteration.
    process_registry.completion_queue.put({
        "type": "completion",
        "session_id": "proc_poller_test",
        "command": "echo hello",
        "exit_code": 0,
        "output": "hello",
    })
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_poll", sess)

        # Should have emitted a status.update with kind=process
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) >= 1
        assert status_calls[0][2]["kind"] == "process"

        # Should have triggered an agent turn
        assert len(turns) == 1
        assert "[IMPORTANT: Background process proc_poller_test completed" in turns[0]
    finally:
        server._sessions.pop("sid_poll", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_poller_skips_consumed(monkeypatch):
    """Already-consumed completions are not dispatched by the poller."""
    from tools.process_registry import process_registry

    turns = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            turns.append(prompt)
            return {"final_response": "ok", "messages": []}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
        def start(self):
            self._target()

    sess = _session(agent=_Agent())
    server._sessions["sid_skip"] = sess
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    process_registry._completion_consumed.add("proc_already_done")
    process_registry.completion_queue.put({
        "type": "completion",
        "session_id": "proc_already_done",
        "command": "echo x",
        "exit_code": 0,
        "output": "x",
    })

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_skip", sess)
        assert len(turns) == 0
    finally:
        server._sessions.pop("sid_skip", None)
        process_registry._completion_consumed.discard("proc_already_done")
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_poller_requeues_when_busy(monkeypatch):
    """When the agent is busy, the poller requeues the event."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    emitted = []

    sess = _session(running=True)  # agent is busy
    server._sessions["sid_busy"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))

    # Isolate the completion queue for the duration of this test. The poller
    # reads process_registry.completion_queue by attribute at runtime, so a
    # fresh Queue here means no concurrently-running test in the same xdist
    # worker can put/get on the shared singleton mid-run and drain the event
    # we expect to be requeued. monkeypatch restores the original on teardown.
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_busy_test")

    evt = {
        "type": "completion",
        "session_id": "proc_busy_test",
        "command": "make build",
        "exit_code": 0,
        "output": "ok",
    }
    isolated_queue.put(evt)

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_busy", sess)

        # Status update was emitted (user sees it)
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 1

        # Event was requeued (agent was busy, no turn triggered)
        assert not isolated_queue.empty()
        requeued = isolated_queue.get_nowait()
        assert requeued["session_id"] == "proc_busy_test"
    finally:
        server._sessions.pop("sid_busy", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_session_save_writes_under_hermes_home_with_system_prompt(monkeypatch, tmp_path):
    """TUI /save (session.save RPC) must snapshot under the Hermes profile
    home — not the project/workspace CWD — and include the system prompt,
    mirroring the classic CLI /save and the dashboard save export.

    Regression: the gateway handler wrote ``hermes_conversation_*.json`` to
    ``os.path.abspath(...)`` (the workspace CWD) and only exported ``model``
    and ``messages``, so ``system_prompt`` was missing.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Run from a different CWD to prove the snapshot does NOT leak there.
    work = tmp_path / "workspace"
    work.mkdir()
    monkeypatch.chdir(work)

    sid = "save-sid"
    agent = types.SimpleNamespace(
        model="hermes-test",
        session_id="20260101_120000_abc123",
        session_start=datetime(2026, 1, 1, 12, 0, 0),
        _cached_system_prompt="You are Hermes.",
    )
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    server._sessions[sid] = {
        "agent": agent,
        "session_key": "save-key",
        "history": history,
        "history_lock": threading.Lock(),
        "created_at": 1735732800.0,
    }
    try:
        resp = server._methods["session.save"]("1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "result" in resp, resp
    saved_file = Path(resp["result"]["file"])

    # Must NOT leak into the workspace/project CWD.
    assert not list(work.glob("hermes_conversation_*.json"))

    saved_dir = home / "sessions" / "saved"
    assert saved_file.parent == saved_dir
    assert saved_file.exists()

    payload = json.loads(saved_file.read_text())
    assert payload["model"] == "hermes-test"
    assert payload["session_id"] == "20260101_120000_abc123"
    assert payload["session_start"] == "2026-01-01T12:00:00"
    assert payload["system_prompt"] == "You are Hermes."
    assert payload["messages"] == history


def test_notification_event_dedup_key_preserves_distinct_watch_matches():
    """Watch-match identity includes match content, not just session/type."""
    base = {
        "type": "watch_match",
        "session_id": "proc_watch",
        "command": "tail -f app.log",
        "pattern": "READY",
        "output": "READY on port 8000",
        "suppressed": 0,
    }

    identical = dict(base)
    distinct_output = {**base, "output": "READY on port 9000"}
    distinct_pattern = {**base, "pattern": "MIGRATION_DONE"}

    base_key = server._notification_event_dedup_key(base)
    assert server._notification_event_dedup_key(identical) == base_key
    assert server._notification_event_dedup_key(distinct_output) != base_key
    assert server._notification_event_dedup_key(distinct_pattern) != base_key


def test_notification_poller_emits_distinct_watch_matches_once(monkeypatch):
    """Distinct watch matches from one process emit; exact replay is deduped."""
    from tools.process_registry import process_registry

    turns = []
    emitted = []

    def _fake_run_prompt_submit(rid, sid, session, text):
        turns.append(text)
        with session["history_lock"]:
            session["running"] = False

    sess = _session()
    server._sessions["sid_watch_dedup"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_run_prompt_submit", _fake_run_prompt_submit)

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    base = {
        "type": "watch_match",
        "session_id": "proc_watch_dedup",
        "command": "tail -f app.log",
        "pattern": "READY",
        "output": "READY on port 8000",
        "suppressed": 0,
    }
    process_registry.completion_queue.put(base)
    process_registry.completion_queue.put({**base, "output": "READY on port 9000"})
    process_registry.completion_queue.put(dict(base))

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_watch_dedup", sess)
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 2
        status_text = "\n".join(call[2]["text"] for call in status_calls)
        assert "READY on port 8000" in status_text
        assert "READY on port 9000" in status_text
        assert len(turns) == 3
    finally:
        server._sessions.pop("sid_watch_dedup", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_event_dedup_key_keeps_completions_one_shot():
    """Completion identity remains process-session scoped to avoid floods."""
    first = {
        "type": "completion",
        "session_id": "proc_done",
        "command": "make build",
        "exit_code": 0,
        "output": "first output",
    }
    replay = {
        "type": "completion",
        "session_id": "proc_done",
        "command": "make build --again",
        "exit_code": 1,
        "output": "different output should not change completion key",
    }

    assert server._notification_event_dedup_key(first) == server._notification_event_dedup_key(
        replay
    )


# --- image.attach_bytes / pdf.attach (remote-client byte upload) -------------

# Smallest valid 1x1 PNG, base64-encoded.
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _attach_bytes_cli(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    monkeypatch.setitem(sys.modules, "cli", fake_cli)


def test_image_attach_bytes_writes_to_gateway_dir(monkeypatch, tmp_path):
    """Remote client uploads base64 bytes; gateway writes them to its own disk."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx",
                "content_base64": _PNG_1X1_B64,
                "filename": "shot.png",
            },
        }
    )

    res = resp["result"]
    assert res["attached"] is True
    written = Path(res["path"])
    assert written.is_file()
    assert written.parent == tmp_path / "images"
    assert written.read_bytes().startswith(b"\x89PNG")
    assert len(server._sessions["abx"]["attached_images"]) == 1
    assert res["bytes"] > 0


def test_image_attach_bytes_accepts_data_url_prefix(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx2"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx2",
                "content_base64": f"data:image/png;base64,{_PNG_1X1_B64}",
            },
        }
    )
    assert resp["result"]["attached"] is True


def test_image_attach_bytes_data_alias_and_magic_sniff(monkeypatch, tmp_path):
    """Older desktop builds send `data` (not content_base64); ext sniffed from bytes."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx3"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx3", "data": _PNG_1X1_B64},
        }
    )
    res = resp["result"]
    assert res["attached"] is True
    assert Path(res["path"]).suffix == ".png"  # sniffed from magic bytes


def test_image_attach_bytes_rejects_invalid_base64(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx4"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx4", "content_base64": "!!!not base64!!!"},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4017


def test_image_attach_bytes_rejects_oversize(monkeypatch, tmp_path):
    import base64 as _b64

    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_ATTACH_BYTES_MAX_BYTES", 10)
    server._sessions["abx5"] = _session()

    big = _b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 100).decode("ascii")
    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx5", "content_base64": big},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4018


def test_image_attach_bytes_rejects_unsupported_extension(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx6"] = _session()

    # filename hint forces a non-image extension; magic sniff is bypassed by hint
    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx6",
                "content_base64": _PNG_1X1_B64,
                "filename": "evil.exe",
            },
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4016


def test_pdf_attach_requires_poppler(monkeypatch, tmp_path):
    """Without pdftoppm on PATH, pdf.attach returns a clear 5028."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    server._sessions["pdf1"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "pdf.attach",
            "params": {"session_id": "pdf1", "content_base64": "JVBERi0xLjQK"},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 5028


def test_pdf_attach_rejects_non_pdf_bytes(monkeypatch, tmp_path):
    import base64 as _b64

    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["pdf2"] = _session()

    not_pdf = _b64.b64encode(b"this is not a pdf").decode("ascii")
    resp = server.handle_request(
        {
            "id": "1",
            "method": "pdf.attach",
            "params": {"session_id": "pdf2", "content_base64": not_pdf},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4017


def test_pdf_attach_requires_path_or_bytes(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["pdf3"] = _session()

    resp = server.handle_request(
        {"id": "1", "method": "pdf.attach", "params": {"session_id": "pdf3"}}
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4015


def test_decode_attach_base64_helper():
    import base64 as _b64

    raw = _b64.b64encode(b"hello").decode("ascii")
    assert server._decode_attach_base64(raw, mime_prefix="image/") == b"hello"
    assert (
        server._decode_attach_base64(f"data:image/png;base64,{raw}", mime_prefix="image/")
        == b"hello"
    )
    # whitespace inside payload is tolerated
    assert server._decode_attach_base64(raw[:4] + "\n" + raw[4:], mime_prefix="image/") == b"hello"
    assert server._decode_attach_base64("@@@", mime_prefix="image/") is None


def test_sniff_image_ext_magic_and_filename():
    assert server._sniff_image_ext(b"\x89PNG\r\n\x1a\n") == ".png"
    assert server._sniff_image_ext(b"\xff\xd8\xff\xe0") == ".jpg"
    assert server._sniff_image_ext(b"GIF89a....") == ".gif"
    assert server._sniff_image_ext(b"RIFF1234WEBPxxxx") == ".webp"
    assert server._sniff_image_ext(b"BM......") == ".bmp"
    assert server._sniff_image_ext(b"unknown") == ".png"  # fallback
    # filename hint wins over magic bytes
    assert server._sniff_image_ext(b"\x89PNG", "photo.jpeg") == ".jpeg"


def test_slash_worker_close_reaps_zombie_and_closes_fds():
    """A hung worker is SIGKILLed, the zombie reaped, all pipes closed — once."""
    calls = {k: 0 for k in ("terminate", "kill", "wait", "stdin", "stdout", "stderr")}

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls[self.name] += 1

    class FakeProc:
        stdin, stdout, stderr = (FakeStream(n) for n in ("stdin", "stdout", "stderr"))

        def poll(self):
            return None  # always alive -> forces terminate then kill

        def terminate(self):
            calls["terminate"] += 1

        def kill(self):
            calls["kill"] += 1

        def wait(self, timeout=None):
            calls["wait"] += 1
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    worker = object.__new__(server._SlashWorker)
    worker.proc = FakeProc()

    worker.close()
    worker.close()  # idempotent

    assert calls["terminate"] == 1
    assert calls["kill"] == 1
    assert calls["wait"] >= 2  # reaped after both terminate and kill
    assert calls["stdin"] == calls["stdout"] == calls["stderr"] == 1


def test_close_session_by_id_is_idempotent_and_full(monkeypatch):
    """One call tears the session down fully; a second is a no-op."""
    calls = {"worker": 0, "agent": 0, "unreg": 0, "finalize": 0}

    class W:
        def close(self):
            calls["worker"] += 1

    class A:
        def close(self):
            calls["agent"] += 1

    def _fake_finalize(s, end_reason="tui_close"):
        # Real _finalize_session is the single chokepoint that closes the
        # slash-worker; mirror that here so the test exercises the actual
        # teardown contract (worker close lives in finalize, not the caller).
        calls["finalize"] += 1
        w = s.get("slash_worker")
        if w:
            w.close()

    monkeypatch.setattr(server, "_finalize_session", _fake_finalize)
    monkeypatch.setattr(
        "tools.approval.unregister_gateway_notify",
        lambda key: calls.__setitem__("unreg", calls["unreg"] + 1), raising=False,
    )
    server._sessions["sid-1"] = {"session_key": "k1", "agent": A(), "slash_worker": W()}

    assert server._close_session_by_id("sid-1", end_reason="ws_disconnect") is True
    assert server._close_session_by_id("sid-1", end_reason="ws_disconnect") is False
    assert calls == {"worker": 1, "agent": 1, "unreg": 1, "finalize": 1}
    assert "sid-1" not in server._sessions


def test_attach_worker_closes_orphan_when_session_already_torn_down():
    """A worker built after its session was reaped must be closed, not orphaned."""
    closed = []

    class W:
        def close(self):
            closed.append(True)

    server._sessions.pop("gone", None)
    detached = {"session_key": "k"}  # not in _sessions -> already torn down
    server._attach_worker("gone", detached, W())

    assert closed == [True]
    assert "slash_worker" not in detached
    assert "gone" not in server._sessions


def test_attach_worker_stores_worker_on_live_session():
    class W:
        def close(self):
            raise AssertionError("must not close a worker for a live session")

    live = {"session_key": "k"}
    server._sessions["live"] = live
    worker = W()
    try:
        server._attach_worker("live", live, worker)
        assert live["slash_worker"] is worker
    finally:
        server._sessions.pop("live", None)


def test_restart_slash_worker_closes_orphan_when_session_reaped(monkeypatch):
    """Post-turn restart of a session reaped mid-flight (e.g. close_on_disconnect
    fired while `running` flipped false) must close the fresh worker, not orphan it."""
    closed = []

    class _FakeWorker:
        def __init__(self, *a, **k):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    server._sessions.pop("reaped", None)
    reaped = {"session_key": "k"}  # not in _sessions -> torn down concurrently
    server._restart_slash_worker("reaped", reaped)

    assert closed == [True]
    assert reaped.get("slash_worker") is None
    assert "reaped" not in server._sessions


def test_restart_slash_worker_stores_on_live_session(monkeypatch):
    class _FakeWorker:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    live = {"session_key": "k", "slash_worker": None}
    server._sessions["live-restart"] = live
    try:
        server._restart_slash_worker("live-restart", live)
        assert isinstance(live["slash_worker"], _FakeWorker)
    finally:
        server._sessions.pop("live-restart", None)


def test_session_close_rpc_delegates_to_close_session_by_id(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: bool(seen.append((sid, end_reason))) or True,
    )
    resp = server.handle_request(
        {"id": "1", "method": "session.close", "params": {"session_id": "s9"}}
    )
    assert resp["result"] == {"closed": True}
    assert seen == [("s9", "tui_close")]


def test_close_sessions_for_transport_closes_flagged_repoints_rest(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: bool(seen.append((sid, end_reason))) or True,
    )
    # Detached session "b" would schedule a real grace-reap threading.Timer that
    # outlives the test; grace=0 short-circuits it so no thread lingers.
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    transport = object()  # the disconnecting transport
    server._sessions.clear()
    server._sessions["a"] = {"transport": transport, "close_on_disconnect": True}
    server._sessions["b"] = {"transport": transport, "close_on_disconnect": False}
    try:
        server._close_sessions_for_transport(transport, end_reason="ws_disconnect")
        assert seen == [("a", "ws_disconnect")]  # only the flagged one closed
        assert server._sessions["b"]["transport"] is server._detached_ws_transport  # re-pointed
    finally:
        server._sessions.clear()


def test_session_create_records_close_on_disconnect_flag(monkeypatch):
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    server._sessions.clear()
    try:
        on = server.handle_request(
            {"id": "1", "method": "session.create", "params": {"close_on_disconnect": True}}
        )["result"]["session_id"]
        off = server.handle_request(
            {"id": "2", "method": "session.create", "params": {}}
        )["result"]["session_id"]
        assert server._sessions[on]["close_on_disconnect"]
        assert not server._sessions[off]["close_on_disconnect"]
    finally:
        server._sessions.clear()


def test_shutdown_sessions_closes_every_session_via_helper(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: seen.append((sid, end_reason)),
    )
    server._sessions.clear()
    server._sessions["a"] = {}
    server._sessions["b"] = {}
    try:
        server._shutdown_sessions()
        assert sorted(sid for sid, _ in seen) == ["a", "b"]
        assert {reason for _, reason in seen} == {"tui_shutdown"}
    finally:
        server._sessions.clear()


def _idle_evictable_session(now):
    """A session that satisfies every eviction precondition."""
    ready = threading.Event()
    ready.set()
    old = now - 10 * 3600  # well past the 6h TTL
    return {
        "running": False,
        "agent_ready": ready,
        "transport": server._detached_ws_transport,  # dead/detached
        "last_active": old,
        "created_at": old,
    }


def test_session_is_evictable_when_idle_dead_and_quiescent(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    now = time.time()
    assert server._session_is_evictable("s", _idle_evictable_session(now), now) is True


def test_session_not_evictable_violating_each_exemption(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    now = time.time()
    live_transport = type("T", (), {"_closed": False})()

    running = _idle_evictable_session(now) | {"running": True}
    assert server._session_is_evictable("s", running, now) is False

    starting = _idle_evictable_session(now)
    starting["agent_ready"] = threading.Event()  # not set -> still starting
    assert server._session_is_evictable("s", starting, now) is False

    on_socket = _idle_evictable_session(now) | {"transport": live_transport}
    assert server._session_is_evictable("s", on_socket, now) is False

    recent = _idle_evictable_session(now) | {"last_active": now}
    assert server._session_is_evictable("s", recent, now) is False

    young = _idle_evictable_session(now) | {"created_at": now}
    assert server._session_is_evictable("s", young, now) is False

    # Pending input request, even when everything else looks idle.
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "input")
    assert server._session_is_evictable("s", _idle_evictable_session(now), now) is False


def test_reap_idle_sessions_closes_only_evictable(monkeypatch):
    closed = []
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: closed.append((sid, end_reason)),
    )
    now = time.time()
    server._sessions.clear()
    server._sessions["stale"] = _idle_evictable_session(now)
    server._sessions["fresh"] = _idle_evictable_session(now) | {"last_active": now}
    try:
        server._reap_idle_sessions()
        assert closed == [("stale", "idle_timeout")]
    finally:
        server._sessions.clear()
