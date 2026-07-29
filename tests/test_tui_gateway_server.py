import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_cli.browser_connect import ChromeDebugLaunch
from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


def test_session_slot_is_claimed_on_first_turn_not_on_create(monkeypatch, tmp_path):
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

        # Opening a chat must NOT take a slot. Every tile paint and every
        # background reconnect-resume calls session.create, and an unprompted
        # draft has no DB row and is filtered out of the sidebar — so a slot
        # held here is invisible to the user while still starving the other
        # surfaces that share this cap.
        first = server._methods["session.create"]("r1", {"cols": 80})
        second = server._methods["session.create"]("r2", {"cols": 80})
        assert "result" in first and "result" in second
        sid = first["result"]["session_id"]
        other = second["result"]["session_id"]
        assert active_session_registry_snapshot() == []

        # The first turn is what claims the slot, and is re-entrant.
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert len(active_session_registry_snapshot()) == 1

        blocked = server._ensure_active_session_slot(other, server._sessions[other])
        assert "active session limit (1/1)" in blocked

        closed = server._methods["session.close"]("r3", {"session_id": sid})
        assert closed["result"]["closed"] is True
        assert active_session_registry_snapshot() == []

        assert server._ensure_active_session_slot(other, server._sessions[other]) is None
    finally:
        _clear_server_sessions()
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None
        reset_hermes_home_override(token)




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
















def test_prompt_submit_golden_transcript_matches_flag_off_and_on(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    class _Agent:
        model = "gold-model"
        provider = "gold-provider"
        session_id = "session-key"
        session_input_tokens = 10
        session_output_tokens = 5
        session_prompt_tokens = 10
        session_completion_tokens = 5
        session_total_tokens = 15
        session_api_calls = 1
        context_compressor = None

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            if stream_callback is not None:
                stream_callback("hi")
            return {
                "final_response": "hi",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "hi"},
                ],
            }

    fixed_info = {"model": "gold-model", "provider": "gold-provider", "usage": {"total": 15}}
    usage = server._get_usage(_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: dict(fixed_info))
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    fake_title = types.ModuleType("agent.title_generator")
    setattr(fake_title, "maybe_auto_title", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "agent.title_generator", fake_title)

    def run_flag_off():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": False}})
        server._sessions["sid"] = _session(
            agent=_Agent(), model_override={"model": "gold-model", "provider": "gold-provider"}
        )
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    def run_flag_on():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})

        class _FakeSupervisor:
            def submit_turn(self, frame, *, on_complete=None):
                sid = frame["sid"]
                server._emit("message.start", sid)
                server._emit("message.delta", sid, {"text": "hi"})
                server._emit("message.complete", sid, {"text": "hi", "usage": usage, "status": "complete"})
                server._emit("session.info", sid, dict(fixed_info))
                if on_complete is not None:
                    on_complete(
                        {
                            "type": "turn.end",
                            "sid": sid,
                            "request_id": frame["request_id"],
                            "session_key": "session-key",
                            "history_version": 1,
                            "message_count": 2,
                            "session_info": dict(fixed_info),
                            "session_info_emitted": True,
                        }
                    )
                return frame["request_id"]

        monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _FakeSupervisor())
        session = _session(
            agent=None,
            agent_ready=threading.Event(),
            _compute_host_active=True,
            model_override={"model": "gold-model", "provider": "gold-provider"},
        )
        session["agent"] = None
        server._sessions["sid"] = session
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    assert run_flag_on() == run_flag_off()




def _write_profile_cfg(home: Path, cwd: str | None) -> Path:
    import yaml

    home.mkdir(parents=True, exist_ok=True)
    cfg = {"terminal": {"cwd": cwd}} if cwd is not None else {}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home






















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






def test_tui_verbose_tool_details_fail_closed_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    setattr(redact_module, "redact_sensitive_text", fail_redaction)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    assert server._redact_tui_verbose_text("api_key=secret") == ""
    assert server._tool_args_text({"api_key": "secret"}) == ""
    assert server._tool_result_text("token=secret") == ""


















def test_system_battery_returns_reading(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "agent.battery",
        types.SimpleNamespace(
            read_battery=lambda: types.SimpleNamespace(
                available=True, percent=77, plugged=False
            ),
            battery_category=lambda _s: "good",
        ),
    )

    resp = server.dispatch({"id": "b1", "method": "system.battery", "params": {}})

    assert resp["result"] == {
        "available": True,
        "percent": 77,
        "plugged": False,
        "category": "good",
    }




































def test_load_enabled_toolsets_prefers_tui_env(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, terminal, ,memory")

    assert server._load_enabled_toolsets() == ["web", "terminal", "memory"]


























































def test_live_visible_history_keeps_candidate_and_fresh_tail():
    """The hard case: the persisted candidate (missing from in-memory) AND a
    not-yet-flushed live turn (missing from the DB) must BOTH survive."""
    # Persisted display: has the verification candidate, lags the newest turn.
    db_display = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
    ]
    # In-memory model history: candidate collapsed out, but has a fresh turn 2.
    in_memory = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]

    class DB:
        def get_messages_as_conversation(self, key, include_ancestors=False, repair_alternation=False):
            return list(db_display)

    result = server._live_visible_history({"session_key": "s1"}, DB(), in_memory)
    assert result == [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]


































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
























def test_startup_runtime_uses_tui_provider_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "nous")
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)

    assert server._resolve_startup_runtime() == ("nous/hermes-test", "nous")


















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


















def test_session_title_creates_row_and_sets_immediately_when_not_ready(monkeypatch):
    """An explicit /title before the first message must persist NOW, not queue.

    Regression: the desktop deferred the DB row to the first prompt, so a
    /title typed before any message only stashed ``pending_title`` and relied
    on a post-turn apply block. When that turn never landed under the session
    key, the title was silently lost and the sidebar fell back to the message
    preview. The handler now creates the row up front (mirroring the messaging
    gateway) so an explicit /title takes effect immediately.
    """
    state = {"row": None, "title": None, "ensured": False}

    class _FakeDB:
        def get_session_title(self, _key):
            return state["title"]

        def get_session(self, _key):
            return state["row"]

        def set_session_title(self, _key, title):
            # Mirrors SessionDB: UPDATE affects 0 rows until the row exists.
            if state["row"] is None:
                return False
            state["title"] = title
            return True

    fake_db = _FakeDB()

    def _fake_ensure_row(_session):
        # The real _ensure_session_db_row does an INSERT OR IGNORE.
        state["ensured"] = True
        state["row"] = {"id": "session-key", "title": None}

    import contextlib

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", _fake_ensure_row)
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "my-custom-name"},
            }
        )

        # No longer queued — the row is created and the title set immediately.
        assert set_resp["result"]["pending"] is False
        assert set_resp["result"]["title"] == "my-custom-name"
        assert state["ensured"] is True, "the row must be created up front"
        assert state["title"] == "my-custom-name"
        assert server._sessions["sid"]["pending_title"] is None

        # A subsequent read reflects the persisted title.
        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "my-custom-name"
    finally:
        server._sessions.pop("sid", None)


















class _StopAfterOneNotificationPoll:
    def __init__(self):
        self._checks = 0

    def is_set(self):
        self._checks += 1
        return self._checks > 1












def _configure_immediate_prompt_run(
    monkeypatch, tmp_path, *, immediate_threads=True
):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            if self._target is not None:
                self._target()

        def is_alive(self):
            return False

    if immediate_threads:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: None)


class _RecordingAgent:
    model = "test-model"
    provider = "test-provider"

    def __init__(self, turns):
        self._turns = turns

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
        self._turns.append(prompt)
        return {"final_response": "", "messages": []}












def test_ensure_session_db_row_persists_explicit_cwd(monkeypatch, tmp_path):
    """An explicitly chosen workspace is persisted as the session cwd."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path), "explicit_cwd": True})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "model_config": None, "cwd": str(tmp_path)}
    ]












































































































def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "low",
            },
        }
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert server._sessions["sid"]["create_reasoning_override"] == {"enabled": True, "effort": "low"}
    assert server._load_cfg()["agent"]["reasoning_effort"] == "medium"

    resp_status = server.handle_request(
        {
            "id": "5",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_status["result"]["value"] == "low"

    resp_global_status = server.handle_request(
        {"id": "6", "method": "config.get", "params": {"key": "reasoning"}}
    )
    assert resp_global_status["result"]["value"] == "medium"

    del server._sessions["sid"]["create_reasoning_override"]
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    resp_agent_status = server.handle_request(
        {
            "id": "7",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_agent_status["result"]["value"] == "high"

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

    # /reasoning full | clamp — parity with the classic CLI reasoning_full
    # toggle. In the TUI these map to the thinking section's expand/collapse
    # rendering (no fixed 10-line recap exists here).
    resp_full = server.handle_request(
        {
            "id": "4",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "full"},
        }
    )
    assert resp_full["result"]["value"] == "full"
    cfg_full = server._load_cfg()
    assert cfg_full["display"]["reasoning_full"] is True
    assert cfg_full["display"]["sections"]["thinking"] == "expanded"

    resp_clamp = server.handle_request(
        {
            "id": "5",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "clamp"},
        }
    )
    assert resp_clamp["result"]["value"] == "clamp"
    cfg_clamp = server._load_cfg()
    assert cfg_clamp["display"]["reasoning_full"] is False
    assert cfg_clamp["display"]["sections"]["thinking"] == "collapsed"








































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
























# ---------------------------------------------------------------------------
# History-mutating commands must reject while session.running is True.
# Without these guards, prompt.submit's post-run history write either
# clobbers the mutation (version matches) or silently drops the agent's
# output (version mismatch) — both produce UI<->backend state desync.
# ---------------------------------------------------------------------------












def test_prompt_submit_sanitizes_bracketed_paste_before_agent(monkeypatch):
    """prompt.submit must sanitize corrupted user text before run_conversation."""
    captured: dict[str, str] = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            captured["prompt"] = prompt
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    corrupted = "hello[" + "~[[e" * 8
    server._sessions["sid"] = _session(agent=_Agent())
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *a, **k: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *a, **k: None)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": corrupted},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert captured["prompt"] == "hello"
    finally:
        server._sessions.pop("sid", None)






# ---------------------------------------------------------------------------
# session.interrupt must only cancel pending prompts owned by the calling
# session — it must not blast-resolve clarify/sudo/secret prompts on
# unrelated sessions sharing the same tui_gateway process.  Without
# session scoping the other sessions' prompts silently resolve to empty
# strings, unblocking their agent threads as if the user cancelled.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# session.interrupt must only cancel pending prompts owned by the calling
# session — it must not blast-resolve clarify/sudo/secret prompts on
# unrelated sessions sharing the same tui_gateway process.  Without
# session scoping the other sessions' prompts silently resolve to empty
# strings, unblocking their agent threads as if the user cancelled.
# ---------------------------------------------------------------------------










def test_interrupt_before_agent_ready_prevents_late_turn_start(monkeypatch):
    """Stop during lazy agent startup must not start the turn after init finishes."""
    threads = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(server, "_wait_agent", lambda session, rid: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"
        assert session["running"] is True
        assert len(threads) == 1

        stop = server.handle_request(
            {"id": "2", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )
        assert stop.get("result"), f"got error: {stop.get('error')}"

        threads[0].target()

        assert calls["run_prompt"] == 0
        assert session["running"] is False
        assert session.get("inflight_turn") is None
    finally:
        server._sessions.pop("sid", None)
























# ---------------------------------------------------------------------------
# /model switch and other agent-mutating commands must reject while the
# session is running.  agent.switch_model() mutates self.model, self.provider,
# self.base_url, self.client etc. in place — the worker thread running
# agent.run_conversation is reading those on every iteration.  Same class of
# bug as the session.undo / session.compress mid-run silent-drop; same fix
# pattern: reject with 4009 while running.
# ---------------------------------------------------------------------------












_PARTIAL_FAKE_HISTORY = [
    {"role": "user", "content": "msg1"},
    {"role": "assistant", "content": "resp1"},
    {"role": "user", "content": "msg2"},
    {"role": "assistant", "content": "resp2"},
    {"role": "user", "content": "keep this"},
    {"role": "assistant", "content": "keep this too"},
]
_PARTIAL_COMPRESSED_HEAD = [
    {"role": "user", "content": "[summary]"},
    {"role": "assistant", "content": "ok"},
]


def _partial_compress_agent(compress_context_calls):
    """Agent stub whose _compress_context records (history, focus_topic)."""
    agent = types.SimpleNamespace(
        _cached_system_prompt=None,
        tools=None,
        session_id="s1",
        context_compressor=None,  # keep _get_usage on the simple path
    )

    def _fake_compress_context(history, sys, approx_tokens=0, focus_topic=None, **kw):
        compress_context_calls.append((list(history), focus_topic))
        return list(_PARTIAL_COMPRESSED_HEAD), {}

    agent._compress_context = _fake_compress_context
    return agent














# ---------------------------------------------------------------------------
# session.create / session.close race: fast /new churn must not orphan the
# global approval-notify registration. (Slash workers are no longer pre-warmed
# by the build thread — slash.exec spawns them on demand — so the build thread
# must ALSO never construct one here.)
# ---------------------------------------------------------------------------


@pytest.mark.real_agent_prewarm
def test_session_create_close_race_does_not_orphan_worker(monkeypatch):
    """Regression guard: if session.close runs while session.create's
    _build thread is still constructing the agent, the build thread
    must detect the orphan and unregister the notify registration it's
    about to install.  It must also never pre-warm a slash worker (each
    worker forks the full stdio MCP fleet; spawn is on-demand in
    slash.exec) — a worker constructed here would be a regression."""
    import threading

    created_workers: list[str] = []
    closed_workers: list[str] = []
    unregistered_keys: list[str] = []

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key
            self._closed = False
            created_workers.append(key)

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

    def _slow_make_agent(sid, key, session_id=None, session_db=None, **_kwargs):
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

    # At this point session.close saw slash_worker=None (never eagerly
    # installed) so it had nothing to close.  Release the build thread
    # and let it finish — it should detect the orphan and unregister
    # the notify, without ever having constructed a worker.
    release_build.set()

    # Give the build thread a moment to run through its finally.
    for _ in range(100):
        if unregistered_keys:
            break
        import time

        time.sleep(0.02)

    assert created_workers == [], (
        f"build thread pre-warmed a slash worker (spawn must stay on-demand "
        f"in slash.exec) — created_workers={created_workers}"
    )
    # Notify may be unregistered by both session.close (unconditional)
    # and the orphan-cleanup path; the key guarantee is that the build
    # thread does at least one unregister call (any prior close
    # already popped the callback; the duplicate is a no-op).
    assert len(unregistered_keys) >= 1, (
        f"orphan notify registration was not unregistered — "
        f"unregistered_keys={unregistered_keys}"
    )












# --------------------------------------------------------------------------
# session.delete — TUI resume picker `d` key
# --------------------------------------------------------------------------






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












# --------------------------------------------------------------------------
# session.* profile scoping (app-global remote mode) — #62503
# --------------------------------------------------------------------------






















# --------------------------------------------------------------------------
# model.options — curated-list parity with `hermes model` and classic /model
# --------------------------------------------------------------------------












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
        model = "gpt-5.6-sol"
        provider = "openai-codex"
        base_url = "https://chatgpt.example.test/backend-api/codex"
        api_key = object()
        api_mode = "codex_responses"

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
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
    assert mock_title.call_args.kwargs["main_runtime"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.example.test/backend-api/codex",
        "api_key": _Agent.api_key,
        "api_mode": "codex_responses",
    }










# ── active live TUI sessions ─────────────────────────────────────────







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

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
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






# ── session.most_recent ──────────────────────────────────────────────










# ── verification.status ──────────────────────────────────────────────






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
    assert resp["result"]["messages"] == [
        "Chromium-family browser is already listening at http://127.0.0.1:9222"
    ]
    assert os.environ.get("BROWSER_CDP_URL") == "http://127.0.0.1:9222"
    # First cleanup runs against the OLD env (none here), second against the NEW.
    assert cleanup_calls == ["", "http://127.0.0.1:9222"]




































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
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})


def test_make_agent_reads_nested_max_turns(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {"agent": {"max_turns": 200}})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 200












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












def test_notification_poller_delivers_completion(monkeypatch):
    """Poller picks up completion events and triggers agent turns."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    turns = []
    emitted = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
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

    # Isolate the completion queue for the duration of this test. The poller
    # reads process_registry.completion_queue by attribute at runtime; the
    # event below carries no session_key, so any *other* poller (a leaked
    # daemon thread from another test, or a concurrent one in the same xdist
    # worker) is allowed to dequeue and dispatch it to its own session — whose
    # agent may be a fixture double without run_conversation. A fresh Queue
    # here fully isolates this test; monkeypatch restores the original on
    # teardown. (Same pattern as test_notification_poller_requeues_when_busy.)
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_poller_test")

    stop = threading.Event()

    # Put event on queue, then immediately signal stop so the poller
    # runs exactly one iteration.
    isolated_queue.put({
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
        assert "[IMPORTANT: Background process proc_poller_test completed normally" in turns[0]
    finally:
        server._sessions.pop("sid_poll", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()












def test_notification_poller_emits_distinct_watch_matches_once(monkeypatch):
    """Distinct watch matches from one process emit; exact replay is deduped."""
    import queue as _queue_mod

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

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)

    base = {
        "type": "watch_match",
        "session_id": "proc_watch_dedup",
        "command": "tail -f app.log",
        "pattern": "READY",
        "output": "READY on port 8000",
        "suppressed": 0,
    }
    isolated_queue.put(base)
    isolated_queue.put({**base, "output": "READY on port 9000"})
    isolated_queue.put(dict(base))

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
















def test_session_close_rpc_claims_then_tears_down(monkeypatch):
    seen = []
    claimed = {"session_key": "k"}
    monkeypatch.setattr(server, "_pop_session_by_id", lambda sid: seen.append(sid) or claimed)
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: seen.append((session, end_reason)) or True,
    )
    resp = server.handle_request(
        {"id": "1", "method": "session.close", "params": {"session_id": "s9"}}
    )
    assert resp["result"] == {"closed": True}
    assert seen == ["s9", (claimed, "tui_close")]


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


def test_session_create_records_ui_model_as_session_override(monkeypatch):
    """The desktop composer owns its model as plain UI state and ships it on
    session.create. The gateway must record it as a PER-SESSION override (built
    into the agent), never a global config write — picking a model for a new chat
    must not mutate the profile default.
    """
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    # Don't run the real deferred build in this storage-focused test.
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    try:
        resp = server._methods["session.create"](
            "r1",
            {
                "cols": 80,
                "model": "claude-sonnet-4.6",
                "provider": "anthropic",
                "reasoning_effort": "high",
                "fast": True,
            },
        )
        sid = resp["result"]["session_id"]
        sess = server._sessions[sid]
        assert sess["model_override"] == {"model": "claude-sonnet-4.6", "provider": "anthropic"}
        assert sess["create_reasoning_override"] is not None
        assert sess["create_service_tier_override"] == "priority"
        # The immediate response reflects the override (not the global default) so
        # the client never clobbers its sticky pick before the build lands.
        assert resp["result"]["info"]["model"] == "claude-sonnet-4.6"
        assert resp["result"]["info"]["provider"] == "anthropic"

        # Explicit false is not the same as omission: it must suppress a Fast
        # profile default for this session's first request.
        normal = server._methods["session.create"](
            "r2", {"cols": 80, "fast": False}
        )
        normal_sess = server._sessions[normal["result"]["session_id"]]
        assert normal_sess["create_service_tier_override"] == ""

        # No knobs → no overrides; the session builds from the profile default.
        plain = server._methods["session.create"]("r3", {"cols": 80})
        plain_sess = server._sessions[plain["result"]["session_id"]]
        assert plain_sess["model_override"] is None
        assert plain_sess["create_reasoning_override"] is None
        assert plain_sess["create_service_tier_override"] is None
    finally:
        server._sessions.clear()




# ── billing/subscription state + error serialization ─────────────────




@pytest.mark.parametrize(
    "card,expected",
    [
        ("canonical", {"kind": "canonical"}),
        (
            "distinct",
            {
                "kind": "distinct",
                "payment_method_id": "pm_auto",
                "brand": None,
                "last4": None,
            },
        ),
        ("none", {"kind": "none"}),
    ],
)
def test_billing_state_serializes_auto_reload_card_union(monkeypatch, card, expected):
    from agent.billing_view import AutoReload, AutoReloadCard, BillingState

    monkeypatch.setattr(server, "_usage_payload", lambda state: {"available": False})
    auto_reload_card = AutoReloadCard(
        kind=card,
        payment_method_id="pm_auto" if card == "distinct" else None,
    )
    state = BillingState(
        logged_in=True,
        auto_reload=AutoReload(enabled=True, card=auto_reload_card),
    )

    result = server._serialize_billing_state(state)

    assert result["auto_reload"]["card"] == expected




class _BillingHeaders:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)






# ── subscription change RPCs (V3): preview + pending-change + upgrade ──


def _sub_rpc(method, params):
    # These RPCs are in _LONG_HANDLERS (pool-routed → dispatch returns None and the
    # worker writes via the transport), so drive the inline handler directly.
    return server.handle_request({"id": "1", "method": method, "params": params})["result"]


def test_subscription_preview_serializes_quote(monkeypatch):
    import hermes_cli.nous_billing as nb

    monkeypatch.setattr(
        nb,
        "post_subscription_preview",
        lambda subscription_type_id: {
            "effect": "charge_now",
            "reason": None,
            "currentTierId": "plus",
            "currentTierName": "Plus",
            "targetTierId": "ultra",
            "targetTierName": "Ultra",
            "monthlyCreditsDelta": "6000",
            "amountDueNowCents": 1234,
            "effectiveAt": None,
        },
    )
    res = _sub_rpc("subscription.preview", {"subscription_type_id": "ultra"})
    assert res["ok"] is True
    assert res["effect"] == "charge_now"
    assert res["amount_due_now_cents"] == 1234
    assert res["target_tier_name"] == "Ultra"
    assert res["monthly_credits_delta"] == "6000"














def test_subscription_upgrade_echoes_status_and_idempotency(monkeypatch):
    import hermes_cli.nous_billing as nb

    seen = {}

    def _upgrade(*, subscription_type_id, idempotency_key):
        seen["key"] = idempotency_key
        return {"status": "upgraded", "targetTierId": "ultra", "targetTierName": "Ultra"}

    monkeypatch.setattr(nb, "post_subscription_upgrade", _upgrade)
    res = _sub_rpc("subscription.upgrade", {"subscription_type_id": "ultra", "idempotency_key": "k-1"})
    assert res["ok"] is True
    assert res["status"] == "upgraded"
    assert res["target_tier_name"] == "Ultra"
    assert res["idempotency_key"] == "k-1"
    assert seen["key"] == "k-1"


# ── _get_usage active_subagents (TUI status-bar ⛓ indicator) ──────────────
# Mirrors the classic CLI status bar: _get_usage embeds a live count of
# background/async subagents from tools.async_delegation.active_count() so the
# Ink status bar can render ⛓ N. Source of truth is the same registry the CLI
# reads; the field rides the existing per-update `usage` payload.


class _BareAgent:
    """Agent stub with no compressor — exercises the active_subagents path
    independent of the `if comp:` context-percent block."""

    model = "x"


def test_get_usage_includes_active_subagents(monkeypatch):
    import tools.async_delegation as ad_mod
    monkeypatch.setattr(ad_mod, "active_count", lambda: 4)
    usage = server._get_usage(_BareAgent())
    assert usage["active_subagents"] == 4










# ---------------------------------------------------------------------------
# _resolve_runtime_with_fallback — init-time provider fallback
# ---------------------------------------------------------------------------

class TestResolveRuntimeWithFallback:
    """Tests for _resolve_runtime_with_fallback(): init-time provider
    fallback when the primary provider raises AuthError."""

    def test_primary_success_returns_runtime(self, monkeypatch):
        """When primary resolve succeeds, return its result directly."""
        expected = {"provider": "openai", "api_key": "tok"}
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: expected,
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai"}
        )
        assert resolution.runtime == expected
        assert resolution.selected_model is None
        assert resolution.used_fallback is False

    def test_auth_error_tries_fallback_chain(self, monkeypatch):
        """On AuthError from primary, walk fallback_providers chain."""
        from hermes_cli.auth import AuthError

        fallback_runtime = {"provider": "deepseek", "api_key": "fb-tok"}

        def fake_resolve(**kwargs):
            if kwargs.get("requested") == "openai-codex":
                raise AuthError("No Codex credentials stored")
            return fallback_runtime

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [{"provider": "deepseek", "model": "deepseek-v4-pro"}],
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai-codex"},
        )
        assert resolution.runtime == fallback_runtime
        assert resolution.selected_model == "deepseek-v4-pro"
        assert resolution.used_fallback is True













# ---------------------------------------------------------------------------
# Streaming TTS — per-turn pipeline + barge-in
# ---------------------------------------------------------------------------

def _fake_tts_modules(monkeypatch, *, requirements=True, playback_stops=None, listen=None, transcribe=None):
    """Install lightweight tools.tts_tool / tools.voice_mode fakes."""
    started = {}

    def fake_stream(text_queue, stop, done, **_kw):
        started["queue"] = text_queue
        stop.wait(5)
        done.set()

    def default_listen(should_stop, capture=False, on_trigger=None, **_kw):
        return None if capture else False

    def default_fd_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        return None

    monkeypatch.setitem(
        sys.modules,
        "tools.tts_tool",
        types.SimpleNamespace(
            check_tts_requirements=lambda: requirements,
            stream_tts_to_speaker=fake_stream,
            _get_provider=lambda cfg: "edge",
            _load_tts_config=lambda: {},
            get_env_value=lambda key, default="": default,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            stop_playback=lambda: (playback_stops.append(True) if playback_stops is not None else None),
            listen_for_speech=listen or default_listen,
            full_duplex_listen=listen or default_fd_listen,
            is_audio_output_active=lambda: False,
            transcribe_recording=transcribe or (lambda path, model=None: {"success": True, "transcript": ""}),
        ),
    )
    # Fresh listener slot per test — the arm is idempotent per process.
    monkeypatch.setattr(server, "_fd_listener_active", False)
    return started


def test_tts_stream_begin_requires_voice_tts(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")
    assert server._tts_stream_begin() is None




def test_tts_stream_begin_and_stop_lifecycle(monkeypatch):
    """begin() spawns the consumer; stop() cuts it and clears the slot."""
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "0")  # no barge-in monitor (no mic)
    playback_stops: list = []
    started = _fake_tts_modules(monkeypatch, playback_stops=playback_stops)

    text_queue = server._tts_stream_begin()
    assert text_queue is not None
    assert started["queue"] is text_queue

    with server._tts_stream_lock:
        state = server._tts_stream_state
    assert state is not None and not state["stop"].is_set()

    server._tts_stream_stop()
    assert state["stop"].is_set()
    assert playback_stops == [True]
    with server._tts_stream_lock:
        assert server._tts_stream_state is None








def test_tts_stream_vad_barge_in_cuts_pipeline_and_submits_capture(monkeypatch, tmp_path):
    """User speech during playback cuts TTS at the moment of detection
    (voice.interrupted), then the captured interruption is transcribed and
    emitted as voice.transcript so the TUI submits it — complete from its
    first syllable, no re-record round trip. The cut also latches the
    speech-interrupted note for the next turn."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "barge.wav"
    wav.write_bytes(b"RIFF")

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        on_trigger("playback")  # playback cut happens at detection
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "stop, actually—"},
    )

    server._tts_stream_begin()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and wav.exists():
        time.sleep(0.01)  # unlink (finally) runs after the transcript emit
    assert ("voice.interrupted", None) in events
    assert ("voice.transcript", {"text": "stop, actually—"}) in events
    assert not wav.exists()  # capture temp file cleaned up
    assert ts.take_speech_interrupted() is True  # VAD cut latches the model note
    server._tts_stream_stop()


def test_full_duplex_generation_phase_interrupts_running_turn(monkeypatch, tmp_path):
    """Speech DURING LLM generation (no TTS audio yet) must interrupt the
    in-flight agent turn via the same seam session.interrupt uses, and the
    captured interjection is emitted as voice.transcript. This is the
    half-duplex gap: previously no listener existed until playback started."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "interject.wav"
    wav.write_bytes(b"RIFF")

    interrupted = threading.Event()
    fake_agent = types.SimpleNamespace(interrupt=lambda: interrupted.set())
    fake_session = {"running": True, "agent": fake_agent}
    monkeypatch.setattr(server, "_sessions", {"sid-fd": fake_session})

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        assert is_playing is not None and is_playing() is False  # generation phase
        on_trigger("generation")
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "wait, try another way"},
    )

    server._arm_full_duplex_listener()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and wav.exists():
        time.sleep(0.01)
    assert interrupted.is_set()  # the running turn was interrupted
    assert ("voice.interrupted", None) in events
    assert ("voice.transcript", {"text": "wait, try another way"}) in events
    assert not wav.exists()














def test_build_persist_message_with_image_refs_without_images_returns_text(monkeypatch):
    """#70720: when no images are attached the persisted message is the raw
    prompt — no @image directive prefix is introduced."""
    assert server._build_persist_message_with_image_refs("what is this?", []) == "what is this?"
    assert server._build_persist_message_with_image_refs("", []) == ""


















