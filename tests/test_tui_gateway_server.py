import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

from tui_gateway import server


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
        types.SimpleNamespace(discover_plugins=lambda: discovered.update({"ready": True})),
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
    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}})

    assert server._load_enabled_toolsets() == ["memory"]
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

    monkeypatch.setattr(config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}})

    assert server._load_enabled_toolsets() == ["memory"]
    assert "using configured CLI toolsets" in capsys.readouterr().err


def test_load_enabled_toolsets_warns_when_config_fallback_fails(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert server._load_enabled_toolsets() is None
    assert "could not be loaded" in capsys.readouterr().err


def test_load_enabled_toolsets_honors_builtin_env_if_config_fails(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web")

    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert server._load_enabled_toolsets() == ["web"]


def test_load_enabled_toolsets_all_env_means_all(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "all")

    assert server._load_enabled_toolsets() is None


def test_load_enabled_toolsets_all_env_warns_about_ignored_extra_entries(monkeypatch, capsys):
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
        lambda agent: {"model": "test", "tools": {}, "skills": {}},
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
    unblock_agent = threading.Event()

    class _FakeWorker:
        def __init__(self, key, model):
            self.key = key

        def close(self):
            return None

    class _FakeAgent:
        model = "x"
        provider = "openrouter"
        base_url = ""
        api_key = ""

    class _FakeDB:
        def create_session(self, _key, source="tui", model=None):
            return None

        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    def _make_agent(_sid, _key):
        unblock_agent.wait(timeout=2.0)
        return _FakeAgent()

    monkeypatch.setattr(server, "_make_agent", _make_agent)
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_session_info", lambda _a: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    session["pending_title"] = "duplicate title"
    unblock_agent.set()
    session["agent_ready"].wait(timeout=2.0)

    assert session["pending_title"] is None
    server._sessions.pop(sid, None)


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
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})
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
    assert set_toggle["result"] == {"key": "mouse", "value": "on"}
    assert writes == [("display.mouse_tracking", True)]

    cfg["display"] = {"mouse_tracking": 0, "tui_mouse": True}
    get_canonical = server.handle_request(
        {"id": "3", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_canonical["result"]["value"] == "off"

    cfg["display"] = {"mouse_tracking": None, "tui_mouse": False}
    get_null = server.handle_request(
        {"id": "4", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_null["result"]["value"] == "on"


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


def test_complete_slash_includes_provider_alias():
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/pro"}}
    )

    assert any(item["text"] == "provider" for item in resp["result"]["items"])


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


def test_config_set_model_uses_live_switch_path(monkeypatch):
    server._sessions["sid"] = _session()
    seen = {}

    def _fake_apply(sid, session, raw):
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
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
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


def test_config_set_model_syncs_inference_provider_env(monkeypatch):
    """After an explicit provider switch, HERMES_INFERENCE_PROVIDER must
    reflect the user's choice so ambient re-resolution (credential pool
    refresh, aux clients) picks up the new provider instead of the original
    one persisted in config or shell env.

    Regression: a TUI user switched openrouter → anthropic and the TUI kept
    trying openrouter because the env-var-backed resolvers still saw the old
    provider.
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

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

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

    assert os.environ["HERMES_INFERENCE_PROVIDER"] == "anthropic"


def test_config_set_model_syncs_tui_provider_unconditionally(monkeypatch):
    """Regression for #16857: /model must set HERMES_TUI_PROVIDER even when
    it wasn't pre-set on launch, so a later /new (which re-runs
    _resolve_startup_runtime) honours the user's explicit provider choice
    instead of falling through to static-catalog detection and picking a
    coincidentally-matching native provider.
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

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

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

    # Both env vars must reflect the user's choice. HERMES_TUI_PROVIDER is
    # the canonical explicit-this-process carrier consumed by
    # _resolve_startup_runtime() on /new.
    assert os.environ["HERMES_TUI_PROVIDER"] == "custom:xuanji"
    assert os.environ["HERMES_INFERENCE_PROVIDER"] == "custom:xuanji"


def test_config_set_model_syncs_tui_provider_env(monkeypatch):
    class Agent:
        model = "gpt-5.3-codex"
        provider = "openai-codex"
        base_url = ""
        api_key = ""

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]

    agent = Agent()
    server._sessions["sid"] = _session(agent=agent)
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "openai-codex")
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
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
        assert os.environ["HERMES_TUI_PROVIDER"] == "anthropic"
        assert os.environ["HERMES_MODEL"] == "anthropic/claude-sonnet-4.6"
        assert os.environ["HERMES_INFERENCE_MODEL"] == "anthropic/claude-sonnet-4.6"
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


def test_config_set_personality_resets_history_and_returns_info(monkeypatch):
    session = _session(
        agent=types.SimpleNamespace(),
        history=[{"role": "user", "text": "hi"}],
        history_version=4,
    )
    new_agent = types.SimpleNamespace(model="x")
    emits = []

    server._sessions["sid"] = session
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    monkeypatch.setattr(
        server, "_make_agent", lambda sid, key, session_id=None: new_agent
    )
    monkeypatch.setattr(
        server, "_session_info", lambda agent: {"model": getattr(agent, "model", "?")}
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(server, "_write_config_key", lambda path, value: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "personality", "value": "helpful"},
        }
    )

    assert resp["result"]["history_reset"] is True
    assert resp["result"]["info"] == {"model": "x"}
    assert session["history"] == []
    assert session["history_version"] == 5
    assert ("session.info", "sid", {"model": "x"}) in emits


def test_session_compress_uses_compress_helper(monkeypatch):
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server,
        "_compress_session_history",
        lambda session, focus_topic=None, **_kw: (2, {"total": 42}),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})

    with patch("tui_gateway.server._emit") as emit:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )

    assert resp["result"]["removed"] == 2
    assert resp["result"]["usage"]["total"] == 42
    emit.assert_any_call("session.info", "sid", {"model": "x"})
    # Final status.update clears the pinned "compressing" indicator so the
    # status bar can revert to the neutral state when compaction finishes.
    emit.assert_any_call(
        "status.update", "sid", {"kind": "status", "text": "ready"}
    )


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
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})
    restart_calls = []
    monkeypatch.setattr(
        server, "_restart_slash_worker", lambda s: restart_calls.append(s)
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

    info = server._session_info(types.SimpleNamespace(tools=[], model=""))

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

    def _fake_apply(sid, session, raw):
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

    def _fake_apply(sid, session, raw):
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
    monkeypatch.setattr(server, "_session_info", lambda _agent: {"model": "x"})
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

    def _slow_make_agent(sid, key, session_id=None):
        build_started.set()
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
    monkeypatch.setattr(server, "_session_info", lambda _a: {"model": "x"})
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

    monkeypatch.setattr(server, "_make_agent", lambda sid, key: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(create_session=lambda *a, **kw: None),
    )
    monkeypatch.setattr(server, "_session_info", lambda _a: {"model": "x"})
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

    monkeypatch.setattr(server, "_make_agent", lambda sid, key: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_info", lambda _a: {"model": "x"})
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
    assert resp["result"]["messages"] == ["Chrome is already listening on port 9222"]
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
    assert resp["result"]["messages"] == ["Chrome is already listening on port 9222"]
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
        == "Chrome isn't running with remote debugging — attempting to launch..."
    )
    assert any(
        "No Chrome/Chromium executable was found" in line
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
        "Chrome isn't running with remote debugging — attempting to launch...",
        "Chrome launched and listening on port 9222",
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
        resp = server.handle_request(
            {"id": "1", "method": "reload.env", "params": {}}
        )

    assert resp["result"] == {"updated": 7}
    assert calls["n"] == 1


def test_reload_env_rpc_surfaces_errors(monkeypatch):
    def _broken():
        raise RuntimeError("env path locked")

    fake = types.SimpleNamespace(reload_env=_broken)
    with patch.dict(sys.modules, {"hermes_cli.config": fake}):
        resp = server.handle_request(
            {"id": "1", "method": "reload.env", "params": {}}
        )

    assert "error" in resp
    assert "env path locked" in resp["error"]["message"]
