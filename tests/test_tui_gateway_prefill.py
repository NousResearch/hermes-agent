"""Tests for TUI gateway prefill messages support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import tui_gateway.server as server


def test_tui_load_prefill_messages_top_level(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    prefill_data = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    prefill_file = tmp_path / "prefill.json"
    prefill_file.write_text(json.dumps(prefill_data), encoding="utf-8")

    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(prefill_file)})
    loaded = server._load_prefill_messages()
    assert loaded == prefill_data


def test_tui_load_prefill_messages_legacy_agent_key(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    prefill_data = [{"role": "system", "content": "persona"}]
    prefill_file = tmp_path / "legacy.json"
    prefill_file.write_text(json.dumps(prefill_data), encoding="utf-8")

    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"prefill_messages_file": str(prefill_file)}})
    loaded = server._load_prefill_messages()
    assert loaded == prefill_data


def test_tui_load_prefill_messages_prefers_env(monkeypatch, tmp_path):
    env_data = [{"role": "user", "content": "from env"}]
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps(env_data), encoding="utf-8")

    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps([{"role": "user", "content": "from cfg"}]), encoding="utf-8")

    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", str(env_file))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(cfg_file)})

    loaded = server._load_prefill_messages()
    assert loaded == env_data


def test_tui_load_prefill_messages_missing_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(tmp_path / "nonexistent.json")})

    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_invalid_json(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(bad_file)})

    assert server._load_prefill_messages() == []


def test_tui_make_agent_injects_prefill(monkeypatch, tmp_path):
    prefill_data = [{"role": "system", "content": "DR.TEST"}]
    prefill_file = tmp_path / "prefill.json"
    prefill_file.write_text(json.dumps(prefill_data), encoding="utf-8")

    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(prefill_file)})
    monkeypatch.setattr(
        server,
        "_resolve_runtime_with_fallback",
        lambda *args, **kwargs: SimpleNamespace(
            runtime={"provider": "custom", "base_url": "http://localhost", "api_key": "test"},
            used_fallback=False,
            selected_model=None,
        ),
    )

    captured_kwargs = {}
    def mock_agent(**kwargs):
        captured_kwargs.update(kwargs)
        mock = MagicMock()
        mock.prefill_messages = kwargs.get("prefill_messages")
        return mock

    monkeypatch.setattr("run_agent.AIAgent", mock_agent)

    agent = server._make_agent("test_sid", "test_key")
    assert captured_kwargs.get("prefill_messages") == prefill_data
    assert agent.prefill_messages == prefill_data


def test_tui_background_agent_kwargs_injects_prefill(monkeypatch, tmp_path):
    prefill_data = [{"role": "system", "content": "DR.BG"}]
    prefill_file = tmp_path / "prefill.json"
    prefill_file.write_text(json.dumps(prefill_data), encoding="utf-8")

    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"prefill_messages_file": str(prefill_file)})

    class FakeAgent:
        pass

    kwargs = server._background_agent_kwargs(FakeAgent(), "test_task")
    assert kwargs.get("prefill_messages") == prefill_data
