"""Regression tests for Desktop/TUI prefill messages injection (tui_gateway.server).

Desktop agents are built in ``_make_agent`` (never through the CLI's config bootstrap), so the
CLI-side prefill resolution never runs for them and a configured ``prefill_messages_file`` was
silently ignored (#60456). These tests pin the two properties that make the Desktop injection
work:

1. ``_load_prefill_messages`` resolves the file from env → top-level config → legacy
   ``agent.*`` key, with CLI precedence.
2. ``_make_agent`` passes ``prefill_messages`` through to ``AIAgent``.
"""

from __future__ import annotations

import json

from tui_gateway import server


# ── _load_prefill_messages resolution ──────────────────────────────────────

def test_tui_load_prefill_messages_from_env(monkeypatch, tmp_path):
    data = [{"role": "user", "content": "from env"}]
    f = tmp_path / "env.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", str(f))
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {})
    assert server._load_prefill_messages() == data


def test_tui_load_prefill_messages_top_level(monkeypatch, tmp_path):
    data = [{"role": "system", "content": "top"}]
    f = tmp_path / "top.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(f)})
    assert server._load_prefill_messages() == data


def test_tui_load_prefill_messages_legacy_agent_key(monkeypatch, tmp_path):
    data = [{"role": "user", "content": "legacy"}]
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(
        "tui_gateway.server._load_cfg", lambda: {"agent": {"prefill_messages_file": str(f)}})
    assert server._load_prefill_messages() == data


def test_tui_load_prefill_messages_env_wins_over_cfg(monkeypatch, tmp_path):
    env_data = [{"role": "user", "content": "env"}]
    cfg_data = [{"role": "user", "content": "cfg"}]
    env_f = tmp_path / "env.json"
    cfg_f = tmp_path / "cfg.json"
    env_f.write_text(json.dumps(env_data), encoding="utf-8")
    cfg_f.write_text(json.dumps(cfg_data), encoding="utf-8")
    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", str(env_f))
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(cfg_f)})
    assert server._load_prefill_messages() == env_data


def test_tui_load_prefill_messages_unset_is_empty(monkeypatch):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {})
    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_missing_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(
        "tui_gateway.server._load_cfg",
        lambda: {"prefill_messages_file": str(tmp_path / "nope.json")})
    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_non_array_is_empty(monkeypatch, tmp_path):
    f = tmp_path / "obj.json"
    f.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(f)})
    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_relative_to_hermes_home(monkeypatch, tmp_path):
    data = [{"role": "user", "content": "relative"}]
    (tmp_path / "prefill.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": "prefill.json"})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path, raising=True)
    assert server._load_prefill_messages() == data


def test_tui_load_prefill_messages_bad_json_is_empty(monkeypatch, tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(f)})
    assert server._load_prefill_messages() == []


# ── _make_agent passes prefill_messages through ────────────────────────────

def _agent_kwargs(monkeypatch):
    """Call the real ``_make_agent`` with every seam mocked; return the AIAgent kwargs."""
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    for mod, name, fn in (
        ("tui_gateway.synthetic_turn", "maybe_build_synthetic_agent", lambda *a, **k: None),
        ("agent.shell_hooks", "register_from_config", lambda cfg: None),
    ):
        monkeypatch.setattr(f"{mod}.{name}", fn, raising=False)
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda cfg, task_id: "", raising=False)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda m, p: ("m", {}), raising=False)
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {}, raising=False)
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda p: "cli", raising=False)
    monkeypatch.setattr(server, "_get_db", lambda: None, raising=False)
    monkeypatch.setattr(server, "_load_fallback_model", lambda: None, raising=False)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {}, raising=False)
    monkeypatch.setattr(server, "_cfg_max_turns", lambda cfg, d: 500, raising=False)
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None, raising=False)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None, raising=False)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: None, raising=False)
    monkeypatch.setattr(server, "_context_cwd_is_launch_artifact", lambda session: False, raising=False)
    server._make_agent("sid", "key")
    return captured


def test_make_agent_passes_prefill_messages(monkeypatch, tmp_path):
    """The factory must forward the resolved prefill messages to AIAgent (#60456)."""
    sentinel = [{"role": "user", "content": "prefill"}]
    monkeypatch.setattr(server, "_load_prefill_messages", lambda: sentinel)
    assert _agent_kwargs(monkeypatch).get("prefill_messages") == sentinel


def test_make_agent_forwards_empty_when_unset(monkeypatch):
    assert _agent_kwargs(monkeypatch).get("prefill_messages") == []
