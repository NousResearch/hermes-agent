"""Regression tests for Desktop/TUI prefill messages injection (tui_gateway.server).

Desktop agents are built in ``_make_agent`` (never through ``CLIAgentSetupMixin``), so the
CLI-side prefill resolution never runs for them. These tests pin the two properties that
make the Desktop injection work:

1. ``_load_prefill_messages`` resolves the file from env → top-level config → legacy
   ``agent.*`` key, mirroring ``cli._load_prefill_messages``.
2. ``_make_agent`` passes ``prefill_messages`` through to ``AIAgent``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


def test_tui_load_prefill_messages_no_file_returns_empty(monkeypatch):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {})
    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_invalid_json_returns_empty(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(bad)})
    assert server._load_prefill_messages() == []


def test_tui_load_prefill_messages_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(
        "tui_gateway.server._load_cfg",
        lambda: {"prefill_messages_file": str(tmp_path / "nope.json")})
    assert server._load_prefill_messages() == []


# ── shared resolver / loader (hermes_cli.prefill_messages) ─────────────────

def test_shared_resolver_env_wins_over_config(monkeypatch):
    from hermes_cli import prefill_messages as shared

    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", "env.json")
    assert shared.resolve_prefill_messages_file(
        {"prefill_messages_file": "cfg.json"}) == "env.json"


def test_shared_resolver_top_level_then_legacy_then_empty(monkeypatch):
    from hermes_cli import prefill_messages as shared

    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    assert shared.resolve_prefill_messages_file(
        {"prefill_messages_file": "top.json",
         "agent": {"prefill_messages_file": "legacy.json"}}) == "top.json"
    assert shared.resolve_prefill_messages_file(
        {"agent": {"prefill_messages_file": "legacy.json"}}) == "legacy.json"
    assert shared.resolve_prefill_messages_file({}) == ""


def test_shared_resolver_reads_active_config_when_not_passed(monkeypatch):
    """No config argument: the resolver reads the active profile's config itself."""
    from hermes_cli import prefill_messages as shared

    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"prefill_messages_file": "from-config.json"})
    assert shared.resolve_prefill_messages_file() == "from-config.json"


def test_shared_loader_follows_hermes_home_at_call_time(monkeypatch, tmp_path):
    """Profile safety: relative configured paths resolve against HERMES_HOME per call."""
    from hermes_cli import prefill_messages as shared

    data = [{"role": "system", "content": "profile-a"}]
    (tmp_path / "prefill.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert shared.load_configured_prefill_messages(
        {"prefill_messages_file": "prefill.json"}) == data

    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert shared.load_configured_prefill_messages(
        {"prefill_messages_file": "prefill.json"}) == []


def test_shared_loader_explicit_base_dir_overrides_home(monkeypatch, tmp_path):
    from hermes_cli import prefill_messages as shared

    data = [{"role": "user", "content": "gateway-home"}]
    base = tmp_path / "gateway-home"
    base.mkdir()
    (base / "prefill.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert shared.load_prefill_messages("prefill.json", base_dir=base) == data


# ── _make_agent injects prefill_messages into AIAgent ──────────────────────

def _make_agent_capturing_prefill(monkeypatch):
    """Build an agent through the real ``_make_agent`` and capture the AIAgent kwargs."""
    captured: dict = {}

    def mock_agent(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()
        mock.prefill_messages = kwargs.get("prefill_messages")
        return mock

    with patch("tui_gateway.server._startup_system_prompt", return_value=""):
        with patch("tui_gateway.server._resolve_agent_model_runtime", return_value=("mock", {})):
            with patch("tui_gateway.server._load_provider_routing", return_value={}):
                with patch("tui_gateway.server._resolve_agent_platform", return_value="desktop"):
                    with patch("tui_gateway.server._cfg_max_turns", return_value=500):
                        with patch("tui_gateway.server._load_reasoning_config", return_value=None):
                            with patch("tui_gateway.server._load_service_tier", return_value=None):
                                with patch("tui_gateway.server._load_enabled_toolsets", return_value=[]):
                                    with patch("tui_gateway.server._load_fallback_model", return_value=None):
                                        with patch("tui_gateway.server._agent_cbs", return_value={}):
                                            with patch("tui_gateway.server._context_cwd_is_launch_artifact", return_value=True):
                                                with patch("tui_gateway.synthetic_turn.maybe_build_synthetic_agent", return_value=None):
                                                    with patch("run_agent.AIAgent", side_effect=mock_agent):
                                                        agent = server._make_agent("test_sid", "test_key")
    return captured, agent


def test_tui_make_agent_injects_prefill(monkeypatch, tmp_path):
    data = [{"role": "system", "content": "DR.TEST"}]
    f = tmp_path / "prefill.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {"prefill_messages_file": str(f)})

    captured, agent = _make_agent_capturing_prefill(monkeypatch)
    assert captured.get("prefill_messages") == data
    assert agent.prefill_messages == data


def test_tui_make_agent_reads_prefill_from_temporary_hermes_home(monkeypatch, tmp_path):
    """Desktop factory under a temporary HERMES_HOME: a relative configured path is honoured."""
    data = [{"role": "system", "content": "DR.PROFILE"}]
    (tmp_path / "prefill.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tui_gateway.server._load_cfg",
                        lambda: {"prefill_messages_file": "prefill.json"})

    captured, agent = _make_agent_capturing_prefill(monkeypatch)
    assert captured.get("prefill_messages") == data
    assert agent.prefill_messages == data


def test_tui_make_agent_without_prefill_passes_none(monkeypatch):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {})

    captured, agent = _make_agent_capturing_prefill(monkeypatch)
    assert captured.get("prefill_messages") is None
