"""Regression tests for CF-59: unpinned default-model sessions adopt config.yaml model.provider changes."""
from __future__ import annotations

import json
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_bound_helpers():
    af_code = (REPO / "tui_gateway" / "agent_factory.py").read_text(encoding="utf-8")
    ms_code = (REPO / "tui_gateway" / "model_switch.py").read_text(encoding="utf-8")
    env_af: dict = {"__name__": "tui_gateway.agent_factory", "__package__": "tui_gateway"}
    exec(compile(af_code, "agent_factory.py", "exec"), env_af)
    env_ms: dict = {"__name__": "tui_gateway.model_switch", "__package__": "tui_gateway"}
    exec(compile(ms_code, "model_switch.py", "exec"), env_ms)

    for env in (env_af, env_ms):
        env["json"] = json
        env["logger"] = logging.getLogger("test_cf59")
        env["_config_model_target"] = lambda: ("google/gemini-4-argon", "openrouter")
        env["_is_routable_provider"] = lambda p: p in ("openrouter", "vertex")
        env["_persist_live_session_runtime"] = lambda s: None
    return env_af, env_ms


def test_stored_session_on_default_model_adopts_configured_provider():
    env_af, _ = _load_bound_helpers()
    row = {
        "model": "google/gemini-4-argon",
        "billing_provider": "vertex",
        "model_config": '{"provider":"vertex","reasoning_config":{"enabled":true,"effort":"high"}}',
        "title": "Normal Desktop Session",
        "hidden": 0,
    }
    overrides = env_af["_stored_session_runtime_overrides"](row)
    assert overrides["model_override"]["provider"] == "openrouter"
    assert overrides["provider_override"] == "openrouter"


def test_in_memory_session_on_default_model_syncs_provider_on_turn_start():
    _, env_ms = _load_bound_helpers()
    switched: list[str] = []
    env_ms["_apply_model_switch"] = lambda sid, session, raw, **kw: switched.append(raw)

    class DummyAgent:
        model = "google/gemini-4-argon"
        provider = "vertex"

    session = {
        "agent": DummyAgent(),
        "config_model_seen": ("google/gemini-4-argon", "vertex"),
        "model_override": {"model": "google/gemini-4-argon", "provider": "vertex"},
        "composer_override_profile": None,
    }
    env_ms["_sync_agent_model_with_config"]("sid_1", session)
    assert switched == ["google/gemini-4-argon --provider openrouter"]


def test_explicit_different_model_override_is_preserved():
    _, env_ms = _load_bound_helpers()
    switched: list[str] = []
    env_ms["_apply_model_switch"] = lambda sid, session, raw, **kw: switched.append(raw)

    class CustomAgent:
        model = "anthropic/claude-opus-4-6"
        provider = "openrouter"

    session = {
        "agent": CustomAgent(),
        "config_model_seen": ("google/gemini-4-argon", "vertex"),
        "model_override": {"model": "anthropic/claude-opus-4-6", "provider": "openrouter"},
        "composer_override_profile": None,
    }
    env_ms["_sync_agent_model_with_config"]("sid_2", session)
    assert switched == []
