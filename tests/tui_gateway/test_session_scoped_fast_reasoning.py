"""Session-scoping for the /fast and /reasoning gateway config handlers.

``config.set`` for ``fast`` / ``reasoning`` used to write the user's GLOBAL
config (``agent.service_tier`` / ``agent.reasoning_effort``) unconditionally,
so toggling either for a single session clobbered the profile default. These
toggles are session-scoped: with a ``session_id`` in scope the pick is parked
on the session's existing ``create_service_tier_override`` /
``create_reasoning_override`` (the same fields ``session.create`` seeds and
``_start_agent_build`` applies at agent BUILD time, mirroring
``model_override``) and global config is never touched.

Without a ``session_id`` the global path is unchanged.
"""

from __future__ import annotations

import yaml

from hermes_constants import (
    parse_reasoning_effort,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tui_gateway import server


# A model that supports OpenAI Priority Processing.
_OPENAI_FAST_MODEL = "gpt-5.4"


class _FakeAgent:
    """Minimal stand-in carrying just the runtime attrs the handlers touch."""

    def __init__(self, model: str = _OPENAI_FAST_MODEL):
        self.model = model
        self.service_tier = None
        self.request_overrides: dict = {}
        self.reasoning_config = None


def _home(tmp_path, monkeypatch, *, model: str = _OPENAI_FAST_MODEL) -> object:
    """Pin a clean HERMES_HOME so global config reads/writes hit tmp_path.

    ``_load_cfg`` honors the home override; ``_save_cfg`` writes to the module
    ``_hermes_home``. Pin both, and reset the cfg cache so reads see fresh
    state. Returns a token to pass to :func:`reset_hermes_home_override`.
    """
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    token = set_hermes_home_override(str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_resolve_model", lambda: model)
    server._cfg_cache = None
    server._cfg_mtime = None
    server._cfg_path = None
    return token


def _global_cfg(tmp_path) -> dict:
    path = tmp_path / ".hermes" / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ── /fast ────────────────────────────────────────────────────────────────────


def test_fast_without_session_writes_global(tmp_path, monkeypatch):
    """No session_id -> global config path is unchanged."""
    token = _home(tmp_path, monkeypatch)
    try:
        server._sessions.clear()
        res = server._methods["config.set"]("r1", {"key": "fast", "value": "fast"})
        assert res["result"]["value"] == "fast"
        # Global config WAS written (legacy global path).
        assert _global_cfg(tmp_path).get("agent", {}).get("service_tier") == "fast"
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


def test_fast_with_session_parks_override_and_skips_global(tmp_path, monkeypatch):
    """With a session_id (no live agent yet) -> park the override, never write global."""
    token = _home(tmp_path, monkeypatch)
    try:
        sid = "sid-fast"
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k-fast",
            "agent": None,
            "create_service_tier_override": None,
        }
        res = server._methods["config.set"](
            "r1", {"key": "fast", "value": "fast", "session_id": sid}
        )
        assert res["result"]["value"] == "fast"
        # Parked on the session's existing build-time override field.
        assert server._sessions[sid]["create_service_tier_override"] == "priority"
        # Global config was NOT touched.
        assert "service_tier" not in _global_cfg(tmp_path).get("agent", {})
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


def test_fast_normal_with_session_parks_none_and_skips_global(tmp_path, monkeypatch):
    """Setting normal in a session parks None (no explicit pick) and skips global."""
    token = _home(tmp_path, monkeypatch)
    try:
        sid = "sid-fast-off"
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        res = server._methods["config.set"](
            "r1", {"key": "fast", "value": "normal", "session_id": sid}
        )
        assert res["result"]["value"] == "normal"
        assert server._sessions[sid]["create_service_tier_override"] is None
        assert "service_tier" not in _global_cfg(tmp_path).get("agent", {})
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


def test_fast_with_live_agent_updates_agent_not_global(tmp_path, monkeypatch):
    """A live agent is updated in place (service_tier + request_overrides); global untouched."""
    token = _home(tmp_path, monkeypatch)
    try:
        sid = "sid-live"
        agent = _FakeAgent(model=_OPENAI_FAST_MODEL)
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k",
            "agent": agent,
            "create_service_tier_override": None,
        }
        monkeypatch.setattr(server, "_persist_live_session_runtime", lambda s: None)
        monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_info", lambda a, s: {})
        res = server._methods["config.set"](
            "r1", {"key": "fast", "value": "fast", "session_id": sid}
        )
        assert res["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert agent.request_overrides == {"service_tier": "priority"}
        assert server._sessions[sid]["create_service_tier_override"] == "priority"
        assert "service_tier" not in _global_cfg(tmp_path).get("agent", {})
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


# ── /reasoning ───────────────────────────────────────────────────────────────


def test_reasoning_without_session_writes_global(tmp_path, monkeypatch):
    """No session_id -> global config path is unchanged."""
    token = _home(tmp_path, monkeypatch)
    try:
        server._sessions.clear()
        res = server._methods["config.set"](
            "r1", {"key": "reasoning", "value": "high"}
        )
        assert res["result"]["value"] == "high"
        assert _global_cfg(tmp_path).get("agent", {}).get("reasoning_effort") == "high"
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


def test_reasoning_with_session_parks_override_and_skips_global(tmp_path, monkeypatch):
    """With a session_id (no live agent) -> park parsed override, never write global."""
    token = _home(tmp_path, monkeypatch)
    try:
        sid = "sid-reason"
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k",
            "agent": None,
            "create_reasoning_override": None,
        }
        res = server._methods["config.set"](
            "r1", {"key": "reasoning", "value": "high", "session_id": sid}
        )
        assert res["result"]["value"] == "high"
        assert server._sessions[sid]["create_reasoning_override"] == parse_reasoning_effort("high")
        assert "reasoning_effort" not in _global_cfg(tmp_path).get("agent", {})
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


def test_reasoning_with_live_agent_updates_agent_not_global(tmp_path, monkeypatch):
    """A live agent's reasoning_config is updated; global config untouched."""
    token = _home(tmp_path, monkeypatch)
    try:
        sid = "sid-reason-live"
        agent = _FakeAgent()
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k",
            "agent": agent,
            "create_reasoning_override": None,
        }
        monkeypatch.setattr(server, "_persist_live_session_runtime", lambda s: None)
        monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_info", lambda a, s: {})
        res = server._methods["config.set"](
            "r1", {"key": "reasoning", "value": "low", "session_id": sid}
        )
        assert res["result"]["value"] == "low"
        assert agent.reasoning_config == parse_reasoning_effort("low")
        assert server._sessions[sid]["create_reasoning_override"] == parse_reasoning_effort("low")
        assert "reasoning_effort" not in _global_cfg(tmp_path).get("agent", {})
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)


# ── config.get reads back the session pick ───────────────────────────────────


def test_config_get_reflects_parked_session_overrides(tmp_path, monkeypatch):
    """config.get fast/reasoning read the session's parked pick, not the global default."""
    token = _home(tmp_path, monkeypatch)
    try:
        # Global default differs from the per-session pick.
        server._write_config_key("agent.service_tier", "normal")
        server._write_config_key("agent.reasoning_effort", "medium")
        sid = "sid-get"
        server._sessions.clear()
        server._sessions[sid] = {
            "session_key": "k",
            "agent": None,
            "create_service_tier_override": "priority",
            "create_reasoning_override": parse_reasoning_effort("high"),
        }
        fast = server._methods["config.get"]("r1", {"key": "fast", "session_id": sid})
        assert fast["result"]["value"] == "fast"
        reasoning = server._methods["config.get"](
            "r2", {"key": "reasoning", "session_id": sid}
        )
        assert reasoning["result"]["value"] == "high"
    finally:
        server._sessions.clear()
        server._cfg_cache = server._cfg_mtime = server._cfg_path = None
        reset_hermes_home_override(token)
