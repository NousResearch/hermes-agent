"""Fast-mode (service tier) session scoping in the TUI gateway (desktop backend).

Sibling of test_reasoning_session_scope.py — the ``reasoning`` key was made
session-scoped when a session is targeted, but ``fast`` kept writing the
global ``agent.service_tier`` to config.yaml on every call. The desktop's
per-model presets call ``config.set key=fast`` on every model selection, so
toggling fast in ONE session silently flipped the tier for every other
session, profile, CLI, and gateway build ("switch one session, switches
everywhere").

Contract under test:

1. ``config.set key=fast`` with a session must NOT write config.yaml; it pins
   ``create_service_tier_override`` ("priority" / "" for explicit normal) so
   lazily-built sessions and rebuilds keep the choice.
2. Without a session it persists globally, unchanged.
3. ``config.get key=fast`` must read a pre-build session's pin.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import tui_gateway.server as server
from hermes_constants import get_hermes_home

FAST_OVERRIDES = {"service_tier": "priority"}


def _agent(service_tier=None):
    return SimpleNamespace(
        reasoning_config=None,
        service_tier=service_tier,
        request_overrides={},
        model="gpt-6",
        provider="openai",
        session_id="sess-key",
    )


def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)


def _get(params: dict) -> dict:
    return server._methods["config.get"]("rid-1", params)


class TestConfigSetFastSessionScope:
    """Session-targeted fast changes must never touch global config."""

    def test_session_scoped_fast_skips_global_write(self) -> None:
        agent = _agent()
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s1", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()


    def test_lazy_session_pins_create_override(self) -> None:
        """A pre-build (agent=None) session must keep the change for the
        deferred agent build instead of dropping it."""
        session = {
            "session_key": "k3",
            "agent": None,
            "model_override": {"model": "gpt-6", "provider": "openai"},
        }
        with patch.dict(server._sessions, {"s3": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s3", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()


    def test_toggle_flips_prebuild_pin(self) -> None:
        """An empty value toggles from the session's pin, not the global."""
        session = {
            "session_key": "k5",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s5": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "session_id": "s5", "value": ""})
        assert resp["result"]["value"] == "normal"
        assert session["create_service_tier_override"] == ""
        write_key.assert_not_called()

    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "value": "normal"})
        assert resp["result"]["value"] == "normal"
        write_key.assert_called_once_with("agent.service_tier", "normal")


class TestConfigGetFastSessionScope:
    def test_reads_prebuild_pin(self) -> None:
        session = {
            "session_key": "k6",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s6": session}, clear=False):
            resp = _get({"key": "fast", "session_id": "s6"})
        assert resp["result"]["value"] == "fast"


    def test_falls_back_to_global(self) -> None:
        with patch.object(server, "_load_service_tier", return_value="priority"):
            resp = _get({"key": "fast"})
        assert resp["result"]["value"] == "fast"


class TestFastSlashSyncsRequestOverrides:
    def test_config_set_fast_then_slash_normal_drops_tier_keys(self) -> None:
        from agent.chat_completion_helpers import _effective_request_overrides

        agent = _agent()
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key"), \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}):
            _set({"key": "fast", "session_id": "s1", "value": "fast"})
            assert _effective_request_overrides(agent).get("service_tier") == "priority"
            server._mirror_slash_side_effects("s1", session, "/fast off")
        wire = _effective_request_overrides(agent)
        assert "service_tier" not in wire
        assert "speed" not in wire


def _reset_tui_cfg_cache() -> None:
    server._cfg_cache = None
    server._cfg_mtime = None
    server._cfg_path = None


def _write_service_tier_cfg(*, global_tier, overrides) -> None:
    payload = {
        "agent": {
            "service_tier": global_tier,
            "service_tier_overrides": overrides,
        }
    }
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    _reset_tui_cfg_cache()


class TestLazySessionHonorsPerModelTier:
    """Pre-build (agent=None) /fast must use the session model overlay."""

    def test_get_and_toggle_use_per_model_normal_not_global_priority(self) -> None:
        model = "openai/gpt-5"
        _write_service_tier_cfg(
            global_tier="priority",
            overrides={model: "normal"},
        )
        session = {
            "session_key": "k-lazy-normal",
            "agent": None,
            "model_override": {"model": model, "provider": "openai"},
        }
        with patch.dict(server._sessions, {"s-lazy-n": session}, clear=False), \
                patch.object(server, "_hermes_home", get_hermes_home()), \
                patch.object(server, "_write_config_key") as write_key, \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            got = _get({"key": "fast", "session_id": "s-lazy-n"})
            status = _set(
                {"key": "fast", "session_id": "s-lazy-n", "value": "status"}
            )
            toggled = _set(
                {"key": "fast", "session_id": "s-lazy-n", "value": "toggle"}
            )
        assert got["result"]["value"] == "normal"
        assert status["result"]["value"] == "normal"
        assert toggled["result"]["value"] == "fast"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()

    def test_get_and_toggle_use_per_model_flex_when_global_empty(self) -> None:
        model = "openai/gpt-5"
        _write_service_tier_cfg(
            global_tier="",
            overrides={model: "flex"},
        )
        session = {
            "session_key": "k-lazy-flex",
            "agent": None,
            "model_override": {"model": model, "provider": "openai"},
        }
        with patch.dict(server._sessions, {"s-lazy-f": session}, clear=False), \
                patch.object(server, "_hermes_home", get_hermes_home()), \
                patch.object(server, "_write_config_key") as write_key:
            got = _get({"key": "fast", "session_id": "s-lazy-f"})
            status = _set(
                {"key": "fast", "session_id": "s-lazy-f", "value": "status"}
            )
            toggled = _set(
                {"key": "fast", "session_id": "s-lazy-f", "value": "toggle"}
            )
        assert got["result"]["value"] == "flex"
        assert status["result"]["value"] == "flex"
        assert toggled["result"]["value"] == "normal"
        assert session["create_service_tier_override"] == ""
        write_key.assert_not_called()

    def test_scalar_model_override_uses_per_model_overlay(self) -> None:
        model = "openai/gpt-5"
        _write_service_tier_cfg(
            global_tier="priority",
            overrides={model: "flex"},
        )
        session = {
            "session_key": "k-scalar",
            "agent": None,
            "model_override": model,
        }
        with patch.dict(server._sessions, {"s-scalar": session}, clear=False), \
                patch.object(server, "_hermes_home", get_hermes_home()), \
                patch.object(server, "_write_config_key") as write_key:
            got = _get({"key": "fast", "session_id": "s-scalar"})
            status = _set(
                {"key": "fast", "session_id": "s-scalar", "value": "status"}
            )
        assert got["result"]["value"] == "flex"
        assert status["result"]["value"] == "flex"
        write_key.assert_not_called()

    def test_lazy_named_profile_reads_own_tier(self) -> None:
        model = "openai/gpt-5"
        launch = get_hermes_home()
        _write_service_tier_cfg(
            global_tier="priority",
            overrides={},
        )
        other = Path(launch).parent / "named-profile-fast"
        other.mkdir(parents=True, exist_ok=True)
        (other / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "agent": {
                        "service_tier": "",
                        "service_tier_overrides": {model: "flex"},
                    }
                }
            ),
            encoding="utf-8",
        )
        session = {
            "session_key": "k-named",
            "agent": None,
            "profile_home": str(other),
            "model_override": model,
        }
        with patch.dict(server._sessions, {"s-named": session}, clear=False), \
                patch.object(server, "_hermes_home", launch), \
                patch.object(server, "_write_config_key"):
            got = _get({"key": "fast", "session_id": "s-named"})
            status = _set(
                {"key": "fast", "session_id": "s-named", "value": "status"}
            )
        assert got["result"]["value"] == "flex"
        assert status["result"]["value"] == "flex"
