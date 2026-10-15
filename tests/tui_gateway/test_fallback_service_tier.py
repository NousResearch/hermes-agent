"""Focused TUI agent-construction coverage for pre-construction fallback tier policy."""

import builtins
import types
from pathlib import Path
from unittest.mock import patch

from hermes_cli.auth import AuthError
from tests.agent.fallback_policy_wire import first_wire_from_runtime


def test_make_agent_carries_selected_fallback_policy(monkeypatch):
    # The frozen base misses the Path import used by an unrelated onboarding annotation.
    # Supply it only while importing the TUI facade so this regression stays scoped to fallback routing.
    monkeypatch.setattr(builtins, "Path", Path, raising=False)
    from tui_gateway import server

    captured = {}

    def resolve(**kwargs):
        if kwargs.get("requested") in (None, "openai-codex"):
            raise AuthError("no key")
        return {
            "provider": "deepseek",
            "api_key": "k",
            "base_url": "https://fallback.example/v1",
        }

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "model": {"default": "primary", "provider": "openai-codex"},
        "fallback_providers": [{
            "provider": "deepseek",
            "model": "fallback",
            "service_tier_override": "normal",
        }],
    })
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_args, **_kwargs: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    with patch(
        "run_agent.AIAgent",
        side_effect=lambda **kwargs: captured.update(kwargs) or types.SimpleNamespace(model=kwargs["model"]),
    ):
        server._make_agent("sid", "key")  # type: ignore[attr-defined]

    assert captured["fallback_service_tier_override"] == "normal"
    wire, snapshot = first_wire_from_runtime(captured)
    assert snapshot["fallback_service_tier_override"] == "normal"
    assert "service_tier" not in wire and "speed" not in wire
    assert "service_tier" not in wire["extra_body"] and "speed" not in wire["extra_body"]
