"""TUI/Desktop ``_make_agent`` carries ``openrouter.min_coding_score`` (#153086)."""
import sys
import types

from tests.tui_gateway import test_protocol as _protocol

server = _protocol.server  # the shared tui_gateway.server import fixture


def test_make_agent_passes_openrouter_min_coding_score(server, monkeypatch):
    """TUI/Desktop agents carry ``openrouter.min_coding_score``; without it ``openrouter/pareto-code``
    requests go out with no ``pareto-router`` plugin and route to the top tier (#153086)."""
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model", "")

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_Agent))
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", types.SimpleNamespace(
        resolve_runtime_provider=lambda **_kwargs: {"provider": "openrouter", "base_url": None, "api_key": None,
                                                     "api_mode": None}))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"openrouter": {"min_coding_score": 0.4},
                                                     "provider_routing": {"sort": "price"}})
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("openrouter/pareto-code", "openrouter"))
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "session-key", session_id="session-key")

    assert captured["provider_sort"] == "price"
    assert captured["openrouter_min_coding_score"] == 0.4
