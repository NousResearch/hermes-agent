"""Desktop/TUI agents honor ``prefill_messages_file`` from the active profile.

Regression for #60456: ``tui_gateway.server._make_agent`` (the Desktop / ``hermes serve`` agent
factory) never passed ``prefill_messages``, so a configured prefill file only worked in the
classic CLI and the messaging gateway.
"""

from __future__ import annotations

import json
import types

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


def _write_home(root, name: str, content: str):
    home = root / name
    home.mkdir()
    (home / "config.yaml").write_text("prefill_messages_file: prefill.json\n", encoding="utf-8")
    (home / "prefill.json").write_text(json.dumps([{"role": "user", "content": content}]), encoding="utf-8")
    return home


def _built_prefill(monkeypatch, home):
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    token = set_hermes_home_override(home)
    try:
        server._make_agent("sid", "session-key")
    finally:
        reset_hermes_home_override(token)
    return captured["prefill_messages"]


def test_make_agent_injects_the_active_profiles_prefill(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_a: ("test-model", {}))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    home_a = _write_home(tmp_path, "a", "prefill from A")
    home_b = _write_home(tmp_path, "b", "prefill from B")

    # A -> B -> A: the relative path resolves against whichever profile builds the agent.
    assert _built_prefill(monkeypatch, home_a) == [{"role": "user", "content": "prefill from A"}]
    assert _built_prefill(monkeypatch, home_b) == [{"role": "user", "content": "prefill from B"}]
    assert _built_prefill(monkeypatch, home_a) == [{"role": "user", "content": "prefill from A"}]
