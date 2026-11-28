"""#51797: the legacy TUI factory forwards rule isolation without disabling memory tools."""
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("isolated", [False, True])
def test_tui_factory_forwards_memory_prompt_gate(monkeypatch, tmp_path, isolated):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1" if isolated else "0")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform: ["memory", "file"])
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: (
        "fixture", {"provider": "custom", "base_url": "http://127.0.0.1:9/v1", "api_key": "fixture"}))
    with patch("run_agent.AIAgent") as constructor:
        server._make_agent("memory-gate-session", "memory-gate-key", context_cwd_is_launch_artifact=False)
    kwargs = constructor.call_args.kwargs
    assert kwargs["skip_memory_prompt"] is isolated
    assert kwargs["skip_memory"] is isolated
    assert "memory" in kwargs["enabled_toolsets"]
