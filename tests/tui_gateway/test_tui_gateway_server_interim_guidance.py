"""A between-turns TUI prompt rebuild keeps the interim preamble line without touching the live callback."""
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_persist_live_session_system_prompt_interim_guidance(monkeypatch, enabled):
    from agent.prompt_builder import INTERIM_PREAMBLE_GUIDANCE
    from agent.system_prompt import _guidance_parts, _interim_surface

    class FakeAgent:
        valid_tool_names = ["read_file"]
        model = "claude"
        _tool_use_enforcement = False
        interim_assistant_callback = None
        _session_db = Mock()

        def _build_system_prompt(self, system_message=None):
            return "\n\n".join(part for part in _guidance_parts(self) if part)

    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: enabled)
    agent = FakeAgent()
    server._persist_live_session_system_prompt({"agent": agent, "session_key": "interim-test"})

    prompt = agent._session_db.update_system_prompt.call_args.args[1]
    assert (INTERIM_PREAMBLE_GUIDANCE in prompt) is enabled
    assert agent.interim_assistant_callback is None and not _interim_surface.get()
