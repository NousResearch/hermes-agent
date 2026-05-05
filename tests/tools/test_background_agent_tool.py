"""Tests for tools/background_agent_tool.py."""

import json
from unittest.mock import patch


def test_spawn_background_agent_writes_prompt_and_starts_tracked_process(tmp_path):
    from tools import background_agent_tool as mod

    def fake_terminal_tool(**kwargs):
        assert kwargs["background"] is True
        assert kwargs["notify_on_complete"] is True
        assert kwargs["timeout"] == 10
        assert "hermes chat -q" in kwargs["command"]
        assert "--toolsets terminal,file" in kwargs["command"]
        assert "--model gpt-5.5" in kwargs["command"]
        return json.dumps({
            "output": "Background process started",
            "session_id": "proc_abc123",
            "pid": 42,
            "exit_code": 0,
            "notify_on_complete": True,
        })

    with patch.object(mod, "get_hermes_home", return_value=tmp_path), \
         patch.object(mod, "terminal_tool", side_effect=fake_terminal_tool):
        result = json.loads(mod.spawn_background_agent(
            prompt="Write the copy package to /tmp/job/copy.json",
            toolsets=["terminal", "file"],
            model="gpt-5.5",
        ))

    assert result["success"] is True
    assert result["session_id"] == "proc_abc123"
    assert result["prompt_path"].startswith(str(tmp_path / "background_agents" / "prompts"))
    assert (tmp_path / "background_agents" / "prompts").exists()
    prompt_path = next((tmp_path / "background_agents" / "prompts").glob("*.txt"))
    assert prompt_path.read_text(encoding="utf-8") == "Write the copy package to /tmp/job/copy.json"


def test_spawn_background_agent_rejects_empty_prompt():
    from tools.background_agent_tool import spawn_background_agent

    result = json.loads(spawn_background_agent(prompt="   "))

    assert "error" in result
    assert "prompt" in result["error"]
