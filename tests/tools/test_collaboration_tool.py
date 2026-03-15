import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tools.collaboration_tool import check_collaboration_requirements, collaborate_with_agent


@pytest.fixture
def collaboration_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "collaboration:\n"
        "  enabled: true\n"
        "  targets:\n"
        "    hermes-2:\n"
        "      platform: webhook\n"
        "      chat_id: hermes-2\n"
        "      display_name: Research\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("WEBHOOK_PORT", "4568")
    return hermes_home


def _make_parent(runner: GatewayRunner) -> SimpleNamespace:
    source = SessionSource(
        platform=runner.config.platforms and next(iter(runner.config.platforms.keys())) or None,
        chat_id="hermes-1",
        chat_type="dm",
        user_id="hermes-1",
        user_name="Caller",
    )
    source.platform = source.platform or __import__("gateway.config", fromlist=["Platform"]).Platform.WEBHOOK
    return SimpleNamespace(
        gateway_runner=runner,
        gateway_session_key=build_session_key(source),
        gateway_source=source,
        _collaboration_lineage=[],
    )


def test_check_collaboration_requirements_reads_config(collaboration_home):
    assert check_collaboration_requirements() is True


def test_collaborate_requires_parent_agent():
    result = json.loads(collaborate_with_agent(target_agent="hermes-2", task="Research"))
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_collaborate_with_agent_waits_for_result(collaboration_home):
    config = GatewayConfig(sessions_dir=collaboration_home / "sessions")
    config.collaboration = {
        "enabled": True,
        "targets": {
            "hermes-2": {"platform": "webhook", "chat_id": "hermes-2", "display_name": "Research"}
        },
    }
    runner = GatewayRunner(config)
    runner._event_loop = asyncio.get_running_loop()

    async def fake_run_agent(message, context_prompt, history, source, session_id, session_key=None):
        assert "Hermes collaboration request" in message
        return {"final_response": "Artifact ready"}

    runner._run_agent = fake_run_agent
    parent = _make_parent(runner)

    result = await asyncio.to_thread(
        lambda: json.loads(
            collaborate_with_agent(
                target_agent="hermes-2",
                task="Research OpenClaw's announce flow",
                parent_agent=parent,
            )
        )
    )

    assert result["status"] == "completed"
    assert result["result"] == "Artifact ready"


@pytest.mark.asyncio
async def test_collaborate_with_agent_rejects_cycle(collaboration_home):
    config = GatewayConfig(sessions_dir=collaboration_home / "sessions")
    config.collaboration = {
        "enabled": True,
        "targets": {
            "hermes-2": {"platform": "webhook", "chat_id": "hermes-2", "display_name": "Research"}
        },
    }
    runner = GatewayRunner(config)
    parent = _make_parent(runner)
    parent._collaboration_lineage = ["agent:main:webhook:dm:hermes-2"]

    result = json.loads(
        collaborate_with_agent(
            target_agent="hermes-2",
            task="Loop",
            parent_agent=parent,
        )
    )

    assert result["status"] == "error"
    assert "cycle" in result["error"].lower()
