"""send_message honours the profile's Slack posting policy too (education
design 5b, defence in depth if `messaging` is ever granted)."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import send_message_tool


def _run_async_immediately(coro):
    import asyncio
    return asyncio.run(coro)


@pytest.mark.parametrize("policy,target,sent", [
    ({}, "slack:C0OTHER01", True),
    ({"slack": {"post_gated_channels": ["C0REL0001"]}}, "slack:C0REL0001", False),
    ({"slack": {"post_allowed_channels": ["C0HOME001"]}}, "slack:C0OTHER01", False),
    ({"slack": {"post_allowed_channels": ["C0HOME001"]}}, "slack:C0HOME001", True),
])
def test_slack_send_respects_policy(policy, target, sent):
    slack_cfg = SimpleNamespace(enabled=True, token="xoxb", extra={})
    config = SimpleNamespace(platforms={Platform.SLACK: slack_cfg}, get_home_channel=lambda _p: None)
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", return_value=None), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("agent.boundary_config.read_boundary_config", return_value=policy), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(send_message_tool({"action": "send", "target": target, "message": "hi"}))
    assert send_mock.await_count == (1 if sent else 0)
    if not sent:
        assert "error" in result
