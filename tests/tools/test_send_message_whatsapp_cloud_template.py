"""WhatsApp Cloud template delivery through the existing send_message tool."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.platforms.base import SendResult
from tools.send_message_tool import SEND_MESSAGE_SCHEMA, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _template_env():
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={Platform.WHATSAPP_CLOUD: pconfig},
        get_home_channel=lambda _platform: None,
    )
    adapter = SimpleNamespace(
        send=AsyncMock(
            side_effect=AssertionError("ordinary send must not handle templates")
        ),
        send_template=AsyncMock(
            return_value=SendResult(
                success=True,
                message_id="wamid.tool-template",
            )
        ),
    )
    runner = SimpleNamespace(adapters={Platform.WHATSAPP_CLOUD: adapter})
    return config, adapter, runner


def test_schema_exposes_typed_whatsapp_cloud_template_fields():
    properties = SEND_MESSAGE_SCHEMA["parameters"]["properties"]

    assert "send_template" in properties["action"]["enum"]
    assert properties["template_name"]["type"] == "string"
    assert properties["template_language"]["type"] == "string"
    assert properties["template_components"]["type"] == "array"
    component_properties = properties["template_components"]["items"]["properties"]
    assert component_properties["type"]["enum"] == ["header", "body", "button"]
    parameter_properties = component_properties["parameters"]["items"]["properties"]
    assert "document" in parameter_properties
    assert "payload" in parameter_properties


def test_send_template_routes_typed_data_to_live_whatsapp_cloud_adapter():
    config, adapter, runner = _template_env()
    components = [
        {
            "type": "body",
            "parameters": [{"type": "text", "text": "Juan"}],
        },
        {
            "type": "button",
            "sub_type": "quick_reply",
            "index": 0,
            "parameters": [{"type": "payload", "payload": "BOOK_VISIT"}],
        },
    ]

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch("gateway.run._gateway_runner_ref", return_value=runner),
        patch("model_tools._run_async", side_effect=_run_async_immediately),
    ):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send_template",
                    "target": "whatsapp_cloud:15551234567",
                    "template_name": "quote_follow_up",
                    "template_language": "es_MX",
                    "template_components": components,
                }
            )
        )

    assert result == {
        "success": True,
        "message_id": "wamid.tool-template",
    }
    adapter.send_template.assert_awaited_once_with(
        "15551234567",
        "quote_follow_up",
        "es_MX",
        components,
    )
    adapter.send.assert_not_awaited()


def test_send_template_rejects_non_whatsapp_cloud_target():
    result = json.loads(
        send_message_tool(
            {
                "action": "send_template",
                "target": "telegram:12345",
                "template_name": "quote_follow_up",
                "template_language": "es_MX",
            }
        )
    )

    assert "whatsapp_cloud" in result["error"]


def test_send_template_requires_named_template_and_language():
    result = json.loads(
        send_message_tool(
            {
                "action": "send_template",
                "target": "whatsapp_cloud:15551234567",
            }
        )
    )

    assert "template_name" in result["error"]
    assert "template_language" in result["error"]


def test_send_template_is_not_skipped_by_cron_auto_delivery_guard():
    config, adapter, runner = _template_env()

    with (
        patch.dict(
            "os.environ",
            {
                "HERMES_CRON_AUTO_DELIVER_PLATFORM": "whatsapp_cloud",
                "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "15551234567",
            },
            clear=False,
        ),
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch("gateway.run._gateway_runner_ref", return_value=runner),
        patch("model_tools._run_async", side_effect=_run_async_immediately),
    ):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send_template",
                    "target": "whatsapp_cloud:15551234567",
                    "template_name": "quote_follow_up",
                    "template_language": "es_MX",
                }
            )
        )

    assert result["success"] is True
    assert "skipped" not in result
    assert "[SILENT]" in result["note"]
    adapter.send_template.assert_awaited_once()


def test_send_template_uses_one_shot_adapter_when_gateway_is_not_running():
    config, _adapter, _runner = _template_env()
    standalone_result = SendResult(
        success=True,
        message_id="wamid.cron-template",
    )

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch("gateway.run._gateway_runner_ref", return_value=None),
        patch(
            "gateway.platforms.whatsapp_cloud.send_template_standalone",
            new=AsyncMock(return_value=standalone_result),
        ) as standalone,
        patch("model_tools._run_async", side_effect=_run_async_immediately),
    ):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send_template",
                    "target": "whatsapp_cloud:15551234567",
                    "template_name": "quote_follow_up",
                    "template_language": "en_US",
                }
            )
        )

    assert result == {
        "success": True,
        "message_id": "wamid.cron-template",
    }
    standalone.assert_awaited_once_with(
        config.platforms[Platform.WHATSAPP_CLOUD],
        "15551234567",
        "quote_follow_up",
        "en_US",
        None,
    )
