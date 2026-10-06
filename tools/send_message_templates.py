"""Approved WhatsApp Cloud template delivery for CLI and scheduled sends."""

import json
import logging

from tools.registry import tool_error

logger = logging.getLogger(__name__)


def handle_template_request(args):
    from tools.send_message_tool import (
        _home_chat_id, _resolve_platform_config, _resolve_tool_target,
    )
    target = args.get("target", "")
    if not target:
        return tool_error("'target' is required when action='send_template'")
    platform_name, chat_id, thread_id, resolution_error = _resolve_tool_target(target)
    if resolution_error:
        return tool_error(resolution_error)
    if platform_name != "whatsapp_cloud":
        return tool_error("action='send_template' is only supported for whatsapp_cloud targets")
    if not args.get("template_name") or not args.get("template_language"):
        return tool_error("Both 'template_name' and 'template_language' are required when action='send_template'")
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")
    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
    except Exception as exc:
        logger.exception("Failed to load template gateway config")
        return tool_error(f"Failed to load gateway config: {exc}")
    platform, pconfig, _entry, error = _resolve_platform_config(platform_name, config)
    if error:
        return tool_error(error)
    if not chat_id:
        chat_id, error = _home_chat_id(config, platform, platform_name)
        if error:
            return tool_error(error)
    return _handle_template_send(args, platform, pconfig, chat_id, thread_id)


def _handle_template_send(args, platform, pconfig, chat_id, thread_id):
    from model_tools import _run_async
    from tools.send_message_tool import _error, _maybe_skip_cron_duplicate_send, _sanitize_error_text
    try:
        result = _run_async(_send_whatsapp_cloud_template(
            platform, pconfig, chat_id, args["template_name"], args["template_language"],
            args.get("template_components")))
    except Exception as exc:
        logger.exception("WhatsApp Cloud template send failed")
        return json.dumps(_error(f"Template send failed: {exc}"))
    if isinstance(result, dict) and "error" in result:
        result["error"] = _sanitize_error_text(result["error"])
    if (isinstance(result, dict) and result.get("success")
            and _maybe_skip_cron_duplicate_send("whatsapp_cloud", chat_id, thread_id)):
        result["note"] = (
            "Template sent directly to this cron job's delivery target. Return [SILENT] "
            "as the final response so the scheduler does not send a second free-form message.")
    return json.dumps(result)


async def _send_whatsapp_cloud_template(platform, pconfig, chat_id, template_name, language_code, components):
    from tools.send_message_tool import _dispatch_on_gateway_loop, _error, _live_adapter
    runner, adapter = _live_adapter(platform)
    if adapter is not None:
        result = await _dispatch_on_gateway_loop(
            runner, lambda: adapter.send_template(chat_id, template_name, language_code, components),
            "send_message: failed to schedule template send on gateway loop")
    else:
        from gateway.platforms.whatsapp_cloud import send_template_standalone
        result = await send_template_standalone(pconfig, chat_id, template_name, language_code, components)
    if isinstance(result, dict):
        return result
    if not result.success:
        return _error(result.error or "Template send failed")
    return {"success": True, "message_id": result.message_id}
