"""Feishu Chat Tool -- get group chat members.

Provides ``get_chat_members`` so the agent can list the people in a Feishu
group chat (with their open_id) before @-mentioning specific members. Resolves
the live adapter via ``_gateway_runner_ref`` (gateway mode only).
"""

import importlib.util
import json
import logging

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


GET_CHAT_MEMBERS_SCHEMA = {
    "name": "get_chat_members",
    "description": (
        "Get the member list of the current Feishu/Lark group chat — returns "
        "each member's name and open_id. Call this before @-mentioning specific "
        "people in a group so you know their open_id. Only works in Feishu group "
        "chats under the gateway."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Optional chat_id; defaults to the current group chat from session context.",
            },
        },
        "required": [],
    },
}


def _check_feishu() -> bool:
    # Mirrors feishu_doc_tool: cheap importability probe (avoids the ~5s cost
    # of executing lark_oapi's __init__ at every tool-availability check). The
    # real import + adapter resolution happens in the handler.
    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


async def _handle_get_chat_members(args: dict, **kwargs) -> str:
    from gateway.session_context import get_session_env

    chat_id = (args.get("chat_id") or "").strip() or get_session_env(
        "HERMES_SESSION_CHAT_ID", ""
    )
    if not chat_id:
        return tool_error("No chat_id available (not in a Feishu group chat context)")

    from gateway.config import Platform
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    adapter = runner.adapters.get(Platform.FEISHU) if runner is not None else None
    if adapter is None or not hasattr(adapter, "get_chat_members"):
        return tool_error(
            "Feishu adapter unavailable — this tool only works in gateway mode on the Feishu platform"
        )

    members = await adapter.get_chat_members(chat_id)
    if not members:
        return tool_error(
            "No members returned — the app likely lacks the im:chat.member:read "
            "scope (grant it in the Feishu developer console and publish a new "
            "app version), or the chat has no members"
        )
    return tool_result(
        success=True,
        content=json.dumps(
            {"members": members, "count": len(members)}, ensure_ascii=False
        ),
    )


registry.register(
    name="get_chat_members",
    toolset="feishu_chat",
    schema=GET_CHAT_MEMBERS_SCHEMA,
    handler=_handle_get_chat_members,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=True,
    description="Get Feishu group chat members",
    emoji="\U0001f465",
)
