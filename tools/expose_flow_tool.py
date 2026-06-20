"""Native EasyHermes tools — expose a local Langflow flow as an MCP tool.

The copilot (the full EasyHermes agent embedded beside the Langflow canvas)
uses these to turn a Chat Input / Chat Output flow into a callable tool without
leaving chat:

  - ``list_flows``           — see the local flows + which are already exposed.
  - ``expose_flow_as_tool``  — flip a flow's ``mcp_enabled`` (+ tool name /
    description) so Langflow's project MCP server lists it, then refresh the
    in-process ``kari_flows`` MCP connection so the new tool appears at once.

Auth: the embedded Langflow runs with ``LANGFLOW_AUTO_LOGIN=true``, so we mint a
keyless token via ``GET /api/v1/auto_login`` and call the flows API with it.
Only flows owned by the auto-login user (the ones the user built in the canvas)
are editable; starter examples (``user_id is None``) are read-only → 404 on PATCH.

Tool contract: ``handle_call_tool`` on the Langflow side feeds the single
``input_value`` argument to the flow's Chat Input and returns the Chat Output
text, so a good tool flow has exactly one Chat Input and a Chat Output.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from tools.flow_chat import is_chat_flow
from tools.registry import registry, tool_error

_TOOLSET = "kari_flows"
# MCP server name in ~/.hermes/config.yaml that points at the project MCP SSE
# endpoint; refreshing it in-process is what makes a newly-exposed flow appear.
_KARI_FLOWS_SERVER = "kari_flows"


def _langflow_base() -> str:
    """scheme://host:port of the local Langflow, derived from the canvas op URL."""
    raw = os.environ.get("KARI_CANVAS_OP_URL", "http://127.0.0.1:7860")
    for marker in ("/kari/", "/api/"):
        idx = raw.find(marker)
        if idx > 0:
            raw = raw[:idx]
            break
    return raw.rstrip("/")


def _host_port() -> Tuple[str, int]:
    parsed = urlparse(_langflow_base())
    return (parsed.hostname or "127.0.0.1"), (parsed.port or 7860)


def _langflow_reachable() -> bool:
    """Fast TCP probe — gates these tools out when Langflow is not running.

    check_fn results are TTL-cached by the registry, so this runs at most once
    every ~30s rather than on every tool-schema build.
    """
    host, port = _host_port()
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


def _load_httpx():
    import httpx

    return httpx


def _auto_login_token(client) -> str:
    resp = client.get(f"{_langflow_base()}/api/v1/auto_login", timeout=10.0)
    resp.raise_for_status()
    token = (resp.json() or {}).get("access_token")
    if not token:
        raise RuntimeError("auto_login 未返回 access_token(LANGFLOW_AUTO_LOGIN 是否开启?)")
    return token


def _list_flows(client, token: str) -> List[dict]:
    resp = client.get(
        f"{_langflow_base()}/api/v1/flows/",
        params={"header_flows": "true", "get_all": "true"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else (data.get("flows") or [])


def _get_flow(client, token: str, flow_id: str) -> "dict | None":
    """取单个 flow 的完整数据(含 data.nodes,用于校验对话流)。失败 → None(此时不阻断暴露)。"""
    try:
        resp = client.get(
            f"{_langflow_base()}/api/v1/flows/{flow_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - 取不到就当无法校验,放行(见调用处说明)
        return None


def _refresh_kari_flows() -> str:
    """Re-list the kari_flows MCP server's tools in-process (best-effort)."""
    try:
        from tools.mcp_tool import _lock, _run_on_mcp_loop, _servers
    except Exception:  # noqa: BLE001 - mcp client optional
        return "skipped (mcp client unavailable)"
    with _lock:
        task = _servers.get(_KARI_FLOWS_SERVER)
    if task is None:
        return "skipped (kari_flows not connected)"
    try:
        _run_on_mcp_loop(task._refresh_tools, timeout=30.0)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        return f"refresh failed: {exc}"
    return "refreshed"


# --------------------------------------------------------------------------- #
# list_flows
# --------------------------------------------------------------------------- #
LIST_FLOWS_SCHEMA: Dict[str, Any] = {
    "name": "list_flows",
    "description": (
        "List the local Langflow flows: id, name, whether each is already exposed "
        "as a tool (mcp_enabled), and its action (tool) name. Use this to find a "
        "flow's id before calling expose_flow_as_tool. Note: starter example flows "
        "are read-only and cannot be exposed — only flows the user built can."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional case-insensitive substring to filter flow names.",
            }
        },
        "required": [],
    },
}


def _handle_list_flows(args: Dict[str, Any], **_kw) -> str:
    query = str(args.get("query") or "").strip().lower()
    httpx = _load_httpx()
    try:
        with httpx.Client(timeout=30.0) as client:
            token = _auto_login_token(client)
            flows = _list_flows(client, token)
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"读取 Langflow flows 失败: {exc}")

    out = []
    for flow in flows:
        name = flow.get("name") or ""
        if query and query not in name.lower():
            continue
        out.append(
            {
                "id": flow.get("id"),
                "name": name,
                "mcp_enabled": bool(flow.get("mcp_enabled")),
                "action_name": flow.get("action_name"),
            }
        )
    return json.dumps({"count": len(out), "flows": out}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# expose_flow_as_tool
# --------------------------------------------------------------------------- #
EXPOSE_FLOW_SCHEMA: Dict[str, Any] = {
    "name": "expose_flow_as_tool",
    "description": (
        "Expose a local Langflow flow as a callable tool for the agent (or hide it "
        "again). The flow should have one Chat Input (the tool's text argument) and "
        "a Chat Output (the tool's text result). This sets the flow's mcp_enabled "
        "flag plus a tool name/description and refreshes the live tool list so the "
        "tool appears immediately. Find flow_id with list_flows first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "flow_id": {
                "type": "string",
                "description": "Flow id (UUID) from list_flows. Either flow_id or flow_name is required.",
            },
            "flow_name": {
                "type": "string",
                "description": "Exact flow name; resolved to an id when flow_id is omitted.",
            },
            "action_name": {
                "type": "string",
                "description": "Tool name shown to the agent (snake_case recommended). Defaults to the flow name.",
            },
            "action_description": {
                "type": "string",
                "description": "One-line description of what the tool does. Strongly recommended.",
            },
            "enabled": {
                "type": "boolean",
                "description": "True to expose (default), False to hide the flow as a tool.",
            },
        },
        "required": [],
    },
}


def _handle_expose_flow(args: Dict[str, Any], **_kw) -> str:
    flow_id = str(args.get("flow_id") or "").strip()
    flow_name = str(args.get("flow_name") or "").strip()
    raw_enabled = args.get("enabled", True)
    if isinstance(raw_enabled, str):
        # Models occasionally send "false"/"0" as strings — treat those as off.
        enabled = raw_enabled.strip().lower() not in ("false", "0", "no", "off", "")
    else:
        enabled = True if raw_enabled is None else bool(raw_enabled)
    action_name = args.get("action_name")
    action_description = args.get("action_description")

    if not flow_id and not flow_name:
        return tool_error("需要 flow_id 或 flow_name 其一(先用 list_flows 查 id)。")

    httpx = _load_httpx()
    try:
        with httpx.Client(timeout=30.0) as client:
            token = _auto_login_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            if not flow_id:
                matches = [f for f in _list_flows(client, token) if (f.get("name") or "") == flow_name]
                if not matches:
                    return tool_error(f"未找到名为 '{flow_name}' 的 flow(用 list_flows 查看可用名)。")
                if len(matches) > 1:
                    return tool_error(f"有多个 flow 叫 '{flow_name}',请改用 flow_id。")
                # The list API does not return user_id, so ownership (owned flow
                # vs read-only starter example) can't be checked up front — the
                # PATCH 404 handler below reports read-only flows clearly instead.
                flow_id = matches[0].get("id")

            # 硬约定门控:工具流必须是「ChatInput + ChatOutput」一进一出对话流(与资源采集
            # 同一判据 tools.flow_chat)。取得完整 flow 且确认不合规 → 直接拒;取不到(瞬时
            # 错误)则放行不阻断(langflow 侧 MCP 调用本就只喂 ChatInput/读 ChatOutput,
            # 非对话流暴露了也跑不通,这道校验是给用户的清晰提示而非安全边界)。
            if enabled:
                flow_obj = _get_flow(client, token, flow_id)
                if flow_obj is not None and not is_chat_flow(flow_obj):
                    return tool_error(
                        "该 flow 不是对话流,不能注册成工具 —— 需要恰好一个 Chat Input(工具的文本入参)"
                        "和一个 Chat Output(工具的文本结果)。请在画布里加上对话入口/出口再试。"
                    )

            payload: Dict[str, Any] = {"mcp_enabled": enabled}
            if enabled:
                if action_name:
                    payload["action_name"] = str(action_name)
                if action_description:
                    payload["action_description"] = str(action_description)
            else:
                payload["action_name"] = None
                payload["action_description"] = None

            resp = client.patch(
                f"{_langflow_base()}/api/v1/flows/{flow_id}", json=payload, headers=headers
            )
            if resp.status_code == 404:
                return tool_error(
                    "Langflow 返回 404 —— 该 flow 不属于当前用户(只读示例不可改)。"
                    "请在画布中新建或另存为自己的 flow 再试。"
                )
            resp.raise_for_status()
            result = resp.json()
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"设置 flow mcp_enabled 失败: {exc}")

    refresh = _refresh_kari_flows()
    name = result.get("name")
    resolved_action = result.get("action_name")
    return json.dumps(
        {
            "success": True,
            "flow_id": result.get("id", flow_id),
            "name": name,
            "mcp_enabled": result.get("mcp_enabled"),
            "action_name": resolved_action,
            "tool_refresh": refresh,
            "message": (
                f"已{'暴露' if enabled else '隐藏'} flow「{name}」"
                + (f",工具名 {resolved_action}" if resolved_action else "")
                + f"。kari_flows {refresh}。"
                # 注册即提示授权:暴露后该工作流会作为「工作流」资源上报,主账号可在
                # 团队账号面板按角色勾选谁能用(③ 注册即问哪个角色可用)。
                + ("该工作流将出现在团队账号的可授权资源里,去给对应角色勾选可用即可。" if enabled else "")
            ),
        },
        ensure_ascii=False,
    )


registry.register(
    name="list_flows",
    toolset=_TOOLSET,
    schema=LIST_FLOWS_SCHEMA,
    handler=_handle_list_flows,
    check_fn=_langflow_reachable,
    requires_env=[],
    is_async=False,
    emoji="🧩",
)

registry.register(
    name="expose_flow_as_tool",
    toolset=_TOOLSET,
    schema=EXPOSE_FLOW_SCHEMA,
    handler=_handle_expose_flow,
    check_fn=_langflow_reachable,
    requires_env=[],
    is_async=False,
    emoji="🧩",
)
