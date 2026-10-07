"""MCP Apps host RPCs (stable spec 2026-01-26): the ``mcp.app.*`` requests of one model tool
call's view (contracts: ``contracts/mcp_apps.py``; the view record: ``tools/mcp_app_host.py``).

Every request resolves its call in three steps: the caller's transport must own the live session
(compute-host sessions are refused); the record comes from memory while the call runs, else from
the session's newest tool row answering it; the live ``Tool`` comes from the record's server inside
the session's profile scope. So the server is always the one the call ran on (spec 399-402)."""

import json
import logging
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_MCP_APP_CONTEXT_KEY = "mcp_app_context"  # session dict: tool_call_id -> the view's latest model context
_mcp_app_context_lock = threading.Lock()


class _McpAppRefusal(Exception):
    """An ``mcp.app.*`` request answered with a JSON-RPC error: ``(code, message)``."""


def _mcp_app_rpc(rid, params, model, body):
    """Validate *params* against its contract, resolve the caller's session and run
    ``body(session, request)`` in that session's scope. One log line per request, never its
    arguments or content; an unexpected failure answers and logs only its type."""
    from pydantic import ValidationError

    try:
        request = model.model_validate(params)
    except ValidationError:
        return _err(rid, 4000, f"invalid params for {_current_rpc_method.get()}")
    level, outcome = logging.INFO, "ok"
    try:
        _, session = _current_session_steer_authority(request.session_id)
        if session is None or session.get("_finalized"):
            raise _McpAppRefusal(4001, "session not found or not owned by this transport")
        if _session_uses_compute_host(session):
            raise _McpAppRefusal(5033, "MCP App views are not served for compute-host sessions")
        with _session_rpc_scope(session, request.session_id):
            return _ok(rid, body(session, request))
    except _McpAppRefusal as refusal:
        code, message = refusal.args
        outcome = f"refused {code}"
        return _err(rid, code, message)
    except ProfileUnavailableError:
        outcome = "profile unavailable"
        raise
    except Exception as exc:
        level, outcome = logging.WARNING, f"failed ({type(exc).__name__})"
        return _err(rid, 5034, "MCP App request failed")
    finally:
        logger.log(level, "%s session=%s call=%s: %s",
                   _current_rpc_method.get(), request.session_id, request.tool_call_id, outcome)


def _view_record(session, tool_call_id):
    """``(status, record)`` of the view's call: running from memory, else its newest tool row."""
    from tools import mcp_app_host

    agent_session_id = getattr(session.get("agent"), "session_id", None) or session["session_key"]
    record = mcp_app_host.running_record(agent_session_id, tool_call_id)
    if record is not None:
        return "running", record
    with _session_db(session) as db:
        metadata = db.get_tool_call(agent_session_id, tool_call_id) if db is not None else None
    record = (metadata or {}).get("mcp_app")
    if not isinstance(record, dict):
        raise _McpAppRefusal(4064, "this tool call has no MCP App view")
    return ("result" if "result" in record else "cancelled"), record


def _live_tool(server_name, tool_name):
    """The connected server and its live ``Tool`` named *tool_name*."""
    from tools import mcp_app_host
    from tools.mcp_tool_discovery import _get_connected_server_for_call

    server = _get_connected_server_for_call(server_name)
    tool = mcp_app_host.live_tool(server, tool_name)
    if tool is None:
        raise _McpAppRefusal(4064, f"MCP server '{server_name}' serves no tool '{tool_name}'")
    return server, tool


def _mcp_result(result):
    """An SDK result as it travelled, or the refusal for the (already sanitized) ``tool_error``
    the MCP layer answered with."""
    from tools import mcp_app_host

    if isinstance(result, str):
        raise _McpAppRefusal(5034, json.loads(result).get("error") or "MCP request failed")
    return mcp_app_host.wire(result)


@method("mcp.app.view")
def _(rid, params):
    from tui_gateway.contracts.mcp_apps import McpAppCallParams

    def view(session, request):
        from tools import mcp_app_host

        status, record = _view_record(session, request.tool_call_id)
        _, tool = _live_tool(record["server"], record["tool"])
        return {"status": status, "arguments": record["arguments"],
                "result": record["result"] if status == "result" else None, "tool": mcp_app_host.wire(tool)}
    return _mcp_app_rpc(rid, params, McpAppCallParams, view)


@method("mcp.app.read_resource")
def _(rid, params):
    from tui_gateway.contracts.mcp_apps import McpAppReadResourceParams

    def read(session, request):
        from tools.mcp_tool_handlers import read_mcp_resource

        _, record = _view_record(session, request.tool_call_id)
        server, _ = _live_tool(record["server"], record["tool"])
        return _mcp_result(read_mcp_resource(record["server"], request.uri, server.tool_timeout))
    return _mcp_app_rpc(rid, params, McpAppReadResourceParams, read)


@method("mcp.app.call_tool")
def _(rid, params):
    from tui_gateway.contracts.mcp_apps import McpAppCallToolParams

    def call(session, request):
        from tools import mcp_app_host
        from tools.mcp_tool_handlers import call_mcp_tool
        from tools.mcp_tool_registration import _make_tool_filter

        _, record = _view_record(session, request.tool_call_id)
        server, tool = _live_tool(record["server"], request.name)
        operator_allowed = _make_tool_filter(record["server"], server._config)(tool.name)
        if not (mcp_app_host.visible_to(tool, "app") and operator_allowed):
            raise _McpAppRefusal(4030, f"MCP tool '{tool.name}' is not callable by an MCP App view")
        return _mcp_result(call_mcp_tool(record["server"], tool.name, request.arguments or {}, server.tool_timeout,
                                         lambda result, _server_name: result))
    return _mcp_app_rpc(rid, params, McpAppCallToolParams, call)


@method("mcp.app.update_model_context")
def _(rid, params):
    from tui_gateway.contracts.mcp_apps import McpAppUpdateModelContextParams

    def update(session, request):
        from mcp.types import ContentBlock
        from pydantic import TypeAdapter, ValidationError

        try:
            content = TypeAdapter(list[ContentBlock]).validate_python(request.content or [])
        except ValidationError:
            raise _McpAppRefusal(4000, "content must be MCP content blocks") from None
        _, record = _view_record(session, request.tool_call_id)
        with _mcp_app_context_lock:
            pending = session.setdefault(_MCP_APP_CONTEXT_KEY, {})
            if content or request.structured_content:
                pending[request.tool_call_id] = (record["server"], record["tool"], content, request.structured_content)
            else:  # each update replaces the last (spec 1093): an empty one clears it
                pending.pop(request.tool_call_id, None)
        return {}
    return _mcp_app_rpc(rid, params, McpAppUpdateModelContextParams, update)


def _pending_mcp_app_context(session: dict) -> str:
    """Note block for the views' latest model context since the last turn (each view's last
    update only, spec 1101), or "": model input only, taken once, wrapped as untrusted MCP output
    and capped like that tool's result."""
    with _mcp_app_context_lock:
        pending = session.pop(_MCP_APP_CONTEXT_KEY, None)
    if not pending:
        return ""
    from mcp.types import CallToolResult

    from agent.tool_dispatch_helpers import _maybe_wrap_untrusted
    from agent.tool_executor import _budget_for_agent
    from tools.mcp_tool_content import _truncate_mcp_text_result
    from tools.mcp_tool_handlers import _render_content_blocks
    from tools.mcp_tool_schema import mcp_prefixed_tool_name

    budget = _budget_for_agent(session.get("agent"))
    notes = []
    for server_name, tool_name, content, structured in pending.values():
        text, _ = _render_content_blocks(CallToolResult(content=content), server_name)
        if structured is not None:
            text = "\n".join(filter(None, (text, json.dumps(structured, ensure_ascii=False))))
        name = mcp_prefixed_tool_name(server_name, tool_name)
        text = _truncate_mcp_text_result(text, int(budget.resolve_threshold(name)))
        notes.append(f"[Latest context from the MCP App view of {name}]\n{_maybe_wrap_untrusted(name, text)}")
    return "\n\n".join(notes)


def register(server):
    bind_module(globals(), server, skip=("_",))
    server._LONG_HANDLERS = server._LONG_HANDLERS | _registry.names()
