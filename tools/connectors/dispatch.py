"""Route connector calls through the normal dispatch policy pipeline."""

import json
from dataclasses import asdict

from tools.registry import tool_error
from tools.connectors.gateway.config import MAX_CALLS_PER_DISPATCH
from tools.connectors.gateway.merge import assemble_results, partition_calls


def dispatch_connector_call(name, arguments, tool_call_id):
    from tools.connectors.gateway.bridge import run_remote

    partition = partition_calls([{"name": name, "arguments": arguments}])
    entries = run_remote(partition.remote, tool_call_id, availability=None, client_factory=None)
    entry = entries[0]
    return json.dumps({key: value for key, value in entry.items() if key in {"response", "error"}},
                      ensure_ascii=False)


def dispatch_connector_batch(calls, ids, *, user_task, enabled_tools,
                             middleware_trace, enabled_toolsets, disabled_toolsets):
    from model_tools import handle_function_call
    from tools.interrupt import is_interrupted

    if len(calls) > MAX_CALLS_PER_DISPATCH:
        return tool_error(f"too many calls: {len(calls)} > max {MAX_CALLS_PER_DISPATCH}. "
                          "Retry with fewer calls per batch.")
    partition = partition_calls(calls)
    entries = list(partition.errors)
    plans = {
        plan.position: (plan.name, plan.arguments, False)
        for plan in partition.remote
    }
    plans.update({
        position: (str(call.get("name") or ""), dict(call.get("arguments") or {}), True)
        for position, call in partition.local
    })
    ordered = sorted(plans.items())
    for offset, (position, (name, arguments, is_local)) in enumerate(ordered):
        if is_interrupted():
            # Check before every entry so /stop prevents unstarted side effects.
            entries.extend({
                "index": pending_position,
                "name": pending_name,
                "error": {"code": "INTERRUPTED", "message": "Stopped by the user before this call was made."},
            } for pending_position, (pending_name, _pending_args, _is_local) in ordered[offset:])
            break
        # Each entry must run its own policy and middleware.
        dispatch_name = "tool_call" if is_local else name
        dispatch_args = {"calls": [{"name": name, "arguments": arguments}]} if is_local else arguments
        payload = handle_function_call(
            dispatch_name, dispatch_args, **asdict(ids), user_task=user_task,
            enabled_tools=enabled_tools, tool_request_middleware_trace=list(middleware_trace),
            skip_pre_tool_call_hook=False, skip_tool_request_middleware=False,
            skip_tool_execution_middleware=False,
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        )
        try:
            value = json.loads(payload) if isinstance(payload, str) else payload
        except ValueError:
            value = payload
        entry = {"index": position, "name": name}
        if isinstance(value, dict) and "error" in value:
            error = value["error"]
            entry["error"] = error if isinstance(error, dict) else {"code": "TOOL_ERROR", "message": str(error)}
        else:
            entry["response"] = value.get("response", value) if isinstance(value, dict) else value
        entries.append(entry)
    return json.dumps(assemble_results(len(calls), entries), ensure_ascii=False)
