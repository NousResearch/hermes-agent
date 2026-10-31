"""Plugin policy at the file tools' resolved, backend-bound write boundary."""
from __future__ import annotations

import copy
import uuid


def _targets(tool_name, args, resolved, entries):
    from tools.file_tools import _V4A_SINGLE_HEADER_RE, _V4A_MOVE_HEADER_RE

    if tool_name == "write_file" or args.get("mode", "replace") == "replace":
        return [{"operation": "write", "raw_path": args.get("path"), "path": resolved.get(args.get("path"))}]
    targets = []
    for match in _V4A_SINGLE_HEADER_RE.finditer(args.get("patch") or ""):
        op, raw = match.group(2).lower(), match.group(3).strip()
        targets.append({"operation": op, "raw_path": raw, "path": (entries if op == "delete" else resolved).get(raw)})
    for match in _V4A_MOVE_HEADER_RE.finditer(args.get("patch") or ""):
        targets.extend({"operation": op, "raw_path": raw.strip(), "path": entries.get(raw.strip())}
                       for op, raw in zip(("move_source", "move_destination"), match.group(2, 3)))
    return targets


def check_file_write_policy(tool_name, args, *, task_id, session_id, file_ops, resolved, entries=None):
    """Return a refusal, or None. No policy hook preserves the existing tool behavior.

    The caller retains its resolved strings, effective content and file_ops object
    across this call and executes those same values. Callback arguments are copies.
    Other filesystem namespaces are explicitly unsupported when a policy is present.
    """
    from hermes_cli import plugins
    from tools.file_tools import _file_ops_uses_host_paths
    from tools import approval_context

    try:
        if not plugins.has_hook("pre_file_write"):
            return None
        if not _file_ops_uses_host_paths(file_ops):
            return "BLOCKED: file write policy requires a supported host filesystem namespace"
        targets = _targets(tool_name, args, resolved, entries or {})
        if not targets or any(not t["path"] for t in targets):
            return "BLOCKED: file write policy requires resolved targets"
        snapshot = {"tool_name": tool_name, "args": copy.deepcopy(args), "targets": targets,
                    "namespace": "host", "task_id": task_id,
                    "session_id": approval_context._approval_session_id.get() or session_id,
                    "turn_id": approval_context._approval_turn_id.get(),
                    "tool_call_id": approval_context._approval_tool_call_id.get(),
                    "write_id": uuid.uuid4().hex}
        # Keep the execution/approval snapshot private from callback mutation.
        results = plugins.invoke_hook("pre_file_write", **copy.deepcopy(snapshot))
        approval = None
        for result in results:
            if result is None:
                continue
            if not isinstance(result, dict) or result.get("action") not in {"allow", "block", "approve"}:
                return "BLOCKED: invalid file write policy directive"
            action = result["action"]
            if action == "block":
                message = result.get("message")
                return message if isinstance(message, str) and message.strip() else "BLOCKED: file write policy veto"
            if action == "approve":
                policy = result.get("approval_policy", "")
                if policy and policy not in getattr(plugins, "VALID_TOOL_APPROVAL_POLICIES", ()):
                    return "BLOCKED: requested file write approval policy is unavailable"
                if approval is None or policy == "mandatory":
                    approval = dict(result)
        if approval is not None:
            from tools.approval import request_tool_approval
            kwargs = {"rule_key": approval.get("rule_key") or tool_name}
            if approval.get("approval_policy"):
                kwargs.update(approval_policy=approval["approval_policy"], effective_args=snapshot)
            decision = request_tool_approval(tool_name, approval.get("message") or "File write policy", **kwargs)
            if not decision.get("approved"):
                refusal = str(decision.get("message") or "BLOCKED: file write approval denied")
                if approval.get("approval_policy") == "mandatory" and approval.get("message"):
                    return str(approval["message"]) + "\n" + refusal
                return refusal
    except Exception:  # health: allow BLE001 -- a failed policy cannot authorize a write
        return "BLOCKED: file write policy failed"
    return None
