"""Non-bypassable origin gates while remote MCP skill guidance is active."""
from __future__ import annotations

import json
from typing import Any

from hermes_constants import get_hermes_home
from tools.mcp_skills_registry import (
    active_skills, has_execution_consent, record_execution_consent,
)
from tools.registry import tool_error

_EXECUTION_TOOLS = {"terminal", "execute_code", "browser_exec"}


def enforce_skill_activation_gate(record: dict[str, Any], *, home, session_id: str) -> str | None:
    """Require first-use consent for one exact remote manifest."""
    from tools.mcp_skills_registry import is_active
    if is_active(record, home, session_id):
        return None
    try:
        from tools.approval_prompt import request_elicitation_consent
        frontmatter = record["frontmatter"]
        origins = active_skills(home, session_id)
        active_note = ""
        if origins:
            active_note = " Active remote origins: " + ", ".join(
                f"{item['server']}:{item['uri']}" for item in origins) + "."
        answer = request_elicitation_consent(
            f"Remote MCP skill activation requested for server {record['server']!r}, "
            f"skill URI {record['uri']!r}, name {frontmatter.get('name')!r}, "
            f"description {frontmatter.get('description')!r}, manifest {record['manifest_fingerprint']}."
            f"{active_note}",
            "Approve loading the main SKILL.md body for this exact remote manifest once for this session. "
            "When another remote origin is active, this same explicit decision also approves only this exact "
            "main-body read for this call. It does not approve host execution or grant remote allowed-tools permissions.",
            surface="mcp-skill-activation")
    except Exception:
        answer = "decline"
    if answer != "accept":
        return tool_error(
            f"BLOCKED: activation was not approved for remote MCP skill {record['uri']!r} "
            f"from server {record['server']!r}. Silence, timeout, and missing approval channels are not consent.")
    return None


def _is_host_execution(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in _EXECUTION_TOOLS:
        return True
    return tool_name == "process_manage" and str(args.get("action") or "").lower() in {
        "write", "submit", "close", "kill",
    }


def _session_id(task_id: str | None, session_id: str | None) -> str:
    return str(session_id or task_id or "")


def _is_delegate_execution(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name != "delegate_task":
        return False
    return str(args.get("action") or "spawn").strip().lower() in {"", "spawn", "steer"}


def _display_action(tool_name: str, args: dict[str, Any]) -> str:
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    if len(raw) > 4000:
        raw = raw[:4000] + "…"
    return f"{tool_name}({raw})"


def enforce_remote_skill_gate(tool_name: str, args: dict[str, Any], *,
                              task_id: str | None = None, session_id: str | None = None) -> str | None:
    """Return a tool error when SEP execution/cross-origin consent is absent."""
    sid = _session_id(task_id, session_id)
    if not sid:
        return None
    target_server = None
    target_uri = ""
    if tool_name.endswith("__read_resource"):
        try:
            from tools import mcp_tool as core
            target_server = core._mcp_tool_server_names.get(tool_name)
        except Exception:
            target_server = None
        target_uri = str(args.get("uri") or "")
    records = active_skills(get_hermes_home(), sid)
    if not records:
        return None

    if _is_host_execution(tool_name, args) or _is_delegate_execution(tool_name, args):
        from tools.approval_prompt import request_elicitation_consent
        from tools.mcp_skills_registry import scan_materialized_targets
        from tools.mcp_skills_scan import RemoteSkillSecurityError
        action = _display_action(tool_name, args)
        for record in records:
            try:
                scan_materialized_targets(record, get_hermes_home(), sid)
            except (RemoteSkillSecurityError, OSError, ValueError):
                return tool_error(
                    "BLOCKED: a provenance-owned materialized remote skill file failed current-byte Skills Guard scanning.")
            if has_execution_consent(record, get_hermes_home(), sid):
                continue
            message = (
                f"Remote MCP skill execution consent required for server {record['server']!r}, "
                f"skill URI {record['uri']!r}, manifest {record['manifest_fingerprint']}.\n"
                f"Requested host action: {action}")
            description = (
                "The active remote skill supplied instructions but cannot authorize host code execution. "
                "Approve once to allow host execution for this exact skill manifest for the remainder of this session. "
                "Normal command and tool guards still apply independently.")
            answer = request_elicitation_consent(message, description, surface="mcp-skill-execution")
            if answer != "accept":
                return tool_error(
                    f"BLOCKED: host execution was not approved for remote MCP skill {record['uri']!r} "
                    f"from server {record['server']!r}. Silence and timeout are not consent.")
            record_execution_consent(record, get_hermes_home(), sid)

    # Existing generated/native MCP read_resource tools are a separate resource
    # route. Exact held same-origin resources are safe; every other origin/URI is
    # per-call consent and is deliberately never remembered.
    if tool_name.endswith("__read_resource"):
        same_origin = bool(records) and all(
            target_server == record["server"] and any(r.get("uri") == target_uri for r in record["resources"])
            for record in records
        )
        if not same_origin:
            from tools.approval_prompt import request_elicitation_consent
            origins = ", ".join(f"{r['server']}:{r['uri']}" for r in records)
            answer = request_elicitation_consent(
                f"Cross-origin MCP resource read requested while remote skill(s) are active. "
                f"Active origins: {origins}. Target: server={target_server!r}, uri={target_uri!r}.",
                "Approve this resource read once. The approval is not cached and does not extend the pinned manifest.",
                surface="mcp-skill-cross-origin-read")
            if answer != "accept":
                return tool_error("BLOCKED: cross-origin MCP resource read was not explicitly approved for this call.")
    return None


def enforce_remote_skill_block_message(tool_name: str, args: dict[str, Any], *,
                                       task_id: str | None = None,
                                       session_id: str | None = None) -> str | None:
    """Plain block message for agent-level dispatchers that wrap their own error JSON."""
    blocked = enforce_remote_skill_gate(
        tool_name, args, task_id=task_id, session_id=session_id)
    if blocked is None:
        return None
    try:
        payload = json.loads(blocked)
    except (TypeError, ValueError):
        return str(blocked)
    message = payload.get("error") if isinstance(payload, dict) else None
    return str(message or blocked)


def enforce_native_skill_read_gate(record: dict[str, Any], resource: dict[str, Any], *,
                                   session_id: str) -> str | None:
    """Per-call consent when native skill_view crosses any active remote origin."""
    records = active_skills(get_hermes_home(), session_id)
    if not records:
        return None
    same_unambiguous_origin = len(records) == 1 and (
        records[0]["server"], records[0]["uri"], records[0]["manifest_fingerprint"]
    ) == (record["server"], record["uri"], record["manifest_fingerprint"])
    if same_unambiguous_origin:
        return None
    from tools.approval_prompt import request_elicitation_consent
    origins = ", ".join(f"{r['server']}:{r['uri']}" for r in records)
    answer = request_elicitation_consent(
        "Cross-origin native MCP skill resource read requested. "
        f"Active origins: {origins}. Target: server={record['server']!r}, "
        f"skill={record['uri']!r}, resource={resource['uri']!r}.",
        "Approve this skill_view resource read once. Multiple active manifests are treated as ambiguous; "
        "the approval is not cached.",
        surface="mcp-skill-cross-origin-read")
    if answer != "accept":
        return tool_error("BLOCKED: cross-origin native MCP skill read was not explicitly approved for this call.")
    return None


def enforce_delegate_for_agent(agent: Any, args: dict[str, Any]) -> str | None:
    return enforce_remote_skill_gate("delegate_task", args, task_id=getattr(agent, "session_id", None),
                                     session_id=getattr(agent, "session_id", None))
