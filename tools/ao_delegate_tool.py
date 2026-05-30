"""Hermes tool for delegating durable coding work to Agent Orchestrator."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from gateway.dev_execution import build_profiled_prompt, resolve_launch_defaults
from tools.ao_bridge import AOBridge, AOBridgeError, AOSession
from tools.registry import registry, tool_error, tool_result


AO_DELEGATE_TASK_SCHEMA = {
    "name": "ao_delegate_task",
    "description": (
        "Spawn an Agent Orchestrator worker in an isolated worktree and stream "
        "its live status as subagent events. Use this for durable coding work "
        "that should run under AO, not for short reasoning-only subtasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Short title for the AO worker row.",
            },
            "prompt": {
                "type": "string",
                "description": "Self-contained instructions for the AO worker.",
            },
            "project_id": {
                "type": "string",
                "description": (
                    "AO project id from agent-orchestrator.yaml. "
                    "Use OrynWorkspace for Oryn app work, OrynPlatform for platform/Hermes work, or Oryn for Oryn.ai app work."
                ),
                "default": "OrynWorkspace",
            },
            "profile_id": {
                "type": "string",
                "description": "Optional Dev launch profile id such as workspace.inspect, workspace.implement, platform.inspect, or platform.implement.",
            },
            "model": {
                "type": "string",
                "description": "Optional model override for this AO worker.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Optional reasoning effort override: low, medium, high, or xhigh.",
            },
            "permissions": {
                "type": "string",
                "description": "Optional permission contract label stored with the worker, e.g. read_only, edit, verify, review_only.",
            },
            "launch_plan_id": {
                "type": "string",
                "description": "Optional Dev execution plan id to link this worker.",
            },
            "launch_task_id": {
                "type": "string",
                "description": "Optional Dev execution task id to link this worker.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional acceptance criteria stored with the worker.",
            },
            "issue_id": {
                "type": "string",
                "description": "Optional Linear/GitHub issue id to link the AO session.",
            },
            "branch": {
                "type": "string",
                "description": (
                    "Optional existing local or remote branch for the AO worktree. "
                    "Use this for local validation branches that must include unmerged changes."
                ),
            },
            "max_wait_seconds": {
                "type": "integer",
                "description": "Maximum time to keep watching the AO worker before returning.",
                "default": 1800,
            },
        },
        "required": ["prompt"],
    },
}


AO_DELEGATE_BATCH_SCHEMA = {
    "name": "ao_delegate_batch",
    "description": (
        "Spawn 2-5 Agent Orchestrator workers from one deterministic tool call. "
        "Use this when the user asks for multiple AO workers or parallel board testing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "description": "AO worker task specs to spawn.",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Short title for this AO worker row."},
                        "prompt": {"type": "string", "description": "Self-contained instructions for this AO worker."},
                        "project_id": {"type": "string", "description": "AO project id.", "default": "OrynWorkspace"},
                        "profile_id": {"type": "string", "description": "Optional Dev launch profile id."},
                        "model": {"type": "string", "description": "Optional model override."},
                        "reasoning_effort": {"type": "string", "description": "Optional reasoning effort override."},
                        "permissions": {"type": "string", "description": "Optional permission contract label."},
                        "launch_plan_id": {"type": "string", "description": "Optional Dev execution plan id."},
                        "launch_task_id": {"type": "string", "description": "Optional Dev execution task id."},
                        "acceptance_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional acceptance criteria.",
                        },
                        "issue_id": {"type": "string", "description": "Optional issue id."},
                        "branch": {"type": "string", "description": "Optional branch for the AO worktree."},
                    },
                    "required": ["prompt"],
                },
            },
        },
        "required": ["tasks"],
    },
}


def _progress_callback(parent_agent):
    return getattr(parent_agent, "tool_progress_callback", None) if parent_agent else None


def build_ao_worker_prompt(prompt: str, *, goal: Optional[str] = None) -> str:
    """Wrap an AO prompt so the delegated brief wins over generic AO lifecycle rules."""
    prompt = (prompt or "").strip()
    if not prompt or "## Hermes AO Delegation Contract" in prompt:
        return prompt

    parts = [
        "## Hermes AO Delegation Contract",
        "You are an AO worker delegated by Hermes.",
        "The task brief below is the authoritative assignment. If it conflicts with generic Agent Orchestrator lifecycle or project rules, follow the task brief.",
        "Do not invent implementation, branch, build, PR, or file-inspection work unless the task brief explicitly asks for it.",
        "For read-only or diagnostic tasks, do not edit files, create branches, run builds, or open PRs unless explicitly requested.",
        "When the brief gives an exact final phrase, prefix, or short output requirement, follow it exactly.",
    ]
    if goal:
        parts.extend(["", f"Goal: {goal.strip()}"])
    parts.extend(["", "## Task Brief", prompt])
    return "\n".join(parts)


def _emit(cb, event_type: str, session: AOSession, goal: str, preview: str = "", tool_name: str = "ao_delegate_task", **extra) -> None:
    if not cb:
        return
    fields = {
        "subagent_id": f"ao:{session.id}",
        "depth": 0,
        "goal": goal,
        **session.event_fields(),
        **extra,
    }
    cb(event_type, tool_name=tool_name, preview=preview, **fields)


def _persist_start_event_direct(
    *,
    session: AOSession,
    goal: str,
    prompt: str,
    project_id: str,
    issue_id: Optional[str],
    branch: Optional[str],
    preview: str,
    tool_name: str,
    task_index: Optional[int] = None,
    task_count: Optional[int] = None,
    event_store: Any = None,
    launch_profile_id: Optional[str] = None,
    launch_plan_id: Optional[str] = None,
    launch_task_id: Optional[str] = None,
    permissions: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    runtime_selection: Optional[Dict[str, Any]] = None,
    selected_runtime: Optional[str] = None,
    runtime_selection_reason: Optional[str] = None,
    runtime_fallback_reason: Optional[str] = None,
    runtime_policy_evidence: Optional[Dict[str, Any]] = None,
    runtime_policy_status: Optional[str] = None,
    runtime_policy_reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Persist AO start metadata when this tool runs without a parent callback."""
    try:
        if event_store is None:
            from gateway.subagent_events import SubagentEventStore

            event_store = SubagentEventStore()
        event_store.upsert_ao_prompt(
            ao_session_id=session.id,
            project_id=project_id,
            prompt=prompt,
            goal=goal,
            issue_id=issue_id,
            branch=branch or session.branch,
            agent=session.agent,
            model=session.model,
            reasoning_effort=session.reasoning_effort,
            launch_profile_id=launch_profile_id,
            launch_plan_id=launch_plan_id,
            launch_task_id=launch_task_id,
            permissions=permissions,
            acceptance_criteria=acceptance_criteria,
            runtime_selection=runtime_selection,
            selected_runtime=selected_runtime,
            runtime_selection_reason=runtime_selection_reason,
            runtime_fallback_reason=runtime_fallback_reason,
        )
        payload: Dict[str, Any] = {
            "event": "subagent.start",
            "subagent_id": f"ao:{session.id}",
            "depth": 0,
            "goal": goal,
            "tool": tool_name,
            "tool_name": tool_name,
            "preview": preview,
            "message": preview,
            "timestamp": time.time(),
            **session.event_fields(),
            "launch_profile_id": launch_profile_id,
            "launch_plan_id": launch_plan_id,
            "launch_task_id": launch_task_id,
            "permissions": permissions,
            "acceptance_criteria": acceptance_criteria or [],
        }
        for key, value in {
            "runtime_selection": runtime_selection,
            "selected_runtime": selected_runtime,
            "runtime_selection_reason": runtime_selection_reason,
            "runtime_fallback_reason": runtime_fallback_reason,
            "runtime_policy_evidence": runtime_policy_evidence,
            "runtime_policy_status": runtime_policy_status,
            "runtime_policy_reason": runtime_policy_reason,
        }.items():
            if value is not None:
                payload[key] = value
        if task_index is not None:
            payload["task_index"] = task_index
        if task_count is not None:
            payload["task_count"] = task_count
        return event_store.append_event(payload)
    except Exception:
        return None


def _summary_from_output(output: str, limit: int = 500) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    text = "\n".join(lines[-6:])
    return text[-limit:]


def _is_separator(line: str) -> bool:
    return line.strip().startswith("──")


def _output_indicates_codex_complete(output: str) -> bool:
    lines = [line.rstrip() for line in output.splitlines()]
    separator_indexes = [idx for idx, line in enumerate(lines) if _is_separator(line)]
    if not separator_indexes:
        return False
    tail = [line.strip() for line in lines[separator_indexes[-1] + 1 :] if line.strip()]
    if any("Working (" in line for line in tail):
        return False
    return any(line.startswith("› ") for line in tail)


def _summary_from_completed_output(output: str, limit: int = 1200) -> str:
    lines = [line.rstrip() for line in output.splitlines()]
    separator_indexes = [idx for idx, line in enumerate(lines) if _is_separator(line)]
    if len(separator_indexes) >= 2:
        start = separator_indexes[-2] + 1
        end = separator_indexes[-1]
        block = lines[start:end]
    else:
        block = lines

    cleaned = []
    for line in block:
        text = line.strip()
        if not text or _is_separator(text):
            continue
        if text.startswith("• "):
            text = text[2:].strip()
        cleaned.append(text)
    return "\n".join(cleaned)[-limit:]


def ao_delegate_task(
    *,
    prompt: str,
    goal: Optional[str] = None,
    project_id: str = "OrynWorkspace",
    profile_id: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    permissions: Optional[str] = None,
    launch_plan_id: Optional[str] = None,
    launch_task_id: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    issue_id: Optional[str] = None,
    branch: Optional[str] = None,
    max_wait_seconds: int = 1800,
    parent_agent=None,
    bridge: Optional[AOBridge] = None,
    event_store: Any = None,
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return tool_error("ao_delegate_task requires a non-empty prompt")

    original_prompt = prompt
    goal = (goal or original_prompt.splitlines()[0])[:180]
    profile = resolve_launch_defaults(
        profile_id=profile_id,
        project_id=project_id,
        model=model,
        reasoning_effort=reasoning_effort,
        permissions=permissions,
    )
    project_id = str(profile.get("project_id") or "OrynWorkspace").strip()
    profiled_prompt = build_profiled_prompt(
        original_prompt,
        goal=goal,
        profile=profile,
        acceptance_criteria=acceptance_criteria or [],
    )
    launch_prompt = build_ao_worker_prompt(profiled_prompt, goal=goal)
    max_wait_seconds = max(5, min(int(max_wait_seconds or 1800), 7200))
    bridge = bridge or AOBridge()
    cb = _progress_callback(parent_agent)

    try:
        session = bridge.spawn(
            project_id=project_id,
            prompt=launch_prompt,
            issue_id=issue_id,
            branch=branch,
            agent=profile.get("agent"),
            model=profile.get("model"),
            reasoning_effort=profile.get("reasoning_effort"),
        )
    except Exception as exc:
        return tool_error(f"AO spawn failed: {exc}")

    _emit(
        cb,
        "subagent.start",
        session,
        goal,
        preview=f"AO session {session.id} spawned",
        _ao_prompt_metadata={
            "prompt": original_prompt,
            "goal": goal,
            "project_id": project_id,
            "issue_id": issue_id,
            "branch": branch,
            "agent": session.agent,
            "model": session.model,
            "reasoning_effort": session.reasoning_effort,
            "launch_profile_id": profile.get("launch_profile_id"),
            "launch_plan_id": launch_plan_id,
            "launch_task_id": launch_task_id,
            "permissions": profile.get("permissions"),
            "acceptance_criteria": acceptance_criteria or [],
        },
        launch_profile_id=profile.get("launch_profile_id"),
        launch_plan_id=launch_plan_id,
        launch_task_id=launch_task_id,
        permissions=profile.get("permissions"),
        acceptance_criteria=acceptance_criteria or [],
    )
    if not cb:
        _persist_start_event_direct(
            session=session,
            goal=goal,
            prompt=original_prompt,
            project_id=project_id,
            issue_id=issue_id,
            branch=branch,
            preview=f"AO session {session.id} spawned",
            tool_name="ao_delegate_task",
            event_store=event_store,
            launch_profile_id=profile.get("launch_profile_id"),
            launch_plan_id=launch_plan_id,
            launch_task_id=launch_task_id,
            permissions=profile.get("permissions"),
            acceptance_criteria=acceptance_criteria or [],
        )

    started = time.monotonic()
    last_signature = None
    pending_complete_summary = None
    final_output = ""

    while time.monotonic() - started < max_wait_seconds:
        try:
            current = bridge.status(session.id) or session
        except AOBridgeError:
            current = session
            current.status = "killed"
        output = bridge.capture_output(current, lines=50)
        final_output = output or final_output
        summary = _summary_from_output(output)
        inferred_complete = _output_indicates_codex_complete(output)
        if inferred_complete and not current.is_terminal:
            completed_summary = _summary_from_completed_output(output) or current.summary or summary
            if completed_summary and completed_summary == pending_complete_summary:
                current.status = "done"
                current.summary = completed_summary
                summary = current.summary or summary
            else:
                pending_complete_summary = completed_summary
                inferred_complete = False
        elif not inferred_complete:
            pending_complete_summary = None
        signature = (current.status, current.activity, summary)

        if signature != last_signature:
            last_signature = signature
            _emit(
                cb,
                "subagent.progress",
                current,
                goal,
                preview=summary or f"AO session {current.id}: {current.status or 'running'}",
                output_tail=[{"tool": "tmux", "preview": summary, "is_error": False}] if summary else [],
            )

        session = current
        if current.is_terminal:
            break
        time.sleep(3)

    timed_out = not session.is_terminal
    result_summary = session.summary or _summary_from_output(final_output, limit=1200)
    final_status = "running" if timed_out else session.display_status

    if timed_out:
        _emit(
            cb,
            "subagent.progress",
            session,
            goal,
            preview=f"AO session {session.id} is still running after {max_wait_seconds}s",
            status="running",
        )
    else:
        _emit(
            cb,
            "subagent.complete",
            session,
            goal,
            preview=result_summary or f"AO session {session.id} finished with {session.status}",
            summary=result_summary or session.summary,
        )

    return tool_result(
        {
            "ok": True,
            "runtime": "ao",
            "status": final_status,
            "timed_out": timed_out,
            "session": session.event_fields(),
            "launch_profile_id": profile.get("launch_profile_id"),
            "launch_plan_id": launch_plan_id,
            "launch_task_id": launch_task_id,
            "permissions": profile.get("permissions"),
            "summary": result_summary or session.summary,
        }
    )


def ao_delegate_batch(
    *,
    tasks: List[Dict[str, Any]],
    parent_agent=None,
    bridge: Optional[AOBridge] = None,
    event_store: Any = None,
) -> str:
    if not isinstance(tasks, list) or not tasks:
        return tool_error("ao_delegate_batch requires a non-empty tasks array")
    if len(tasks) > 5:
        return tool_error("ao_delegate_batch supports at most 5 tasks")

    bridge = bridge or AOBridge()
    cb = _progress_callback(parent_agent)
    sessions: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for index, raw_task in enumerate(tasks, start=1):
        prompt = str((raw_task or {}).get("prompt") or "").strip()
        if not prompt:
            failures.append({"task_index": index, "error": "Task prompt is required"})
            continue
        original_prompt = prompt
        goal = str((raw_task or {}).get("goal") or original_prompt.splitlines()[0])[:180]
        project_id = str((raw_task or {}).get("project_id") or "OrynWorkspace").strip()
        profile = resolve_launch_defaults(
            profile_id=(raw_task or {}).get("profile_id"),
            project_id=project_id,
            model=(raw_task or {}).get("model"),
            reasoning_effort=(raw_task or {}).get("reasoning_effort"),
            permissions=(raw_task or {}).get("permissions"),
        )
        project_id = str(profile.get("project_id") or project_id).strip()
        acceptance_criteria = [
            str(item).strip()
            for item in ((raw_task or {}).get("acceptance_criteria") or [])
            if str(item).strip()
        ]
        profiled_prompt = build_profiled_prompt(
            original_prompt,
            goal=goal,
            profile=profile,
            acceptance_criteria=acceptance_criteria,
        )
        launch_prompt = build_ao_worker_prompt(profiled_prompt, goal=goal)
        issue_id = (raw_task or {}).get("issue_id")
        branch = (raw_task or {}).get("branch")
        try:
            session = bridge.spawn(
                project_id=project_id,
                prompt=launch_prompt,
                issue_id=issue_id,
                branch=branch,
                agent=profile.get("agent"),
                model=profile.get("model"),
                reasoning_effort=profile.get("reasoning_effort"),
            )
        except Exception as exc:
            failures.append({"task_index": index, "goal": goal, "error": str(exc)})
            continue

        _emit(
            cb,
            "subagent.start",
            session,
            goal,
            preview=f"AO batch session {session.id} spawned",
            tool_name="ao_delegate_batch",
            task_index=index,
            task_count=len(tasks),
            _ao_prompt_metadata={
                "prompt": original_prompt,
                "goal": goal,
                "project_id": project_id,
                "issue_id": issue_id,
                "branch": branch,
                "agent": session.agent,
                "model": session.model,
                "reasoning_effort": session.reasoning_effort,
                "launch_profile_id": profile.get("launch_profile_id"),
                "launch_plan_id": (raw_task or {}).get("launch_plan_id"),
                "launch_task_id": (raw_task or {}).get("launch_task_id"),
                "permissions": profile.get("permissions"),
                "acceptance_criteria": acceptance_criteria,
            },
            launch_profile_id=profile.get("launch_profile_id"),
            launch_plan_id=(raw_task or {}).get("launch_plan_id"),
            launch_task_id=(raw_task or {}).get("launch_task_id"),
            permissions=profile.get("permissions"),
            acceptance_criteria=acceptance_criteria,
        )
        if not cb:
            _persist_start_event_direct(
                session=session,
                goal=goal,
                prompt=original_prompt,
                project_id=project_id,
                issue_id=issue_id,
                branch=branch,
                preview=f"AO batch session {session.id} spawned",
                tool_name="ao_delegate_batch",
                task_index=index,
                task_count=len(tasks),
                event_store=event_store,
                launch_profile_id=profile.get("launch_profile_id"),
                launch_plan_id=(raw_task or {}).get("launch_plan_id"),
                launch_task_id=(raw_task or {}).get("launch_task_id"),
                permissions=profile.get("permissions"),
                acceptance_criteria=acceptance_criteria,
            )
        sessions.append({
            "task_index": index,
            "goal": goal,
            "session": session.event_fields(),
            "launch_profile_id": profile.get("launch_profile_id"),
            "launch_plan_id": (raw_task or {}).get("launch_plan_id"),
            "launch_task_id": (raw_task or {}).get("launch_task_id"),
        })

    return tool_result({
        "ok": bool(sessions),
        "runtime": "ao",
        "status": "spawned" if sessions else "failed",
        "session_count": len(sessions),
        "failure_count": len(failures),
        "sessions": sessions,
        "failures": failures,
    })


def _handle_ao_delegate_task(args: Dict[str, Any], **kwargs) -> str:
    return ao_delegate_task(
        prompt=args.get("prompt") or args.get("goal") or "",
        goal=args.get("goal"),
        project_id=args.get("project_id") or "OrynWorkspace",
        profile_id=args.get("profile_id"),
        model=args.get("model"),
        reasoning_effort=args.get("reasoning_effort"),
        permissions=args.get("permissions"),
        launch_plan_id=args.get("launch_plan_id"),
        launch_task_id=args.get("launch_task_id"),
        acceptance_criteria=args.get("acceptance_criteria") or [],
        issue_id=args.get("issue_id"),
        branch=args.get("branch"),
        max_wait_seconds=args.get("max_wait_seconds") or 1800,
        parent_agent=kwargs.get("parent_agent"),
    )


def _handle_ao_delegate_batch(args: Dict[str, Any], **kwargs) -> str:
    return ao_delegate_batch(
        tasks=args.get("tasks") or [],
        parent_agent=kwargs.get("parent_agent"),
    )


registry.register(
    name="ao_delegate_task",
    toolset="delegation",
    schema=AO_DELEGATE_TASK_SCHEMA,
    handler=_handle_ao_delegate_task,
    emoji="AO",
    max_result_size_chars=20_000,
)

registry.register(
    name="ao_delegate_batch",
    toolset="delegation",
    schema=AO_DELEGATE_BATCH_SCHEMA,
    handler=_handle_ao_delegate_batch,
    emoji="AO",
    max_result_size_chars=30_000,
)
