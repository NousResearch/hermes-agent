"""Pre-dispatch approval for every tool call (``approvals.scope: all``) and the resolution of
plugin ``pre_tool_call`` directives.

Every model tool call path (both agent-loop executors, ``invoke_tool``, ``handle_function_call``
for execute_code / connector entries / the MCP bridge) resolves its pre-call verdict through
:func:`resolve_pre_tool_block`, so the gate sees reads, writes, searches and meta tools alike,
always under the real tool name (``tool_call`` is unwrapped before the hook fires).

A session grant is bound to a tool and a scope: file tools to the task workspace or the exact
out-of-workspace paths, ``terminal`` to one exact command, everything else to the tool.
Unattended contexts (-q / -z, cron, unattended platforms) keep the pattern-only behaviour; only a
present human is asked. A delegated child inherits the parent's approval context, so its calls
reach the same human (gateway) or the subagent callback (``delegation.subagent_auto_approve``).

Every decision is recorded: logged, and kept on the tool message's ``display_metadata`` as
``approval`` (see :func:`pop_record`), so the transcript shows who allowed or refused the call and
how (once, session grant, ...) without the model reading it.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, Optional

from tools.approval_tool_gate_preview import patch_preview, write_file_preview

logger = logging.getLogger(__name__)

# Pure catalog reads / UI with no side effect. ``tool_call`` reaches the gate only when the
# executor kept the wrapper (connector batches, unresolvable calls); each entry is gated on
# its own dispatch.
_EXEMPT_TOOLS = frozenset({"tool_search", "tool_describe", "tool_call", "todo_list", "clarify"})

_PREVIEW_CHARS = 300


@dataclass(frozen=True)
class _Scope:
    key: str           # allowlist grain, stored as ``tool_scope:<key>``
    target: str        # what the prompt shows
    grant: str         # what "allow for this session / always" covers
    warning: str = ""  # dangerous-pattern match: a flag in the prompt, never the decision
    preview: Optional[Callable[[], str]] = None  # replaces ``target``; built only when a human is asked

    def shown(self) -> str:
        if self.preview is None:
            return self.target
        try:
            return self.preview() or self.target
        except Exception:  # health: allow BLE001 -- best-effort preview; the prompt falls back to the paths
            logger.debug("approval preview failed for %s", self.target, exc_info=True)
            return self.target


def _host_real(path):
    """Symlinks resolved on host paths so a workspace link cannot widen a workspace grant."""
    return Path(os.path.realpath(path)) if isinstance(path, Path) else path


def _split_workspace(paths: list[str], task_id: str) -> tuple[str, list[str]]:
    """``(workspace_root, outside_paths)``; an unresolvable path counts as outside."""
    from tools.file_tools_paths import _resolve_base_dir, _resolve_path_for_task
    try:
        root = _host_real(_resolve_base_dir(task_id))
    except Exception:  # health: allow BLE001 -- unknown root: every path is scoped on its own
        logger.debug("workspace root unresolvable for task %s", task_id, exc_info=True)
        root = None
    outside: list[str] = []
    for raw in paths:
        try:
            resolved = _host_real(_resolve_path_for_task(raw, task_id))
        except Exception:  # health: allow BLE001 -- fall back to the raw path (stricter grant)
            logger.debug("path unresolvable for the approval gate: %s", raw, exc_info=True)
            outside.append(raw)
            continue
        if root is None or not (resolved == root or root in resolved.parents):
            outside.append(str(resolved))
    return ("" if root is None else str(root)), sorted(set(outside))


def _path_scope(tool_name: str, paths: list[str], task_id: str, preview=None) -> _Scope:
    root, outside = _split_workspace(paths, task_id)
    target = f"{tool_name} {', '.join(paths)}"
    if not outside:
        return _Scope(f"{tool_name}@ws:{root}", target, f"{tool_name} calls inside the workspace {root}",
                      preview=preview)
    return _Scope(f"{tool_name}@path:{'|'.join(outside)}", target,
                  f"{tool_name} calls on {', '.join(outside)} only", preview=preview)


def _patch_paths(args: dict) -> list[str]:
    paths = [args["path"]] if isinstance(args.get("path"), str) and args["path"] else []
    if args.get("mode") == "patch" and isinstance(args.get("patch"), str):
        from tools.file_tools import _collect_v4a_header_paths
        collected = _collect_v4a_header_paths(args["patch"])
        if not isinstance(collected, str):  # a rejected header fails in the tool itself
            paths += collected[0]
    return paths or ["."]


def _str_arg(args: dict, name: str, default: str) -> str:
    value = args.get(name)
    return value if isinstance(value, str) and value.strip() else default


def _command_digest(command: str) -> str:
    return hashlib.sha256(command.strip().encode("utf-8")).hexdigest()[:16]


def _terminal_scope(tool_name: str, args: dict, task_id: str) -> _Scope:
    from tools.approval_detection import detect_dangerous_command
    command = _str_arg(args, "command", "")
    dangerous, _key, pattern = detect_dangerous_command(command)
    return _Scope(f"terminal:cmd:{_command_digest(command)}", command, "this exact command only",
                  warning=f"⚠️ Matches a dangerous pattern: {pattern}." if dangerous else "")


def _edit_scope(preview):
    def build(tool_name: str, args: dict, task_id: str) -> _Scope:
        paths = _patch_paths(args) if tool_name == "patch" else [_str_arg(args, "path", ".")]
        return _path_scope(tool_name, paths, task_id, preview=lambda: preview(args, task_id))
    return build


def _tool_scope(tool_name: str, args: dict, task_id: str) -> _Scope:
    from agent.redact import redact_sensitive_text
    preview = json.dumps(args, ensure_ascii=False, default=str)
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[:_PREVIEW_CHARS] + "…"
    return _Scope(f"tool:{tool_name}", f"{tool_name} {redact_sensitive_text(preview)}", f"every {tool_name} call")


_SCOPE_BUILDERS = {
    "read_file": lambda n, a, t: _path_scope(n, [_str_arg(a, "path", ".")], t),
    "search_files": lambda n, a, t: _path_scope(n, [_str_arg(a, "path", ".")], t),
    "write_file": _edit_scope(write_file_preview),
    "patch": _edit_scope(patch_preview),
    "terminal": _terminal_scope,
}


# --- Terminal hand-off: one approval per command ---------------------------------------------------------------------
# A gate-approved terminal command must not be asked again by the dangerous-pattern prompt inside
# the handler; floors (hardline, approvals.deny, sudo, runtime self-delete) still run first there.

_approved_commands: dict[str, set[str]] = {}
_approved_commands_lock = threading.Lock()


def _remember_approved_command(command: str) -> None:
    from tools.approval_context import get_current_session_key
    with _approved_commands_lock:
        _approved_commands.setdefault(get_current_session_key(), set()).add(_command_digest(command))


def consume_gate_approved_command(command: str) -> bool:
    """True once for a command this session's tool gate just approved."""
    from tools.approval_context import get_current_session_key
    session_key, digest = get_current_session_key(), _command_digest(command)
    with _approved_commands_lock:
        approved = _approved_commands.get(session_key)
        if not approved or digest not in approved:
            return False
        approved.discard(digest)
        return True


def forget_session(session_key: str) -> None:
    with _approved_commands_lock:
        _approved_commands.pop(session_key, None)


# --- Gate ------------------------------------------------------------------------------------------------------------

def _human_presence() -> Optional[bool]:
    """``None``: no human context (unattended, unchanged). ``True``: a surface can show the
    prompt now. ``False``: a human context whose surface is gone, e.g. a background subagent
    still running after the parent's turn unregistered its gateway approval callback."""
    from tools import approval
    from tools.approval_context import _should_fall_through_to_cli_approval, get_current_session_key
    callback, is_cli, is_gateway, is_ask = approval._presence()
    if not (is_cli or is_gateway or is_ask):
        return None
    if not (is_gateway or is_ask):
        return True
    notify_cb = approval._gateway_notify_cb(get_current_session_key())
    return notify_cb is not None or _should_fall_through_to_cli_approval(
        is_cli=is_cli, approval_callback=callback, notify_cb=notify_cb)


def asks_every_call() -> bool:
    """True when this context puts every tool call to a human who can answer now. A background
    subagent would outlive the turn that hosts its prompts, so ``delegate_task`` runs it
    synchronously instead."""
    from tools import approval
    from tools.approval_context import _get_approval_mode, _get_approval_scope
    if _get_approval_scope() != "all" or approval._yolo_active() or _get_approval_mode() == "off":
        return False
    return _human_presence() is True


def _terminal_floor_block(command: str) -> Optional[str]:
    """The handler's own floors (approvals.deny, hardline, sudo, runtime self-delete), run BEFORE
    the prompt: a command that can never run must not be put to the human first."""
    from tools import approval
    from tools.terminal_tool import _docker_has_host_access, _get_env_config
    config = _get_env_config()
    if approval._should_skip_container_guards(config.get("env_type", "local"), _docker_has_host_access(config)):
        blocked = approval._user_deny_block(command)
    else:
        blocked = approval._floor_block(command, sudo_guard=True)
    return None if blocked is None else str(blocked.get("message") or "BLOCKED: this command is not allowed.")


def _prior_decision(pattern_key: str) -> Optional[str]:
    """How the gate will allow this call without asking, if it will."""
    from tools import approval
    from tools.approval_context import _get_approval_mode, get_current_session_key
    if approval._yolo_active() or _get_approval_mode() == "off":
        return "bypass"
    if approval.is_approved(get_current_session_key(), pattern_key):
        return "permanent_grant" if approval._is_permanently_approved(pattern_key) else "session_grant"
    return None


def _answered_by(choice: Optional[str], prior: Optional[str]) -> str:
    if not choice:
        return prior or "auto"
    from tools import approval
    from tools.delegate_tool_config import _subagent_auto_approve
    callback, _is_cli, is_gateway, _is_ask = approval._presence()
    return "subagent_auto_approve" if callback is _subagent_auto_approve and not is_gateway else "user"


# --- Approval record: logged and attached to the tool result ---------------------------------------------------------

_records: dict[tuple[str, str], dict] = {}
_records_lock = threading.Lock()
_RECORDS_MAX = 256


def _record(tool_name: str, tool_call_id: str, record: dict) -> None:
    logger.info("tool approval: %s %s (%s)", tool_name, record["decision"], record)
    if not tool_call_id:
        return
    with _records_lock:
        _records[(tool_call_id, tool_name)] = record
        while len(_records) > _RECORDS_MAX:
            _records.pop(next(iter(_records)))


def pop_record(tool_name: str, tool_call_id: Optional[str]) -> Optional[dict]:
    """This call's approval record, once. The executor stores it in the tool message's
    ``display_metadata`` (durable, never sent to the model), so recording a decision does not
    change what the model reads."""
    with _records_lock:
        return _records.pop((tool_call_id or "", tool_name), None)


def _evaluate(tool_name: str, args: dict, task_id: str, tool_call_id: str) -> Optional[str]:
    from agent.i18n import t
    from tools.approval_context import _get_approval_scope
    if tool_name in _EXEMPT_TOOLS or _get_approval_scope() != "all":
        return None
    presence = _human_presence()
    if presence is None:
        return None
    from tools import approval
    if tool_name == "terminal":
        floor = _terminal_floor_block(_str_arg(args, "command", ""))
        if floor is not None:
            _record(tool_name, tool_call_id, {"decision": "blocked", "by": "rule"})
            return floor
    scope = _SCOPE_BUILDERS.get(tool_name, _tool_scope)(tool_name, args, task_id)
    pattern_key = f"tool_scope:{scope.key}"
    prior = _prior_decision(pattern_key)
    if not presence and prior is None:
        _record(tool_name, tool_call_id, {"decision": "denied", "by": "unreachable", "scope": scope.grant})
        return (f"BLOCKED: {tool_name} needs the user's approval, but the user cannot be reached from here "
                "(no approval prompt is attached, e.g. a background subagent after the parent's turn ended). "
                "The call was denied and did not run. Do NOT retry it or try another route; report that it "
                "needs approval and stop.")
    description = f"Tool call: {tool_name}. Allowing it for this session (or always) covers {scope.grant}."
    if scope.warning:
        title = t("cli.tui.approval_title")
        description = f"{scope.warning} {description}"
    else:
        title = t("approval.command_title") if tool_name == "terminal" else t("approval.tool_title", tool=tool_name)
    blocked = f"BLOCKED: {tool_name} was not approved ({scope.grant})."
    result = approval._run_approval_gate(
        pattern_key=pattern_key, description=description,
        display_target=scope.target if prior is not None else scope.shown(),
        subject=f"Tool '{tool_name}' requires approval", advice="Find an alternative approach.",
        autoapprove_log_prefix=f"tool call '{tool_name}'", fail_closed_when_no_human=True,
        no_human_block_message=blocked, title=title,
    )
    if not result.get("approved"):
        _record(tool_name, tool_call_id, {"decision": "denied", "scope": scope.grant})
        return str(result.get("message") or blocked)
    choice = result.get("choice")
    _record(tool_name, tool_call_id, {
        "decision": "approved", "by": _answered_by(choice, prior),
        **({"choice": choice} if choice else {}), "scope": scope.grant,
        **({"flag": scope.warning} if scope.warning else {}),
    })
    if tool_name == "terminal":
        _remember_approved_command(_str_arg(args, "command", ""))
    return None


def evaluate_tool_gate(tool_name: str, args: Any, task_id: str = "", tool_call_id: str = "") -> Optional[str]:
    """Block message when a present human did not approve this call, else ``None``. Fails
    closed: every caller treats an exception from the pre-call verdict as "proceed"."""
    try:
        return _evaluate(tool_name, args if isinstance(args, dict) else {}, task_id or "default", tool_call_id)
    except Exception:  # health: allow BLE001 -- fail-closed boundary, traceback logged
        logger.exception("tool approval gate failed for %s", tool_name)
        return f"BLOCKED: the approval gate failed for {tool_name}; the call was not run."


def _resolve_directive(details, tool_name: str) -> tuple[Optional[str], bool]:
    """``(block_message, human_approved)`` for a plugin directive: ``block`` blocks with its
    message; an ``approve`` whose gate errors, denies, or times out is blocked."""
    if details.action == "block":
        return details.message, False
    if details.action != "approve":
        return None, False
    try:
        from tools.approval import request_tool_approval
        result = request_tool_approval(tool_name, details.message or "", rule_key=details.rule_key or tool_name)
    except Exception:  # health: allow BLE001 -- fail-closed: never run a plugin-flagged action ungated
        logger.exception("plugin approval gate failed for %s", tool_name)
        return f"BLOCKED: plugin approval gate failed for {tool_name}", False
    if not result.get("approved"):
        return str(result.get("message") or f"BLOCKED: plugin approval required for {tool_name}"), False
    return None, True


def resolve_pre_tool_block(details, tool_name: str, args: Any, *, task_id: str = "", turn_id: str = "",
                           tool_call_id: str = "", session_id: str = "") -> Optional[str]:
    """The ONE pre-call verdict: the plugin directive, then the full-scope tool gate on the
    final (possibly plugin-modified) args. A plugin ``approve`` the human granted already
    covers this call, so it is not asked twice."""
    from tools.approval_context import reset_current_observability_context, set_current_observability_context
    tokens = None
    try:
        tokens = set_current_observability_context(turn_id=turn_id, tool_call_id=tool_call_id, session_id=session_id)
    except Exception:  # health: allow BLE001 -- observability only; the verdict must still run
        logger.debug("approval observability context unavailable", exc_info=True)
    try:
        block, human_approved = _resolve_directive(details, tool_name)
        if block is not None or human_approved:
            return block
        final_args = details.modified_args if details.modified_args is not None else args
        return evaluate_tool_gate(tool_name, final_args, task_id, tool_call_id)
    finally:
        if tokens is not None:
            reset_current_observability_context(tokens)
