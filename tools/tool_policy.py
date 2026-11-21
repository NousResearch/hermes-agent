"""Per-tool approval policy: ``approvals.tools`` maps tool names (fnmatch globs) to allow / ask / deny.

Enforced at both tool dispatch funnels (``agent.tool_executor._dispatch_authorized_once`` for the agent
loop, ``model_tools._pre_dispatch_guards`` for direct callers: execute_code RPC, connector batches, Tool
Search re-dispatch) on the FINAL arguments, after plugin ``modify`` hooks.

Two invariants:

- An ask approval is bound to the exact invocation. The grant key carries a digest of the canonical
  arguments and the prompt shows the redacted arguments, so a session/always answer for
  ``send_message(to=alice)`` never covers ``send_message(to=bob)``.
- A policy that cannot be established blocks. A reader or evaluator error, a non-mapping
  ``approvals.tools`` or an unknown verb refuses the call; only an absent or empty map means "no policy".
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_VERBS = ("allow", "ask", "deny")
_DISPLAY_ARGS_CHARS = 600


class ToolPolicyUnavailable(Exception):
    """``approvals.tools`` is configured but cannot be read or understood."""


def _policy_map() -> dict[str, str]:
    from hermes_cli.config import load_config_readonly

    # A parse failure serves the last-known-good config (which still carries the user's policy); a
    # FailedConfigRead of bare defaults only happens when no copy of the file ever parsed, so no
    # policy from it was ever in force.
    raw = (load_config_readonly().get("approvals") or {}).get("tools")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ToolPolicyUnavailable(f"approvals.tools must be a mapping of tool name -> allow/ask/deny, got {type(raw).__name__}")
    policy: dict[str, str] = {}
    for key, value in raw.items():
        verb = value.strip().lower() if isinstance(value, str) else value
        if verb not in _VERBS:
            raise ToolPolicyUnavailable(f"approvals.tools[{key!r}] = {value!r} is not one of allow/ask/deny")
        if isinstance(key, str) and key.strip():
            policy[key.strip().lower()] = verb
    return policy


def resolve_tool_policy(tool_name: str) -> str | None:
    """The verb for *tool_name*: an exact (case-insensitive) key beats globs; globs in declaration order."""
    policy = _policy_map()
    name = tool_name.lower()
    if name in policy:
        return policy[name]
    return next((verb for pattern, verb in policy.items() if fnmatch.fnmatchcase(name, pattern)), None)


def _invocation_digest(tool_name: str, args: Any) -> str:
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _display_invocation(tool_name: str, args: Any) -> str:
    from agent.redact import redact_approval_display

    shown = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    if len(shown) > _DISPLAY_ARGS_CHARS:
        shown = shown[:_DISPLAY_ARGS_CHARS] + f"… ({len(shown)} chars)"
    return redact_approval_display(f"{tool_name} {shown}", force=True)


def _unavailable_message(tool_name: str, reason: str) -> str:
    return (f"BLOCKED: '{tool_name}' was not run because the per-tool approval policy (approvals.tools in "
            f"config.yaml) could not be evaluated: {reason}. Do NOT retry; ask the user to fix the policy.")


def tool_policy_block(tool_name: str, args: Any, approval_callback=None) -> str | None:
    """Block message for this call under ``approvals.tools``, or None when it may run."""
    try:
        verb = resolve_tool_policy(tool_name)
    except ToolPolicyUnavailable as exc:
        logger.warning("approvals.tools unusable: %s", exc)
        return _unavailable_message(tool_name, str(exc))
    except Exception as exc:  # reader fault: a configured deny/ask must not vanish with it
        logger.warning("approvals.tools could not be read: %s", exc, exc_info=True)
        return _unavailable_message(tool_name, f"{type(exc).__name__}: {exc}")
    if verb is None or verb == "allow":
        return None
    if verb == "deny":
        return (f"BLOCKED: the '{tool_name}' tool is set to 'deny' in the user's per-tool approval policy "
                "(approvals.tools in config.yaml). It cannot be used, not even with --yolo or approvals.mode: off. "
                "Do NOT retry this tool.")
    from tools.approval import _run_approval_gate

    description = f"approvals.tools: '{tool_name}' asks before every run"
    result = _run_approval_gate(
        pattern_key=f"tool_policy:{tool_name}:{_invocation_digest(tool_name, args)}",
        description=description, display_target=_display_invocation(tool_name, args),
        approval_callback=approval_callback, subject=f"Tool '{tool_name}' requires approval ({description})",
        advice="Find an alternative approach.", require_human=True,
        autoapprove_log_prefix=f"approvals.tools ask for '{tool_name}'",
        no_human_block_message=(f"BLOCKED: '{tool_name}' is set to 'ask' in approvals.tools and no user is "
                                "present to approve it. Do NOT retry; report that it needs the user's approval."),
    )
    if result.get("approved"):
        return None
    return result.get("message") or f"BLOCKED: '{tool_name}' was not approved (approvals.tools: ask)."
