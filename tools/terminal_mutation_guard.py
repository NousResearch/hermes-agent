"""Canonical terminal registration wrapper for interpreter-backed mutation effects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tools import terminal_tool as _terminal_owner
from tools.mutation_effect_guard import MutationEffect, MutationEffectGuard
from tools.registry import registry

logger = logging.getLogger(__name__)


def _terminal_effect(
    args: dict[str, Any],
    kwargs: dict[str, Any],
    *,
    source_root: Path | None = None,
    active: bool | None = None,
) -> MutationEffect | None:
    """Resolve and inspect the exact local terminal context before delegation."""

    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    from tools.self_repo_guard import get_running_source_root, guard_active

    if active is None:
        active = guard_active()
    if not active:
        return None

    config = _terminal_owner._get_env_config()
    if config.get("env_type") != "local":
        return None

    root = source_root if source_root is not None else get_running_source_root()
    if root is None:
        return None

    from tools.approval import get_current_session_key

    task_id = kwargs.get("task_id") or ""
    overrides = _terminal_owner.resolve_task_overrides(task_id)
    default_cwd = (
        overrides.get("cwd")
        or _terminal_owner.get_session_cwd(task_id)
        or config.get("cwd")
    )
    session_key = get_current_session_key(default="") or task_id
    guard_cwd = _terminal_owner._resolve_command_cwd(
        workdir=args.get("workdir"),
        default_cwd=default_cwd,
        session_key=session_key,
    )
    return MutationEffectGuard(Path(root)).detect(command, guard_cwd)


def _blocked_result(effect: MutationEffect) -> str:
    message = effect.message
    if effect.origin != "terminal command":
        message = (
            f"{message} Interpreter or script indirection does not change this "
            "boundary; the recovered operation was not executed."
        )
    return json.dumps(
        {
            "output": "",
            "exit_code": 1,
            "error": message,
            "status": "blocked",
        },
        ensure_ascii=False,
    )


def _handle_terminal_with_effect_guard(args: dict[str, Any], **kwargs: Any) -> str:
    effect = _terminal_effect(args, kwargs)
    # Preserve the existing direct-command owner and its exact terminal timing.
    # This wrapper owns only effects hidden behind an interpreter boundary.
    if effect is not None and effect.origin != "terminal command":
        logger.warning(
            "Blocked interpreter-backed self-repo mutation: %s (%s)",
            effect.operation,
            effect.origin,
        )
        return _blocked_result(effect)
    return _terminal_owner._handle_terminal(args, **kwargs)


# This module is discovered as a built-in tool registration. Importing the
# terminal owner above materializes its schema/handler first; this registration
# then replaces only the handler while retaining the canonical contract.
registry.register(
    name="terminal",
    toolset="terminal",
    schema=_terminal_owner.TERMINAL_SCHEMA,
    handler=_handle_terminal_with_effect_guard,
    check_fn=_terminal_owner.check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
