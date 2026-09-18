"""Discovery shim: tools.registry auto-imports ``tools/*.py``, so this top-level
module registers ``computer_use`` for the ``tools/computer_use/`` package."""

from __future__ import annotations

from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import (
    check_computer_use_requirements, handle_computer_use, release_computer_use_session, set_approval_callback,
)
from tools.registry import registry


registry.register(
    name="computer_use",
    toolset="computer_use",
    schema=COMPUTER_USE_SCHEMA,
    handler=lambda args, **kw: handle_computer_use(args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=(
        "Desktop and browser control. For a multi-step GUI/browser goal prefer "
        "action=run_goal so System-One (Jev when keyed) loops without a frontier "
        "model; fail-open falls back to capture/click. Background by default — "
        "does not steal the user's cursor or keyboard focus."
    ),
)


__all__ = ["handle_computer_use", "release_computer_use_session", "set_approval_callback", "check_computer_use_requirements"]
