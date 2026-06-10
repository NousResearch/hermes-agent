"""Shim for tool discovery. Registers `computer_use` with tools.registry.

The real implementation lives in the `tools/computer_use/` package to keep
the file structure clean. This shim exists because tools.registry auto-imports
`tools/*.py` — we need a top-level module to trigger the registration.
"""

from __future__ import annotations

from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import (
    check_computer_use_requirements,
    handle_computer_use,
    set_approval_callback,
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
        "Universal desktop control. Works with any tool-capable model "
        "(Anthropic, OpenAI, OpenRouter, local vLLM, etc.). macOS: "
        "background computer-use via cua-driver (does NOT steal the user's "
        "cursor or keyboard focus). Windows: UI Automation + SendInput "
        "(actions briefly foreground the target window)."
    ),
)


__all__ = [
    "handle_computer_use",
    "set_approval_callback",
    "check_computer_use_requirements",
]
