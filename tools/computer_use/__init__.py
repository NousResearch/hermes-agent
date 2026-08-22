"""Computer use toolset — universal (any-model) macOS desktop control.

Architecture
------------
This toolset drives macOS apps through cua-driver's background computer-use
primitive (SkyLight private SPIs for focus-without-raise + pid-scoped event
posting). Unlike #4562's pyautogui backend, it does NOT steal the user's
cursor, keyboard focus, or Space — the agent and the user can co-work on the
same machine.

Unlike #4562's Anthropic-native `computer_20251124` tool, the schema here is
a plain OpenAI function-calling schema that every tool-capable model can
drive. Vision models get SOM (set-of-mark) captures — a screenshot with
numbered overlays on every interactable element plus the AX tree — so they
click by element index instead of pixel coordinates. Non-vision models can
drive via the AX tree alone.

Wiring
------
* `tool.py`       — registers the `computer_use` tool via tools.registry.
* `backend.py`    — abstract `ComputerUseBackend`; swappable implementation.
* `cua_backend.py`— default backend; speaks MCP over stdio to `cua-driver`.
* `schema.py`     — shared schema + docstring for the generic `computer_use`
                    tool. Model-agnostic.
* `capture.py`    — screenshot post-processing (PNG coercion, sizing, SOM
                    overlay if the backend did not).

The outer integration points (multimodal tool-result plumbing, screenshot
eviction in the Anthropic adapter, image-aware token estimation, the
COMPUTER_USE_GUIDANCE prompt block, approval hook, and the skill) live
alongside this package. See agent/anthropic_adapter.py and
agent/prompt_builder.py for the salvaged hunks from PR #4562.
"""

from __future__ import annotations

# Install Linux native-Wayland policy before the tool factory first imports the
# CUA backend. The installer is a no-op off Linux and idempotent on repeated
# package imports.
from tools.computer_use.wayland_runtime import (
    install_wayland_runtime_policy as _install_wayland_runtime_policy,
)

_install_wayland_runtime_policy()
del _install_wayland_runtime_policy

# Re-export the public surface so `from tools.computer_use import ...` works.
from tools.computer_use.tool import (  # noqa: F401,E402
    handle_computer_use,
    release_computer_use_session,
    set_approval_callback,
    check_computer_use_requirements,
    get_computer_use_schema,
    release_computer_use_session,
)
