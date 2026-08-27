"""Computer use toolset — universal (any-model) desktop control via cua-driver.

Drives apps through cua-driver's background primitive (focus-without-raise + pid-scoped
event posting): it does NOT steal the user's cursor, keyboard focus, or Space. Plain
OpenAI function-calling schema; vision models get SOM captures (numbered overlays + AX
tree) and click by index, non-vision models use the AX tree alone. Model-facing guidance
lives in the schema description and each action result's `verdict`.

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
eviction in the Anthropic adapter, image-aware token estimation, approval
hook, and the skill) live alongside this package. See
agent/anthropic_adapter.py for the salvaged hunks from PR #4562. Model-facing
guidance (workflow, background-first, the escalate ladder, safety) lives in
the tool's schema description and each action result's `verdict`, not a
separate system-prompt block.
"""


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from __future__ import annotations  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'check_computer_use_requirements': ('tools.computer_use.tool', 'check_computer_use_requirements'),
    'get_computer_use_schema': ('tools.computer_use.tool', 'get_computer_use_schema'),
    'handle_computer_use': ('tools.computer_use.tool', 'handle_computer_use'),
    'release_computer_use_session': ('tools.computer_use.tool', 'release_computer_use_session'),
    'set_approval_callback': ('tools.computer_use.tool', 'set_approval_callback'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
