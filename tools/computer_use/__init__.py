"""Computer use toolset — universal (any-model) desktop control via cua-driver.

Drives apps through cua-driver's background primitive (focus-without-raise + pid-scoped
event posting): it does NOT steal the user's cursor, keyboard focus, or Space. Plain
OpenAI function-calling schema; vision models get SOM captures (numbered overlays + AX
tree) and click by index, non-vision models use the AX tree alone. Model-facing guidance
lives in the schema description and each action result's `verdict`.

Modules: `tool.py` (handler, approval gate, response shaping), `backend.py` (abstract
`ComputerUseBackend` + result dataclasses), `cua_backend.py` (default MCP-over-stdio
backend + `cua_backend_parse`/`_session`/`_daemon` siblings), `schema.py` (byte-frozen).
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
