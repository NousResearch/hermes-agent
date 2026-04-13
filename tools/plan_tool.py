"""Plan Mode tool for Hermes Agent (Phase B2).

Allows entering/exiting plan mode and querying status.
"""

import importlib.util
import json
import os
import sys

from tools.registry import registry

PLAN_MODE_HOOK_MODULE_NAME = "plan_mode_hook"
PLAN_MODE_HOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "plugins",
    "hongxing-enhancements",
    "plan_mode_hook.py",
)

PLAN_MODE_SCHEMA = {
    "name": "plan_mode",
    "description": "Enter or exit plan mode. In plan mode only read-only and planning tools are available.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["enter", "exit", "status"],
                "description": "Action to perform: enter plan mode, exit plan mode, or check status.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session identifier. Defaults to 'default'.",
            },
        },
        "required": ["action"],
    },
}


def _load_plan_mode_hook():
    """Load the plan mode hook once and reuse the module instance."""
    mod = sys.modules.get(PLAN_MODE_HOOK_MODULE_NAME)
    if mod is not None:
        return mod

    spec = importlib.util.spec_from_file_location(
        PLAN_MODE_HOOK_MODULE_NAME,
        PLAN_MODE_HOOK_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {PLAN_MODE_HOOK_MODULE_NAME} from {PLAN_MODE_HOOK_PATH}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[PLAN_MODE_HOOK_MODULE_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(PLAN_MODE_HOOK_MODULE_NAME, None)
        raise
    return mod


def plan_mode_handler(args: dict, **kwargs) -> str:
    """Handle plan_mode tool calls."""
    action = args.get("action", "status")
    session_id = args.get("session_id") or "default"
    mod = _load_plan_mode_hook()

    if action == "enter":
        mod.enter_plan_mode(session_id=session_id)
        return json.dumps({"success": True, "plan_mode": True}, ensure_ascii=False)
    elif action == "exit":
        mod.exit_plan_mode(session_id=session_id)
        return json.dumps({"success": True, "plan_mode": False}, ensure_ascii=False)
    else:
        return json.dumps({"plan_mode": mod.is_active(session_id=session_id)}, ensure_ascii=False)


registry.register(
    name="plan_mode",
    toolset="core",
    schema=PLAN_MODE_SCHEMA,
    handler=plan_mode_handler,
    description="Enter or exit plan mode",
    emoji="📋",
    allowed_in_plan_mode_default=True,
)
