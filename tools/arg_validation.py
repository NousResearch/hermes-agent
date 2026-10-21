"""Reject missing required arguments and report unsupported arguments that were not used.

The model-facing schema is deliberately smaller than some handlers' accepted input.
Compatibility rules below preserve those known arguments/forms without exempting any
whole tool from validation. Internal dispatch kwargs are never model arguments.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Accepted but unadvertised inputs, documented by the corresponding handlers.
# These extend accepted keys only; other unknown keys are omitted with a notice.
_COMPATIBILITY_ARGUMENTS = {
    "write_file": frozenset({"cross_profile"}),  # sandbox-mirror override
    "patch": frozenset({"cross_profile", "mode", "patch"}),  # model-dependent V4A schema
    "terminal": frozenset({"notify_on_complete", "watch_patterns"}),  # legacy notify interface
    "delegate_task": frozenset({"goal", "context", "output_schema", "role", "max_iterations"}),
    "clarify": frozenset({"question", "choices", "multi_select"}),  # legacy single question
    "cronjob_manage": frozenset({"include_disabled", "monitor_script", "monitor_url"}),
    "image_generate": frozenset({"image_url", "reference_image_urls", "upscale"}),  # capability-gated
    "video_generate": frozenset({"image_url", "reference_image_urls", "negative_prompt", "audio", "seed", "upscale"}),
    "browser_exec": frozenset({"local"}),  # dynamically advertised backend selector
    "kanban_create": frozenset({"project_id", "session_id"}),
    "kanban_attach_url": frozenset({"title"}),  # legacy filename alias
    # Shipped A2A plugin's explicitly supported aliases.
    "a2a_call": frozenset({"agent_name", "name", "text", "task", "contextId"}),
    "a2a_history": frozenset({"contextId"}),
    "a2a_orchestrate": frozenset({"task"}),
}

_REQUIRED_ALIASES = {
    "clarify": {"questions": ("question",)},
    "delegate_task": {"tasks": ("goal",)},
    "a2a_call": {"agent": ("agent_name", "name"), "message": ("text", "task")},
    "a2a_history": {"context_id": ("contextId",)},
    "a2a_orchestrate": {"message": ("task",)},
}


def _validation_parameters(tool_name: str) -> Optional[dict]:
    schema = registry.get_schema(tool_name)
    if not isinstance(schema, dict):
        return None  # Unknown/unregistered tools keep their existing routing error.
    fn = schema.get("function") if schema.get("type") == "function" else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict):
        return None
    # Dynamic schemas can advertise parameters not in the static registration.
    # Retain static properties too: some dynamic schemas hide accepted legacy inputs.
    entry = registry.get_entry(tool_name)
    override = getattr(entry, "dynamic_schema_overrides", None)
    if override is not None:
        try:
            dynamic = override()
            live = dynamic.get("parameters") if isinstance(dynamic, dict) else None
            if isinstance(live, dict):
                params = {**params, **live, "properties": {
                    **(params.get("properties") or {}), **(live.get("properties") or {}),
                }}
        except Exception:
            logger.debug("Dynamic argument schema unavailable for %s; using registration", tool_name, exc_info=True)
    return params


def _argument_error(tool_name: str, args: dict[str, Any], message: str) -> str:
    """Keep the shared rejection, appending optional correction advice owned by the tool."""
    entry = registry.get_entry(tool_name)
    hint_fn = getattr(entry, "argument_error_hint", None)
    if hint_fn is not None:
        try:
            hint = hint_fn(dict(args))
            if isinstance(hint, str) and hint.strip():
                message += " " + hint.strip()
        except Exception:
            # Broken advice must never bypass validation or obscure the original error.
            logger.debug("Argument correction hint unavailable for %s", tool_name, exc_info=True)
    return tool_error(message)


def validate_tool_args(tool_name: str, args: dict[str, Any]) -> Optional[str]:
    """Check required argument names; return an error JSON or ``None``.

    Runs after coercion and middleware. Parameter values remain the handler's contract
    (and deferred tools' existing JSON Schema validator). No alias guessing or automatic
    retry. Unsupported keys are handled by split_tool_args.
    """
    params = _validation_parameters(tool_name)
    if params is None or not isinstance(args, dict):
        return None
    properties = params.get("properties")
    if not isinstance(properties, dict):
        return None  # No declared key set to close; open object schemas remain open.

    required = params.get("required") or []
    if tool_name == "patch":
        # Both forms are accepted regardless of the main model's advertised schema.
        required = ["patch"] if args.get("mode", "replace") == "patch" else ["path", "old_string", "new_string"]
    aliases = _REQUIRED_ALIASES.get(tool_name, {})
    missing = [key for key in required
               if key not in args and not any(alias in args for alias in aliases.get(key, ()))]
    valid = ", ".join(properties) or "none"
    if missing:
        return _argument_error(
            tool_name, args,
            f"{tool_name} is missing required argument(s): {', '.join(missing)}. "
            f"The tool was NOT invoked. Valid arguments: {valid}.")

    return None


def split_tool_args(tool_name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Separate unsupported top-level keys without mutating the caller's payload."""
    params = _validation_parameters(tool_name)
    if params is None or not isinstance(args, dict):
        return args, []
    properties = params.get("properties")
    if not isinstance(properties, dict):
        return args, []
    additional = params.get("additionalProperties", False)
    if additional is True or isinstance(additional, dict):
        return args, []
    patterns = params.get("patternProperties") or {}
    extra_keys = _COMPATIBILITY_ARGUMENTS.get(tool_name, ())
    ignored = [key for key in args
               if key not in properties and key not in extra_keys
               and not any(re.search(pattern, key) for pattern in patterns)]
    if not ignored:
        return args, []
    return {key: value for key, value in args.items() if key not in ignored}, ignored


def with_ignored_arguments(result: Any, tool_name: str, ignored: list[str]) -> Any:
    """Preserve the tool result and explicitly state which options did not take effect."""
    if not ignored:
        return result
    notice = (f"Unsupported argument(s) for {tool_name}: {', '.join(ignored)}. "
              "These arguments WERE NOT USED. The tool was invoked with supported arguments only; "
              "do not assume the requested options took effect.")
    if isinstance(result, dict):
        payload = dict(result)
    else:
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            payload = result
        if not isinstance(payload, dict):
            payload = {"output": payload}
    if payload.get("_multimodal") is True and isinstance(payload.get("content"), list):
        payload["content"] = [*payload["content"], {"type": "text", "text": notice}]
        if payload.get("text_summary"):
            payload["text_summary"] = str(payload["text_summary"]) + "\n" + notice
    previous = payload.get("_warning")
    payload["_warning"] = f"{previous}\n{notice}" if previous else notice
    return payload if isinstance(result, dict) else json.dumps(payload, ensure_ascii=False)
