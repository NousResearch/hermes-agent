"""Canonical effect contracts. Unknown tools are write-capable, never implicit reads.

Registry metadata is operator/plugin supplied; MCP readOnlyHint remains only a
hint and does not bypass MCP trust gates. Idempotency requires a concrete key
contract, not merely an idempotentHint or a tool name.
"""
from enum import Enum
import json


class ToolEffect(str, Enum):
    PURE_READ = "PURE_READ"
    DISCOVERY = "DISCOVERY"
    IDEMPOTENT_WRITE = "IDEMPOTENT_WRITE"
    MUTATION = "MUTATION"
    INTERACTIVE = "INTERACTIVE"
    COGNITIVE = "COGNITIVE"


# Single compatibility table for built-ins predating registry effect metadata.
_BUILTINS = {
    **dict.fromkeys(("read_file", "search_files", "browser_snapshot", "browser_get_images",
                     "session_search"), ToolEffect.PURE_READ),
    **dict.fromkeys(("tool_search", "tool_describe", "web_search", "web_extract",
                     "browser_extract_items"), ToolEffect.DISCOVERY),
    "clarify": ToolEffect.INTERACTIVE,
    "delegate_task": ToolEffect.COGNITIVE,
}
READ_EFFECTS = frozenset({ToolEffect.PURE_READ, ToolEffect.DISCOVERY})
WRITE_EFFECTS = frozenset({ToolEffect.MUTATION, ToolEffect.IDEMPOTENT_WRITE})


def tool_contract(name, *, scope=None, schema=None):
    from tools.registry import registry
    entry = registry.get_entry(name, scope=scope)
    metadata = dict(schema or {})
    if entry is not None:
        metadata = {**entry.schema, "effect": entry.effect or entry.schema.get("effect"),
                    "idempotency_key": entry.idempotency_key or entry.schema.get("idempotency_key"),
                    "routes": entry.routes or entry.schema.get("routes")}
    effect = metadata.get("effect") or metadata.get("x-hermes-effect")
    if effect:
        try:
            effect = ToolEffect(effect)
        except (ValueError, TypeError):
            effect = ToolEffect.MUTATION
        if effect == ToolEffect.IDEMPOTENT_WRITE and not isinstance(metadata.get("idempotency_key"), str):
            effect = ToolEffect.MUTATION
    elif isinstance(metadata.get("annotations"), dict) and metadata["annotations"].get("readOnlyHint") is True:
        effect = ToolEffect.PURE_READ
    else:
        effect = _BUILTINS.get(name, ToolEffect.MUTATION)
    return effect, metadata


def tool_effect(name, **kwargs):
    return tool_contract(name, **kwargs)[0]


def unwrap_call(call):
    name = call.function.name
    args = call.function.arguments
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except (TypeError, ValueError):
        args = {}
    if name == "tool_call":
        return args.get("name", ""), args.get("arguments", {})
    return name, args
