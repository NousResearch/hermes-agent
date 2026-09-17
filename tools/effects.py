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
                     "session_search", "skill_view", "skills_list", "read_terminal"), ToolEffect.PURE_READ),
    **dict.fromkeys(("tool_search", "tool_describe", "web_search", "web_extract",
                     "browser_extract_items"), ToolEffect.DISCOVERY),
    "clarify": ToolEffect.INTERACTIVE,
    "delegate_task": ToolEffect.COGNITIVE,
    "vision_analyze": ToolEffect.COGNITIVE,
}
READ_EFFECTS = frozenset({ToolEffect.PURE_READ, ToolEffect.DISCOVERY})
WRITE_EFFECTS = frozenset({ToolEffect.MUTATION, ToolEffect.IDEMPOTENT_WRITE})


def builtin_effect(name):
    """Explicit conservative default for authenticated first-party registrations.

    A new builtin is write-capable until its owner supplies safer metadata. This
    never classifies tools by get/list prefixes or grants trust to plugins.
    """
    return _BUILTINS.get(name, ToolEffect.MUTATION)


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
    metadata.setdefault("name", name)
    metadata.setdefault("result_schema", None)
    metadata.setdefault("default_verifier", "external_readback" if effect in WRITE_EFFECTS else "result_contract")
    metadata.setdefault("supports", {"cancellation": False, "event_completion": False,
                                    "retry": effect in READ_EFFECTS, "rollback": False,
                                    "semantic_readiness": False})
    metadata.setdefault("artifact_encoding", "structured_reference")
    metadata.setdefault("runtime_family", "unknown")
    metadata.setdefault("idempotency", effect == ToolEffect.IDEMPOTENT_WRITE)
    return effect, metadata


def tool_effect(name, **kwargs):
    return tool_contract(name, **kwargs)[0]


def capability_key(provider, channel, method):
    """Read and write evidence never share an authority key."""
    return json.dumps([provider, channel, method.upper()], separators=(",", ":"))


def observe_capability(capabilities, name, args, raw):
    """Owner-declared method semantics only; never infer authority from arbitrary JS."""
    _, contract = tool_contract(name)
    declaration = contract.get("capability")
    if not isinstance(declaration, dict):
        return
    method = declaration.get("method") or args.get(declaration.get("method_field"))
    if not isinstance(method, str) or not declaration.get("provider") or not declaration.get("channel"):
        return
    try:
        result = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return
    if not isinstance(result, dict):
        return
    status = result.get("status_code", result.get("http_status", result.get("status")))
    if not isinstance(status, int):
        return
    key = capability_key(declaration["provider"], declaration["channel"], method)
    if status in {401, 403, 405}:
        capabilities[key] = {"status": "REJECTED", "http_status": status}
    elif 200 <= status < 300 and capabilities.get(key, {}).get("status") != "REJECTED":
        capabilities[key] = {"status": "VERIFIED", "http_status": status}


def operation_capability(capabilities, name, args):
    _, contract = tool_contract(name)
    declaration = contract.get("capability") or {}
    method = declaration.get("method") or args.get(declaration.get("method_field"))
    if not isinstance(method, str):
        return "UNKNOWN"
    return capabilities.get(capability_key(declaration.get("provider"), declaration.get("channel"), method), {}).get("status", "UNKNOWN")


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
