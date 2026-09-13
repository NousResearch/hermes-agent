"""MCP tool schema conversion and naming: JSON-schema normalisation for provider
compatibility, mcp__server__tool naming, utility-tool schemas, include/exclude filters and
description injection scanning."""

import logging
import fnmatch
import re
from typing import Any, List
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import mcp_field

logger = logging.getLogger("tools.mcp_tool")

# Prompt-injection indicators in MCP tool descriptions. WARNING-level only: log but never
# block, since false positives would break legitimate servers.
_MCP_INJECTION_PATTERNS = [
    (re.compile(pattern, re.I), reason)
    for pattern, reason in (
        (r"ignore\s+(all\s+)?previous\s+instructions", "prompt override attempt ('ignore previous instructions')"),
        (r"you\s+are\s+now\s+a", "identity override attempt ('you are now a...')"),
        (r"your\s+new\s+(task|role|instructions?)\s+(is|are)", "task override attempt"),
        (r"system\s*:\s*", "system prompt injection attempt"),
        (r"<\s*(system|human|assistant)\s*>", "role tag injection attempt"),
        (r"do\s+not\s+(tell|inform|mention|reveal)", "concealment instruction"),
        (r"(curl|wget|fetch)\s+https?://", "network command in description"),
        (r"base64\.(b64decode|decodebytes)", "base64 decode reference"),
        (r"exec\s*\(|eval\s*\(", "code execution reference"),
        (r"import\s+(subprocess|os|shutil|socket)", "dangerous import reference"))]


def _scan_mcp_description(server_name: str, tool_name: str, description: str) -> List[str]:
    """Scan a tool description for injection patterns; returns finding strings (empty =
    clean) and logs a warning when any match."""
    if not description:
        return []
    findings = [reason for pattern, reason in _MCP_INJECTION_PATTERNS if pattern.search(description)]
    if findings:
        logger.warning("MCP server '%s' tool '%s': suspicious description content — %s. Description: %.200s",
                       server_name, tool_name, "; ".join(findings), description)
    return findings


_EMPTY_OBJECT_SCHEMA = {"type": "object", "properties": {}}


def _rewrite_local_refs(node):
    """Promote legacy ``definitions`` to ``$defs`` (Moonshot rejects the draft-07 form) — ONLY
    where it is a JSON Schema meta-keyword, never as a property NAME inside
    ``properties``/``patternProperties``: a parameter legitimately named ``definitions``
    rewritten to ``$defs`` would 400 the whole tool array (Anthropic/OpenAI forbid ``$`` in
    property names)."""
    if isinstance(node, list):
        return [_rewrite_local_refs(item) for item in node]
    if not isinstance(node, dict):
        return node
    normalized = {}
    for key, value in node.items():
        if key in ("properties", "patternProperties") and isinstance(value, dict):
            normalized[key] = {name: _rewrite_local_refs(schema) for name, schema in value.items()}
        else:
            normalized["$defs" if key == "definitions" else key] = _rewrite_local_refs(value)
    ref = normalized.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/definitions/"):
        normalized["$ref"] = "#/$defs/" + ref[len("#/definitions/"):]
    return normalized


def _repair_object_shape(node):
    """Recursively fill a missing object ``type``, ensure ``properties`` (so ``required``
    can't dangle) and prune ``required`` to names present in ``properties`` (Gemini 400s
    otherwise)."""
    if isinstance(node, list):
        return [_repair_object_shape(item) for item in node]
    if not isinstance(node, dict):
        return node
    repaired = {k: _repair_object_shape(v) for k, v in node.items()}
    if not repaired.get("type") and ("properties" in repaired or "required" in repaired):
        repaired["type"] = "object"
    if repaired.get("type") == "object":
        if not isinstance(repaired.get("properties"), dict):
            repaired["properties"] = {}
        required = repaired.get("required")
        if isinstance(required, list):
            props = repaired.get("properties") or {}
            valid = [r for r in required if isinstance(r, str) and r in props]
            if len(valid) != len(required):
                if valid:
                    repaired["required"] = valid
                else:
                    repaired.pop("required", None)
    return repaired


# Lazy (schema-cache registered) servers are available: the first real call spawns/connects them (#56832).
def _normalize_mcp_input_schema(schema: dict | None) -> dict:
    """Normalize MCP input schemas so one form is valid on OpenAI, Anthropic, Gemini and
    Moonshot. Order matters: ``definitions`` -> ``$defs``; nullable ``anyOf`` unions collapsed
    to the non-null branch (Anthropic rejects nullable branches; optionality lives in the
    parent's ``required``; the ``nullable: true`` hint is kept so runtime coercion can map a
    model-emitted ``"null"`` string to ``None``); same-typed const unions -> enum (AFTER the
    nullable strip); then object-shape repair.

    * Missing or ``null`` ``type`` on an object-shaped node is coerced to ``"object"`` (some servers omit
    it). See PR #4897. * When an ``object`` node lacks ``properties``, an empty ``properties`` dict is added
    so ``required`` entries don't dangle. * ``required`` arrays are pruned to only names that exist in
    ``properties``; otherwise Google AI Studio / Gemini 400s with ``property is not defined``. See PR #4651.
    * MCP/Pydantic optional fields commonly arrive as ``anyOf: [{...}, {"type": "null"}], default: null``.
    """
    if not schema:
        return dict(_EMPTY_OBJECT_SCHEMA)
    from tools.schema_sanitizer import collapse_const_unions, strip_nullable_unions
    normalized = _rewrite_local_refs(schema)
    normalized = strip_nullable_unions(normalized, keep_nullable_hint=True)
    normalized = collapse_const_unions(normalized)
    normalized = _repair_object_shape(normalized)
    if not isinstance(normalized, dict):
        return dict(_EMPTY_OBJECT_SCHEMA)
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized = {**normalized, "properties": {}}
    return normalized


def sanitize_mcp_name_component(value: str) -> str:
    """Replace every char outside ``[A-Za-z0-9_]`` with ``_`` (hyphens included, the
    historical behavior) so generated names pass provider validation."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


# ``mcp__<server>__<tool>``: the convention shared by Claude Code, Codex and OpenCode. The
# double underscore disambiguates the server/tool boundary even when either contains
# underscores, and matches the Anthropic-OAuth wire form.
# Native MCP tool-name prefix. It also aligns native registration with the Anthropic-OAuth wire form
# (``_MCP_TOOL_PREFIX`` in anthropic_adapter.py), removing the single->double rewrite that path previously
# had to perform. See #33533.
MCP_TOOL_NAME_PREFIX = "mcp__"


def mcp_prefixed_tool_name(server_name: str, tool_name: str) -> str:
    """Registry/wire name: ``mcp__<sanitizedServer>__<sanitizedTool>``."""
    return f"{MCP_TOOL_NAME_PREFIX}{sanitize_mcp_name_component(server_name)}__{sanitize_mcp_name_component(tool_name)}"


def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """Convert an MCP ``Tool`` (``.input_schema``, or ``.inputSchema`` before mcp 2.0) to a
    ``registry.register(schema=...)`` dict."""
    return {
        "name": mcp_prefixed_tool_name(server_name, mcp_tool.name),
        "description": strip_unicode_tags(mcp_tool.description or f"MCP tool {mcp_tool.name} from {server_name}"),
        "parameters": _normalize_mcp_input_schema(mcp_field(mcp_tool, "input_schema", "inputSchema")),
    }


# Utility tools generated per server: handler_key -> (description template, parameter
# properties, required names). Schemas are FROZEN wire bytes — the key order emitted by
# ``_build_utility_schemas`` must not change.
_UTILITY_TOOL_SPECS = (
    ("list_resources", "List available resources from MCP server '{server}'", {}, None),
    ("read_resource", "Read a resource by URI from MCP server '{server}'",
     {"uri": {"type": "string", "description": "URI of the resource to read"}}, ["uri"]),
    ("list_prompts", "List available prompts from MCP server '{server}'", {}, None),
    ("get_prompt", "Get a prompt by name from MCP server '{server}'",
     {
         "name": {"type": "string", "description": "Name of the prompt to retrieve"},
         "arguments": {
             "type": "object",
             "description": "Optional arguments to pass to the prompt",
             "properties": {},
             "additionalProperties": True},
     }, ["name"]))


def _build_utility_schemas(server_name: str) -> List[dict]:
    """Schemas for the resource/prompt utility tools as ``{schema, handler_key}`` dicts."""
    out = []
    for handler_key, description, properties, required in _UTILITY_TOOL_SPECS:
        parameters = {"type": "object", "properties": {k: dict(v) for k, v in properties.items()}}
        if required is not None:
            parameters["required"] = list(required)
        out.append({
            "schema": {
                "name": mcp_prefixed_tool_name(server_name, handler_key),
                "description": description.format(server=server_name),
                "parameters": parameters},
            "handler_key": handler_key})
    return out


def _normalize_name_filter(value: Any, label: str) -> set[str]:
    """Normalize include/exclude config to a set of exact names or fnmatch globs."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP config %s must be a string or list of strings; ignoring %r", label, value)
    return set()


def matches_name_filter(tool_name: str, patterns: set[str]) -> bool:
    """True if ``tool_name`` matches any entry: exact names literally, entries with
    ``*``/``?``/``[`` as case-sensitive globs (same semantics as ``approvals.deny``). Exact
    membership is checked first so big lists stay O(1)."""
    if not patterns:
        return False
    if tool_name in patterns:
        return True
    return any(fnmatch.fnmatchcase(tool_name, p) for p in patterns if "*" in p or "?" in p or "[" in p)


# Utility handler -> capability key that must be non-None on the server's ``initialize``
# response for the handler to be registered. Without this gate a tools-only server got all
# four stubs and every call returned JSON-RPC -32601, making the model conclude the server
# was broken.
# Source of truth: MCP spec — capabilities.resources / capabilities.prompts are present on the response only
# when the server actually implements those request families. Context7 @upstash/context7-mcp, which
# advertises only ``tools``) had all four utility stubs registered and every model call to them came back
# with JSON-RPC ``-32601 Method not found``, which made the model conclude the server was broken even when
# the real tools worked. See #18051.
_UTILITY_CAPABILITY_ATTRS = {
    "list_resources": "resources", "read_resource": "resources",
    "list_prompts": "prompts", "get_prompt": "prompts"}
