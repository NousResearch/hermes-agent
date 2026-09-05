"""Sanitize tool JSON schemas for strict LLM backends. llama.cpp's grammar converter fails on
``{"type": "object"}`` without ``properties``, bare-string schemas and ``type`` arrays; Anthropic
rejects nullable ``anyOf`` at the top of ``input_schema``; Fireworks rejects ``default`` beside
``$ref``; Codex rejects top-level combinators. Walks a deep copy and fixes only those shapes."""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Anthropic (and Bedrock/Vertex/Azure fronting it) reject property keys not matching this; one bad
# key anywhere in the tools array 400s the request (Cloudflare's MCP ships 61).
_PROP_KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
_PROP_KEY_BAD_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")
_UNION_KEYS = ("anyOf", "oneOf")
_UNION_META_KEYS = ("title", "description", "default", "examples")  # copied onto replacements


def _empty_object() -> dict:
    return {"type": "object", "properties": {}}


def _rewrite(schema: Any, fn: Callable[[dict], Any]) -> Any:
    """Bottom-up map over a schema tree: lists/dicts recurse, then *fn* sees each dict."""
    if isinstance(schema, list):
        return [_rewrite(item, fn) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return fn({k: _rewrite(v, fn) for k, v in schema.items()})


def sanitize_property_key(key: str) -> str:
    """Deterministically map an arbitrary property key to a conforming one."""
    return _PROP_KEY_BAD_CHARS.sub("_", key)[:64] or "param"


def _rename_property_keys(props: dict, path: str) -> dict[str, str]:
    """{original_key: conforming_key} for one properties dict (identity entries omitted).
    Deterministic (insertion order, numeric suffixes on collision) so the model-visible schema
    and the dispatch-time reverse map from the registry's original schema agree."""
    renames: dict[str, str] = {}
    taken = {k for k in props if _PROP_KEY_RE.match(k)}
    for key in (k for k in props if not _PROP_KEY_RE.match(k)):
        base = sanitize_property_key(key)
        candidate, i = base, 2
        while candidate in taken:
            candidate, i = base[: 64 - len(f"_{i}")] + f"_{i}", i + 1
        taken.add(candidate)
        renames[key] = candidate
        logger.debug("schema_sanitizer[%s]: renamed property key %r -> %r "
                     "(provider key-pattern compat)", path, key, candidate)
    return renames


def unrename_tool_args(params_schema: Any, args: Any) -> Any:
    """Map sanitized keys in model-emitted args back to wire names. ``params_schema`` is the
    ORIGINAL registry schema; recurses into objects/array items; unknown keys pass through."""
    props = params_schema.get("properties") if isinstance(params_schema, dict) else None
    if not isinstance(props, dict) or not isinstance(args, dict):
        return args
    reverse = {v: k for k, v in _rename_property_keys(props, "<unrename>").items()}
    out = {}
    for key, value in args.items():
        orig = reverse.get(key, key)
        sub = props.get(orig) if isinstance(props.get(orig), dict) else {}
        if isinstance(value, dict) and sub:
            value = unrename_tool_args(sub, value)
        elif isinstance(value, list) and isinstance(sub.get("items"), dict):
            value = [unrename_tool_args(sub["items"], item) if isinstance(item, dict) else item
                     for item in value]
        out[orig] = value
    return out


def sanitize_tool_schemas(tools: list[dict]) -> list[dict]:
    """Deep-copied ``tools`` (OpenAI format) with sanitized parameter schemas; safe to mutate."""
    return [_sanitize_single_tool(tool) for tool in tools] if tools else tools


def _sanitize_single_tool(tool: dict) -> dict:
    out = copy.deepcopy(tool)
    fn = out.get("function") if isinstance(out, dict) else None
    if not isinstance(fn, dict):
        return out
    params = fn.get("parameters")
    if not isinstance(params, dict):  # missing / non-dict → minimal valid shape
        fn["parameters"] = _empty_object()
        return out
    name = fn.get("name", "<tool>")
    top = _sanitize_node(params, path=name)
    top = top if isinstance(top, dict) else {}  # guarantee an object with properties on top
    top["type"] = "object"
    if not isinstance(top.get("properties"), dict):
        top["properties"] = {}
    # The recursive pass only handles array-form ``type: [X, "null"]``; collapse anyOf unions
    # here, keeping ``nullable: true`` so ``tools.arg_coercion._schema_allows_null`` still coerces.
    top = strip_nullable_unions(top, keep_nullable_hint=True)
    top = _strip_top_level_combinators(top, path=name)
    fn["parameters"] = _strip_ref_siblings(top)
    return out


_REF_FORBIDDEN_SIBLINGS = frozenset({"default"})  # strict validators reject these beside ``$ref``


def _strip_ref_siblings(node: Any) -> Any:
    """Recursively drop forbidden siblings of ``$ref`` (Fireworks rejects ``default`` there)."""
    def strip(out: dict) -> dict:
        for key in _REF_FORBIDDEN_SIBLINGS if "$ref" in out else ():
            out.pop(key, None)
        return out
    return _rewrite(node, strip)


_TOP_LEVEL_FORBIDDEN_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def _strip_top_level_combinators(params: dict, *, path: str = "<tool>") -> dict:
    """Drop combinators from the TOP level only (Codex rejects them there). They are usually
    conditional-required hints, so validity is unchanged (handlers re-validate); nested ones
    stay."""
    if not isinstance(params, dict):
        return params
    out = dict(params)
    for key in [k for k in _TOP_LEVEL_FORBIDDEN_KEYS if k in out]:
        logger.debug("schema_sanitizer[%s]: stripped top-level %r combinator "
                     "from tool parameters (strict-backend compat)", path, key)
        del out[key]
    return out


def _is_null_branch(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "null"


def _carry_union_meta(outer: dict, replacement: dict, *, skip_default_on_ref: bool) -> None:
    """Copy outer-union metadata onto *replacement* where absent (``default`` is illegal beside
    ``$ref`` on strict backends, hence ``skip_default_on_ref``)."""
    for meta_key in _UNION_META_KEYS:
        if meta_key in outer and meta_key not in replacement and not (
                skip_default_on_ref and meta_key == "default" and "$ref" in replacement):
            replacement[meta_key] = outer[meta_key]


def strip_nullable_unions(schema: Any, *, keep_nullable_hint: bool = True) -> Any:
    """Collapse ``anyOf``/``oneOf`` nullable unions (MCP/Pydantic optional fields) to the single
    non-null branch: Anthropic rejects the null branch and optionality already lives in the parent's
    ``required``. Only when a null branch was dropped AND exactly one non-null branch survives.
    ``keep_nullable_hint`` sets ``nullable: true`` for runtime ``"null"`` → ``None`` coercion."""
    def collapse(stripped: dict) -> Any:
        for key in _UNION_KEYS:
            variants = stripped.get(key)
            if not isinstance(variants, list):
                continue
            non_null = [item for item in variants if not _is_null_branch(item)]
            if len(non_null) == 1 and len(non_null) != len(variants):
                replacement = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
                if keep_nullable_hint:
                    replacement.setdefault("nullable", True)
                _carry_union_meta(stripped, replacement, skip_default_on_ref=True)
                return _rewrite(replacement, collapse)  # the survivor may itself be a union
        return stripped
    return _rewrite(schema, collapse)


_CONST_PRIMITIVE_TYPES: dict[type, str] = {
    bool: "boolean", int: "integer", float: "number", str: "string"}


def _const_branch_type(branch: Any) -> str | None:
    """Primitive JSON-Schema type of a pure ``const`` branch (declared ``type``, if any, must match;
    only ``title``/``description`` may accompany it), else None."""
    if not isinstance(branch, dict) or "const" not in branch \
            or set(branch) - {"const", "type", "title", "description"}:
        return None
    # ``type(value)`` lookup (not isinstance): bool is a subclass of int.
    json_type = _CONST_PRIMITIVE_TYPES.get(type(branch["const"]))
    return json_type if branch.get("type") in (None, json_type) else None


def collapse_const_unions(schema: Any) -> Any:
    """Collapse ``anyOf``/``oneOf`` unions of same-typed consts (Rust/TS MCP servers emit
    ``{"anyOf": [{"const": "red"}, {"const": "green"}]}``) to ``enum``; ported from block/goose
    ``tool_schema_normalize.rs`` (Apache-2.0). Only when EVERY non-null branch is a pure ``const``
    of one primitive type (``bool`` never merges with ``integer``); one ``{"type": "null"}`` branch
    is tolerated as ``nullable: true``. Branch order kept; outer metadata carried; input never
    mutated."""
    def collapse(out: dict) -> Any:
        for key in _UNION_KEYS:
            variants = out.get(key)
            if not isinstance(variants, list) or not variants:
                continue
            null_branches = [i for i in variants if _is_null_branch(i) and "const" not in i]
            const_branches = [item for item in variants if item not in null_branches]
            if len(null_branches) > 1 or not const_branches:
                continue
            branch_types = {_const_branch_type(item) for item in const_branches}
            if len(branch_types) != 1 or None in branch_types:
                continue
            replacement: dict = {
                "type": branch_types.pop(), "enum": [item["const"] for item in const_branches]}
            if null_branches:
                replacement["nullable"] = True
            _carry_union_meta(out, replacement, skip_default_on_ref=False)
            return replacement
        return out
    return _rewrite(schema, collapse)


_BARE_TYPE_NAMES = frozenset({"object", "string", "number", "integer", "boolean", "array", "null"})
# Values that are NOT schemas (recursing would treat a required name like "path" as a bare schema).
_NON_SCHEMA_LIST_KEYS = frozenset({"required", "enum", "examples", "dependentRequired"})


def _normalize_type_array(value: list, out: dict) -> None:
    """Normalize a ``type: [...]`` array into *out* (llama.cpp and Gemini-via-OpenAI reject arrays).
    Per AI-SDK: one non-null type → ``type: X`` (+ ``nullable`` if ``null`` present); several →
    ``anyOf`` of single-type schemas so EVERY branch survives; none → ``null``/object fallback."""
    has_null = "null" in value
    non_null = [t for t in value if isinstance(t, str) and t != "null"]
    if not non_null:
        out["type"] = "null" if has_null else "object"
        return
    if len(non_null) == 1:
        out["type"] = non_null[0]
    else:
        out["anyOf"] = [{"type": t} for t in non_null]
    if has_null:
        out.setdefault("nullable", True)


def _sanitize_node(node: Any, path: str) -> Any:
    """Recursively sanitize a JSON-Schema fragment: bare-string schemas → ``{"type": <value>}``
    (unknown strings → permissive object); object nodes gain ``properties: {}``; ``type`` arrays
    are normalized; property keys are renamed to the provider-safe pattern and ``required``
    follows, with entries missing from ``properties`` pruned.

    - Normalizes ``type: [X, "null"]`` arrays to single ``type: X`` (keeping ``nullable: true`` as a hint),
    and multi-type arrays like ``["number", "string"]`` to an ``anyOf`` of single-type schemas so no branch
    is dropped (ported from anomalyco/opencode#31877). - Recurses into ``properties``, ``items``,
    ``additionalProperties``, ``anyOf``, ``oneOf``, ``allOf``, and ``$defs`` / ``definitions``.
    """
    if isinstance(node, str):
        if node in _BARE_TYPE_NAMES:
            logger.debug("schema_sanitizer[%s]: replacing bare-string schema %r with {'type': %r}",
                         path, node, node)
            return _empty_object() if node == "object" else {"type": node}
        logger.debug("schema_sanitizer[%s]: replacing non-schema string %r "
                     "with empty object schema", path, node)
        return _empty_object()
    if isinstance(node, list):
        return [_sanitize_node(item, f"{path}[{i}]") for i, item in enumerate(node)]
    if not isinstance(node, dict):
        return node
    # Renames computed up front so ``required`` remaps even when it precedes ``properties``.
    props_in = node.get("properties")
    prop_renames = (_rename_property_keys(props_in, f"{path}.properties")
                    if isinstance(props_in, dict) else {})
    out: dict = {}
    for key, value in node.items():
        # JSON Schema ``type`` arrays (e.g. ``["number", "string"]``, common in MCP tool schemas) are
        # rejected by several tool-call backends: * llama.cpp's grammar generator only accepts a singular
        # string type. * Gemini (including OpenAI-compatible transports such as GitHub Copilot proxying to
        # Gemini) rejects the array form outright — plain @ai-sdk/google rewrites it, but the
        # OpenAI-compatible path forwards it verbatim and the backend 400s. Normalize per the SDK's
        # behavior: * single non-null type → ``type: X`` (+ ``nullable: true`` if the array also contained
        # "null"). No data lost. * multiple non-null types → ``anyOf`` of single-type schemas, so EVERY
        # branch survives instead of silently dropping all but the first. ``null`` is lifted into
        # ``nullable: true``. * all-null / empty → ``type: "null"`` (or object fallback). Ported from
        # anomalyco/opencode#31877.
        if key == "type" and isinstance(value, list):
            _normalize_type_array(value, out)
        elif key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            renames = prop_renames if key == "properties" else {}
            out[key] = {
                renames.get(k, k): _sanitize_node(v, f"{path}.{key}.{renames.get(k, k)}")
                for k, v in value.items()}
        elif key in {"items", "additionalProperties"}:
            # Bool ``additionalProperties`` is valid; bool ``items`` is non-standard but preserved.
            out[key] = value if isinstance(value, bool) else _sanitize_node(value, f"{path}.{key}")
        elif key in _NON_SCHEMA_LIST_KEYS:
            if key == "required" and prop_renames and isinstance(value, list):
                out[key] = [prop_renames.get(r, r) if isinstance(r, str) else r for r in value]
            else:
                out[key] = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
        else:  # anyOf/oneOf/allOf and any other nested schema recurse (lists index the path)
            out[key] = _sanitize_node(value, f"{path}.{key}") if isinstance(value, (dict, list)) else value
    if out.get("type") == "object":
        if not isinstance(out.get("properties"), dict):
            out["properties"] = {}
        if isinstance(out.get("required"), list):
            valid = [r for r in out["required"] if isinstance(r, str) and r in out["properties"]]
            if valid:
                out["required"] = valid
            else:
                del out["required"]
    return out


# ---- Reactive strips — only invoked after a backend rejects a schema ----
_STRIP_ON_RECOVERY_KEYS = frozenset({"pattern", "format"})
_SCHEMA_MARKERS = frozenset({"type", "anyOf", "oneOf", "allOf"})  # a node with one IS a schema


def _dict_nodes(node: Any):
    """Pre-order walk over every dict node (yielded before its values, so it may be mutated)."""
    if isinstance(node, dict):
        yield node
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    for child in children:
        yield from _dict_nodes(child)


def _reactive_strip(
    tools: list[dict], strip_node: Callable[[dict], int], log_msg: str) -> tuple[list[dict], int]:
    """Apply *strip_node* (-> keywords removed) to every dict node of each tool's parameters, in
    place; OpenAI (``{"function": {"parameters"}}``) and Responses (``{"parameters"}``) formats."""
    stripped = 0
    for tool in tools or ():
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        params = fn.get("parameters") if isinstance(fn, dict) else None
        params = params if isinstance(params, dict) else tool.get("parameters")
        if isinstance(params, dict):
            stripped += sum(strip_node(node) for node in _dict_nodes(params))
    if stripped:
        logger.info(log_msg, stripped)
    return tools, stripped


def strip_pattern_and_format(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``pattern``/``format`` in place — reactive, only after llama.cpp's grammar converter
    rejected a schema (its regex engine is a small ECMAScript subset); cloud providers use these as
    prompting hints. Only beside ``type``/combinators, so a property *named* ``pattern`` stays."""
    def _strip(node: dict) -> int:
        is_schema = bool(node.keys() & _SCHEMA_MARKERS)
        hits = [k for k in node if k in _STRIP_ON_RECOVERY_KEYS] if is_schema else []
        for k in hits:
            del node[k]
        return len(hits)
    return _reactive_strip(
        tools, _strip,
        "schema_sanitizer: stripped %d pattern/format keyword(s) from "
        "tool schemas (llama.cpp grammar-parse recovery)")


def strip_slash_enum(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``enum`` keywords whose string values contain ``/``, in place: xAI's grammar compiler
    rejects them (HTTP 400 before any token) — typically MCP enums of HuggingFace model IDs."""
    def _strip(node: dict) -> int:
        enum_val = node.get("enum")
        if isinstance(enum_val, list) and any(isinstance(v, str) and "/" in v for v in enum_val):
            del node["enum"]
            return 1
        return 0
    return _reactive_strip(
        tools, _strip,
        "schema_sanitizer: stripped %d enum keyword(s) containing '/' "
        "from tool schemas (xAI Responses grammar-compile recovery)")
