"""Local argument validation for ``tool_call`` against a deferred tool's schema."""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import tool_error
from tools.tool_search_catalog import BRIDGE_TOOL_NAMES, _registry_entry

logger = logging.getLogger("tools.tool_search")

_SCHEMA_LITERAL_KEYS = frozenset({"const", "default", "enum", "example", "examples"})


def _schema_for_local_validation(node: Any) -> Any:
    """JSON-Schema-compatible copy honoring OpenAPI ``nullable: true`` (the normal coercion
    path accepts that shape, so local validation must too)."""
    if isinstance(node, list):
        return [_schema_for_local_validation(item) for item in node]
    if not isinstance(node, dict):
        return node
    # Literal keywords hold instance data, not schemas: copy byte-for-byte.
    normalized = {key: (copy.deepcopy(value) if key in _SCHEMA_LITERAL_KEYS
                        else _schema_for_local_validation(value))
                  for key, value in node.items() if key != "nullable"}
    if node.get("nullable") is not True:
        return normalized
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            normalized["type"] = [*schema_type, "null"]
        return normalized
    # No ``type`` to extend ($ref/combinator): wrap so local refs still resolve from the
    # root while null stays an explicit alternative.
    return {"anyOf": [normalized, {"type": "null"}]}


def _schema_has_external_ref(node: Any) -> bool:
    """True when *node* contains a non-local ``$ref`` — local validation must never turn a
    tool call into an implicit network/file fetch (fail open)."""
    if isinstance(node, list):
        return any(_schema_has_external_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    return (isinstance(ref, str) and not ref.startswith("#")) or any(
        _schema_has_external_ref(value) for key, value in node.items()
        if key not in _SCHEMA_LITERAL_KEYS)


def _validation_path(error: Any) -> str:
    """Format a jsonschema error path as a compact argument path."""
    path = "arguments"
    for part in getattr(error, "absolute_path", ()):
        if isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{part if isinstance(part, int) else json.dumps(part, ensure_ascii=False)}]"
    return path


def _validation_error(message: str, *, path: str, constraint: str, parameters: Any) -> str:
    return tool_error(
        message, path=path, constraint=constraint, parameters=parameters,
        hint="Retry tool_call with 'arguments' matching the parameters schema above.")


def validate_deferred_call_args(name: str, args: dict[str, Any]) -> Optional[str]:
    """Validate ``tool_call`` arguments against the deferred tool's schema. Models invoke
    deferred tools "blind" (schema unseen) and omit required args; without this, the opaque
    downstream failure makes cheap models loop. Required-field probe first, then the same
    schema-guided coercion normal dispatch applies, then jsonschema on the repaired copy.
    Missing/malformed schemas, no validator, and external refs all fail OPEN. Returns a JSON
    error string when invalid, ``None`` when the call should dispatch.

    This restores the concrete-schema checks that the provider cannot perform through the generic
    ``arguments: object`` bridge. See #5149.
    """
    try:
        from tools.registry import registry as _registry
        schema = _registry.get_schema(name)
        if not isinstance(schema, dict):
            return None
        fn = schema.get("function") if schema.get("type") == "function" else schema
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            return None
        required = params.get("required")
        missing = ([r for r in required if isinstance(r, str) and r not in args]
                   if isinstance(required, list) else [])
        if missing:
            return _validation_error(
                f"tool_call to '{name}' is missing required argument(s): "
                f"{', '.join(missing)}. The tool was NOT invoked.",
                path="arguments", constraint="required", parameters=params)
        validation_schema = _schema_for_local_validation(params)
        if _schema_has_external_ref(validation_schema):
            logger.debug("Skipping local deferred-argument validation for %s: external $ref", name)
            return None
        # Validate the repaired shape dispatch will see; copy because coerce_tool_args may
        # normalize in place (dispatch re-coerces canonically).
        try:
            from model_tools import coerce_tool_args
            candidate_args = coerce_tool_args(name, dict(args))
        except Exception:
            logger.debug("Deferred-argument coercion failed for %s", name, exc_info=True)
            candidate_args = dict(args)
        try:
            from jsonschema.exceptions import best_match
            from jsonschema.validators import validator_for
        except ImportError:
            logger.debug("jsonschema unavailable; keeping required-only validation for %s", name)
            return None
        validator_cls = validator_for(validation_schema)
        validator_cls.check_schema(validation_schema)
        validation_error = best_match(validator_cls(validation_schema).iter_errors(candidate_args))
        if validation_error is None:
            return None
        path = _validation_path(validation_error)
        constraint = str(getattr(validation_error, "validator", None) or "schema")
        detail = re.sub(r"\s+", " ", str(validation_error.message)).strip()
        if len(detail) > 600:
            detail = detail[:597] + "..."
        return _validation_error(
            f"tool_call to '{name}' failed argument validation at {path} "
            f"({constraint}): {detail}. The tool was NOT invoked.",
            path=path, constraint=constraint, parameters=params)
    except Exception:  # pragma: no cover — never block dispatch on validator bugs
        logger.debug("validate_deferred_call_args failed for %s", name, exc_info=True)
        return None


# The only prefix either repair accepts in front of the single arguments object: the
# opened array/entry, optionally carrying the entry's own "name" key first. Anything
# else ahead of "arguments" (prose, another key, an injected instruction) means the
# string is not one of the observed mangles, so it is never repaired.
_REPAIR_PREFIX_RE = re.compile(r'^\s*\[?\s*\{\s*(?:"name"\s*:\s*"([^"]*)"\s*,\s*)?"arguments"\s*:\s*\{')
# Family-A tail: after the balanced arguments object only the entry/array closers
# (either may be missing) may follow.
_FAMILY_A_TAIL_RE = re.compile(r'^\s*\}?\s*\]?\s*$')
# Family-D tail: after the balanced arguments object the string may
# only contain orphaned closers, the relocated "name" key, and the (never-closed)
# entry/array braces:  `] , "name": "TOOL" } ]`  (variants: trailing ']' missing,
# stray '"' before the comma).
_FAMILY_D_TAIL_RE = re.compile(r'^\s*\]?\s*"?\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}\s*\]?\s*$')
# Tool names are tool_call identifiers: mcp__server__tool, snake_case, dotted,
# colon-namespaced. Anything outside this charset in a mangled payload is a
# model artifact, not a tool name — never repair on it.
_REPAIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def _balanced_args_span(raw: str) -> Optional[tuple[int, int]]:
    """(start, end_exclusive) of the balanced {...} following the single 'arguments'
    key, or None. The key must sit right after the anchored entry prefix
    (``_REPAIR_PREFIX_RE``). String-state aware: braces inside JSON strings are
    skipped, so a balanced slice here is the actual arguments object."""
    if raw.count('"arguments"') != 1:
        return None  # multi-entry batch or key echoed in content: not recoverable with certainty
    m = _REPAIR_PREFIX_RE.match(raw)
    if not m:
        return None
    start = m.end() - 1
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
    return None


def _prefix_name(raw: str) -> Optional[str]:
    """The entry's own "name" written ahead of "arguments", if any (prefix already anchored)."""
    m = _REPAIR_PREFIX_RE.match(raw)
    return m.group(1).strip() if m and m.group(1) is not None else None


def _parse_args_slice(raw: str, span: tuple[int, int]) -> Optional[dict[str, Any]]:
    try:
        args = json.loads(raw[span[0]:span[1]])
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def _repairable_name(name: str) -> bool:
    return bool(name) and bool(_REPAIR_NAME_RE.match(name)) and name not in BRIDGE_TOOL_NAMES


def _try_reconstruct_single_call(
    raw_calls: str, outer_args: dict[str, Any]
) -> Optional[list[dict[str, Any]]]:
    """One-shot reconstruction of the observed glm-5.3-flash mangle (family A):
    'calls' string-encoded with the entry object left unclosed and 'name' promoted
    to a sibling of 'calls' in the outer arguments. Accepted ONLY when the call is
    recoverable with certainty: an exact, charset-valid outer sibling 'name'; the
    string is exactly ``[{`` + optional ``"name": "<same name>",`` + one balanced,
    strictly-parseable 'arguments' object + optional closers — nothing else before
    or after. A name inside the string that differs from the sibling, or any other
    content around the arguments object, means the args may have been written for
    another tool. Any doubt -> None (caller emits the specific unparseable error;
    the model retries as a native array)."""
    name = str(outer_args.get("name") or "").strip()
    if not _repairable_name(name):
        return None
    span = _balanced_args_span(raw_calls)
    if span is None:
        return None
    in_string_name = _prefix_name(raw_calls)
    if in_string_name is not None and in_string_name != name:
        return None
    if not _FAMILY_A_TAIL_RE.match(raw_calls[span[1]:]):
        return None
    args = _parse_args_slice(raw_calls, span)
    if args is None:
        return None
    logger.warning(
        "normalize_tool_call_entries: reconstructed string-encoded 'calls' for %s "
        "(glm dangling-entry mangle family A); repaired to single native call",
        name)
    return [{"name": name, "arguments": args}]


def _try_reconstruct_family_d(
    raw_calls: str, outer_args: dict[str, Any]
) -> Optional[list[dict[str, Any]]]:
    """One-shot reconstruction of the evolved glm-5.3-flash mangle (family D):
    'calls' string-encoded with the entry object left unclosed after its (balanced,
    strictly-parseable) 'arguments' object, and 'name' relocated INSIDE the string
    ahead of orphaned closers — ``[{"arguments": {ARGS}] , "name": "TOOL" } ]``.
    Accepted ONLY when: the string opens with the anchored entry prefix, exactly
    one '"arguments"' occurrence, the args object is balanced and strictly parses
    to a dict, the remainder of the string fully matches the family-D tail
    (nothing else allowed), the recovered name is non-empty, charset-valid and not
    a bridge tool (a recovered name of 'tool_call' means the model mangled the tool
    name itself — untrustworthy), and neither an outer sibling 'name' nor a name in
    the prefix contradicts it. Never re-serializes model JSON: the native entry is
    rebuilt from the parsed args. Any doubt -> None."""
    span = _balanced_args_span(raw_calls)
    if span is None:
        return None
    m = _FAMILY_D_TAIL_RE.match(raw_calls[span[1]:])
    if not m:
        return None
    name = m.group(1).strip()
    if not _repairable_name(name):
        return None
    in_string_name = _prefix_name(raw_calls)
    if in_string_name is not None and in_string_name != name:
        return None
    sibling = outer_args.get("name")
    if isinstance(sibling, str) and sibling.strip() and sibling.strip() != name:
        return None
    args = _parse_args_slice(raw_calls, span)
    if args is None:
        return None
    for key, value in outer_args.items():
        if key not in ("calls", "name") and key not in args:
            args[key] = value
    logger.warning(
        "normalize_tool_call_entries: reconstructed string-encoded 'calls' for %s "
        "(glm unclosed-entry mangle family D); repaired to single native call",
        name)
    return [{"name": name, "arguments": args}]


def normalize_tool_call_entries(args: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Normalize ``tool_call`` arguments into a ``calls[]`` list of entries.

    Accepts the advertised batch shape ``{"calls": [{"name", "arguments"}, ...]}``
    and, tolerantly, the legacy single shape ``{"name": ..., "arguments": ...}``
    (a single call is a batch of one). Each entry's ``arguments`` is coerced to
    a dict (JSON strings parsed, ``None`` → ``{}``). Returns ``(entries, None)``
    or ``([], error_message)``.
    """
    raw_calls = args.get("calls")
    if raw_calls is None:
        # Legacy single shape.
        if not str(args.get("name") or "").strip():
            return [], "tool_call requires 'calls' (an array of {name, arguments})"
        raw_calls = [{"name": args.get("name"), "arguments": args.get("arguments")}]
    if isinstance(raw_calls, str):
        # Tolerate the model emitting the batch envelope as a JSON string —
        # mirror the per-entry `arguments` handling below (#114484).
        try:
            raw_calls = json.loads(raw_calls)
        except json.JSONDecodeError as e:
            # Narrow one-shot repairs for the two observed glm-5.3-flash mangles
            # (family A, family D). No broad auto-repair of arbitrary malformed
            # JSON — any doubt fails closed to the re-emit-native error. A string
            # that does not parse is a DIFFERENT failure from an empty/non-array
            # 'calls': say so specifically, or the model retries the same bytes.
            repaired = _try_reconstruct_single_call(raw_calls, args)
            if repaired is None:
                repaired = _try_reconstruct_family_d(raw_calls, args)
            if repaired is None:
                return [], (
                    "tool_call 'calls' was emitted as a JSON-encoded string and that string "
                    f"is not valid JSON: {e.msg} at char {e.pos} of {len(raw_calls)}. Re-emit "
                    "'calls' as a native JSON array (not a string), with every entry closed "
                    'before the next begins: [{"name": "…", "arguments": {"…": …}}, …]. '
                    "For large payloads, drop optional params or split the call rather than "
                    "string-encoding the array."
                )
            raw_calls = repaired
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list) or not raw_calls:
        return [], "tool_call 'calls' must be a non-empty array of {name, arguments}"

    entries: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            return [], f"tool_call calls[{position}] must be an object with 'name' and 'arguments'"
        name = str(raw.get("name") or "").strip()
        if not name:
            return [], f"tool_call calls[{position}] requires a 'name'"
        if name in BRIDGE_TOOL_NAMES:
            return [], f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
        raw_args = raw.get("arguments")
        if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
            # "" / whitespace is how some OpenAI-compatible gateways spell "no arguments" for a
            # parameterless tool (#83937); the loop already treats an empty outer arguments string
            # as {} (turn_tool_validation), and a missing required param still surfaces below via
            # validate_deferred_call_args instead of an opaque JSON parse error.
            raw_args = {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                return [], f"tool_call calls[{position}].arguments is not valid JSON: {e}"
        if not isinstance(raw_args, dict):
            return [], f"tool_call calls[{position}].arguments must be an object"
        entries.append({"name": name, "arguments": raw_args})
    return entries, None


_ECHO_ARGS_MAX_CHARS = 1500


def local_batch_error(entries: list[dict[str, Any]]) -> str:
    """Rejection for a multi-entry batch that names a local tool. Restates the valid
    shape with the caller's OWN first entry: small models re-send an identical batch
    when told only the constraint, and the echoed payload is what gets them unstuck."""
    first = entries[0]
    args = json.dumps(first.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
    if len(args) > _ECHO_ARGS_MAX_CHARS:
        args = "{...}"  # keep the correction readable; the model still has its own arguments
    retry = '{{"calls":[{{"name":{},"arguments":{}}}]}}'.format(json.dumps(first["name"], ensure_ascii=False), args)
    remaining = (f" then issue the remaining {len(entries) - 1} call(s) as separate tool_call invocations"
                 if len(entries) > 1 else "")
    return (
        f"tool_call takes exactly one entry for local tools; you sent {len(entries)}. "
        f"Retry with only: {retry}{remaining}. Only connectors__ names may be batched together."
    )


def not_deferrable_error(name: str) -> str:
    """Rejection for a ``tool_call`` naming something that is not a deferred tool.
    Two different mistakes reach here and need opposite corrections: a directly-listed
    tool (call it without the bridge) vs. an unknown name — typically a deferred MCP tool
    cited by its bare suffix instead of the full ``mcp__<server>__{tool}`` name. Telling
    the second group 'call it directly' is the opposite of what they must do."""
    from tools.tool_search import _core_tool_names  # late: tool_search imports this module
    if name in _core_tool_names() or _registry_entry(name) is not None:
        return (f"'{name}' is a directly-listed tool, not a deferred one. "
                "Call it directly instead of via tool_call.")
    suffix = f"__{name}"
    try:
        from tools.registry import registry
        candidates = sorted(n for n in registry.get_all_tool_names() if n.endswith(suffix))
    except Exception:
        candidates = []
    hint = (f" Did you mean {', '.join(repr(c) for c in candidates)}?" if candidates
            else " Use tool_search to find the exact name.")
    return (f"'{name}' is not a known tool name. Deferred tools must be invoked through tool_call "
            f"by the exact name tool_search returns (e.g. mcp__<server>__{{tool}}).{hint}")
