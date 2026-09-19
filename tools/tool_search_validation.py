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


def validate_deferred_call_args(name: str, args: Dict[str, Any]) -> Optional[str]:
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



_ARGS_KEY_RE = re.compile(r'"arguments"\s*:\s*\{')
# Family-D tail (t_09babe1a): after the balanced arguments object the string may
# only contain orphaned closers, the relocated "name" key, and the (never-closed)
# entry/array braces:  `] , "name": "TOOL" } ]`  (variants: trailing ']' missing,
# stray '"' before the comma).
_FAMILY_D_TAIL_RE = re.compile(r'^\s*\]?\s*"?\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}\s*\]?\s*$')
# Tool names are tool_call identifiers: mcp__server__tool, snake_case, dotted,
# colon-namespaced. Anything outside this charset in a mangled payload is a
# model artifact, not a tool name — never repair on it.
_REPAIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def _balanced_args_span(raw: str) -> Optional[Tuple[int, int]]:
    """(start, end_exclusive) of the balanced {...} following the single 'arguments'
    key, or None. String-state aware: braces inside JSON strings are skipped, so a
    balanced slice here is the actual arguments object, not an escaping artifact."""
    if raw.count('"arguments"') != 1:
        return None  # multi-entry batch or key echoed in content: not recoverable with certainty
    m = _ARGS_KEY_RE.search(raw)
    if not m:
        return None
    start = raw.index("{", m.end() - 1)
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


def _extract_balanced_args(raw: str) -> Optional[str]:
    """Return the balanced {...} slice following the single 'arguments' key, or None."""
    span = _balanced_args_span(raw)
    return raw[span[0]:span[1]] if span is not None else None


def _try_reconstruct_single_call(
    raw_calls: str, outer_args: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """One-shot reconstruction of the observed glm-5.3-flash mangle (t_86acfa96, family A):
    'calls' string-encoded with the entry object left unclosed and 'name' promoted
    to a sibling of 'calls' in the outer arguments. Accepted ONLY when the call is
    recoverable with certainty: an exact outer sibling 'name', exactly one
    '"arguments"' occurrence in the string, and a balanced, strictly-parseable args
    object. Any doubt -> None (caller emits the specific unparseable error; the
    model retries as a native array, which has a 0% failure rate post-fix)."""
    name = str(outer_args.get("name") or "").strip()
    if not name or name in BRIDGE_TOOL_NAMES:
        return None
    span = _balanced_args_span(raw_calls)
    if span is None:
        return None
    args_slice = raw_calls[span[0]:span[1]]
    try:
        args = json.loads(args_slice)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    # Ambiguity guard: an in-string "name" in the tail that disagrees with the
    # sibling means the payload is internally contradictory — never guess.
    m = re.search(r'"name"\s*:\s*"([^"]*)"', raw_calls[span[1]:])
    if m and m.group(1).strip() and m.group(1).strip() != name:
        return None
    logger.warning(
        "normalize_tool_call_entries: reconstructed string-encoded 'calls' for %s "
        "(glm dangling-entry mangle family A, t_86acfa96); repaired to single native call",
        name)
    return [{"name": name, "arguments": args}]


def _try_reconstruct_family_d(
    raw_calls: str, outer_args: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """One-shot reconstruction of the evolved glm-5.3-flash mangle (t_09babe1a, family D):
    'calls' string-encoded with the entry object left unclosed after its (balanced,
    strictly-parseable) 'arguments' object, and 'name' relocated INSIDE the string
    ahead of orphaned closers — ``[{"arguments": {ARGS}] , "name": "TOOL" } ]``.
    Accepted ONLY when: exactly one '"arguments"' occurrence, the args object is
    balanced and strictly parses to a dict, the remainder of the string fully
    matches the family-D tail (nothing else allowed), the recovered name is
    non-empty, charset-valid and not a bridge tool (a recovered name of 'tool_call'
    means the model mangled the tool name itself — untrustworthy), and any outer
    sibling 'name' does not contradict it. Never re-serializes model JSON: the
    native entry is rebuilt from the parsed args. Any doubt -> None."""
    span = _balanced_args_span(raw_calls)
    if span is None:
        return None
    try:
        args = json.loads(raw_calls[span[0]:span[1]])
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    m = _FAMILY_D_TAIL_RE.match(raw_calls[span[1]:])
    if not m:
        return None
    name = m.group(1).strip()
    if not name or not _REPAIR_NAME_RE.match(name) or name in BRIDGE_TOOL_NAMES:
        return None
    sibling = outer_args.get("name")
    if isinstance(sibling, str) and sibling.strip() and sibling.strip() != name:
        return None
    for key, value in outer_args.items():
        if key not in ("calls", "name") and key not in args:
            args[key] = value
    logger.warning(
        "normalize_tool_call_entries: reconstructed string-encoded 'calls' for %s "
        "(glm unclosed-entry mangle family D, t_09babe1a); repaired to single native call",
        name)
    return [{"name": name, "arguments": args}]


def normalize_tool_call_entries(args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
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
        # Models occasionally emit 'calls' as a JSON-encoded string (observed in the
        # wild: glm-5.3-flash on large multi-param MCP payloads, t_99484a2c). Extend
        # the same tolerance the per-entry 'arguments' field gets below (#114484).
        # A string that does not parse is a DIFFERENT failure from an empty/non-array
        # 'calls' — say so specifically; the generic empty-array error drove a
        # byte-stable 4x retry loop because it told the model nothing was wrong
        # with its structure.
        try:
            raw_calls = json.loads(raw_calls)
        except json.JSONDecodeError as e:
            # Narrow one-shot repairs (t_86acfa96 family A, t_09babe1a family D):
            # only the two reconstructable shapes above. Everything else still gets
            # the specific error below. No broad auto-repair of arbitrary malformed
            # JSON — any doubt fails closed to the re-emit-native error, which has
            # a 100% native-retry success rate post-fix.
            repaired = _try_reconstruct_single_call(raw_calls, args)
            if repaired is None:
                repaired = _try_reconstruct_family_d(raw_calls, args)
            if repaired is not None:
                raw_calls = repaired
            else:
                return [], (
                    "tool_call 'calls' was emitted as a JSON-encoded string and that string "
                    f"is not valid JSON: {e.msg} at char {e.pos} of {len(raw_calls)}. Re-emit "
                    "'calls' as a native JSON array (not a string), with every entry closed "
                    'before the next begins: [{"name": "…", "arguments": {"…": …}}, …]. '
                    "For large payloads, drop optional params or split the call rather than "
                    "string-encoding the array."
                )
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list) or not raw_calls:
        return [], "tool_call 'calls' must be a non-empty array of {name, arguments}"

    entries: List[Dict[str, Any]] = []
    for position, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            return [], f"tool_call calls[{position}] must be an object with 'name' and 'arguments'"
        name = str(raw.get("name") or "").strip()
        if not name:
            return [], f"tool_call calls[{position}] requires a 'name'"
        if name in BRIDGE_TOOL_NAMES:
            return [], f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
        raw_args = raw.get("arguments")
        if raw_args is None:
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


def local_batch_error(entries: List[Dict[str, Any]]) -> str:
    """Rejection for a multi-entry batch that names a local tool. Restates the valid
    shape with the caller's OWN first entry: small models re-send an identical batch
    when told only the constraint, and the echoed payload is what gets them unstuck."""
    first = entries[0]
    args = json.dumps(first.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
    if len(args) > _ECHO_ARGS_MAX_CHARS:
        args = "{...}"  # keep the correction readable; the model still has its own arguments
    retry = '{"calls":[{"name":%s,"arguments":%s}]}' % (json.dumps(first["name"], ensure_ascii=False), args)
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
    cited by its bare suffix instead of the full ``mcp__<server>__<tool>`` name. Telling
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
            f"by the exact name tool_search returns (e.g. mcp__<server>__<tool>).{hint}")
