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


_MAX_STRINGIFIED_REPAIRS = 8


def _repair_stringified_json(text: str) -> Any:
    """Repair an under-closed JSON payload a model emitted as a string.

    Some models stringify the batch envelope (or a nested ``arguments`` value)
    and lose a structural delimiter at the tail — e.g.
    ``[{"arguments": {…}, "name": "x"}]`` with the entry object's closing brace
    missing before the final ``]``. The parser then reports
    "Expecting ',' delimiter" at the *last character*, at any payload length, and
    the model re-sends the same shape until the failure guardrail halts the turn.

    Repair inserts ONLY delimiters the open containers already imply — never
    content, never a guessed value — and only when every string in the payload is
    terminated. An unterminated string means the model was cut off mid-value;
    that case must not be guessed at, so it returns None. Callers treat None as
    "not repairable" and surface the original decode error.
    """
    closers = {"[": "]", "{": "}"}
    stack: List[str] = []
    out: List[str] = []
    insertions = 0
    in_string = False
    escaped = False

    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if in_string:
            out.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch in "[{":
            stack.append(closers[ch])
            out.append(ch)
            continue
        if ch in "]}":
            # Close whatever the model left open so this closer applies to the
            # container it was written for.
            while stack and stack[-1] != ch:
                out.append(stack.pop())
                insertions += 1
                if insertions > _MAX_STRINGIFIED_REPAIRS:
                    return None
            if not stack:
                return None  # stray closer: not the under-close shape
            stack.pop()
            out.append(ch)
            continue
        out.append(ch)

    if in_string:
        return None  # truncated mid-value: never guess content
    while stack:
        out.append(stack.pop())
        insertions += 1
        if insertions > _MAX_STRINGIFIED_REPAIRS:
            return None
    if not insertions:
        return None  # nothing missing: a genuine syntax error, not under-closing
    try:
        return json.loads("".join(out))
    except json.JSONDecodeError:
        return None


def _is_known_tool_name(name: str) -> bool:
    """True when *name* resolves to a real tool — directly listed, in the registry, or a
    deferred catalog entry. Used to gate the nested-name hoist below: a string that
    happens to sit under an ``arguments.name`` key must only be read as the invocation
    target when it actually names something callable."""
    if not name:
        return False
    try:
        from tools.tool_search import _core_tool_names  # late: tool_search imports this module
        if name in _core_tool_names():
            return True
    except Exception:  # pragma: no cover — never block dispatch on a lookup bug
        pass
    return _registry_entry(name) is not None


def _calls_string_error(exc: json.JSONDecodeError, text: str) -> str:
    """Actionable rejection for a ``calls`` payload that arrived as a malformed JSON
    string. The bare decode error read as opaque tail-truncation, so a small model
    re-sent the same shape twelve times until the guardrail halted the turn; state the
    required shape and echo the tail it actually produced."""
    tail = text.strip()[-80:]
    return (
        f"tool_call 'calls' must be a JSON array of {{name, arguments}} objects, not a "
        f"JSON string: {exc}. Payload tail: {tail!r}. Send 'calls' as an array."
    )


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
            repaired = _repair_stringified_json(raw_calls)
            if repaired is None:
                return [], _calls_string_error(e, raw_calls)
            logger.warning(
                "tool_call 'calls' arrived as an under-closed JSON string; repaired "
                "the missing delimiter(s) and continued: %s", e)
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
            # Some models nest the tool name INSIDE `arguments` and under-close
            # the entry object (observed when a parameter is itself a nested
            # JSON value). Hoist it only when it names a real tool, so a legit
            # "name" field in a tool's own arguments is never read as the
            # invocation target.
            nested_args = raw.get("arguments")
            candidate = nested_args.get("name") if isinstance(nested_args, dict) else None
            if isinstance(candidate, str) and _is_known_tool_name(candidate.strip()):
                name = candidate.strip()
                logger.warning(
                    "tool_call calls[%d] nested the tool name inside 'arguments'; "
                    "hoisted %r", position, name)
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
                # Same hazard as the envelope above: a stringified `arguments`
                # value can lose a tail delimiter. Repair only delimiters.
                repaired_args = _repair_stringified_json(raw_args)
                if repaired_args is None:
                    return [], f"tool_call calls[{position}].arguments is not valid JSON: {e}"
                logger.warning(
                    "tool_call calls[%d].arguments arrived as an under-closed JSON "
                    "string; repaired the missing delimiter(s) and continued: %s",
                    position, e)
                raw_args = repaired_args
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
