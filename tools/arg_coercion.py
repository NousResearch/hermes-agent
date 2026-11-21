"""Tool-argument type coercion: repair string-typed values the model emitted against a tool's JSON Schema.

Models emit "42" for integers, "true" for booleans, JSON-encoded strings for
arrays/objects (also nested inside containers), and bare scalars where an array
is expected (wrapped in a one-element list). Coercion is schema-guided and
conservative: originals are kept whenever a repair is not unambiguous.
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from tools.schema_sanitizer import resolve_local_schema_ref

# Logger name kept as "model_tools": these messages were always emitted under
# that name and log-based tooling filters on it.
logger = logging.getLogger("model_tools")


def coerce_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce string-typed args to their JSON-Schema types; originals kept on failure."""
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    parameters = (schema or {}).get("parameters") or {}
    properties = parameters.get("properties")
    if not properties:
        return args

    # The model saw the SANITIZED schema (provider-illegal property keys were
    # renamed); map those keys back to the registry's wire names first.
    try:
        from tools.schema_sanitizer import unrename_tool_args
        args = unrename_tool_args(parameters, args)
    except Exception:  # pragma: no cover — never break dispatch
        pass

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        prop_schema = resolve_local_schema_ref(prop_schema, parameters)
        expected = prop_schema.get("type") or _union_branch_types(prop_schema, parameters)
        is_container = isinstance(value, (list, tuple))

        # Bare non-list value for an array schema. Strings go through
        # _coerce_value first so a JSON-encoded array is parsed and a nullable
        # "null" becomes None (not ["null"]). None itself is preserved: the tool's
        # own default handling decides between "omit" and "empty list".
        if expected == "array" and value is not None and not is_container:
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    args[key] = coerced
                    continue
                if value.strip().startswith("["):
                    logger.warning("coerce_tool_args: %s.%s looks like a JSON array string "
                                   "but could not be parsed — model may have emitted a "
                                   "JSON-encoded string instead of a native array. "
                                   "Falling back to single-element list.", tool_name, key)
                args[key] = [value]
                logger.info("coerce_tool_args: wrapped bare string in list for %s.%s", tool_name, key)
                continue
            args[key] = [value]
            logger.info("coerce_tool_args: wrapped bare %s in list for %s.%s", type(value).__name__, tool_name, key)
            continue

        if not isinstance(value, str):
            # Native container: still normalize JSON-encoded elements/sub-fields.
            if (expected == "array" and is_container) or (expected == "object" and isinstance(value, dict)):
                args[key] = _normalize_json_strings_for_schema(
                    value, prop_schema, root_schema=parameters
                )
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(
                    coerced, prop_schema, root_schema=parameters
                )

    return args


def _union_branch_types(schema: dict, root_schema: Any) -> str | list | None:
    """Branch types of a type-less ``anyOf``/``oneOf`` property, desugared to the
    equivalent ``type`` list so ``anyOf[X, Y]`` reuses the type-list coercion machinery
    (bare-value wrap, null guard, ``_coerce_value``'s union path). Only the branch
    *types* are derived: deep element normalization of a native container does not
    descend into union branches (no ``items``/``properties`` propagation). That is
    irrelevant for string-item arrays like ``notify`` and is not a regression — the
    whole union was skipped before.

    The main loop only read ``prop_schema["type"]``; a union without a top-level
    ``type`` (terminal ``notify: anyOf[boolean, array]``, a non-nullable MCP
    ``Union[bool, list[str]]`` rendered as ``anyOf[boolean, array]``, or a plugin
    tool registered with a raw ``anyOf``) yielded ``expected=None`` and was skipped,
    leaving string-typed args uncoerced (#123345). Deriving the branch types here
    lets the existing type-list machinery (the bare-value wrap branch, the null
    guard, and ``_coerce_value``'s union path) handle unions unchanged.

    Nullable MCP ``Optional[X]`` unions are collapsed to ``{type: X, nullable: true}``
    by ``_normalize_mcp_input_schema`` before registration, so they already reach
    coercion as a plain type; this helper covers the unions that survive normalization.

    A branch that is a local ``$ref`` counts with the type it points at. Returns
    ``None`` (leave the value alone) when the union node itself or a branch is constrained
    by ``enum``/``const`` (even with a plain type — coercion does not validate members, so
    it must not fabricate a disallowed value), when the node or a branch composes constraints
    (``allOf``/``anyOf``/``oneOf``/``not``/``if``, or both ``anyOf`` and ``oneOf`` on the node),
    has no plain scalar type (a nested type-list, an ``allOf`` wrapper, or a dangling/cyclic
    ``$ref``), or when any branch accepts ``string``: a string value is then already valid
    (or ambiguous) and must not be reinterpreted ("007" stays "007").

    Scope of the guard (deliberate): only membership constraints (``enum``/``const``) and
    composition stop coercion. Range/format keywords (``minimum``/``maximum``/``pattern``/
    ``multipleOf``/``minLength`` ...) do NOT — parity with the plain-type path, which also
    coerces through them and leaves enforcement to the handler. A known cost of the
    presence-based guard: a *valid* member of a constrained union is left unrepaired too
    ("2" stays "2" for ``enum: [1, 2, 3]``); validating membership after coercion, and the
    pre-existing plain-type path's same fabrication, are tracked as a follow-up, not here.

    A single remaining branch collapses to its scalar type, so pure ``anyOf[array]``
    behaves like ``type: array`` (bare-string wrapping included); ``anyOf[array, null]``
    stays a list and does NOT wrap a bare string, exactly like ``type: [array, null]``.
    Types are emitted in a canonical order (scalars before containers) to keep log
    noise deterministic; the order does not affect the coerced value.
    """
    # The union node itself may carry constraints (enum/const) or composition (allOf/not/if,
    # or both anyOf+oneOf) that coerce past the branch types — the same fabrication class as a
    # constrained branch, one level up. Coercion validates none of it, so leave the value alone.
    if any(k in schema for k in ("enum", "const", "allOf", "not", "if")) or (
        "anyOf" in schema and "oneOf" in schema
    ):
        return None
    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if not isinstance(branches, list) or not branches:
            continue
        types: list[str] = []
        for branch in branches:
            resolved = resolve_local_schema_ref(branch, root_schema)
            if not isinstance(resolved, dict):
                return None
            if "enum" in resolved or "const" in resolved or any(
                k in resolved for k in ("allOf", "anyOf", "oneOf", "not", "if")
            ):
                # A constrained branch (enum/const, even with a plain type) or a composed
                # branch (allOf/anyOf/oneOf/not carrying constraints): type coercion validates
                # neither members nor composed constraints, so coercing to the branch type
                # could fabricate a disallowed value that reaches the handler. Leave it alone.
                return None
            branch_type = resolved.get("type")
            if not isinstance(branch_type, str):
                return None  # nested type-list, allOf, dangling/cyclic $ref
            types.append(branch_type)
        if "string" in types:
            return None  # string already valid; do not reinterpret
        order = {"null": 0, "boolean": 1, "integer": 2, "number": 3, "array": 4, "object": 5}
        unique = sorted(dict.fromkeys(types), key=lambda t: order.get(t, len(order)))
        return unique[0] if len(unique) == 1 else unique
    return None


def _schema_accepts_kind(
    schema: Any,
    kind: str,
    root_schema: Any = None,
    seen_schema_ids: frozenset[int] = frozenset(),
) -> bool:
    """True when *schema* permits JSON type *kind* via ``type`` or any anyOf/oneOf/allOf branch."""
    if not isinstance(schema, dict):
        return False
    if id(schema) in seen_schema_ids:
        return False
    # Track this branch only: a recursive union must not hide its other alternatives.
    seen_schema_ids = seen_schema_ids | {id(schema)}
    if root_schema is None:
        root_schema = schema
    schema = resolve_local_schema_ref(schema, root_schema)
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    return any(isinstance(branches := schema.get(union_key), list) and any(_schema_accepts_kind(b, kind, root_schema, seen_schema_ids) for b in branches)
               for union_key in ("anyOf", "oneOf", "allOf"))


def _normalize_json_strings_for_schema(
    value: Any,
    schema: Any,
    root_schema: Any = None,
) -> Any:
    """Recursively parse JSON-encoded strings where the schema expects array/object.

    Schema-guided: a string is only parsed when its schema position expects a
    container, so legitimate JSON-looking ``type: string`` fields survive.
    Returns the same object when nothing changed (identity = cheap no-op check).

    Ported from cline/cline#11803, adapted to hermes-agent's coercion layer.
    """
    if not isinstance(schema, dict):
        return value
    if root_schema is None:
        root_schema = schema
    schema = resolve_local_schema_ref(schema, root_schema)

    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array", root_schema)
        expects_object = _schema_accepts_kind(schema, "object", root_schema)
        if not ((expects_array and trimmed.startswith("[")) or (expects_object and trimmed.startswith("{"))):
            return value
        try:
            parsed = json.loads(trimmed)
        except (ValueError, TypeError):
            return value
        if not ((isinstance(parsed, list) and expects_array) or (isinstance(parsed, dict) and expects_object)):
            return value
        value = parsed

    if isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        out = [
            _normalize_json_strings_for_schema(
                item, items_schema, root_schema=root_schema
            )
            for item in value
        ]
        return out if any(n is not o for n, o in zip(out, value)) else value

    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        out = dict(value)
        for k, prop_schema in props.items():
            if k in value and isinstance(prop_schema, dict):
                out[k] = _normalize_json_strings_for_schema(
                    value[k], prop_schema, root_schema=root_schema
                )
        return out if any(out[k] is not v for k, v in value.items()) else value

    return value


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Coerce string *value* to *expected_type* (str or union list); original on failure."""
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        return next((r for t in expected_type if (r := _coerce_value(value, t, schema=schema)) is not value), value)

    coercer = _SCALAR_COERCERS.get(expected_type)
    if coercer is not None:
        return coercer(value)
    return None if expected_type == "null" and value.strip().lower() == "null" else value


def _schema_allows_null(schema: dict | None) -> bool:
    """True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if schema.get("nullable") is True:
        return True
    return any(isinstance(variants := schema.get(union_key), list)
               and any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
               for union_key in ("anyOf", "oneOf"))


def _coerce_json(value: str, expected_python_type: type):
    """json.loads *value* when the schema expects array/object; original string on mismatch."""
    name = expected_python_type.__name__
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        logger.warning("coerce_tool_args: failed to parse string as JSON for expected type %s: %s", name, exc)
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug("coerce_tool_args: coerced string to %s via json.loads", name)
        return parsed
    logger.warning("coerce_tool_args: JSON-parsed value is %s, expected %s — skipping coercion",
                   type(parsed).__name__, name)
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Parse *value* as a number; original string on failure, inf/nan, or decimals when integer_only.

    Integer literals are parsed with ``int`` directly so large integers (Discord
    snowflakes, Telegram ids) keep full precision: a ``float`` round-trip silently
    corrupts values beyond 2**53. This guards integer branches newly coerced through
    type-less unions (#123345) and also fixes the pre-existing precision loss for a
    plain ``type: integer`` (which already corrupted large ids on main). The set of
    accepted strings is unchanged — ``float`` already accepted underscores and
    unicode digits.
    """
    try:
        return int(value.strip())
    except (ValueError, OverflowError):
        pass
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    if f != f or f in (float("inf"), float("-inf")):
        return value  # not JSON-serializable
    if f == int(f):
        return int(f)
    return value if integer_only else f


def _coerce_boolean(value: str):
    """Parse "true"/"false" (case-insensitive); original string otherwise."""
    return {"true": True, "false": False}.get(value.strip().lower(), value)


# JSON-Schema scalar/container type -> coercer; "null" and unions are handled in _coerce_value.
_SCALAR_COERCERS = {
    "integer": lambda v: _coerce_number(v, integer_only=True),
    "number": _coerce_number,
    "boolean": _coerce_boolean,
    "array": lambda v: _coerce_json(v, list),
    "object": lambda v: _coerce_json(v, dict),
}
