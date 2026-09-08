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

# Logger name kept as "model_tools": these messages were always emitted under
# that name and log-based tooling filters on it.
logger = logging.getLogger("model_tools")


def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce string-typed args to their JSON-Schema types; originals kept on failure."""
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    properties = ((schema or {}).get("parameters") or {}).get("properties")
    if not properties:
        return args

    # The model saw the SANITIZED schema (provider-illegal property keys were
    # renamed); map those keys back to the registry's wire names first.
    try:
        from tools.schema_sanitizer import unrename_tool_args
        args = unrename_tool_args(schema.get("parameters"), args)
    except Exception:  # pragma: no cover — never break dispatch
        pass

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")
        union_accepts_string = False
        if not expected:
            # Unions carry their types inside anyOf/oneOf variants; resolve
            # them so the repairs below also apply to union-typed properties
            # (e.g. terminal's notify, the #23129 read_url shape).
            expected = _union_schema_types(prop_schema)
            if isinstance(expected, list) and "string" in expected:
                # A string branch accepts any string, so scalar repairs are
                # ambiguous ("42"/"true" are legitimate strings); keep only
                # the container/null branches so a JSON-encoded array/object
                # string still parses.
                union_accepts_string = True
                expected = [t for t in expected if t in ("array", "object", "null")] or None
        is_container = isinstance(value, (list, tuple))

        # Bare non-list value for an array schema. Strings go through
        # _coerce_value first so a JSON-encoded array is parsed and a nullable
        # "null" becomes None (not ["null"]). None itself is preserved: the tool's
        # own default handling decides between "omit" and "empty list".
        # Unions only get the wrap when no string branch exists: a value that
        # is already a legal string is never unambiguously "meant as a list".
        if ((expected == "array" or (isinstance(expected, list) and "array" in expected))
                and not union_accepts_string and value is not None and not is_container):
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
            if not isinstance(expected, list):
                # Plain array schemas keep the historical bare-scalar wrap.
                # Unions must not: a non-string value that a scalar branch
                # already accepts carries its own meaning. Wrapping terminal's
                # notify=true into [true] silently flipped completion
                # notification to watch-pattern mode (dispatch maps a list to
                # watch_patterns and forces notify_on_complete=False).
                args[key] = [value]
                logger.info("coerce_tool_args: wrapped bare %s in list for %s.%s", type(value).__name__, tool_name, key)
            continue

        if not isinstance(value, str):
            # Native container: still normalize JSON-encoded elements/sub-fields.
            if ((expected == "array" and is_container)
                    or (expected == "object" and isinstance(value, dict))
                    or (isinstance(expected, list) and (is_container or isinstance(value, dict)))):
                args[key] = _normalize_json_strings_for_schema(value, prop_schema)
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced
            if isinstance(coerced, (list, tuple, dict)):
                args[key] = _normalize_json_strings_for_schema(coerced, prop_schema)

    return args


def _union_schema_types(prop_schema: dict) -> str | list[str] | None:
    """Concrete type name(s) declared by anyOf/oneOf variants, in schema order.

    Returns a plain string when the variants agree on a single non-null type
    (nullable unions like ``[{integer}, {null}]`` collapse to ``"integer"``),
    a list when several concrete types appear, and None when the union carries
    no usable ``type`` information. allOf is intentionally skipped: it is an
    intersection, so per-branch types cannot be tried independently.
    """
    for union_key in ("anyOf", "oneOf"):
        variants = prop_schema.get(union_key)
        if isinstance(variants, list) and variants:
            types = [v.get("type") for v in variants
                     if isinstance(v, dict) and isinstance(v.get("type"), str)]
            if types:
                concrete = [t for t in types if t != "null"]
                if not concrete:
                    return "null"
                if len(concrete) == 1:
                    return concrete[0]
                return concrete
    return None


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    """True when *schema* permits JSON type *kind* via ``type`` or any anyOf/oneOf/allOf branch."""
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    return any(isinstance(branches := schema.get(union_key), list) and any(_schema_accepts_kind(b, kind) for b in branches)
               for union_key in ("anyOf", "oneOf", "allOf"))


def _branch_accepting(schema: Any, kind: str) -> Any:
    """Narrow a union schema to its first branch that accepts *kind*.

    Plain (non-union) schemas pass through unchanged, so callers can keep
    reading ``items``/``properties`` off the result either way. Returns the
    original *schema* when no branch matches — repairs then simply find no
    nested schema and keep the value as-is.
    """
    if isinstance(schema, dict):
        for union_key in ("anyOf", "oneOf"):
            branches = schema.get(union_key)
            if isinstance(branches, list):
                return next((b for b in branches
                             if isinstance(b, dict) and _schema_accepts_kind(b, kind)), schema)
    return schema


def _normalize_json_strings_for_schema(value: Any, schema: Any) -> Any:
    """Recursively parse JSON-encoded strings where the schema expects array/object.

    Schema-guided: a string is only parsed when its schema position expects a
    container, so legitimate JSON-looking ``type: string`` fields survive.
    Returns the same object when nothing changed (identity = cheap no-op check).

    Ported from cline/cline#11803, adapted to hermes-agent's coercion layer.
    """
    if not isinstance(schema, dict):
        return value

    if isinstance(value, str):
        trimmed = value.strip()
        expects_array = _schema_accepts_kind(schema, "array")
        expects_object = _schema_accepts_kind(schema, "object")
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
        items_schema = _branch_accepting(schema, "array").get("items")
        if not isinstance(items_schema, dict):
            return value
        out = [_normalize_json_strings_for_schema(item, items_schema) for item in value]
        return out if any(n is not o for n, o in zip(out, value)) else value

    if isinstance(value, dict):
        props = _branch_accepting(schema, "object").get("properties")
        if not isinstance(props, dict):
            return value
        out = dict(value)
        for k, prop_schema in props.items():
            if k in value and isinstance(prop_schema, dict):
                out[k] = _normalize_json_strings_for_schema(value[k], prop_schema)
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
    """Parse *value* as a number; original string on failure, inf/nan, or decimals when integer_only."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    if f != f or f in (float("inf"), float("-inf")):
        return value  # not JSON-serializable
    return int(f) if f == int(f) else value if integer_only else f


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
