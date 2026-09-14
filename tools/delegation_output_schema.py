"""Structured-output contract helpers for delegate_task.

Optional per-task ``output_schema`` (a JSON Schema object) and the simpler
``output_fields`` shorthand both compile to one contract: the child gets an
OUTPUT CONTRACT block appended to its context, the parent validates the final
answer with jsonschema, and on failure sends exactly ONE bounded retry turn
carrying the validation errors verbatim (more retries make frontier models
drop fields that were right the first time; the schema is never re-pasted).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def coerce_output_schema(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(schema, None)`` when usable, ``(None, error)`` when not; ``None`` input
    passes through as ``(None, None)`` (no schema requested)."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        # Models sometimes double-encode the schema as a JSON string.
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None, "output_schema must be a JSON Schema object, got a non-JSON string."
        if not isinstance(raw, dict):
            return None, "output_schema must be a JSON Schema object."
    if not isinstance(raw, dict):
        return None, f"output_schema must be a JSON Schema object, got {type(raw).__name__}."
    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]
        validator_for(raw).check_schema(raw)
    except ImportError:
        # Degrade to accepting the dict as-is so delegation still works without jsonschema.
        logger.debug("jsonschema unavailable; skipping output_schema meta-validation")
    except Exception as exc:
        return None, f"output_schema is not a valid JSON Schema: {exc}"
    return raw, None


_SIMPLE_OUTPUT_TYPES = frozenset({"string", "integer", "number", "boolean", "object"})


def _simple_field_schema(type_name: Any) -> Optional[Dict[str, Any]]:
    """Return one JSON-Schema fragment for the deliberately small field vocabulary."""
    if not isinstance(type_name, str):
        return None
    is_array = type_name.endswith("[]")
    base_type = type_name[:-2] if is_array else type_name
    if base_type not in _SIMPLE_OUTPUT_TYPES:
        return None
    if not is_array:
        return {"type": base_type}
    return {"type": "array", "items": {"type": base_type}}


def compile_output_fields(
    raw_fields: Any, required_fields: Any = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Compile the flat ``output_fields`` shorthand into the existing JSON-Schema contract.

    The shorthand intentionally accepts only primitive types and their array forms. Keeping
    this vocabulary closed prevents it from becoming a second, partial JSON-Schema language;
    callers that need nested constraints can continue using ``output_schema``.
    """
    if raw_fields is None:
        if required_fields is None:
            return None, None
        return None, "required_output_fields requires output_fields."
    if not isinstance(raw_fields, dict):
        return None, "output_fields must be an object mapping field names to type names."
    if not raw_fields:
        return None, "output_fields must declare at least one field."
    if required_fields is None:
        normalized_required: List[str] = []
    elif not isinstance(required_fields, list):
        return None, "required_output_fields must be an array of field names."
    else:
        normalized_required = []
        for field_name in required_fields:
            if not isinstance(field_name, str) or not field_name:
                return None, "required_output_fields entries must be non-empty strings."
            if field_name in normalized_required:
                return None, f"required_output_fields contains duplicate field {field_name!r}."
            normalized_required.append(field_name)

    properties: Dict[str, Any] = {}
    for field_name, type_name in raw_fields.items():
        if not isinstance(field_name, str) or not field_name:
            return None, "output_fields keys must be non-empty strings."
        field_schema = _simple_field_schema(type_name)
        if field_schema is None:
            allowed = ", ".join(sorted(_SIMPLE_OUTPUT_TYPES)) + " and their [] forms"
            return None, f"output_fields[{field_name!r}] must use one of {allowed}; got {type_name!r}."
        properties[field_name] = field_schema

    unknown_required = [field_name for field_name in normalized_required if field_name not in properties]
    if unknown_required:
        return None, f"required_output_fields contains undeclared field(s): {', '.join(unknown_required)}."
    return {"type": "object", "properties": properties, "required": normalized_required}, None


def coerce_output_contract(
    raw_schema: Any, raw_fields: Any = None, required_fields: Any = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve one task's raw or simplified contract without duplicating precedence rules."""
    if raw_schema is not None and (raw_fields is not None or required_fields is not None):
        return None, "output_schema and output_fields are mutually exclusive; provide only one."
    if raw_fields is not None or required_fields is not None:
        return compile_output_fields(raw_fields, required_fields)
    return coerce_output_schema(raw_schema)


def append_output_contract(context: Optional[str], schema: Dict[str, Any]) -> str:
    """Append the explicit output contract block to a child's context."""
    try:
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        schema_text = str(schema)
    block = ("OUTPUT CONTRACT (machine-validated):\n"
             "Your FINAL response must be a single JSON object that validates "
             "against this JSON Schema. No prose before or after the JSON; a "
             "```json code fence is acceptable but not required.\n" f"{schema_text}")
    base = (context or "").rstrip()
    return f"{base}\n\n{block}" if base else block


def extract_json_candidate(text: str) -> str:
    """Strip markdown fences and prose around the outermost ``{...}``/``[...]``."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[: -3]
        raw = raw.strip()
        if raw.lower().startswith("json\n"):
            raw = raw.split("\n", 1)[1]
    for opener, closer in (("{", "}"), ("[", "]")):
        if raw.startswith(opener):
            return raw
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start >= 0 and end > start:
            return raw[start : end + 1]
    return raw


def validate_output(text: str, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """``(True, [])`` or ``(False, errors)`` with strings suitable for the retry turn."""
    candidate = extract_json_candidate(text or "")
    if not candidate.strip():
        return False, ["Response was empty — expected a JSON object matching the schema."]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError) as exc:
        return False, [f"Response is not valid JSON: {exc}"]
    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("jsonschema unavailable; accepting parsed JSON without validation")
        return True, []
    validator = validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
    rendered = [  # bound error volume for the retry prompt
        "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path) + f": {err.message}"
        for err in errors[:10]]
    return not rendered, rendered


def build_retry_message(errors: List[str]) -> str:
    """Single bounded retry turn: errors verbatim, schema deliberately NOT re-pasted."""
    error_block = "\n".join(f"- {e}" for e in errors)
    return ("Your previous final response was rejected by the output contract "
            "validator. Validation errors:\n" f"{error_block}\n\n"
            "Reply with ONLY the corrected JSON object matching the OUTPUT "
            "CONTRACT schema from your task context. No prose, no explanations.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

MAX_SCHEMA_RETRIES = 1
# ---- END PLUGIN-COMPAT ----
