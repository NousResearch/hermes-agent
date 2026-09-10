"""Structured-output schema helpers for delegate_task.

Optional per-task ``output_schema`` (a JSON Schema object): the child gets an
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
