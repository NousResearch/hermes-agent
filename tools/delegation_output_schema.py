"""Structured-output schema helpers for delegate_task (T1-24).

Optional per-task ``output_schema`` (a JSON Schema object): the child is
told about the contract via an OUTPUT CONTRACT block appended to its
context, the parent validates the child's final answer with jsonschema,
and on failure sends exactly ONE bounded retry turn carrying the
validation errors verbatim (per llm-structured-output-schema-design:
max 1 retry, exact errors, no schema re-paste).

Pattern from: github/copilot-cli ctx.agent(prompt, {schema}) — PATTERN
ONLY, zero code/prompt text copied (proprietary).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Exactly one retry turn — bounded by design. More retries make frontier
# models drop fields that were right the first time.
MAX_SCHEMA_RETRIES = 1

_CONTRACT_HEADER = "OUTPUT CONTRACT (machine-validated)"


def coerce_output_schema(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate a model/caller-supplied output_schema value.

    Returns ``(schema, None)`` when usable, ``(None, error)`` when not.
    ``None`` input passes through as ``(None, None)`` (no schema requested).
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        # Models sometimes double-encode the schema as a JSON string.
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None, "output_schema must be a JSON Schema object, got a non-JSON string."
        if not isinstance(parsed, dict):
            return None, "output_schema must be a JSON Schema object."
        raw = parsed
    if not isinstance(raw, dict):
        return None, (
            f"output_schema must be a JSON Schema object, got {type(raw).__name__}."
        )
    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]

        validator_for(raw).check_schema(raw)
    except ImportError:
        # jsonschema is a hard dependency in practice; degrade to accepting
        # the dict as-is so delegation still works without it.
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
    block = (
        f"{_CONTRACT_HEADER}:\n"
        "Your FINAL response must be a single JSON object that validates "
        "against this JSON Schema. No prose before or after the JSON; a "
        "```json code fence is acceptable but not required.\n"
        f"{schema_text}"
    )
    base = (context or "").rstrip()
    return f"{base}\n\n{block}" if base else block


def extract_json_candidate(text: str) -> str:
    """Best-effort extraction of a JSON payload from model output.

    Strips markdown code fences and leading/trailing prose around the
    outermost ``{...}`` / ``[...]`` span. Returns the (possibly unchanged)
    candidate string; parsing errors are reported by validate_output.
    """
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


def validate_output(
    text: str, schema: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate a child's final answer against ``schema``.

    Returns ``(True, [])`` on success or ``(False, errors)`` where errors
    are human-readable strings suitable for the retry turn.
    """
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
    if not errors:
        return True, []
    rendered: List[str] = []
    for err in errors[:10]:  # bound error volume for the retry prompt
        path = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path
        )
        rendered.append(f"{path}: {err.message}")
    return False, rendered


def build_retry_message(errors: List[str]) -> str:
    """Build the single bounded retry turn sent to the child.

    Carries the validation errors verbatim; deliberately does NOT
    re-paste the schema (the child already has it in its context).
    """
    error_block = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous final response was rejected by the output contract "
        "validator. Validation errors:\n"
        f"{error_block}\n\n"
        "Reply with ONLY the corrected JSON object matching the OUTPUT "
        "CONTRACT schema from your task context. No prose, no explanations."
    )
