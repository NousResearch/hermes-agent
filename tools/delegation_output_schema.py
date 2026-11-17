"""Structured-output schema helpers shared by delegate_task, cron jobs and /v1/runs.

Optional ``output_schema`` (a JSON Schema object): the agent gets an OUTPUT
CONTRACT block appended to its task, the host validates the final answer with
jsonschema, and on failure sends exactly ONE bounded retry turn carrying the
validation errors verbatim (more retries make frontier models drop fields that
were right the first time). ``settle_output`` is that validate-retry-validate
step for any host that can run one more turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def coerce_output_schema(raw: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
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


def append_output_contract(context: Optional[str], schema: dict[str, Any]) -> str:
    """Append the explicit output contract block to a child's context."""
    try:
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        schema_text = str(schema)
    block = ("OUTPUT CONTRACT (machine-validated):\n"
             "Your FINAL response must be ONLY the JSON value that validates against this JSON "
             "Schema — no prose before or after it, no code fence, no explanation. Anything else "
             "costs a correction turn and, if it fails again, is handed to the caller unvalidated.\n"
             f"{schema_text}")
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
    # Try each bracket kind's outermost span, earliest opener first, and keep the first that parses:
    # checking "{" before "[" unconditionally sliced a fenced array down to its first..last object and
    # rejected every valid array answer.
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start >= 0 and end > start:
            spans.append((start, raw[start : end + 1]))
    for _start, candidate in sorted(spans):
        try:
            json.loads(candidate)
            return candidate
        except ValueError:
            continue
    return spans[0][1] if spans else raw


def validate_output(text: str, schema: dict[str, Any]) -> tuple[bool, list[str]]:
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


def build_retry_message(errors: list[str]) -> str:
    """Single bounded retry turn: errors verbatim. The caller appends the contract itself, because a
    compaction of a long first turn can leave the original one only in a summary."""
    error_block = "\n".join(f"- {e}" for e in errors)
    return ("Your previous final response was rejected by the output contract "
            "validator. Validation errors:\n" f"{error_block}\n\n"
            "Reply with ONLY the corrected JSON object matching the OUTPUT "
            "CONTRACT schema from your task context. No prose, no explanations.")


def parse_validated_output(text: str, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """``(value, [])`` with the parsed JSON value that satisfies *schema*, else ``(None, errors)``."""
    valid, errors = validate_output(text, schema)
    if not valid:
        return None, errors
    return json.loads(extract_json_candidate(text or "")), []


@dataclass(frozen=True)
class SettledOutput:
    """Verdict of ``settle_output``: ``text`` is the answer the caller keeps, ``value`` its parsed
    JSON when ``valid``."""

    valid: bool
    text: str
    value: Any = None
    errors: list[str] = field(default_factory=list)
    retries: int = 0


def settle_output(
    text: str, schema: dict[str, Any], retry: Optional[Callable[[str], Optional[str]]],
) -> SettledOutput:
    """Validate *text*; when it fails and is non-empty, run ONE correction turn through ``retry``.

    ``retry(message)`` sends *message* (errors verbatim + the contract re-pasted: a compaction of a
    long first turn can leave the original only in a summary) as the next user turn of the same
    conversation and returns its final text, or ``None`` when no turn ran. A blank retry answer
    keeps the first text but is the verdict. ``retry=None`` (interrupted run) only validates.
    """
    value, errors = parse_validated_output(text, schema)
    if not errors or not (text or "").strip() or retry is None:
        return SettledOutput(not errors, text, value, errors)
    retried = retry(append_output_contract(build_retry_message(errors), schema))
    if not isinstance(retried, str):
        return SettledOutput(False, text, None, errors, 1)
    value, errors = parse_validated_output(retried, schema)
    return SettledOutput(not errors, retried if retried.strip() else text, value, errors, 1)
