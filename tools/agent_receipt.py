"""agent_receipt — the structured completion-receipt contract for reusable agent work.

One canonical JSON Schema (``agent_receipt.schema.json``, shipped beside this module)
plus a dependency-free, SCHEMA-DRIVEN validator. The validator reads its rules
(``required`` / ``type`` / ``enum`` / ``properties`` / ``items`` / ``minLength``) FROM
the loaded schema — it never hardcodes them — so editing the schema file changes
validation behavior (and the ``--self-test`` mutation check proves it).

Why stdlib, not ``jsonschema``: ``jsonschema`` is not an installed dependency, and the
one place the codebase uses it (``agent/plugin_llm.py``) SILENTLY SKIPS validation on
ImportError. A gate that silently no-ops is worse than none, so this module implements
the small Draft-07 subset it needs in plain Python.

The receipt is the STRUCTURED SUPERSET of the verify-status contract: each ``sources`` /
``commands`` entry carries its own ``verified|unverified`` status.

Consumed by three producer surfaces:
  - delegate (``tools/delegate_tool.py``): post-run ``receipt_owed`` stamp + optional repair.
  - kanban   (``tools/kanban_tools.py``): HARD reject of ``kanban_complete`` without a
    valid ``metadata.agent_receipt`` (mirrors the phantom-id gate).
  - cron      (audit ``~/.hermes/scripts``): an owed-but-missing receipt is a violation.

FAIL-OPEN, LOUD: any *infrastructure* fault (schema file unreadable AND embedded copy
unusable, or a validator code exception) degrades the gate to PASS rather than bricking
``kanban_complete`` — but it writes a ``receipt_gate_degraded`` marker + WARNING so the
degradation is observable, never silent. A schema *typo* on disk falls back to the
EMBEDDED schema (which still enforces); only a genuine code fault fails fully open.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(_HERE, "agent_receipt.schema.json")

# The 4 anchor fields that must be present AND non-empty (over and above being
# listed in the schema's ``required``). Derived behavior is still schema-driven;
# this set only names which required fields additionally reject empty values.
_ANCHOR_FIELDS = ("claim_id", "producer", "task", "stop_reason")

# Embedded fallback — kept byte-equivalent (by VALUE) to agent_receipt.schema.json.
# A test (test_agent_receipt.py) asserts EMBEDDED_SCHEMA == json.load(SCHEMA_PATH) so
# the fallback can never silently enforce stale rules. If you edit the .json, edit this.
EMBEDDED_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://hermes-agent/agent_receipt.schema.json",
    "title": "agent_receipt",
    "description": (
        "Structured completion receipt for reusable agent work (delegated subagent "
        "reports, Kanban completions, agent-driven cron output). The structured "
        "superset of the verify-status contract: every sources/commands entry carries "
        "its own verified|unverified status."
    ),
    "type": "object",
    "required": [
        "claim_id", "producer", "task", "stop_reason",
        "sources", "touched", "commands", "blockers", "next_owner",
    ],
    "additionalProperties": True,
    "properties": {
        "claim_id": {"type": "string", "minLength": 1, "description": "Stable id for this completion claim (anchor, non-empty)."},
        "producer": {"type": "string", "minLength": 1, "description": "Profile / lane / persona that produced the work (anchor, non-empty). NOT the model id — the harness already records that."},
        "task": {"type": "string", "minLength": 1, "description": "One line: what the work was (anchor, non-empty)."},
        "stop_reason": {"type": "string", "enum": ["completed", "blocked", "partial", "timeout", "aborted"], "description": "Why the work stopped (anchor)."},
        "sources": {
            "type": "array",
            "description": "What was read/consulted. May be [] when nothing was read.",
            "items": {
                "type": "object",
                "required": ["ref", "status"],
                "properties": {
                    "ref": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["verified", "unverified"]},
                },
            },
        },
        "touched": {"type": "array", "description": "Files/URLs created or modified. May be [].", "items": {"type": "string"}},
        "commands": {
            "type": "array",
            "description": "Tests / verifications run WITH pass-fail status. NOT a restatement of the harness tool_trace. May be [].",
            "items": {
                "type": "object",
                "required": ["cmd", "status"],
                "properties": {
                    "cmd": {"type": "string", "minLength": 1},
                    "result": {"type": "string"},
                    "status": {"type": "string", "enum": ["verified", "unverified"]},
                },
            },
        },
        "blockers": {"type": "array", "description": "Open blockers. May be [].", "items": {"type": "string"}},
        "next_owner": {"type": "string", "minLength": 1, "description": "Who/what picks this up next, or the literal string \"none\"."},
    },
}

# Default marker dir; overridable for tests via HERMES_STATE_DIR (an internal bridge
# env var, not user-facing config — only used to relocate the degraded marker in tests).
_STATE_DIR = os.environ.get("HERMES_STATE_DIR") or os.path.expanduser("~/.hermes/state")
_DEGRADED_MARKER = "receipt_gate_degraded"

# Pull the first fenced ```receipt ... ``` block out of free text.
_RECEIPT_BLOCK_RE = re.compile(r"```receipt\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Schema loading (schema-driven; embedded fallback; never throws to the caller)
# --------------------------------------------------------------------------- #
def load_schema(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the canonical schema.

    Resolution order: explicit ``path`` arg -> on-disk ``SCHEMA_PATH`` -> EMBEDDED_SCHEMA.
    Never raises: a missing/corrupt file falls back to the embedded copy (which still
    enforces the real rules), so a schema-file problem can never brick a hard gate.
    """
    target = path or SCHEMA_PATH
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and data.get("type") == "object":
            return data
        logger.warning("agent_receipt: schema at %s is malformed; using embedded fallback", target)
    except FileNotFoundError:
        # Only warn for an EXPLICIT path miss; the default-path miss is a normal
        # portability fallback (temp HERMES_HOME etc.).
        if path:
            logger.warning("agent_receipt: schema not found at %s; using embedded fallback", target)
    except (OSError, ValueError) as exc:
        logger.warning("agent_receipt: schema at %s unreadable (%s); using embedded fallback", target, exc)
    return EMBEDDED_SCHEMA


def _mark_degraded(reason: str) -> None:
    """Write a durable degraded marker + WARNING so a fail-open is never silent."""
    logger.warning("agent_receipt: gate DEGRADED (fail-open): %s", reason)
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(os.path.join(_STATE_DIR, _DEGRADED_MARKER), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "reason": reason}) + "\n")
    except OSError:
        pass  # marker is best-effort; the WARNING above is the floor


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_receipt(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first fenced ```receipt block out of free text and json.loads it.

    Returns the parsed dict, or None if there is no block or it does not parse.
    """
    if not text or not isinstance(text, str):
        return None
    m = _RECEIPT_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- #
# Schema-driven validation (small Draft-07 subset: type/required/enum/items/minLength)
# --------------------------------------------------------------------------- #
_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _type_ok(value: Any, typ: str) -> bool:
    py = _TYPE_MAP.get(typ)
    if py is None:
        return True  # unknown type keyword -> don't enforce
    if typ == "boolean":
        return isinstance(value, bool)
    if typ in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass; reject it for numeric fields
    return isinstance(value, py)


def _validate_node(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    typ = schema.get("type")
    if typ and not _type_ok(value, typ):
        errors.append(f"{path or 'value'}: expected {typ}, got {type(value).__name__}")
        return

    if typ == "string":
        ml = schema.get("minLength")
        if isinstance(ml, int) and len(value) < ml:
            errors.append(f"{path or 'value'}: must be non-empty (minLength {ml})")
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            errors.append(f"{path or 'value'}: '{value}' not one of {enum}")
        return

    if typ == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate_node(item, item_schema, f"{path}[{i}]", errors)
        return

    if typ == "object" or "properties" in schema or "required" in schema:
        if not isinstance(value, dict):
            return
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path + '.' if path else ''}{req}: required field missing")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value and isinstance(sub, dict):
                _validate_node(value[key], sub, f"{path + '.' if path else ''}{key}", errors)


def validate(obj: Any, schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Validate ``obj`` against the agent_receipt schema.

    Returns ``(ok, errors)``. Reads required/type/enum/items/minLength FROM ``schema``
    (defaults to the canonical loaded schema). Anchor fields additionally must be
    present and non-empty. Never raises: an internal validator fault is caught,
    marks the gate degraded, and returns ``(True, [])`` (fail-open) so a code bug
    can never brick the calling surface.
    """
    try:
        sch = schema if schema is not None else load_schema()
        if not isinstance(obj, dict):
            return False, ["agent_receipt did not match schema: top-level value must be a JSON object"]
        errors: List[str] = []
        _validate_node(obj, sch, "", errors)
        # Anchor non-empty rule (over and above schema minLength, for robustness if a
        # schema variant drops minLength).
        for anchor in _ANCHOR_FIELDS:
            if anchor in obj and isinstance(obj[anchor], str) and not obj[anchor].strip():
                msg = f"{anchor}: anchor field must be non-empty"
                if msg not in errors and all(anchor not in e for e in errors):
                    errors.append(msg)
        if errors:
            return False, [f"agent_receipt did not match schema: {e}" for e in errors]
        return True, []
    except Exception as exc:  # noqa: BLE001 — fail-open is the explicit contract
        _mark_degraded(f"validate() raised: {type(exc).__name__}: {exc}")
        return True, []


def validate_text(text: str, schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Extract a receipt from free text and validate it. Missing block -> invalid."""
    obj = extract_receipt(text)
    if obj is None:
        return False, ["agent_receipt did not match schema: no ```receipt block found / not valid JSON"]
    return validate(obj, schema)


# --------------------------------------------------------------------------- #
# owes_receipt — the KEYSTONE reusable-work signal
# --------------------------------------------------------------------------- #
def owes_receipt(
    *,
    role: Optional[str] = None,
    tool_trace: Optional[List[Any]] = None,
    response_text: Optional[str] = None,
    surface: str = "delegate",
) -> bool:
    """Decide whether a producer's output OWES a receipt.

    This replaces a manual ``expects_receipt`` opt-in (which a careless caller never
    sets — selection bias). It derives from data the harness already has.

    - surface='kanban': ALWAYS owes (an explicit completion of tracked work).
    - surface='delegate': owes IF role == 'orchestrator', OR the child wrote/patched
      files or ran commands (read from its own tool_trace). A pure read-only lookup
      with no side effects does NOT owe — no false receipt_valid:false.
    - surface='cron': owes IF the response makes reusable claims AND is not a
      [SILENT]/status-only run. The actual claimy/[SILENT] thresholds live in the
      audit (``verify-status-audit.py`` ``_looks_claimy`` requires >=2 hits); this
      module only exposes the predicate so callers route through one definition.
    """
    if surface == "kanban":
        return True

    if surface == "delegate":
        if (role or "").strip().lower() == "orchestrator":
            return True
        return _trace_has_side_effects(tool_trace)

    if surface == "cron":
        # Delegated to the audit's looks_claimy_not_silent; default conservative here.
        return _looks_claimy_not_silent(response_text or "")

    return False


# Tool names that indicate a real side effect (file write / command run), as opposed
# to a pure read/lookup. Conservative: matched as a substring, case-insensitive.
_SIDE_EFFECT_HINTS = (
    "write_file", "patch", "terminal", "execute_code", "edit", "str_replace",
    "create_file", "delete", "run", "exec", "bash", "shell",
)


def _trace_has_side_effects(tool_trace: Optional[List[Any]]) -> bool:
    if not tool_trace:
        return False
    for entry in tool_trace:
        name = ""
        if isinstance(entry, dict):
            name = str(entry.get("tool") or entry.get("name") or "")
        else:
            name = str(entry)
        low = name.lower()
        if any(h in low for h in _SIDE_EFFECT_HINTS):
            return True
    return False


# Lightweight local mirror of the audit's owes-filter so cron callers inside the fork
# have a definition without importing the ~/.hermes script. The AUTHORITATIVE thresholds
# live in verify-status-audit.py (_looks_claimy >=2 hits, _SILENT_RE); this is the same
# shape, kept deliberately simple. Cron enforcement is the audit's job, not this module's.
_SILENT_RE = re.compile(r"\[SILENT\]", re.IGNORECASE)
_CLAIM_HINT_RE = re.compile(
    r"\b(http[s]?://|returned\s+\d{3}|status\s+\d{3}|exit(?:ed)?\s+(?:code\s+)?\d|"
    r"\bversion\s+\d|\brepo(?:sitory)?\b|\bpackage\b|\bcommit\b|tests?\s+pass|"
    r"passing|all green|line\s+\d+|:\d+\b)",
    re.IGNORECASE,
)


def _looks_claimy_not_silent(text: str) -> bool:
    if not text or _SILENT_RE.search(text):
        return False
    return len(_CLAIM_HINT_RE.findall(text)) >= 2


# --------------------------------------------------------------------------- #
# Self-test (AC1, AC2, AC5, AC6)
# --------------------------------------------------------------------------- #
def _good_receipt() -> Dict[str, Any]:
    return {
        "claim_id": "rcpt-001",
        "producer": "reviewer-codex",
        "task": "Review the pricing route change",
        "stop_reason": "completed",
        "sources": [{"ref": "app/Pricing.php:42", "status": "verified"}],
        "touched": ["app/Pricing.php"],
        "commands": [{"cmd": "phpunit tests/PricingTest.php", "result": "3 passed", "status": "verified"}],
        "blockers": [],
        "next_owner": "none",
    }


def _self_test(schema_path: Optional[str] = None) -> int:
    schema = load_schema(schema_path)
    failures: List[str] = []

    def expect(label: str, got_ok: bool, want_ok: bool) -> None:
        if got_ok != want_ok:
            failures.append(f"  [{'PASS' if want_ok else 'FAIL'} expected] {label}: got ok={got_ok}")
        else:
            print(f"  ok: {label} -> {'PASS' if got_ok else 'FAIL'} (expected)")

    # --- GOOD cases ---
    expect("well-formed receipt", validate(_good_receipt(), schema)[0], True)

    empty_collections = _good_receipt()
    empty_collections.update({"sources": [], "touched": [], "commands": [], "blockers": [], "next_owner": "none"})
    expect("empty collections + next_owner 'none'", validate(empty_collections, schema)[0], True)

    md = "Here is my work.\n\n```receipt\n" + json.dumps(_good_receipt()) + "\n```\nDone."
    expect("markdown-wrapped fenced block parses+validates", validate_text(md, schema)[0], True)

    # --- BAD cases: missing each required key ---
    for key in schema.get("required", []):
        bad = _good_receipt()
        bad.pop(key, None)
        expect(f"missing required '{key}' rejected", validate(bad, schema)[0], False)

    # --- BAD: wrong type for sources ---
    bad_type = _good_receipt()
    bad_type["sources"] = "not-an-array"
    expect("wrong type for sources rejected", validate(bad_type, schema)[0], False)

    # --- BAD: bad stop_reason enum ---
    bad_enum = _good_receipt()
    bad_enum["stop_reason"] = "finished"
    expect("bad stop_reason enum rejected", validate(bad_enum, schema)[0], False)

    # --- BAD: empty anchor ---
    bad_anchor = _good_receipt()
    bad_anchor["claim_id"] = "  "
    expect("empty claim_id anchor rejected", validate(bad_anchor, schema)[0], False)

    # --- BAD: no receipt block in text ---
    expect("free text with no receipt block rejected", validate_text("just prose, no block", schema)[0], False)

    if failures:
        print("\nSELF-TEST FAILURES:")
        for f in failures:
            print(f)
        return 1
    print("\nself-test: all cases passed")
    return 0


def _self_test_mutation() -> int:
    """Prove the validator is SCHEMA-DRIVEN: drop stop_reason from a temp schema's
    required[] and confirm the 'missing stop_reason' bad-case stops failing — i.e.
    behavior tracks the schema file, not hardcoded rules. Returns 0 if the validator
    is correctly schema-driven."""
    import copy
    mutated = copy.deepcopy(load_schema())
    mutated["required"] = [r for r in mutated.get("required", []) if r != "stop_reason"]
    bad = _good_receipt()
    bad.pop("stop_reason", None)
    ok, _ = validate(bad, mutated)
    if ok:
        print("mutation check: PASS — dropping stop_reason from schema.required made the "
              "missing-stop_reason case validate (validator is schema-driven)")
        return 0
    print("mutation check: FAIL — validator still rejected missing stop_reason after the "
          "schema dropped it from required[] -> rules are HARDCODED, not schema-driven")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="agent_receipt schema-driven validator")
    ap.add_argument("--self-test", action="store_true", help="run the built-in good/bad table")
    ap.add_argument("--mutation-test", action="store_true", help="prove the validator is schema-driven")
    ap.add_argument("--schema", help="path to a schema file (overrides the canonical one)")
    ap.add_argument("--file", help="validate a receipt JSON / markdown file")
    args = ap.parse_args(argv)

    rc = 0
    if args.self_test:
        rc |= _self_test(args.schema)
    if args.mutation_test:
        rc |= _self_test_mutation()
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
        ok, errs = validate_text(text, load_schema(args.schema))
        print("VALID" if ok else "INVALID")
        for e in errs:
            print(" -", e)
        rc |= 0 if ok else 1
    if not (args.self_test or args.mutation_test or args.file):
        ap.print_help()
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
