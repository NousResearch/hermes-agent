#!/usr/bin/env python3
"""
Noun Learner Tool — updates the NDR Draas Google Contacts spreadsheet
after voice corrections and conversation events.

Registered as: noun_learner
Toolset: google_workspace

Actions:
  learn_correction     — add a voice misspelling to the misspellings column
  add_alias            — add a short abbreviation alias (e.g. SLP, RO, DRA)
  update_associations  — update associated contacts/projects/entities/land
  append_history       — append a summary line to conversation_history column
  increment_score      — bump contact_score for a resolved contact

All writes are targeted single-cell/range updates.
Forces a noun_resolver index rebuild after every successful write.
"""

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

SHEET_ID = "1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g"

# Column letters for each sheet (for targeted cell writes)
SHEET_WRITE_COLS = {
    "contacts": {
        "voice_misspellings": "CN",
        "contact_score":      "CO",
        "conversation_history": "CG",
        "people_association": "CF",
        "project_association": "CA",
        "land_association":   "CB",
        "topic_association":  "CC",
        "alias":              "CE",
    },
    "projects": {
        "voice_misspellings":   "C",
        "associated_contacts":  "D",
        "associated_entities":  "E",
        "associated_land":      "F",
        "conversation_history": "I",
        "alias":                "B",
    },
    "land_proposals": {
        "voice_misspellings":   "C",
        "associated_contacts":  "F",
        "associated_projects":  "G",
        "conversation_history": "J",
        "alias":                "B",
    },
    "entities": {
        "voice_misspellings":   "C",
        "associated_contacts":  "E",
        "associated_projects":  "F",
        "conversation_history": "H",
        "alias":                "B",
    },
    "topics": {
        "voice_misspellings":    "C",
        "associated_contacts":   "F",
        "associated_projects":   "G",
        "associated_land":       "H",
        "associated_entities":   "I",
        "conversation_history":  "J",
        "alias":                 "B",
    },
    "vocabulary_corrections": {
        "misspelt": "A",
        "correct": "B",
        "category": "C",
        "first_seen": "D",
        "occurrences": "E",
        "notes": "F",
    },
}

SHEET_TAB_NAMES = {
    "contacts":       "NDR DRAAS Google contacts.csv",
    "projects":       "projects",
    "land_proposals": "land_proposals",
    "entities":       "entities",
    "topics":         "topics",
    "vocabulary_corrections": "vocab_corrections",
}

# Referenced throughout _ensure_vocab_corrections_tab / _upsert_vocab_correction
# below. Was missing entirely -- every learn_correction call NameError'd here
# (caught by the outer try/except in _learn_correction, so it looked like a
# generic "Error writing misspelling" even though the misspelling had already
# been written to the contact row a line earlier).
VOCAB_CORRECTIONS_TAB = SHEET_TAB_NAMES["vocabulary_corrections"]



# ── Credential helper ─────────────────────────────────────────────────────────

def _get_session_user_cfg() -> dict:
    try:
        from gateway.session_context import get_gws_identity_env
        from tools._user_registry import get_user_config
        session_uid = get_gws_identity_env()
        return get_user_config(session_uid) or {}
    except Exception:
        return {}


def _get_sheet_id() -> str:
    cfg = _get_session_user_cfg()
    sheet_id = cfg.get("contacts_sheet_id", "")
    if not sheet_id:
        uid = cfg.get("draas_user_id", "unknown")
        raise RuntimeError(
            f"User {uid!r} has no contacts_sheet_id configured in their profile"
        )
    return sheet_id


def _build_service():
    """Return a Google Sheets API service for the current session user via vault.

    Delegates to ``gws_auth.build_service`` so the token is keyed by the
    canonical vault user_id and loaded as a refresh-capable Credentials object.
    """
    from tools import gws_auth
    from gateway.session_context import get_gws_identity_env

    cfg = _get_session_user_cfg()
    gws_service = cfg.get("gws_service", "")
    # get_gws_identity_env() covers both live-chat sessions and cron jobs
    # (cron deliberately clears HERMES_SESSION_USER_ID and stores the real
    # owner in HERMES_CRON_JOB_OWNER_ID instead).
    session_uid = get_gws_identity_env()

    if not session_uid:
        raise RuntimeError(
            "Cannot identify session user for Google Sheets access. "
            "No live session or cron job owner context available."
        )
    if not gws_service:
        raise RuntimeError(
            f"User {session_uid!r} has no gws_service configured in their profile."
        )

    return gws_auth.build_service("sheets", "v4", service_name=gws_service)


def _cell(tab: str, col: str, row: int) -> str:
    return f"'{tab}'!{col}{row}"


def _read_cell(svc, tab: str, col: str, row: int) -> str:
    resp = svc.spreadsheets().values().get(
        spreadsheetId=_get_sheet_id(),
        range=_cell(tab, col, row),
    ).execute()
    vals = resp.get("values", [])
    return vals[0][0] if vals and vals[0] else ""


def _write_cell(svc, tab: str, col: str, row: int, value: str):
    svc.spreadsheets().values().update(
        spreadsheetId=_get_sheet_id(),
        range=_cell(tab, col, row),
        valueInputOption="RAW",
        body={"values": [[value]]},
    ).execute()


def _append_to_cell(svc, tab: str, col: str, row: int, new_value: str, sep: str = " | "):
    """Read current cell value, append new_value if not already present."""
    current = _read_cell(svc, tab, col, row)
    existing = [v.strip() for v in re.split(r"[|,;]", current) if v.strip()]
    if new_value.strip() in existing:
        return  # already there
    updated = sep.join(existing + [new_value.strip()])
    _write_cell(svc, tab, col, row, updated)


# ── Action handlers ───────────────────────────────────────────────────────────

def _ensure_vocab_corrections_tab(svc):
    """Create vocab_corrections tab with headers if it does not exist."""
    meta = svc.spreadsheets().get(spreadsheetId=_get_sheet_id()).execute()
    existing = {s["properties"]["title"] for s in meta.get("sheets", [])}
    if VOCAB_CORRECTIONS_TAB in existing:
        return
    svc.spreadsheets().batchUpdate(
        spreadsheetId=_get_sheet_id(),
        body={"requests": [{"addSheet": {"properties": {"title": VOCAB_CORRECTIONS_TAB}}}]}
    ).execute()
    svc.spreadsheets().values().update(
        spreadsheetId=_get_sheet_id(),
        range=f"{VOCAB_CORRECTIONS_TAB}!A1:F1",
        valueInputOption="RAW",
        body={"values": [["misspelt", "correct", "category", "first_seen", "occurrences", "notes"]]},
    ).execute()


def _upsert_vocab_correction(svc, misspelling: str, correct: str, category: str):
    """
    Insert or increment a row in vocab_corrections.
    Deduplicates by (misspelt, correct) pair — increments occurrences if already present.
    """
    _ensure_vocab_corrections_tab(svc)
    resp = svc.spreadsheets().values().get(
        spreadsheetId=_get_sheet_id(),
        range=f"{VOCAB_CORRECTIONS_TAB}!A:F",
    ).execute()
    rows = resp.get("values", [])

    misspelling_l = misspelling.lower()
    correct_l = correct.lower()

    # Row 0 is header; data starts at index 1 → sheet row = index + 1
    for i, row in enumerate(rows[1:], start=2):
        if (
            len(row) >= 2
            and row[0].strip().lower() == misspelling_l
            and row[1].strip().lower() == correct_l
        ):
            # Found — increment occurrences (col E = index 4)
            current_count = int(row[4]) if len(row) > 4 and row[4].isdigit() else 0
            svc.spreadsheets().values().update(
                spreadsheetId=_get_sheet_id(),
                range=f"{VOCAB_CORRECTIONS_TAB}!E{i}",
                valueInputOption="RAW",
                body={"values": [[str(current_count + 1)]]},
            ).execute()
            return

    # Not found — append new row
    today = datetime.utcnow().strftime("%Y-%m-%d")
    svc.spreadsheets().values().append(
        spreadsheetId=_get_sheet_id(),
        range=f"{VOCAB_CORRECTIONS_TAB}!A:F",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[misspelling, correct, category, today, "1", ""]]},
    ).execute()


def _learn_correction(args: dict) -> str:
    """
    Add a voice misspelling to the voice_misspellings column AND to vocab_corrections tab.

    Required: sheet_type, row, misspelling
    Optional: correct (canonical correct spelling — inferred from sheet if omitted)
    """
    sheet_type  = args.get("sheet_type", "contacts")
    misspelling = args.get("misspelling", "").strip()
    correct     = args.get("correct", "").strip()

    try:
        row = int(args.get("row", 0))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": "row must be an integer"})
    if row < 2:
        return json.dumps({"success": False, "error": "row must be >= 2 (row 1 is the header)"})

    if not misspelling:
        return "Error: row and misspelling are required"
    if sheet_type not in SHEET_WRITE_COLS:
        return f"Error: unknown sheet_type '{sheet_type}'"

    cols = SHEET_WRITE_COLS[sheet_type]
    if "voice_misspellings" not in cols:
        return f"Error: {sheet_type} has no voice_misspellings column"

    tab = SHEET_TAB_NAMES[sheet_type]
    col = cols["voice_misspellings"]

    try:
        svc = _build_service()
        _append_to_cell(svc, tab, col, row, misspelling)

        # If caller didn't pass correct spelling, read canonical name from col A
        if not correct:
            try:
                correct = _read_cell(svc, tab, "A", row).strip()
            except Exception:
                correct = ""

        # Write to vocab_corrections (deduped with occurrence counting)
        if correct:
            _upsert_vocab_correction(svc, misspelling, correct, sheet_type)

        _trigger_index_rebuild()
        msg = f"Learned: '{misspelling}' added to {sheet_type} row {row} misspellings"
        if correct:
            msg += f" and vocab_corrections ('{misspelling}' → '{correct}')"
        return msg
    except Exception as e:
        return f"Error writing misspelling: {e}"


def _add_alias(args: dict) -> str:
    """
    Add a short alias to the alias column.

    Use when the original voice text is an all-uppercase abbreviation (2–4 chars),
    e.g. 'SLP', 'RO', 'DRA'. These are intentional short-forms, not misspellings.
    contacts → column CE  |  projects/land_proposals/entities/topics → column B

    Required: sheet_type, row, alias
    """
    sheet_type = args.get("sheet_type", "contacts")
    alias_val  = args.get("alias", "").strip()

    try:
        row = int(args.get("row", 0))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": "row must be an integer"})
    if row < 2:
        return json.dumps({"success": False, "error": "row must be >= 2 (row 1 is the header)"})
    if not alias_val:
        return "Error: alias is required"
    if sheet_type not in SHEET_WRITE_COLS:
        return f"Error: unknown sheet_type '{sheet_type}'"

    cols = SHEET_WRITE_COLS[sheet_type]
    if "alias" not in cols:
        return f"Error: {sheet_type} has no alias column configured"

    tab = SHEET_TAB_NAMES[sheet_type]
    col = cols["alias"]

    try:
        svc = _build_service()
        _append_to_cell(svc, tab, col, row, alias_val)
        _trigger_index_rebuild()
        return f"Alias saved: '{alias_val}' added to {sheet_type} row {row} aliases"
    except Exception as e:
        return f"Error writing alias: {e}"


def _update_associations(args: dict) -> str:
    """
    Update one or more association columns for a row.

    Required: sheet_type, row
    Optional: contacts, projects, entities, land_proposals (comma-sep strings)
    """
    sheet_type = args.get("sheet_type", "contacts")

    try:
        row = int(args.get("row", 0))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": "row must be an integer"})
    if row < 2:
        return json.dumps({"success": False, "error": "row must be >= 2 (row 1 is the header)"})

    cols = SHEET_WRITE_COLS.get(sheet_type, {})
    tab  = SHEET_TAB_NAMES.get(sheet_type, "")
    if not tab:
        return f"Error: unknown sheet_type '{sheet_type}'"

    mapping = {
        "contacts":      "associated_contacts",
        "projects":      "associated_projects",
        "entities":      "associated_entities",
        "land_proposals": "associated_land",
    }

    updated = []
    try:
        svc = _build_service()
        for arg_key, col_key in mapping.items():
            value = args.get(arg_key, "").strip()
            if value and col_key in cols:
                for item in re.split(r"[,;]", value):
                    item = item.strip()
                    if item:
                        _append_to_cell(svc, tab, cols[col_key], row, item)
                updated.append(arg_key)
        _trigger_index_rebuild()
        return f"Updated associations ({', '.join(updated)}) for {sheet_type} row {row}"
    except Exception as e:
        return f"Error updating associations: {e}"


def _append_history(args: dict) -> str:
    """
    Append a timestamped summary line to the conversation_history column.

    Required: sheet_type, row, summary
    """
    sheet_type = args.get("sheet_type", "contacts")
    summary    = args.get("summary", "").strip()

    try:
        row = int(args.get("row", 0))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": "row must be an integer"})
    if row < 2:
        return json.dumps({"success": False, "error": "row must be >= 2 (row 1 is the header)"})

    if not summary:
        return "Error: summary is required"

    cols = SHEET_WRITE_COLS.get(sheet_type, {})
    tab  = SHEET_TAB_NAMES.get(sheet_type, "")
    if "conversation_history" not in cols:
        return f"Error: {sheet_type} has no conversation_history column"

    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    entry = f"[{timestamp}] {summary}"

    try:
        svc = _build_service()
        current = _read_cell(svc, tab, cols["conversation_history"], row)
        updated = (current + "\n" + entry).strip() if current else entry
        _write_cell(svc, tab, cols["conversation_history"], row, updated)
        _trigger_index_rebuild()
        return f"History appended to {sheet_type} row {row}"
    except Exception as e:
        return f"Error appending history: {e}"


def _increment_score(args: dict) -> str:
    """
    Increment contact_score for a contacts row.

    Required: row
    Optional: amount (default 1)
    """
    try:
        row = int(args.get("row", 0))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": "row must be an integer"})
    if row < 2:
        return json.dumps({"success": False, "error": "row must be >= 2 (row 1 is the header)"})

    try:
        amount = int(args.get("amount", 1))
    except (ValueError, TypeError):
        amount = 1

    try:
        from tools.noun_resolver import get_resolver
        get_resolver().increment_contact_score(row, amount)
        return f"contact_score incremented by {amount} for row {row}"
    except Exception as e:
        return f"Error incrementing score: {e}"


def _trigger_index_rebuild():
    """Non-blocking index rebuild after a write."""
    try:
        from tools.noun_resolver import get_resolver
        get_resolver().force_refresh()
    except Exception:
        pass


# ── Tool handler ──────────────────────────────────────────────────────────────

_ACTIONS = {
    "learn_correction":    _learn_correction,
    "add_alias":           _add_alias,
    "update_associations": _update_associations,
    "append_history":      _append_history,
    "increment_score":     _increment_score,
}


def _handle_noun_learner(args: dict, **_) -> str:
    action = args.get("action", "")
    handler = _ACTIONS.get(action)
    if not handler:
        return f"Error: unknown action '{action}'. Valid: {', '.join(_ACTIONS)}"
    try:
        return handler(args)
    except Exception as e:
        logger.exception(f"noun_learner action={action} failed")
        return f"Error in noun_learner ({action}): {e}"


def _check_available() -> bool:
    try:
        import googleapiclient  # noqa: F401
        from google.oauth2.credentials import Credentials  # noqa: F401
        return True
    except ImportError:
        return False


# ── Schema ────────────────────────────────────────────────────────────────────

_NOUN_LEARNER_SCHEMA = {
    "name": "noun_learner",
    "description": (
        "Update the NDR Draas Google Contacts spreadsheet with learned noun corrections "
        "and conversation context. Use after resolving a voice noun or when the user "
        "corrects a misrecognized word. Actions: learn_correction, add_alias, "
        "update_associations, append_history, increment_score."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["learn_correction", "add_alias", "update_associations", "append_history", "increment_score"],
                "description": "What to update.",
            },
            "sheet_type": {
                "type": "string",
                "enum": ["contacts", "projects", "land_proposals", "entities", "topics"],
                "description": "Which sheet the row is in.",
            },
            "row": {
                "type": "integer",
                "description": "Spreadsheet row number (from noun resolver result).",
            },
            "misspelling": {
                "type": "string",
                "description": "The incorrect voice recognition text to save (for learn_correction).",
            },
            "correct": {
                "type": "string",
                "description": (
                    "The correct canonical spelling for the misspelling (for learn_correction). "
                    "Pass this explicitly when you know it — e.g. the contact's full name, project name, etc. "
                    "If omitted, the tool reads column A of the target row as the correct form."
                ),
            },
            "alias": {
                "type": "string",
                "description": (
                    "Short abbreviation to save as a permanent alias (for add_alias action). "
                    "Use for all-uppercase 2–4 char tokens like 'SLP', 'RO', 'DRA'."
                ),
            },
            "summary": {
                "type": "string",
                "description": "Brief conversation summary to append to history (for append_history).",
            },
            "contacts": {"type": "string", "description": "Comma-separated contact names to associate."},
            "projects":  {"type": "string", "description": "Comma-separated project names to associate."},
            "entities":  {"type": "string", "description": "Comma-separated entity names to associate."},
            "land_proposals": {"type": "string", "description": "Comma-separated land proposal names to associate."},
            "amount": {"type": "integer", "description": "Score increment amount (default 1, for increment_score)."},
        },
        "required": ["action"],
    },
}


# ── Registration ──────────────────────────────────────────────────────────────

# NOTE: registration must be a BARE top-level call — the auto-discovery AST
# scan (tools/registry.py::_module_registers_tools) only picks up modules with
# a top-level ``registry.register(...)`` expression statement. Wrapping it in
# a function or try-block silently disables the tool (regression fixed
# 2026-08-02, same pattern as tools/smart_browser_tool.py).
from tools.registry import registry

registry.register(
    name="noun_learner",
    toolset="general",
    schema=_NOUN_LEARNER_SCHEMA,
    handler=_handle_noun_learner,
    check_fn=_check_available,
    is_async=False,
    description=_NOUN_LEARNER_SCHEMA["description"],
    emoji="📚",
)
logger.info("noun_learner tool registered")
