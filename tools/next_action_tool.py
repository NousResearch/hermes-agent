"""Next Agent Action — structured CRUD tool over the Next Agent Action sheet.

Toolset: ``next_action``. Replaces the inline ``build_service('sheets','v4',
...)`` Python the cron agent previously wrote fresh every run (see
``hermes-data/skills/productivity/next-agent-action/SKILL.md``), and the
fragile "match the row by Date + Action text" lookup that skill's own notes
flagged as breaking when rows shift.

Every row now carries a stable ``Task_ID`` (short uuid) so lookups/updates
never depend on row position or text matching.

Spreadsheet (fixed, one file, three tabs):
  - ``Actions``        — the live queue the cron agent scans.
                          A:J = Date | Slot | Action | Context | Status |
                          Notes | Task_ID | Session_Key | Context_Keywords |
                          Memory_Refs. Status is Pending or
                          Awaiting_Clarification only — Done/Cancelled rows
                          are logged and removed, never left here.
  - ``Clarifications`` — async Q&A the cron agent parks when it can't
                          proceed; NDR answers by editing the sheet
                          directly (no tool call needed on his side).
                          A:E = Task_ID | Question | Answer | Asked_At |
                          Answered_At. Rows are deleted once consumed.
  - ``Completed Log``  — append-only audit trail, written right before a
                          row is deleted from Actions. A:H = Task_ID |
                          Date | Slot | Action | Context | Final_Status |
                          Notes | Completed_At.

Auth: reuses ``tools.gws_auth.build_service`` exactly as the skill already
did — per-session OAuth only, no new credential path. Note this means
every handler here requires a live session/cron-job context
(``HERMES_SESSION_USER_ID`` or the cron job-owner fallback) — it cannot be
dispatched outside one (verified during the ``next_action_add`` pilot).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from tools.gws_auth import build_service
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

TOOLSET = "next_action"
EMOJI = "\U0001F4C5"  # 📅

SPREADSHEET_ID = "1YR5LMHr4JG42anEKTYSdsIBwEVVcBHRBqOxUTNtIM1g"
SERVICE_NAME = "google-draas"

ACTIONS_TAB = "Actions"
CLARIFICATIONS_TAB = "Clarifications"
COMPLETED_LOG_TAB = "Completed Log"

# Actions!A:J
ACTIONS_RANGE = f"{ACTIONS_TAB}!A:J"
ACTIONS_NUM_COLS = 10
# Clarifications!A:E
CLARIFICATIONS_RANGE = f"{CLARIFICATIONS_TAB}!A:E"
CLARIFICATIONS_NUM_COLS = 5
# 'Completed Log'!A:H  (quoted — tab name has a space)
COMPLETED_LOG_RANGE = f"'{COMPLETED_LOG_TAB}'!A:H"

VALID_SLOTS = ("Morning", "Afternoon", "Evening")
ACTIVE_STATUSES = ("Pending", "Awaiting_Clarification")

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")


def _sheets_service():
    return build_service("sheets", "v4", service_name=SERVICE_NAME)


def _pad(row: list, width: int) -> list:
    return row + [""] * max(0, width - len(row))


def _get_sheet_id(service, title: str) -> int:
    meta = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        if props.get("title") == title:
            return props["sheetId"]
    raise RuntimeError(f"tab {title!r} not found in spreadsheet")


def _existing_task_ids(service) -> set:
    """Read column G (Task_ID) of Actions so a new id can't collide."""
    resp = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=f"{ACTIONS_TAB}!G2:G",
    ).execute()
    rows = resp.get("values", [])
    return {r[0] for r in rows if r}


def _new_task_id(service) -> str:
    existing = _existing_task_ids(service)
    for _ in range(10):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    # 10 collisions in a row on an 8-hex-char space is practically
    # impossible, but fail loudly instead of silently reusing an id.
    raise RuntimeError("could not generate a unique Task_ID after 10 attempts")


def _find_action_row(service, task_id: str):
    """Return (sheet_row_number_1based, padded_row_values) for the Actions
    row matching task_id, or (None, None) if not found. Row 1 is the header.
    """
    resp = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=ACTIONS_RANGE,
    ).execute()
    rows = resp.get("values", [])
    for i, raw in enumerate(rows):
        if i == 0:
            continue  # header
        row = _pad(raw, ACTIONS_NUM_COLS)
        if row[6] == task_id:
            return i + 1, row
    return None, None


def _delete_row(service, sheet_id: int, row_number_1based: int) -> None:
    """Delete a single row (1-based sheet row number, header included)."""
    start = row_number_1based - 1  # deleteDimension uses 0-based half-open range
    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={"requests": [{
            "deleteDimension": {
                "range": {"sheetId": sheet_id, "dimension": "ROWS",
                          "startIndex": start, "endIndex": start + 1},
            },
        }]},
    ).execute()


def _log_completion(service, task_id: str, row: list, final_status: str, notes_suffix: str) -> None:
    """Append one row to 'Completed Log' before the Actions row is deleted."""
    date, slot, action, context, _status, notes = row[0], row[1], row[2], row[3], row[4], row[5]
    combined_notes = f"{notes} | [{_now_ist_str()}] {notes_suffix}".strip(" |")
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=COMPLETED_LOG_RANGE,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[
            task_id, date, slot, action, context, final_status,
            combined_notes, _now_ist_str(),
        ]]},
    ).execute()


# =============================================================================
# next_action_add
# =============================================================================

ADD_SCHEMA = {
    "name": "next_action_add",
    "description": (
        "Add a new row to the Next Agent Action sheet — schedules a "
        "future-dated action the cron agent will pick up and execute "
        "automatically in the matching date+slot. Use when NDR says 'set a "
        "reminder for X' or a task needs a future-dated follow-up. Returns "
        "the new row's task_id — remember it if you may need to update, "
        "cancel, or ask a clarifying question about this task later in the "
        "same session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "YYYY-MM-DD — the date this action is due.",
            },
            "slot": {
                "type": "string",
                "enum": list(VALID_SLOTS),
                "description": "Time block: Morning (~8am IST), Afternoon (~1pm IST), or Evening (~6pm IST).",
            },
            "action": {
                "type": "string",
                "description": "Brief description of the action, e.g. 'Follow up on IndusInd card dispatch'.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Full self-contained instructions for the cron agent, "
                    "which has NO memory of this conversation: what to "
                    "check, what to do, Drive doc links, email context, "
                    "counterparty contacts, what to do if an update is "
                    "found vs not."
                ),
            },
            "session_key": {
                "type": "string",
                "description": (
                    "Optional. The originating session's key, format "
                    "'<platform>:<chat_id>' (e.g. 'telegram:123456789'), if "
                    "known — lets the cron agent fall back to "
                    "session_search if context alone isn't enough."
                ),
            },
            "context_keywords": {
                "type": "string",
                "description": (
                    "Optional. A few keywords describing the topic (e.g. "
                    "'Nvidia earnings follow-up'), used as a session_search "
                    "query by the cron agent when it needs more background "
                    "than the context field provides."
                ),
            },
            "memory_refs": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated PENDING: short titles (from "
                    "the memory tool / pending-actions-tracker skill) this "
                    "task ties into, if any."
                ),
            },
        },
        "required": ["date", "slot", "action", "context"],
    },
}


def _handle_add(args: Optional[dict], **kw) -> str:
    args = args or {}
    date = str(args.get("date", "")).strip()
    slot = str(args.get("slot", "")).strip()
    action = str(args.get("action", "")).strip()
    context = str(args.get("context", "")).strip()
    session_key = str(args.get("session_key", "") or "").strip()
    context_keywords = str(args.get("context_keywords", "") or "").strip()
    memory_refs = str(args.get("memory_refs", "") or "").strip()

    if not date:
        return tool_error("date is required (YYYY-MM-DD)")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return tool_error(f"date {date!r} is not in YYYY-MM-DD format")
    if slot not in VALID_SLOTS:
        return tool_error(f"slot must be one of {VALID_SLOTS}, got {slot!r}")
    if not action:
        return tool_error("action is required")
    if not context:
        return tool_error("context is required (must be self-contained — the cron agent has no memory of this conversation)")

    try:
        service = _sheets_service()
        task_id = _new_task_id(service)
        row = [
            date, slot, action, context, "Pending",
            f"[{_now_ist_str()}] Created",
            task_id, session_key, context_keywords, memory_refs,
        ]
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=ACTIONS_RANGE,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
    except Exception as exc:
        logger.exception("next_action_add failed")
        return tool_error(f"failed to add row: {exc}")

    return tool_result(
        success=True,
        task_id=task_id,
        date=date, slot=slot, action=action, status="Pending",
    )


# =============================================================================
# next_action_update
# =============================================================================

UPDATE_SCHEMA = {
    "name": "next_action_update",
    "description": (
        "Update fields on an existing Next Agent Action row, found by "
        "task_id (never by date/action text — rows can shift). Only the "
        "fields you pass are changed. Use append_note (not the raw Notes "
        "column) to add a timestamped log line without erasing prior notes. "
        "Do NOT use this to mark a task Done or Cancelled — use "
        "next_action_complete / next_action_cancel instead, which also "
        "write the audit log row."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task_id returned by next_action_add."},
            "date": {"type": "string", "description": "New date, YYYY-MM-DD, if rescheduling."},
            "slot": {"type": "string", "enum": list(VALID_SLOTS)},
            "action": {"type": "string", "description": "New brief description."},
            "context": {"type": "string", "description": "New full context (overwrites, not appends)."},
            "session_key": {"type": "string"},
            "context_keywords": {"type": "string"},
            "memory_refs": {"type": "string"},
            "append_note": {
                "type": "string",
                "description": "Text to append to Notes as a new timestamped line, preserving history.",
            },
        },
        "required": ["task_id"],
    },
}


def _handle_update(args: Optional[dict], **kw) -> str:
    args = args or {}
    task_id = str(args.get("task_id", "")).strip()
    if not task_id:
        return tool_error("task_id is required")

    if "slot" in args and args["slot"] and args["slot"] not in VALID_SLOTS:
        return tool_error(f"slot must be one of {VALID_SLOTS}, got {args['slot']!r}")
    if "date" in args and args["date"]:
        try:
            datetime.strptime(str(args["date"]), "%Y-%m-%d")
        except ValueError:
            return tool_error(f"date {args['date']!r} is not in YYYY-MM-DD format")

    try:
        service = _sheets_service()
        row_num, row = _find_action_row(service, task_id)
        if row_num is None:
            return tool_error(f"no Actions row found with task_id {task_id!r} (already completed/cancelled?)")

        field_to_col = {
            "date": 0, "slot": 1, "action": 2, "context": 3,
            "session_key": 7, "context_keywords": 8, "memory_refs": 9,
        }
        for field, col in field_to_col.items():
            if field in args and args[field] is not None and str(args[field]).strip() != "":
                row[col] = str(args[field]).strip()

        append_note = args.get("append_note")
        if append_note:
            existing_notes = row[5]
            row[5] = f"{existing_notes} | [{_now_ist_str()}] {append_note}".strip(" |")

        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{ACTIONS_TAB}!A{row_num}:J{row_num}",
            valueInputOption="RAW",
            body={"values": [row]},
        ).execute()
    except Exception as exc:
        logger.exception("next_action_update failed")
        return tool_error(f"failed to update row: {exc}")

    return tool_result(success=True, task_id=task_id, updated_fields=[k for k in field_to_col if k in args] + (["notes"] if append_note else []))


# =============================================================================
# next_action_complete
# =============================================================================

COMPLETE_SCHEMA = {
    "name": "next_action_complete",
    "description": (
        "Mark a Next Agent Action task fully DONE. Appends the row to the "
        "'Completed Log' tab (audit trail) with your summary, then removes "
        "it from the active Actions sheet entirely. Use this only when the "
        "task genuinely needs no further action — for a task that spawns a "
        "follow-up, call next_action_add for the follow-up FIRST, then "
        "complete this one."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task_id to complete."},
            "summary": {"type": "string", "description": "One-line summary of the outcome, recorded in the audit log."},
        },
        "required": ["task_id", "summary"],
    },
}


def _handle_complete(args: Optional[dict], **kw) -> str:
    args = args or {}
    task_id = str(args.get("task_id", "")).strip()
    summary = str(args.get("summary", "")).strip()
    if not task_id:
        return tool_error("task_id is required")
    if not summary:
        return tool_error("summary is required (recorded in the Completed Log audit trail)")

    try:
        service = _sheets_service()
        row_num, row = _find_action_row(service, task_id)
        if row_num is None:
            return tool_error(f"no Actions row found with task_id {task_id!r} (already completed/cancelled?)")

        _log_completion(service, task_id, row, "Done", f"Completed: {summary}")
        sheet_id = _get_sheet_id(service, ACTIONS_TAB)
        _delete_row(service, sheet_id, row_num)
    except Exception as exc:
        logger.exception("next_action_complete failed")
        return tool_error(f"failed to complete task: {exc}")

    return tool_result(success=True, task_id=task_id, final_status="Done")


# =============================================================================
# next_action_cancel
# =============================================================================

CANCEL_SCHEMA = {
    "name": "next_action_cancel",
    "description": (
        "Cancel a Next Agent Action task — use when NDR says to drop/skip "
        "it, or the cron agent determines it's no longer relevant (e.g. "
        "already resolved via another channel). Appends the row to "
        "'Completed Log' with Final_Status=Cancelled (audit trail), then "
        "removes it from Actions entirely."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task_id to cancel."},
            "reason": {"type": "string", "description": "Why it's being cancelled, recorded in the audit log."},
        },
        "required": ["task_id", "reason"],
    },
}


def _handle_cancel(args: Optional[dict], **kw) -> str:
    args = args or {}
    task_id = str(args.get("task_id", "")).strip()
    reason = str(args.get("reason", "")).strip()
    if not task_id:
        return tool_error("task_id is required")
    if not reason:
        return tool_error("reason is required (recorded in the Completed Log audit trail)")

    try:
        service = _sheets_service()
        row_num, row = _find_action_row(service, task_id)
        if row_num is None:
            return tool_error(f"no Actions row found with task_id {task_id!r} (already completed/cancelled?)")

        _log_completion(service, task_id, row, "Cancelled", f"Cancelled: {reason}")
        sheet_id = _get_sheet_id(service, ACTIONS_TAB)
        _delete_row(service, sheet_id, row_num)
    except Exception as exc:
        logger.exception("next_action_cancel failed")
        return tool_error(f"failed to cancel task: {exc}")

    return tool_result(success=True, task_id=task_id, final_status="Cancelled")


# =============================================================================
# next_action_ask
# =============================================================================

ASK_SCHEMA = {
    "name": "next_action_ask",
    "description": (
        "Park an async clarifying question for NDR on a Next Agent Action "
        "task you cannot proceed on right now. Adds a row to the "
        "'Clarifications' tab (blank Answer — NDR fills it in by editing "
        "the sheet directly, whenever he gets to it, no tool call needed "
        "from him) and sets the Actions row to Awaiting_Clarification so "
        "the cron agent skips re-running it until answered. Call "
        "next_action_check_answers on a later run to pick up the answer "
        "and resume."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task_id this question is about."},
            "question": {"type": "string", "description": "The question to ask, self-contained and specific."},
        },
        "required": ["task_id", "question"],
    },
}


def _handle_ask(args: Optional[dict], **kw) -> str:
    args = args or {}
    task_id = str(args.get("task_id", "")).strip()
    question = str(args.get("question", "")).strip()
    if not task_id:
        return tool_error("task_id is required")
    if not question:
        return tool_error("question is required")

    try:
        service = _sheets_service()
        row_num, row = _find_action_row(service, task_id)
        if row_num is None:
            return tool_error(f"no Actions row found with task_id {task_id!r}")

        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=CLARIFICATIONS_RANGE,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [[task_id, question, "", _now_ist_str(), ""]]},
        ).execute()

        row[4] = "Awaiting_Clarification"
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{ACTIONS_TAB}!A{row_num}:J{row_num}",
            valueInputOption="RAW",
            body={"values": [row]},
        ).execute()
    except Exception as exc:
        logger.exception("next_action_ask failed")
        return tool_error(f"failed to record clarification request: {exc}")

    return tool_result(success=True, task_id=task_id, status="Awaiting_Clarification")


# =============================================================================
# next_action_check_answers
# =============================================================================

CHECK_ANSWERS_SCHEMA = {
    "name": "next_action_check_answers",
    "description": (
        "Check the 'Clarifications' tab for answered questions on a task. "
        "Call this FIRST for any task whose Status is Awaiting_Clarification "
        "before deciding what to do next. Returns any answered Q&A pairs "
        "and deletes those rows once returned (per design — answered "
        "clarifications are not kept). If no unanswered questions remain "
        "for this task afterward, its Actions row status flips back to "
        "Pending automatically. If nothing has been answered yet, "
        "still_awaiting is true and you should not re-run the task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task_id to check."},
        },
        "required": ["task_id"],
    },
}


def _handle_check_answers(args: Optional[dict], **kw) -> str:
    args = args or {}
    task_id = str(args.get("task_id", "")).strip()
    if not task_id:
        return tool_error("task_id is required")

    try:
        service = _sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, range=CLARIFICATIONS_RANGE,
        ).execute()
        rows = resp.get("values", [])

        answered_indices = []   # 1-based sheet row numbers to delete
        unanswered_remaining = False
        answers = []
        for i, raw in enumerate(rows):
            if i == 0:
                continue  # header
            row = _pad(raw, CLARIFICATIONS_NUM_COLS)
            if row[0] != task_id:
                continue
            question, answer = row[1], row[2]
            if answer.strip():
                answers.append({"question": question, "answer": answer.strip()})
                answered_indices.append(i + 1)
            else:
                unanswered_remaining = True

        if answered_indices:
            sheet_id = _get_sheet_id(service, CLARIFICATIONS_TAB)
            # Delete bottom-up so earlier indices stay valid.
            requests = [
                {"deleteDimension": {"range": {
                    "sheetId": sheet_id, "dimension": "ROWS",
                    "startIndex": rn - 1, "endIndex": rn,
                }}}
                for rn in sorted(answered_indices, reverse=True)
            ]
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID, body={"requests": requests},
            ).execute()

        still_awaiting = unanswered_remaining
        if answers and not still_awaiting:
            # All clarifications for this task now resolved — resume it.
            row_num, row = _find_action_row(service, task_id)
            if row_num is not None and row[4] == "Awaiting_Clarification":
                row[4] = "Pending"
                service.spreadsheets().values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{ACTIONS_TAB}!A{row_num}:J{row_num}",
                    valueInputOption="RAW",
                    body={"values": [row]},
                ).execute()
    except Exception as exc:
        logger.exception("next_action_check_answers failed")
        return tool_error(f"failed to check clarifications: {exc}")

    return tool_result(
        success=True, task_id=task_id, answers=answers,
        still_awaiting=still_awaiting,
    )


# =============================================================================
# registration
# =============================================================================

registry.register(name="next_action_add", toolset=TOOLSET, schema=ADD_SCHEMA,
                   handler=_handle_add, emoji=EMOJI)
registry.register(name="next_action_update", toolset=TOOLSET, schema=UPDATE_SCHEMA,
                   handler=_handle_update, emoji=EMOJI)
registry.register(name="next_action_complete", toolset=TOOLSET, schema=COMPLETE_SCHEMA,
                   handler=_handle_complete, emoji=EMOJI)
registry.register(name="next_action_cancel", toolset=TOOLSET, schema=CANCEL_SCHEMA,
                   handler=_handle_cancel, emoji=EMOJI)
registry.register(name="next_action_ask", toolset=TOOLSET, schema=ASK_SCHEMA,
                   handler=_handle_ask, emoji="❓")
registry.register(name="next_action_check_answers", toolset=TOOLSET, schema=CHECK_ANSWERS_SCHEMA,
                   handler=_handle_check_answers, emoji=EMOJI)
