"""save_workflow + create_routine tools — the canonical pivot away from
the per-rule ``save_workflow_rule`` store (deprecated, see
hermes-agent-202606-028).

NORTH_STAR primitives are three: **connector**, **workflow**, **routine**.
There is no standalone "rule" any more. Operators authoring automation
from chat compose one ``save_workflow`` call (the *what*) plus optionally
one ``create_routine`` call (the *when*).

Both tools persist YAML files under HERMES_HOME (canonical). Companion
mirrors for UI surfaces are applied by the adapter layer (best effort).

create_routine additionally schedules a real Hermes cron job via
``cron.jobs.create_job`` with ``workflow_ids=[workflow_id]``. When the
cron fires, the scheduler may delegate to workflow dispatch.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error
from tools.routine_storage import save_routine
from tools.workflow_storage import save_workflow as _persist_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# save_workflow
# ---------------------------------------------------------------------------


def save_workflow_handler(
    id: str,
    name_he: str,
    description_he: str = "",
    trigger_kind: str = "manual",
    triage_keywords_he: Optional[List[str]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Persist a workflow YAML to HERMES_HOME/workflows/<id>.yaml."""
    record: Dict[str, Any] = {
        "id": id,
        "name_he": name_he,
        "description_he": description_he,
        "trigger_kind": trigger_kind,
        "triage_keywords_he": list(triage_keywords_he or []),
        "actions": list(actions or []),
    }
    try:
        saved = _persist_workflow(record)
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    except OSError as exc:
        logger.exception("save_workflow failed for id=%s", id)
        return tool_error(f"failed to persist workflow: {exc}", success=False)
    return json.dumps({"success": True, "workflow": saved}, ensure_ascii=False)


SAVE_WORKFLOW_SCHEMA = {
    "name": "save_workflow",
    "description": (
        "Persist a workflow YAML under <HERMES_HOME>/workflows/<id>.yaml . "
        "A workflow declares what should happen when its "
        "trigger fires: an optional triage step that matches incoming "
        "events against Hebrew keywords, followed by an ordered list of "
        "actions (e.g. create_ticket). Pair with create_routine to wire a "
        "cron schedule. Replaces the deprecated save_workflow_rule tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "Stable id for the workflow (filesystem-safe: "
                    "[A-Za-z0-9_-]{1,128})."
                ),
            },
            "name_he": {
                "type": "string",
                "description": "Hebrew display name shown in the desktop UI.",
            },
            "description_he": {
                "type": "string",
                "description": "Optional Hebrew long-form description.",
            },
            "trigger_kind": {
                "type": "string",
                "enum": [
                    "schedule_triggered",
                    "wa_incoming_message",
                    "manual",
                ],
                "description": (
                    "What kicks the workflow off. schedule_triggered = "
                    "fired by a routine, wa_incoming_message = WhatsApp "
                    "intake, manual = operator-launched."
                ),
            },
            "triage_keywords_he": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional Hebrew keywords; when set, the workflow only "
                    "runs if the inbound event text contains one of them."
                ),
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": (
                                "Action kind, e.g. 'create_ticket', "
                                "'send_message'."
                            ),
                        },
                    },
                    "required": ["kind"],
                    "additionalProperties": True,
                },
                "description": "Ordered list of downstream actions.",
            },
        },
        "required": ["id", "name_he", "trigger_kind"],
    },
}


# ---------------------------------------------------------------------------
# create_routine
# ---------------------------------------------------------------------------


def _create_cron_job_for_routine(
    routine_id: str,
    workflow_id: str,
    name_he: str,
    cron_schedule: str,
) -> Optional[str]:
    """Best-effort: register a Hermes cron job that emits schedule_triggered.

    Returns the cron job id on success, ``None`` if the cron module isn't
    available (test environments, sidecar packaging variants). The routine
    record is still persisted on the filesystem so the desktop shell can
    render it; any cron-side wiring can be reconciled later by the
    scheduler boot.
    """
    try:
        from cron.jobs import create_job
    except Exception as exc:  # noqa: BLE001 — defensive at boundary
        logger.warning(
            "create_routine: cron.jobs unavailable, skipping cron wiring (%s)",
            exc,
        )
        return None

    prompt = (
        f"schedule_triggered routine={routine_id} workflow={workflow_id}\n"
        f"Fire workflow '{workflow_id}' on the cron tick. The desktop "
        f"bridge consumes this event and executes the workflow's actions."
    )
    try:
        job = create_job(
            prompt=prompt,
            schedule=cron_schedule,
            name=f"routine:{name_he}",
            workflow_ids=[workflow_id],
        )
    except ValueError as exc:
        raise ValueError(f"invalid cron schedule '{cron_schedule}': {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_routine: create_job failed: %s", exc)
        return None
    job_id = job.get("id") if isinstance(job, dict) else None
    return job_id if isinstance(job_id, str) else None


def create_routine_handler(
    id: str,
    workflow_id: str,
    name_he: str,
    cron_schedule: str,
    natural_language_schedule_he: str = "",
) -> str:
    """Schedule a cron trigger that fires the given workflow_id."""
    try:
        cron_job_id = _create_cron_job_for_routine(
            routine_id=id,
            workflow_id=workflow_id,
            name_he=name_he,
            cron_schedule=cron_schedule,
        )
    except ValueError as exc:
        return tool_error(str(exc), success=False)

    record: Dict[str, Any] = {
        "id": id,
        "workflow_id": workflow_id,
        "name_he": name_he,
        "cron_schedule": cron_schedule,
        "natural_language_schedule_he": natural_language_schedule_he,
        "cron_job_id": cron_job_id,
    }
    try:
        saved = save_routine(record)
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    except OSError as exc:
        logger.exception("create_routine failed for id=%s", id)
        return tool_error(f"failed to persist routine: {exc}", success=False)
    return json.dumps({"success": True, "routine": saved}, ensure_ascii=False)


CREATE_ROUTINE_SCHEMA = {
    "name": "create_routine",
    "description": (
        "Schedule a cron trigger that fires a workflow. Pairs with "
        "save_workflow: save_workflow declares the *what*, create_routine "
        "declares the *when*. On every cron tick, Hermes raises a "
        "schedule_triggered event that the bound workflow_id consumes. "
        "Persists a YAML under <HERMES_HOME>/routines/<id>.yaml AND "
        "registers a real Hermes cron job. Companion mirrors applied by adapter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "Stable id for the routine "
                    "([A-Za-z0-9_-]{1,128})."
                ),
            },
            "workflow_id": {
                "type": "string",
                "description": "The workflow this routine fires.",
            },
            "name_he": {
                "type": "string",
                "description": "Hebrew display name for the routine.",
            },
            "cron_schedule": {
                "type": "string",
                "description": (
                    "Cron expression. Accepts the same shapes as the "
                    "cronjob tool: '0 * * * *' (hourly), 'every 30m', "
                    "'every 2h', 'every 1d', or a one-shot ISO timestamp."
                ),
            },
            "natural_language_schedule_he": {
                "type": "string",
                "description": (
                    "Operator's own Hebrew description of the schedule "
                    "(e.g. 'בכל בוקר ב-9:00') — rendered in the desktop UI."
                ),
            },
        },
        "required": ["id", "workflow_id", "name_he", "cron_schedule"],
    },
}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


registry.register(
    name="save_workflow",
    toolset="workflows",
    schema=SAVE_WORKFLOW_SCHEMA,
    handler=lambda args, **_kw: save_workflow_handler(
        id=args.get("id", ""),
        name_he=args.get("name_he", ""),
        description_he=args.get("description_he", ""),
        trigger_kind=args.get("trigger_kind", "manual"),
        triage_keywords_he=args.get("triage_keywords_he"),
        actions=args.get("actions"),
    ),
    emoji="🧩",
)


registry.register(
    name="create_routine",
    toolset="workflows",
    schema=CREATE_ROUTINE_SCHEMA,
    handler=lambda args, **_kw: create_routine_handler(
        id=args.get("id", ""),
        workflow_id=args.get("workflow_id", ""),
        name_he=args.get("name_he", ""),
        cron_schedule=args.get("cron_schedule", ""),
        natural_language_schedule_he=args.get("natural_language_schedule_he", ""),
    ),
    emoji="⏰",
)
