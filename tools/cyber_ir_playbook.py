"""Incident Response Playbook Tool.

In-memory IR state machine that tracks incidents, timelines, and evidence
across an agent session. One IR store per AIAgent instance.

Actions:
  create        — open a new IR incident
  update        — change status or severity of an existing incident
  add_timeline  — append a timestamped event to the incident timeline
  add_evidence  — log a piece of evidence (IOC, log snippet, screenshot ref, etc.)
  status        — retrieve full incident state
  list          — list all incidents with summary line
  report        — generate a structured markdown IR summary

Incidents persist for the lifetime of the session (one agent run).
For cross-session persistence, pair with the memory tool.

Severity levels: P1 (critical) → P2 (high) → P3 (medium) → P4 (low)
Status values:   open → investigating → contained → eradicated → closed
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_VALID_SEVERITIES = {"P1", "P2", "P3", "P4"}
_VALID_STATUSES   = {"open", "investigating", "contained", "eradicated", "closed"}
_VALID_EVIDENCE   = {"ioc", "log", "screenshot", "artifact", "note", "pcap", "memory_dump", "registry", "other"}


class IRStore:
    """Per-session IR incident store. Injected via kw['store']."""

    def __init__(self) -> None:
        self._incidents: dict[str, dict] = {}
        self._counter   = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"INC-{self._counter:04d}"

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- CRUD ---

    def create(self, title: str, severity: str, description: str = "") -> dict:
        severity = severity.upper()
        if severity not in _VALID_SEVERITIES:
            return {"error": f"Invalid severity {severity!r}. Use P1/P2/P3/P4."}
        inc_id = self._new_id()
        self._incidents[inc_id] = {
            "id":          inc_id,
            "title":       title,
            "description": description,
            "severity":    severity,
            "status":      "open",
            "created":     self._ts(),
            "updated":     self._ts(),
            "timeline":    [],
            "evidence":    [],
            "assignees":   [],
            "tags":        [],
        }
        logger.info("IR incident created: %s — %s (%s)", inc_id, title, severity)
        return {"created": inc_id, "incident": self._incidents[inc_id]}

    def update(self, inc_id: str, status: str | None = None, severity: str | None = None,
               assignee: str | None = None, tags: list[str] | None = None) -> dict:
        inc = self._incidents.get(inc_id)
        if not inc:
            return {"error": f"Incident {inc_id} not found"}
        if status:
            status = status.lower()
            if status not in _VALID_STATUSES:
                return {"error": f"Invalid status {status!r}. Use: {', '.join(sorted(_VALID_STATUSES))}"}
            inc["status"] = status
        if severity:
            severity = severity.upper()
            if severity not in _VALID_SEVERITIES:
                return {"error": f"Invalid severity {severity!r}"}
            inc["severity"] = severity
        if assignee and assignee not in inc["assignees"]:
            inc["assignees"].append(assignee)
        if tags:
            inc["tags"] = list(dict.fromkeys(inc["tags"] + tags))
        inc["updated"] = self._ts()
        return {"updated": inc_id, "incident": inc}

    def add_timeline(self, inc_id: str, event: str, actor: str = "system") -> dict:
        inc = self._incidents.get(inc_id)
        if not inc:
            return {"error": f"Incident {inc_id} not found"}
        entry = {"ts": self._ts(), "actor": actor, "event": event}
        inc["timeline"].append(entry)
        inc["updated"] = self._ts()
        return {"added": entry, "timeline_length": len(inc["timeline"])}

    def add_evidence(self, inc_id: str, evidence_type: str, value: str,
                     source: str = "", notes: str = "") -> dict:
        inc = self._incidents.get(inc_id)
        if not inc:
            return {"error": f"Incident {inc_id} not found"}
        ev_type = evidence_type.lower()
        if ev_type not in _VALID_EVIDENCE:
            ev_type = "other"
        entry = {
            "ts":     self._ts(),
            "type":   ev_type,
            "value":  value,
            "source": source,
            "notes":  notes,
        }
        inc["evidence"].append(entry)
        inc["updated"] = self._ts()
        return {"added": entry, "evidence_count": len(inc["evidence"])}

    def get_status(self, inc_id: str) -> dict:
        inc = self._incidents.get(inc_id)
        if not inc:
            return {"error": f"Incident {inc_id} not found"}
        return inc

    def list_all(self) -> dict:
        if not self._incidents:
            return {"incidents": [], "total": 0}
        summaries = [
            {
                "id":       inc["id"],
                "title":    inc["title"],
                "severity": inc["severity"],
                "status":   inc["status"],
                "created":  inc["created"],
                "timeline_events": len(inc["timeline"]),
                "evidence_items":  len(inc["evidence"]),
            }
            for inc in self._incidents.values()
        ]
        return {"incidents": summaries, "total": len(summaries)}

    def report(self, inc_id: str) -> dict:
        inc = self._incidents.get(inc_id)
        if not inc:
            return {"error": f"Incident {inc_id} not found"}

        iocs = [e["value"] for e in inc["evidence"] if e["type"] == "ioc"]
        logs = [e for e in inc["evidence"] if e["type"] == "log"]

        lines = [
            f"# Incident Report: {inc['id']}",
            f"",
            f"**Title:** {inc['title']}",
            f"**Severity:** {inc['severity']}",
            f"**Status:** {inc['status']}",
            f"**Created:** {inc['created']}",
            f"**Last Updated:** {inc['updated']}",
        ]
        if inc["assignees"]:
            lines.append(f"**Assignees:** {', '.join(inc['assignees'])}")
        if inc["tags"]:
            lines.append(f"**Tags:** {', '.join(inc['tags'])}")
        if inc["description"]:
            lines += ["", f"## Description", f"", inc["description"]]

        if inc["timeline"]:
            lines += ["", "## Timeline", ""]
            for e in inc["timeline"]:
                lines.append(f"- `{e['ts']}` [{e['actor']}] {e['event']}")

        if iocs:
            lines += ["", "## Indicators of Compromise", ""]
            for ioc in iocs:
                lines.append(f"- `{ioc}`")

        if logs:
            lines += ["", "## Key Log Evidence", ""]
            for lg in logs[:5]:
                lines.append(f"- **Source:** {lg.get('source', 'unknown')}  ")
                lines.append(f"  `{lg['value'][:200]}`")

        other_evidence = [e for e in inc["evidence"] if e["type"] not in ("ioc", "log")]
        if other_evidence:
            lines += ["", "## Other Evidence", ""]
            for e in other_evidence:
                lines.append(f"- [{e['type']}] {e['value'][:120]}"
                              + (f" — {e['notes']}" if e["notes"] else ""))

        return {"report_markdown": "\n".join(lines), "incident_id": inc_id}


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def _handle(args: dict, **kw: Any) -> str:
    store: IRStore = kw.get("store")
    if store is None:
        return json.dumps({"error": "IR store not initialised (internal error)"})

    action = args.get("action", "")

    if action == "create":
        result = store.create(
            title=args.get("title", "Untitled Incident"),
            severity=args.get("severity", "P3"),
            description=args.get("description", ""),
        )
    elif action == "update":
        result = store.update(
            inc_id=args.get("incident_id", ""),
            status=args.get("status"),
            severity=args.get("severity"),
            assignee=args.get("assignee"),
            tags=args.get("tags"),
        )
    elif action == "add_timeline":
        result = store.add_timeline(
            inc_id=args.get("incident_id", ""),
            event=args.get("event", ""),
            actor=args.get("actor", "analyst"),
        )
    elif action == "add_evidence":
        result = store.add_evidence(
            inc_id=args.get("incident_id", ""),
            evidence_type=args.get("evidence_type", "other"),
            value=args.get("value", ""),
            source=args.get("source", ""),
            notes=args.get("notes", ""),
        )
    elif action == "status":
        result = store.get_status(args.get("incident_id", ""))
    elif action == "list":
        result = store.list_all()
    elif action == "report":
        result = store.report(args.get("incident_id", ""))
    else:
        result = {
            "error": f"Unknown action: {action!r}.",
            "valid_actions": ["create", "update", "add_timeline", "add_evidence", "status", "list", "report"],
        }

    return json.dumps(result, indent=2)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "ir_incident",
        "description": (
            "Incident Response playbook tool. Creates and manages IR incidents "
            "within the current session: timeline tracking, evidence logging, "
            "severity/status updates, and structured markdown report generation. "
            "Pair with extract_iocs and threat_intel for end-to-end IR workflows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "add_timeline", "add_evidence",
                             "status", "list", "report"],
                    "description": "IR action to perform.",
                },
                "incident_id": {
                    "type": "string",
                    "description": "Incident ID returned by create (e.g. INC-0001). Required for all actions except create/list.",
                },
                "title": {
                    "type": "string",
                    "description": "Short incident title (create only).",
                },
                "severity": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=critical, P2=high, P3=medium, P4=low.",
                },
                "description": {
                    "type": "string",
                    "description": "Narrative description of the incident (create only).",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "investigating", "contained", "eradicated", "closed"],
                    "description": "New status (update only).",
                },
                "assignee": {
                    "type": "string",
                    "description": "Analyst name or handle to assign (update only).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to add to the incident (update only).",
                },
                "event": {
                    "type": "string",
                    "description": "Timeline event description (add_timeline only).",
                },
                "actor": {
                    "type": "string",
                    "description": "Who performed the action: analyst handle, system name, etc. (add_timeline only).",
                },
                "evidence_type": {
                    "type": "string",
                    "enum": ["ioc", "log", "screenshot", "artifact", "note",
                             "pcap", "memory_dump", "registry", "other"],
                    "description": "Category of evidence (add_evidence only).",
                },
                "value": {
                    "type": "string",
                    "description": "Evidence value: IOC string, log line, file path, description, etc. (add_evidence only).",
                },
                "source": {
                    "type": "string",
                    "description": "Where the evidence came from: SIEM, EDR, firewall log, etc. (add_evidence only).",
                },
                "notes": {
                    "type": "string",
                    "description": "Analyst notes about this evidence item (add_evidence only).",
                },
            },
            "required": ["action"],
        },
    },
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402


def _make_handler(agent_instance: Any = None, **_kw: Any):
    """Factory that binds a fresh IRStore to the tool handler per agent instance."""
    # The registry calls the handler with kw['agent'] for store injection.
    # We create one IRStore per session by attaching it to the agent object.
    def bound_handler(args: dict, **kw: Any) -> str:
        agent = kw.get("agent") or agent_instance
        if agent is not None and not hasattr(agent, "_ir_store"):
            agent._ir_store = IRStore()
        store = getattr(agent, "_ir_store", None) or IRStore()
        return _handle(args, store=store, **kw)
    return bound_handler


registry.register(
    name="ir_incident",
    toolset="cyber",
    schema=SCHEMA,
    handler=_make_handler(),
    emoji="🚨",
)
