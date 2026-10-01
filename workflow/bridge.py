"""Hermes's own happenings, as workflow events.

Every session, turn and tool call is already a Relay scope the trace recorder sees. This listener
names the ones a workflow can react to and publishes them — only when some workflow in that
profile listens for the name, so an idle install pays nothing:

    hermes.session.started / hermes.session.ended   {sessionId, parentSessionId}
    hermes.turn.finished                            {sessionId, turnId}
    hermes.tool.<name>                              {sessionId, tool, ok}

A session that belongs to a workflow run carries ``workflowId`` / ``workflowRunId``, which is what
keeps a workflow from re-triggering itself from its own steps.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agent import relay_runtime
from hermes_cli.observability import relay_traces

logger = logging.getLogger(__name__)

_RUN_SESSION = re.compile(r"^wf-(run-\d+-[0-9a-f]+)(?:-.+)?$")


def _run_owner(home: Path, session_id: str) -> dict[str, str]:
    match = _RUN_SESSION.match(session_id)
    if not match:
        return {}
    from workflow.store import workflows_dir

    path = workflows_dir(home) / "runs" / f"{match.group(1)}.json"
    try:
        import json

        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {"workflowRunId": match.group(1)}
    return {"workflowRunId": match.group(1), "workflowId": str(state.get("workflowId") or "")}


def _named(event: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if event.get("kind") != "scope":
        return None
    phase, name = event.get("scope_category"), str(event.get("name") or "")
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    if name == relay_runtime.SESSION_SCOPE and phase in {"start", "end"}:
        verb = "started" if phase == "start" else "ended"
        return f"hermes.session.{verb}", {"parentSessionId": str(metadata.get(relay_runtime.PARENT_SESSION_ID_KEY) or "")}
    if phase != "end":
        return None
    if name == relay_runtime.TURN_SCOPE:
        return "hermes.turn.finished", {"turnId": str(metadata.get(relay_runtime.TURN_ID_KEY) or "")}
    if event.get("category") == "tool":
        ok = str(metadata.get("otel.status_code") or "OK") != "ERROR"
        return f"hermes.tool.{name}", {"tool": name, "ok": ok}
    return None


def _on_event(profile_key: str, root_session_id: str, session_id: str, event: dict[str, Any]) -> None:
    del root_session_id
    named = _named(event)
    if named is None:
        return
    name, payload = named
    try:
        from workflow import events

        home = Path(profile_key)
        if not events.wanted(name, home=home):
            return
        events.publish(name, {"sessionId": session_id, **payload, **_run_owner(home, session_id)},
                       source="hermes", home=home)
    except Exception:
        logger.debug("workflow bridge failed for %s", name, exc_info=True)


relay_traces.add_listener(_on_event)
