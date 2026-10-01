"""Workflow store + run JSON-RPC — the desktop canvas's durable half.

Documents live under ``HERMES_HOME/workflows``. Everything that moves a run (Play, pause, an
answer) is published as a workflow event; the reactor (``workflow/reactor.py``) — the
desktop-spawned backend or the messaging gateway, whichever holds the lease — acts on it. A run's history is its Relay trace:
``workflow.run.events`` returns the recorded events, and every newly recorded event of a run in
this process is broadcast as ``workflow.run`` for the canvas to fold live.

Handlers are rebound onto server.py at install time (see method_ctx.py) so they can call
``_ok`` / ``_err`` / ``_broadcast_global_event``.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("workflow.store.list")
def _(rid, params: dict) -> dict:
    from workflow.store import list_runs, load_documents
    from workflow.triggers import hook_info, secret_for

    payload = load_documents()
    webhooks = {}
    for doc in payload["docs"]:
        if secret_for(doc["id"]):
            webhooks[doc["id"]] = hook_info(doc["id"])
    runs: dict[str, int] = {}
    for run in list_runs():
        wid = run.get("workflowId")
        if isinstance(wid, str) and wid:
            runs[wid] = runs.get(wid, 0) + 1
    return _ok(rid, {**payload, "webhooks": webhooks, "runs": runs})


@method("workflow.store.put")
def _(rid, params: dict) -> dict:
    from workflow.store import save_documents
    from workflow.triggers import sync_triggers

    docs = params.get("docs")
    if not isinstance(docs, list):
        return _err(rid, 4001, "docs must be an array")
    current = params.get("currentId")
    saved = save_documents(docs, None if current is None else str(current))
    try:
        saved["triggers"] = sync_triggers(saved["docs"])
    except Exception as exc:
        saved["triggers"] = {"error": str(exc)}
    return _ok(rid, saved)


@method("workflow.store.remove")
def _(rid, params: dict) -> dict:
    from workflow.store import remove_document
    from workflow.triggers import sync_triggers

    workflow_id = str(params.get("id") or "").strip()
    if not workflow_id:
        return _err(rid, 4001, "id is required")
    saved = remove_document(workflow_id)
    try:
        saved["triggers"] = sync_triggers(saved["docs"])
    except Exception as exc:
        saved["triggers"] = {"error": str(exc)}
    return _ok(rid, saved)


@method("workflow.run.start")
def _(rid, params: dict) -> dict:
    from workflow import events
    from workflow.store import active_run, get_document, new_run_id

    workflow_id = str(params.get("workflowId") or params.get("id") or "").strip()
    if not workflow_id:
        return _err(rid, 4001, "workflowId is required")
    scenario = params.get("scenario")
    if scenario is not None and not isinstance(scenario, dict):
        return _err(rid, 4001, "scenario must be an object")
    if scenario is None and get_document(workflow_id) is None:
        return _err(rid, 4004, f"No workflow called '{workflow_id}'.")
    source = str(params.get("source") or "manual")
    existing = active_run(workflow_id) if source == "manual" else None
    if existing is not None:
        return _ok(rid, {"runId": existing["runId"], "status": existing.get("status")})
    run_id = new_run_id()
    events.publish(events.START, {
        "workflowId": workflow_id, "runId": run_id, "scenario": scenario, "payload": params.get("payload"),
        "source": source, "fake": bool(params.get("fake")),
    }, source="desktop")
    return _ok(rid, {"runId": run_id, "status": "queued"})


@method("workflow.run.events")
def _(rid, params: dict) -> dict:
    from workflow.store import load_run
    from workflow.trace import run_reply

    run_id = str(params.get("runId") or "").strip()
    if not run_id:
        return _err(rid, 4001, "runId is required")
    return _ok(rid, run_reply(load_run(run_id), run_id))


@method("workflow.run.active")
def _(rid, params: dict) -> dict:
    from workflow.store import active_run
    from workflow.trace import run_reply

    workflow_id = str(params.get("workflowId") or params.get("id") or "").strip()
    if not workflow_id:
        return _err(rid, 4001, "workflowId is required")
    state = active_run(workflow_id)
    if state is None:
        return _ok(rid, {"run": None, "events": []})
    return _ok(rid, run_reply(state, state["runId"]))


@method("workflow.run.respond")
def _(rid, params: dict) -> dict:
    from workflow import events
    from workflow.store import load_run

    run_id = str(params.get("runId") or "").strip()
    node_id = str(params.get("nodeId") or "").strip()
    decision = str(params.get("decision") or "").strip()
    if not run_id or not node_id or decision not in {"approved", "denied"}:
        return _err(rid, 4001, "runId, nodeId, and decision ('approved'|'denied') are required")
    state = load_run(run_id)
    park = ((state or {}).get("parks") or {}).get(node_id) or {}
    if park.get("kind") != "human":
        return _err(rid, 4004, "This run is not waiting on that person.")
    events.publish(events.ANSWER, {"runId": run_id, "nodeId": node_id, "decision": decision, "by": params.get("by")},
                   source="desktop")
    return _ok(rid, {"runId": run_id, "status": state.get("status")})


@method("workflow.run.event")
def _(rid, params: dict) -> dict:
    from workflow import events

    name = str(params.get("name") or params.get("event") or "").strip()
    if not name:
        return _err(rid, 4001, "name is required")
    queued = events.publish(name, params.get("payload"), source="desktop")
    return _ok(rid, {"id": queued["id"]})


@method("workflow.run.pause")
def _(rid, params: dict) -> dict:
    from workflow import events

    run_id = str(params.get("runId") or "").strip()
    state = events.publish_for_run(events.PAUSE, run_id) if run_id else None
    if state is None:
        return _err(rid, 4001 if not run_id else 4004, "runId is required" if not run_id else f"No run '{run_id}'.")
    return _ok(rid, {"runId": run_id, "status": state.get("status")})


@method("workflow.run.resume")
def _(rid, params: dict) -> dict:
    from workflow import events

    run_id = str(params.get("runId") or "").strip()
    state = events.publish_for_run(events.RESUME, run_id) if run_id else None
    if state is None:
        return _err(rid, 4001 if not run_id else 4004, "runId is required" if not run_id else f"No run '{run_id}'.")
    return _ok(rid, {"runId": run_id, "status": state.get("status")})


@method("workflow.run.cancel")
def _(rid, params: dict) -> dict:
    from workflow import events

    run_id = str(params.get("runId") or "").strip()
    state = events.publish_for_run(events.CANCEL, run_id) if run_id else None
    if state is None:
        return _err(rid, 4001 if not run_id else 4004, "runId is required" if not run_id else f"No run '{run_id}'.")
    return _ok(rid, {"runId": run_id, "status": state.get("status")})


def register(server) -> None:
    _registry.install(server)
    from hermes_cli.observability import relay_traces

    def forward(profile_key: str, root_session_id: str, session_id: str, event: dict) -> None:
        """Relay publication thread → the canvas, for every event recorded under a workflow run."""
        del profile_key, session_id
        if root_session_id.startswith("wf-run-"):
            server._broadcast_global_event("workflow.run", {"runId": root_session_id[3:], "event": event})

    previous = getattr(server, "_workflow_listener_unsubscribe", None)
    if callable(previous):
        previous()
    server._workflow_listener_unsubscribe = relay_traces.add_listener(forward)
