"""Workflow store + run JSON-RPC — the desktop canvas's durable half.

Documents live under ``HERMES_HOME/workflows``. Play starts a gateway run
that emits the same events the canvas already folds. Handlers are rebound
onto server.py at install time (see method_ctx.py) so they can call
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
    from workflow.runner import start_run

    workflow_id = str(params.get("workflowId") or params.get("id") or "").strip()
    if not workflow_id:
        return _err(rid, 4001, "workflowId is required")
    scenario = params.get("scenario")
    if scenario is not None and not isinstance(scenario, dict):
        return _err(rid, 4001, "scenario must be an object")
    try:
        state = start_run(
            workflow_id,
            scenario=scenario,
            payload=params.get("payload"),
            source=str(params.get("source") or "manual"),
            fake=bool(params.get("fake")),
        )
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    except Exception as exc:
        return _err(rid, 5001, str(exc))
    return _ok(rid, {"runId": state["runId"], "status": state.get("status")})


@method("workflow.run.events")
def _(rid, params: dict) -> dict:
    from workflow.store import load_events, load_run

    run_id = str(params.get("runId") or "").strip()
    if not run_id:
        return _err(rid, 4001, "runId is required")
    state = load_run(run_id)
    if state is None:
        return _err(rid, 4004, f"No run '{run_id}'.")
    after = params.get("after", -1)
    try:
        after_n = int(after)
    except (TypeError, ValueError):
        after_n = -1
    return _ok(rid, {"run": state, "events": load_events(run_id, after_n)})


@method("workflow.run.active")
def _(rid, params: dict) -> dict:
    from workflow.runner import snapshot_active

    workflow_id = str(params.get("workflowId") or params.get("id") or "").strip()
    if not workflow_id:
        return _err(rid, 4001, "workflowId is required")
    snap = snapshot_active(workflow_id)
    if snap is None:
        return _ok(rid, {"run": None, "events": []})
    return _ok(rid, {"run": snap["run"], "events": snap["events"], "runId": snap["run"]["runId"]})


@method("workflow.run.respond")
def _(rid, params: dict) -> dict:
    from workflow.runner import respond

    run_id = str(params.get("runId") or "").strip()
    node_id = str(params.get("nodeId") or "").strip()
    decision = str(params.get("decision") or "").strip()
    if not run_id or not node_id or decision not in {"approved", "denied"}:
        return _err(rid, 4001, "runId, nodeId, and decision ('approved'|'denied') are required")
    try:
        state = respond(run_id, node_id, decision, by=params.get("by"))
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    return _ok(rid, {"runId": state["runId"], "status": state.get("status")})


@method("workflow.run.event")
def _(rid, params: dict) -> dict:
    from workflow.runner import start_matching

    name = str(params.get("name") or params.get("event") or "").strip()
    if not name:
        return _err(rid, 4001, "name is required")
    started = start_matching(event=name, payload=params.get("payload"), source="event")
    return _ok(rid, {"started": [s["runId"] for s in started]})


@method("workflow.run.pause")
def _(rid, params: dict) -> dict:
    from workflow.runner import request_pause

    run_id = str(params.get("runId") or "").strip()
    if not run_id:
        return _err(rid, 4001, "runId is required")
    try:
        state = request_pause(run_id)
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    return _ok(rid, {"runId": state["runId"], "status": state.get("status")})


@method("workflow.run.resume")
def _(rid, params: dict) -> dict:
    from workflow.runner import resume_run

    run_id = str(params.get("runId") or "").strip()
    if not run_id:
        return _err(rid, 4001, "runId is required")
    try:
        state = resume_run(run_id)
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    return _ok(rid, {"runId": state["runId"], "status": state.get("status")})


@method("workflow.run.cancel")
def _(rid, params: dict) -> dict:
    from workflow.runner import cancel_run

    run_id = str(params.get("runId") or "").strip()
    if not run_id:
        return _err(rid, 4001, "runId is required")
    try:
        state = cancel_run(run_id)
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    return _ok(rid, {"runId": state["runId"], "status": state.get("status")})


def register(server) -> None:
    _registry.install(server)
    from workflow.store import set_event_sink
    from workflow.runner import rearm_parked

    set_event_sink(lambda event: server._broadcast_global_event("workflow.run", event))
    try:
        rearm_parked()
    except Exception:
        pass
