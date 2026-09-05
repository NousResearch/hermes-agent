"""Walk a scenario and emit the canvas event log.

Topology is real. Work is real when ``execute_fn`` calls a model; tests
inject a stub. Human and wait steps persist a park so closing the app
does not lose the run — resume via ``respond`` / ``resolve_event`` /
``tick_timers``.

This file is the loop and the doors into it. What it walks over, waits on and
stands in for lives beside it, so each can be read and tested without the loop:

    topology  reading the authored scenario — steps, wires, conditions
    runtime   the per-run plumbing every thread shares — lock, seq, signals
    waits     the steps that stop, and the clocks that start them again
    fake      the recording stand-in that calls no model
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from workflow import fake
from workflow.runtime import (
    absorb_signals,
    clear_signal,
    emit,
    fail_dead_run,
    lock_for,
    signal,
    spawn,
    thread_alive,
)
from workflow.store import (
    active_run,
    get_document,
    list_runs,
    load_documents,
    load_events,
    load_run,
    new_run_id,
    save_run,
    upsert_document,
)
from workflow.topology import (
    between,
    by_id,
    config_of,
    holds,
    kind_of,
    preds,
    scenario_of,
    steps_of,
    succs,
    title_of,
)
from workflow.waits import (
    finish_wait,
    park_human,
    park_wait,
    rearm,
    tick_polls,
    tick_timers,
)

ExecuteFn = Callable[[str, str, Any, dict], dict]

_execute_fn: ExecuteFn | None = None


def set_execute_fn(fn: ExecuteFn | None) -> None:
    global _execute_fn
    _execute_fn = fn


def _context_for(state: dict, node_id: str) -> str:
    parts = []
    for pred in preds(state["scenario"], node_id, loops=True):
        summary = (state.get("summaries") or {}).get(pred)
        output = (state.get("outputs") or {}).get(pred)
        if summary:
            parts.append(f"{pred}: {summary}")
        if output:
            parts.append(f"{pred} output: {output}")
    return "\n".join(parts)


def _fresh_state(workflow_id: str, scenario: dict, payload: Any, source: str, name: str) -> dict:
    return {
        "runId": new_run_id(),
        "workflowId": workflow_id,
        "name": name,
        "scenario": scenario,
        "payload": payload,
        "source": source,
        "status": "running",
        "queue": [],
        "ran": [],
        "satisfied": [],
        "verdicts": {},
        "outputs": {},
        "summaries": {},
        "take": {},
        "loops": 0,
        "park": None,
        "wakeAt": None,
        "waitingEvent": None,
        "pauseRequested": False,
        "seq": 0,
        "startedAt": int(time.time() * 1000),
        "failed": False,
        "tries": {},
        "inFlight": [],
        "sessions": {},
    }


def start_run(
    workflow_id: str,
    *,
    scenario: dict | None = None,
    payload: Any = None,
    source: str = "manual",
    execute_fn: ExecuteFn | None = None,
    background: bool = True,
    fake: bool = False,
) -> dict:
    doc = get_document(workflow_id)
    if doc is not None:
        workflow_id = doc["id"]
    if scenario is None:
        if doc is None:
            raise ValueError(f"No workflow called '{workflow_id}'.")
        scenario = scenario_of(doc)
    else:
        if doc is not None:
            upsert_document({**doc, "scenario": scenario})
        else:
            upsert_document({"id": workflow_id, "name": workflow_id, "scenario": scenario})
            doc = get_document(workflow_id)
    name = (doc or {}).get("name") or workflow_id
    existing = active_run(workflow_id)
    if existing is not None:
        if existing.get("status") == "running" and not thread_alive(existing["runId"]):
            fail_dead_run(existing)
        else:
            return existing

    state = _fresh_state(workflow_id, scenario, payload, source, name)
    if fake:
        state["fake"] = True
    steps = steps_of(scenario)
    entries = [s["id"] for s in steps if not preds(scenario, s["id"])]
    if not entries and steps:
        entries = [steps[0]["id"]]
    state["queue"] = list(entries)
    save_run(state)
    emit(state, "RunStarted", {"scenario": name})
    if background:
        spawn(state["runId"], execute_fn)
    else:
        advance(state["runId"], execute_fn=execute_fn)
    return load_run(state["runId"]) or state


def start_from_trigger(workflow_id: str, *, source: str = "cron", payload: Any = None) -> dict:
    return start_run(workflow_id, payload=payload, source=source)


def start_matching(
    *,
    event: str,
    payload: Any = None,
    source: str = "event",
    background: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> list[dict]:
    """Start every workflow whose trigger listens for this event, and resume parks."""
    started = []
    for run_id in resolve_event(event, payload, background=background, execute_fn=execute_fn):
        parked = load_run(run_id)
        if parked is not None:
            started.append(parked)
    needle = (event or "").strip().lower()
    if not needle:
        return started
    for doc in load_documents()["docs"]:
        scenario = scenario_of(doc)
        for step in steps_of(scenario):
            if kind_of(step) != "trigger":
                continue
            on = config_of(step).get("on") or {}
            if on.get("type") != "event":
                continue
            if str(on.get("spec") or "").strip().lower() != needle:
                continue
            if active_run(doc["id"]) is not None:
                continue
            started.append(
                start_run(
                    doc["id"],
                    payload=payload,
                    source=source,
                    background=background,
                    execute_fn=execute_fn,
                )
            )
            break
    return started


def advance(run_id: str, *, execute_fn: ExecuteFn | None = None) -> dict:
    with lock_for(run_id):
        return _advance(run_id, execute_fn)


def _advance(run_id: str, execute_fn: ExecuteFn | None) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") in {"succeeded", "failed", "cancelled"}:
        return state
    if state.get("status") == "paused":
        return state
    if state.get("park"):
        return state

    fn = execute_fn or _execute_fn
    scenario = state["scenario"]
    steps = by_id(scenario)

    leftover = [node_id for node_id in (state.get("inFlight") or []) if node_id not in state["ran"] and node_id not in state["queue"]]
    if leftover:
        state["queue"] = leftover + state["queue"]
        state["inFlight"] = []
        save_run(state)

    def park_pause() -> dict:
        state["status"] = "paused"
        state["inFlight"] = []
        save_run(state)
        emit(state, "RunPaused", {})
        return state

    while state["queue"] and state.get("status") == "running":
        absorb_signals(state)
        if state.get("pauseRequested"):
            return park_pause()

        ran = set(state["ran"])
        satisfied = set(state["satisfied"])
        ready = [
            node_id
            for node_id in state["queue"]
            if all(pred in ran or pred in satisfied or pred not in steps for pred in preds(scenario, node_id))
        ]
        if not ready:
            break

        state["queue"] = [node_id for node_id in state["queue"] if node_id not in ready]
        state["inFlight"] = list(ready)
        save_run(state)
        routed: list[str] = []
        halted = False

        ready_by_kind: dict[str, list[str]] = {}
        for node_id in ready:
            step = steps.get(node_id)
            if step:
                ready_by_kind.setdefault(kind_of(step), []).append(node_id)

        triggers = ready_by_kind.get("trigger", [])
        agents = ready_by_kind.get("agent", [])
        gates = ready_by_kind.get("gate", [])
        waits = ready_by_kind.get("wait", [])
        humans = ready_by_kind.get("human", [])

        for node_id in triggers:
            step = steps[node_id]
            iteration = int((state["take"].get(node_id) or 0))
            emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
            _run_trigger(state, step, iteration)
            routed.extend(succs(scenario, node_id))
            state["inFlight"] = [x for x in state["inFlight"] if x != node_id]

        if agents:
            extra, stop = _run_agents(state, [steps[n] for n in agents], fn)
            routed.extend(extra)
            halted = halted or stop
            state["inFlight"] = [x for x in state["inFlight"] if x not in agents]

        absorb_signals(state)
        if state.get("pauseRequested"):
            for nxt in routed:
                if nxt in steps and nxt not in state["queue"]:
                    state["queue"].append(nxt)
            return park_pause()

        for node_id in gates:
            step = steps[node_id]
            iteration = int((state["take"].get(node_id) or 0))
            emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
            extra, stop = _run_gate(state, step, iteration, fn)
            routed.extend(extra)
            if stop:
                halted = True
            state["inFlight"] = [x for x in state["inFlight"] if x != node_id]

        parked = False
        for node_id in waits + humans:
            step = steps[node_id]
            iteration = int((state["take"].get(node_id) or 0))
            emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
            if kind_of(step) == "wait":
                if park_wait(state, step, iteration):
                    parked = True
                else:
                    routed.extend(succs(scenario, node_id))
            else:
                park_human(state, step, iteration)
                parked = True
            state["inFlight"] = [x for x in state["inFlight"] if x != node_id]
            if parked:
                for rest in state["inFlight"]:
                    if rest not in state["queue"]:
                        state["queue"].append(rest)
                state["inFlight"] = []
                save_run(state)
                return state

        for nxt in routed:
            if nxt in steps and nxt not in state["queue"]:
                state["queue"].append(nxt)

        state["inFlight"] = []
        save_run(state)
        if halted:
            state["failed"] = True
            break

    if state.get("park") or state.get("status") == "paused":
        save_run(state)
        return state

    leftover = [node_id for node_id in state["queue"] if node_id not in state["ran"]]
    if leftover and not state.get("failed"):
        state["failed"] = True
        emit(
            state,
            "NodeFailed",
            {
                "nodeId": leftover[0],
                "iteration": int((state.get("take") or {}).get(leftover[0]) or 0),
                "error": f"never became ready (still waiting on {', '.join(leftover)})",
            },
        )

    state["status"] = "failed" if state.get("failed") else "succeeded"
    save_run(state)
    emit(state, "RunFinished", {"state": "failed" if state.get("failed") else "succeeded"})
    return load_run(run_id) or state


def _run_trigger(state: dict, step: dict, iteration: int) -> None:
    node_id = step["id"]
    on = config_of(step).get("on") or {"type": "manual", "spec": ""}
    label = f"{on.get('type') or 'manual'}"
    if on.get("spec"):
        label += f" · {on['spec']}"
    emit(
        state,
        "NodeStarted",
        {"nodeId": node_id, "iteration": iteration, "input": label, "maxIters": 0},
    )
    emit(state, "NodeFinished", {"nodeId": node_id, "iteration": iteration})
    state["ran"].append(node_id)
    state["take"][node_id] = iteration + 1
    state["verdicts"][node_id] = None


def _arm_matches(arm: dict, inputs: list[dict], state: dict, execute_fn: ExecuteFn | None) -> bool:
    when = arm.get("when") or {}
    if when.get("mode") != "prose":
        return holds(when, inputs)
    source = str(when.get("source") or "").strip() or "Should this arm be taken? Answer PASS or FAIL."
    context = "\n".join(
        f"{item['nodeId']}: {item.get('verdict') or '—'} · {(state.get('summaries') or {}).get(item['nodeId'], '')}"
        for item in inputs
    )
    if execute_fn is None:
        from workflow.agent import execute_agent_step

        result = execute_agent_step(source, context, state.get("payload"), {"maxIterations": 8})
    else:
        result = execute_fn(source, context, state.get("payload"), {"maxIterations": 8})
    if not result.get("ok", True):
        return False
    verdict = result.get("verdict")
    if verdict:
        return verdict == "PASS"
    text = str(result.get("summary") or "").upper()
    return "PASS" in text or text.startswith("YES")


def _run_gate(state: dict, step: dict, iteration: int, execute_fn: ExecuteFn | None = None) -> tuple[list[str], bool]:
    node_id = step["id"]
    scenario = state["scenario"]
    inputs = [{"nodeId": pred, "verdict": state["verdicts"].get(pred)} for pred in preds(scenario, node_id)]
    emit(
        state,
        "NodeStarted",
        {
            "nodeId": node_id,
            "iteration": iteration,
            "input": " · ".join(f"{i['nodeId']} {i['verdict'] or '—'}" for i in inputs) or "no inputs",
            "maxIters": 8,
        },
    )
    arms = config_of(step).get("arms") or []
    arm = next((a for a in arms if isinstance(a, dict) and _arm_matches(a, inputs, state, execute_fn)), None)
    route = None
    if arm is not None:
        targets = succs(scenario, node_id, arm.get("id"))
        route = targets[0] if targets else None
    culprit = next((i for i in inputs if i.get("verdict") == "FAIL"), None)
    decision = "fail" if culprit else "pass"
    title = title_of(by_id(scenario).get(route) or {"id": route or "", "title": "nowhere"})
    emit(
        state,
        "GateEvaluated",
        {
            "nodeId": node_id,
            "iteration": iteration,
            "inputs": inputs,
            "decision": decision,
            "route": route or "",
            "summary": f"{culprit['nodeId'] + ' FAIL' if culprit else 'group PASS'} → {title}",
        },
    )
    state["ran"].append(node_id)
    state["take"][node_id] = iteration + 1
    state["verdicts"][node_id] = "FAIL" if culprit else "PASS"

    if not route:
        emit(
            state,
            "NodeFailed",
            {
                "nodeId": node_id,
                "iteration": iteration,
                "error": (
                    f'"{arm.get("label") or arm.get("id")}" isn\'t wired anywhere'
                    if arm
                    else "no arm matched, so the work has nowhere to go"
                ),
            },
        )
        state["failed"] = True
        return [], True

    if route in state["ran"]:
        cap = int(config_of(step).get("maxLoops") or 5)
        if state["loops"] >= cap:
            emit(state, "NodeFailed", {"nodeId": node_id, "iteration": iteration, "error": f"gave up after {cap} takes"})
            state["failed"] = True
            return [], True
        state["loops"] += 1
        emit(
            state,
            "LoopAdvanced",
            {
                "loopId": node_id,
                "iteration": state["loops"],
                "to": route,
                "feedback": f"{culprit['nodeId']} feedback" if culprit else "another take",
            },
        )
        body = between(scenario, route, node_id)
        rerun = []
        for item in body:
            if item in state["ran"]:
                state["ran"] = [x for x in state["ran"] if x != item]
            if item not in {route, node_id} and state["verdicts"].get(item) == "PASS":
                if item not in state["satisfied"]:
                    state["satisfied"].append(item)
                emit(
                    state,
                    "NodeSkipped",
                    {
                        "nodeId": item,
                        "iteration": state["loops"],
                        "reason": f"satisfied · PASS on take {state['take'].get(item) or 1}",
                    },
                )
            else:
                rerun.append(item)
        return [route], False

    return [route], False


def _compute_agent(state: dict, step: dict, iteration: int, execute_fn: ExecuteFn | None) -> dict:
    node_id = step["id"]
    cfg = config_of(step)
    goal = str(cfg.get("goal") or "").strip() or title_of(step)
    context = "" if cfg.get("blind") else _context_for(state, node_id)
    traces: list[tuple[str, str]] = []

    def on_tool(name: str, arg: str = "") -> None:
        traces.append((name, arg))

    sessions = state.setdefault("sessions", {})
    existing = sessions.get(node_id)
    session_id = existing or f"wf-{state['runId']}-{node_id}"
    sessions[node_id] = session_id
    save_run(state)
    resume = bool(existing)
    if state.get("fake") and execute_fn is None:
        return fake.play(state, step, iteration)
    if execute_fn is None:
        from workflow.agent import execute_agent_step

        result = execute_agent_step(
            goal,
            context,
            state.get("payload"),
            cfg,
            on_tool=on_tool,
            session_id=session_id,
            resume=resume or bool(state.get("tries", {}).get(node_id)),
        )
    else:
        result = execute_fn(goal, context, state.get("payload"), cfg)
    if traces:
        result = {**result, "_traces": traces}
    return result


def _apply_agent(state: dict, step: dict, iteration: int, result: dict) -> str:
    node_id = step["id"]
    cfg = config_of(step)
    for name, arg in result.get("_traces") or []:
        emit(
            state,
            "AgentTraceEvent",
            {"nodeId": node_id, "iteration": iteration, "tool": {"name": name, "arg": arg}},
        )
    if result.get("_paused"):
        state["pauseRequested"] = True
        return "hold"
    if not result.get("ok", True):
        error = str(result.get("error") or "step failed")
        emit(state, "NodeFailed", {"nodeId": node_id, "iteration": iteration, "error": error})
        from workflow.agent import is_user_fixable

        if is_user_fixable(error):
            emit(state, "UserAsk", {"nodeId": node_id, "prompt": error})
            state["failed"] = True
            return "halt"
        retries = int(state.setdefault("tries", {}).get(node_id) or 0)
        allowed = int(cfg.get("maxRetries") or 0)
        if retries < allowed:
            state["tries"][node_id] = retries + 1
            return "retry"
        state["ran"].append(node_id)
        state["take"][node_id] = iteration + 1
        state["verdicts"][node_id] = "FAIL"
        on_fail = cfg.get("onFail") or "halt"
        if on_fail == "route":
            return "route"
        state["failed"] = True
        return "halt"

    summary = str(result.get("summary") or "done")
    verdict = result.get("verdict")
    output = result.get("output") if isinstance(result.get("output"), dict) else {"text": summary}
    emit(state, "AgentTraceSummary", {"nodeId": node_id, "iteration": iteration, "summary": summary, "verdict": verdict})
    emit(state, "TaskOutput", {"nodeId": node_id, "iteration": iteration, "output": output})
    emit(state, "NodeFinished", {"nodeId": node_id, "iteration": iteration})
    state["ran"].append(node_id)
    state["take"][node_id] = iteration + 1
    state["verdicts"][node_id] = verdict
    state["summaries"][node_id] = summary
    state["outputs"][node_id] = output
    return "ok"


def _run_agents(state: dict, steps: list[dict], execute_fn: ExecuteFn | None) -> tuple[list[str], bool]:
    routed: list[str] = []
    halted = False
    prepared = []
    for step in steps:
        iteration = int((state["take"].get(step["id"]) or 0))
        cfg = config_of(step)
        goal = str(cfg.get("goal") or "").strip() or title_of(step)
        emit(state, "NodePending", {"nodeId": step["id"], "iteration": iteration})
        emit(
            state,
            "NodeStarted",
            {
                "nodeId": step["id"],
                "iteration": iteration,
                "input": goal[:80],
                "maxIters": int(cfg.get("maxIterations") or 20),
                "loop": iteration > 0,
            },
        )
        prepared.append((step, iteration))

    results: dict[str, dict] = {}
    if len(prepared) == 1:
        step, iteration = prepared[0]
        results[step["id"]] = _compute_agent(state, step, iteration, execute_fn)
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(prepared))) as pool:
            futs = {
                pool.submit(_compute_agent, state, step, iteration, execute_fn): step["id"]
                for step, iteration in prepared
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()

    for step, iteration in prepared:
        stop = _apply_agent(state, step, iteration, results[step["id"]])
        if stop == "halt":
            halted = True
        elif stop == "retry":
            state["queue"].append(step["id"])
        elif stop == "hold":
            if step["id"] not in state["queue"]:
                state["queue"].append(step["id"])
        else:
            routed.extend(succs(state["scenario"], step["id"]))
    return routed, halted


def respond(run_id: str, node_id: str, decision: str, *, by: str | None = None, execute_fn: ExecuteFn | None = None) -> dict:
    with lock_for(run_id):
        state = load_run(run_id)
        if state is None:
            raise ValueError(f"No run '{run_id}'.")
        park = state.get("park") or {}
        if park.get("kind") != "human" or park.get("nodeId") != node_id:
            raise ValueError("This run is not waiting on that person.")
        choice = "approved" if decision == "approved" else "denied"
        who = by or park.get("who") or "you"
        iteration = int(park.get("iteration") or 0)
        emit(state, "HumanResponded", {"nodeId": node_id, "iteration": iteration, "decision": choice, "by": who})
        state["park"] = None
        if choice == "approved":
            state["ran"].append(node_id)
            state["take"][node_id] = iteration + 1
            state["verdicts"][node_id] = "PASS"
            state["status"] = "running"
            for nxt in succs(state["scenario"], node_id):
                if nxt not in state["queue"]:
                    state["queue"].append(nxt)
            save_run(state)
        else:
            on_fail = park.get("onFail") or "halt"
            if on_fail == "retry":
                state["status"] = "running"
                if node_id not in state["queue"]:
                    state["queue"].append(node_id)
                save_run(state)
            else:
                state["failed"] = True
                state["ran"].append(node_id)
                state["take"][node_id] = iteration + 1
                state["verdicts"][node_id] = "FAIL"
                state["status"] = "failed"
                save_run(state)
                emit(state, "RunFinished", {"state": "failed"})
                return load_run(run_id) or state
    return advance(run_id, execute_fn=execute_fn)


def resolve_event(
    event: str,
    payload: Any = None,
    *,
    background: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> list[str]:
    """Resume every run parked on this event name. Payload is recorded on the wait."""
    needle = (event or "").strip().lower()
    resumed: list[str] = []
    if not needle:
        return resumed
    for state in list_runs():
        if state.get("status") != "waiting_world":
            continue
        waiting = str(state.get("waitingEvent") or "").strip().lower()
        if waiting != needle:
            continue
        park = state.get("park") or {}
        if park.get("kind") != "wait":
            continue
        with lock_for(state["runId"]):
            live = load_run(state["runId"])
            if live is None or live.get("status") != "waiting_world":
                continue
            if payload is not None:
                live["payload"] = payload
            node_id = park["nodeId"]
            finish_wait(live, node_id, int(park.get("iteration") or 0), "event received")
            for nxt in succs(live["scenario"], node_id):
                if nxt not in live["queue"]:
                    live["queue"].append(nxt)
            save_run(live)
        if background:
            spawn(state["runId"], execute_fn)
        else:
            advance(state["runId"], execute_fn=execute_fn)
        resumed.append(state["runId"])
    return resumed


def request_pause(run_id: str) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") != "running":
        return state
    if state.get("park"):
        return state
    if not thread_alive(run_id):
        return fail_dead_run(state)
    signal(run_id, "pause")
    state["pauseRequested"] = True
    save_run(state)
    return state


def resume_run(run_id: str, *, execute_fn: ExecuteFn | None = None) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") != "paused":
        return state
    clear_signal(run_id)
    state["pauseRequested"] = False
    state["status"] = "running"
    save_run(state)
    if execute_fn is not None:
        return advance(run_id, execute_fn=execute_fn)
    spawn(run_id)
    return load_run(run_id) or state


def cancel_run(run_id: str) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    signal(run_id, "cancel")
    state["status"] = "cancelled"
    state["park"] = None
    state["queue"] = []
    save_run(state)
    emit(state, "RunFinished", {"state": "failed"})
    return load_run(run_id) or state


def rearm_parked() -> None:
    """On gateway start: resume running work, wake due timers, leave humans parked."""
    tick_timers()
    tick_polls()
    for state in list_runs():
        status = state.get("status")
        if status == "running":
            spawn(state["runId"])
        elif status == "waiting_world":
            rearm(state)


def snapshot(run_id: str, after: int = -1) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    return {"run": state, "events": load_events(run_id, after)}


def snapshot_active(workflow_id: str) -> dict | None:
    state = active_run(workflow_id)
    if state is None:
        return None
    return {"run": state, "events": load_events(state["runId"])}
