"""Run a scenario as a reactor: a step starts when what it waits on has happened.

A step is ready when its inputs are in — every incoming wire by default, any one of them with
``join: any`` — and ready steps run at the same time: an agent works while another branch waits on
a person and a third on a timer. The loop thread is the run's only writer. Step workers compute and
hand their result back; everything from outside (an answer, an event, a timer, a pause) arrives as
mail (``runtime.post``) that the loop folds in between steps.

Triggers are entry points and a workflow may have several. A run starts from the trigger that
fired; the others are dormant for that run and never hold up a join. A matching event that arrives
while a run is live fires a dormant trigger inside it; otherwise it starts a run of its own.

The doors in — ``start_run``, ``deliver_event``, ``respond``, ``request_pause``, ``resume_run``,
``cancel_run`` — are what the reactor calls for each inbox event (``workflow/reactor.py``).

This file is the loop and its doors. What it reads, waits on and records through lives beside it:

    topology  reading the authored scenario — steps, wires, conditions
    runtime   the per-run lock, mailbox and loop thread
    waits     the steps that park, and the clocks that end them
    trace     the run's Relay session, where its history is recorded
    fake      the scripted stand-in that calls no model
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from typing import Any, Callable

from workflow import events, fake, trace
from workflow.runtime import (
    clear_stopping,
    emit,
    ensure_running,
    fail_dead_run,
    lock_for,
    post,
    retire,
    take_mail,
    thread_alive,
)
from workflow.store import (
    active_run,
    get_document,
    live_runs,
    load_documents,
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
from workflow.waits import park_human, park_wait, parked_runs, rearm_all

ExecuteFn = Callable[[str, str, Any, dict], dict]

FINAL = frozenset({"succeeded", "failed", "cancelled"})
_TICK = 0.2
_DASH = "\u2014"
_MAX_WORKERS = 8

_execute_fn: ExecuteFn | None = None


def set_execute_fn(fn: ExecuteFn | None) -> None:
    global _execute_fn
    _execute_fn = fn


# ── starting ──────────────────────────────────────────────────────────────────────────────────


def _fresh_state(workflow_id: str, scenario: dict, payload: Any, source: str, name: str, run_id: str | None) -> dict:
    return {
        "runId": run_id or new_run_id(),
        "workflowId": workflow_id,
        "name": name,
        "scenario": scenario,
        "payload": payload,
        "source": source,
        "status": "running",
        "queue": [],
        "ran": [],
        "satisfied": [],
        "dormant": [],
        "verdicts": {},
        "outputs": {},
        "summaries": {},
        "take": {},
        "loops": 0,
        "parks": {},
        "pauseRequested": False,
        "seq": 0,
        "startedAt": int(time.time() * 1000),
        "failed": False,
        "tries": {},
        "inFlight": [],
        "sessions": {},
    }


_SOURCE_TRIGGER = {"manual": "manual", "cli": "manual", "tool": "manual", "webhook": "webhook", "cron": "cron"}


def _trigger_type(step: dict) -> str:
    return str((config_of(step).get("on") or {}).get("type") or "manual")


def _entries(scenario: dict, source: str, trigger: str | None) -> tuple[list[str], list[str]]:
    """(the steps a run starts on, the triggers that did not fire)."""
    steps = steps_of(scenario)
    triggers = [s for s in steps if kind_of(s) == "trigger"]
    if trigger:
        fired = [s["id"] for s in triggers if s["id"] == trigger]
    else:
        want = _SOURCE_TRIGGER.get(source)
        fired = [s["id"] for s in triggers if _trigger_type(s) == want]
        fired = fired or [s["id"] for s in triggers]
    dormant = [s["id"] for s in triggers if s["id"] not in fired]
    free = [s["id"] for s in steps if kind_of(s) != "trigger" and not preds(scenario, s["id"])]
    entries = fired + free
    if not entries and steps:
        entries = [steps[0]["id"]]
    return entries, dormant


def _pinned_payload(scenario: dict, fired: list[str]) -> Any:
    steps = by_id(scenario)
    for node_id in fired:
        pinned = config_of(steps.get(node_id) or {}).get("pinPayload")
        if pinned not in (None, "", {}):
            return pinned
    return None


def start_run(
    workflow_id: str,
    *,
    scenario: dict | None = None,
    payload: Any = None,
    source: str = "manual",
    execute_fn: ExecuteFn | None = None,
    background: bool = True,
    fake: bool = False,
    trigger: str | None = None,
    run_id: str | None = None,
) -> dict:
    doc = get_document(workflow_id)
    if doc is not None:
        workflow_id = doc["id"]
    if scenario is None:
        if doc is None:
            raise ValueError(f"No workflow called '{workflow_id}'.")
        scenario = scenario_of(doc)
    elif doc is not None:
        upsert_document({**doc, "scenario": scenario})
    else:
        upsert_document({"id": workflow_id, "name": workflow_id, "scenario": scenario})
        doc = get_document(workflow_id)
    name = (doc or {}).get("name") or workflow_id

    if source == "manual":
        # Play adopts the run that is already going rather than starting a second one beside it.
        existing = active_run(workflow_id)
        if existing is not None:
            if existing.get("status") == "running" and not thread_alive(existing["runId"]):
                fail_dead_run(existing)
            else:
                return existing

    state = _fresh_state(workflow_id, scenario, payload, source, name, run_id)
    if fake:
        state["fake"] = True
    entries, dormant = _entries(scenario, source, trigger)
    if state["payload"] is None:
        state["payload"] = _pinned_payload(scenario, [e for e in entries if e not in dormant])
    state["queue"], state["dormant"] = entries, dormant
    emit(state, "RunStarted", {"scenario": name, "source": source, **({"trigger": trigger} if trigger else {})})
    save_run(state)
    if background:
        ensure_running(state["runId"], execute_fn)
    else:
        advance(state["runId"], execute_fn=execute_fn)
    return load_run(state["runId"]) or state


def _deliver(run_id: str, mail: dict, *, background: bool, execute_fn: ExecuteFn | None) -> None:
    post(run_id, mail, start=background, execute_fn=execute_fn)
    if not background:
        advance(run_id, execute_fn=execute_fn)


def deliver_event(
    name: str,
    payload: Any = None,
    *,
    source: str = "event",
    background: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> list[str]:
    """Hand one event to everything listening: parked waits resume, matching triggers fire inside a
    live run where they are dormant, or start a run of their own. Returns the runs it touched."""
    touched: list[str] = []
    for state, park in parked_runs():
        if park.get("until") == "event" and events.matches(str(park.get("event") or ""), name):
            _deliver(state["runId"], {"kind": "resolve", "nodeId": park["nodeId"], "by": name, "payload": payload},
                     background=background, execute_fn=execute_fn)
            touched.append(state["runId"])
    # A workflow never re-triggers itself from its own happenings: that is a loop, not a reaction.
    origin = payload.get("workflowId") if isinstance(payload, dict) else None
    for doc in load_documents()["docs"]:
        if doc["id"] == origin:
            continue
        scenario = scenario_of(doc)
        for step in steps_of(scenario):
            spec = str((config_of(step).get("on") or {}).get("spec") or "")
            if kind_of(step) != "trigger" or _trigger_type(step) != "event" or not events.matches(spec, name):
                continue
            host = next((r for r in live_runs(doc["id"]) if step["id"] in (r.get("dormant") or [])), None)
            if host is not None:
                _deliver(host["runId"], {"kind": "fire", "nodeId": step["id"], "payload": payload},
                         background=background, execute_fn=execute_fn)
                touched.append(host["runId"])
            else:
                started = start_run(doc["id"], payload=payload, source=source, trigger=step["id"],
                                    background=background, execute_fn=execute_fn)
                touched.append(started["runId"])
    return touched


def respond(
    run_id: str,
    node_id: str,
    decision: str,
    *,
    by: str | None = None,
    background: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    park = (state.get("parks") or {}).get(node_id) or {}
    if park.get("kind") != "human":
        raise ValueError("This run is not waiting on that person.")
    _deliver(run_id, {"kind": "answer", "nodeId": node_id, "decision": decision, "by": by},
             background=background, execute_fn=execute_fn)
    return load_run(run_id) or state


def request_pause(run_id: str, *, background: bool = True, execute_fn: ExecuteFn | None = None) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") in FINAL or state.get("status") == "paused":
        return state
    _deliver(run_id, {"kind": "pause"}, background=background, execute_fn=execute_fn)
    return load_run(run_id) or state


def resume_run(run_id: str, *, background: bool = True, execute_fn: ExecuteFn | None = None) -> dict:
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") != "paused" and not state.get("pauseRequested"):
        return state
    _deliver(run_id, {"kind": "resume"}, background=background, execute_fn=execute_fn)
    return load_run(run_id) or state


def cancel_run(run_id: str, *, background: bool = True, execute_fn: ExecuteFn | None = None) -> dict:
    """Cancel, and wait (briefly) for the loop to say so, so a Play right after cannot adopt it."""
    state = load_run(run_id)
    if state is None:
        raise ValueError(f"No run '{run_id}'.")
    if state.get("status") in FINAL:
        return state
    _deliver(run_id, {"kind": "cancel"}, background=background, execute_fn=execute_fn)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        live = load_run(run_id) or state
        if live.get("status") in FINAL:
            return live
        time.sleep(0.05)
    return load_run(run_id) or state


def rearm_parked() -> None:
    """When this process becomes the reactor: re-arm parked clocks, and pick up runs a dead
    process left mid-step (their ``inFlight`` steps are queued again)."""
    rearm_all()
    from workflow.store import list_runs

    for state in list_runs():
        if state.get("status") == "running" and not thread_alive(state["runId"]):
            ensure_running(state["runId"])


# ── the loop ──────────────────────────────────────────────────────────────────────────────────


def advance(run_id: str, *, execute_fn: ExecuteFn | None = None) -> dict:
    with lock_for(run_id), trace.recording(run_id):
        return _Loop(run_id, execute_fn or _execute_fn).run()


class _Loop:
    """One pass of a run's loop thread. Holds the live state; nothing else writes it."""

    def __init__(self, run_id: str, execute_fn: ExecuteFn | None) -> None:
        state = load_run(run_id)
        if state is None:
            raise ValueError(f"No run '{run_id}'.")
        self.state = state
        self.run_id = run_id
        self.fn = execute_fn
        self.steps = by_id(state["scenario"])
        self.inflight: dict[Future, tuple[str, int, str, Any]] = {}
        self.pool: ThreadPoolExecutor | None = None

    # -- the pass --

    def run(self) -> dict:
        state = self.state
        if state.get("status") in FINAL:
            take_mail(self.run_id)
            return state
        leftover = [n for n in state.get("inFlight") or [] if n not in state["ran"] and n not in state["queue"]]
        state["queue"] = leftover + state["queue"]
        state["inFlight"] = []
        try:
            while True:
                self.fold_mail()
                if state["status"] in FINAL:
                    break
                holding = state["status"] == "paused" or state.get("pauseRequested") or state.get("failed")
                if not holding:
                    self.dispatch_ready()
                if self.inflight:
                    self.collect()
                    continue
                if not holding and self.ready():
                    continue
                if state.get("failed") or not (state.get("parks") or state.get("pauseRequested") or state["status"] == "paused"):
                    self.finish()
                    break
                self.settle()
                save_run(state)
                if retire(self.run_id):
                    return state
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False)
        save_run(state)
        retire(self.run_id)
        return state

    def settle(self) -> None:
        """Nothing to do until something happens: say what the run is waiting on."""
        state = self.state
        if state.get("pauseRequested") or state["status"] == "paused":
            if state["status"] != "paused":
                state["status"] = "paused"
                emit(state, "RunPaused", {})
            return
        parks = (state.get("parks") or {}).values()
        state["status"] = "waiting_human" if any(p.get("kind") == "human" for p in parks) else "waiting_world"

    def finish(self) -> None:
        state = self.state
        leftover = [n for n in state["queue"] if n not in state["ran"]]
        if leftover and not state.get("failed"):
            state["failed"] = True
            emit(state, "NodeFailed", {
                "nodeId": leftover[0], "iteration": int(state["take"].get(leftover[0]) or 0),
                "error": f"never became ready (still waiting on {', '.join(leftover)})",
            })
        state["parks"] = {}
        state["status"] = "failed" if state.get("failed") else "succeeded"
        emit(state, "RunFinished", {"state": state["status"]})
        save_run(state)
        events.publish_if_wanted(events.RUN_FINISHED, {
            "workflowId": state["workflowId"], "runId": self.run_id, "state": state["status"],
        }, source="workflow")

    # -- readiness and routing --

    def ready(self) -> list[str]:
        state = self.state
        busy = {node for node, *_ in self.inflight.values()} | set(state.get("parks") or {})
        done = set(state["ran"]) | set(state["satisfied"])
        dormant = set(state.get("dormant") or [])
        out = []
        for node_id in state["queue"]:
            step = self.steps.get(node_id)
            if step is None or node_id in busy or node_id in out:
                continue
            inputs = [p for p in preds(state["scenario"], node_id) if p in self.steps and p not in dormant]
            arrived = [p in done for p in inputs]
            joined = any(arrived) if config_of(step).get("join") == "any" else all(arrived)
            if not inputs or joined:
                out.append(node_id)
        return out

    def route(self, targets: list[str]) -> None:
        state = self.state
        for nxt in targets:
            step = self.steps.get(nxt)
            if step is None or nxt in state["queue"]:
                continue
            if config_of(step).get("join") == "any" and nxt in state["ran"]:
                continue  # the first arrival already ran it; later ones are not a second take
            state["queue"].append(nxt)

    # -- dispatch --

    def dispatch_ready(self) -> None:
        state = self.state
        for node_id in self.ready():
            if state.get("pauseRequested") or state.get("failed"):
                return
            state["queue"].remove(node_id)
            step = self.steps[node_id]
            iteration = int(state["take"].get(node_id) or 0)
            handler = {
                "trigger": self.run_trigger, "agent": self.start_agent, "gate": self.start_gate,
                "wait": self.start_wait, "human": self.start_human,
            }.get(kind_of(step), self.start_agent)
            handler(step, iteration)
        state["inFlight"] = [node for node, *_ in self.inflight.values()]
        save_run(state)

    def submit(self, node_id: str, iteration: int, kind: str, work: Callable[[], Any], extra: Any = None) -> None:
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix=f"workflow-{self.run_id}")
        from agent.memory_provider import ctx_bound

        self.inflight[self.pool.submit(ctx_bound(work))] = (node_id, iteration, kind, extra)

    def collect(self) -> None:
        done, _ = wait_futures(list(self.inflight), timeout=_TICK, return_when=FIRST_COMPLETED)
        for fut in done:
            node_id, iteration, kind, extra = self.inflight.pop(fut)
            if self.state["status"] in FINAL:
                continue  # cancelled under it: the answer has nowhere to go
            try:
                result = fut.result()
            except Exception as exc:
                result = {"ok": False, "error": str(exc)} if kind == "agent" else None
            step = self.steps[node_id]
            if kind == "agent":
                self.apply_agent(step, iteration, result)
            else:
                self.apply_gate(step, iteration, extra, result)
        self.state["inFlight"] = [node for node, *_ in self.inflight.values()]
        save_run(self.state)

    # -- triggers --

    def run_trigger(self, step: dict, iteration: int, payload: Any = None) -> None:
        state = self.state
        node_id = step["id"]
        on = config_of(step).get("on") or {"type": "manual", "spec": ""}
        label = f"{on.get('type') or 'manual'}" + (f" \u00b7 {on['spec']}" if on.get("spec") else "")
        if payload is not None:
            state["payload"] = payload
        emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
        emit(state, "NodeStarted", {"nodeId": node_id, "iteration": iteration, "input": label, "maxIters": 0})
        emit(state, "NodeFinished", {"nodeId": node_id, "iteration": iteration})
        state["ran"].append(node_id)
        state["take"][node_id] = iteration + 1
        state["verdicts"][node_id] = None
        self.route(succs(state["scenario"], node_id))

    # -- agents --

    def start_agent(self, step: dict, iteration: int) -> None:
        state = self.state
        node_id = step["id"]
        cfg = config_of(step)
        goal = str(cfg.get("goal") or "").strip() or title_of(step)
        emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
        emit(state, "NodeStarted", {
            "nodeId": node_id, "iteration": iteration, "input": goal[:80],
            "maxIters": int(cfg.get("maxIterations") or 20), "loop": iteration > 0,
            **({"pinned": True} if cfg.get("pin") else {}),
        })
        pinned = cfg.get("pin")
        if isinstance(pinned, dict) and pinned:
            # A pinned step answers with what was frozen on it; nothing runs.
            self.apply_agent(step, iteration, {"ok": True, **pinned, "pinned": True})
            return
        sessions = state.setdefault("sessions", {})
        resume = node_id in sessions or bool(state["tries"].get(node_id))
        sessions[node_id] = sessions.get(node_id) or trace.step_session_id(self.run_id, node_id)
        job = {
            "goal": goal, "context": "" if cfg.get("blind") else self.context_for(node_id),
            "payload": state.get("payload"), "cfg": dict(cfg), "session_id": sessions[node_id],
            "parent_session_id": trace.run_session_id(self.run_id), "resume": resume,
        }
        fn, run_id = self.fn, self.run_id
        if state.get("fake") and fn is None:
            self.submit(node_id, iteration, "agent", lambda: fake.play(run_id, node_id, iteration))
            return
        self.submit(node_id, iteration, "agent", lambda: _compute_agent(job, fn))

    def context_for(self, node_id: str) -> str:
        state = self.state
        parts = []
        for pred in preds(state["scenario"], node_id, loops=True):
            summary = (state.get("summaries") or {}).get(pred)
            output = (state.get("outputs") or {}).get(pred)
            if summary:
                parts.append(f"{pred}: {summary}")
            if output:
                parts.append(f"{pred} output: {output}")
        return "\n".join(parts)

    def apply_agent(self, step: dict, iteration: int, result: dict) -> None:
        state = self.state
        node_id = step["id"]
        cfg = config_of(step)
        if result.get("_paused"):
            state["pauseRequested"] = True
            state["queue"].append(node_id)
            return
        if not result.get("ok", True):
            error = str(result.get("error") or "step failed")
            emit(state, "NodeFailed", {"nodeId": node_id, "iteration": iteration, "error": error})
            from workflow.agent import is_user_fixable

            if is_user_fixable(error):
                emit(state, "UserAsk", {"nodeId": node_id, "iteration": iteration, "prompt": error})
                state["failed"] = True
                return
            retries = int(state["tries"].get(node_id) or 0)
            if retries < int(cfg.get("maxRetries") or 0):
                state["tries"][node_id] = retries + 1
                state["queue"].append(node_id)
                return
            state["ran"].append(node_id)
            state["take"][node_id] = iteration + 1
            state["verdicts"][node_id] = "FAIL"
            if (cfg.get("onFail") or "halt") == "route":
                self.route(succs(state["scenario"], node_id))
            else:
                state["failed"] = True
            return
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
        events.publish_if_wanted(events.STEP_FINISHED, {
            "workflowId": state["workflowId"], "runId": self.run_id, "nodeId": node_id,
            "verdict": verdict, "summary": summary,
        }, source="workflow")
        self.route(succs(state["scenario"], node_id))

    # -- gates --

    def start_gate(self, step: dict, iteration: int) -> None:
        state = self.state
        node_id = step["id"]
        inputs = [{"nodeId": p, "verdict": state["verdicts"].get(p)} for p in preds(state["scenario"], node_id)]
        emit(state, "NodePending", {"nodeId": node_id, "iteration": iteration})
        emit(state, "NodeStarted", {
            "nodeId": node_id, "iteration": iteration, "maxIters": 8,
            "input": " \u00b7 ".join(f"{i['nodeId']} {i['verdict'] or _DASH}" for i in inputs) or "no inputs",
        })
        arms = [a for a in config_of(step).get("arms") or [] if isinstance(a, dict)]
        context = "\n".join(
            f"{i['nodeId']}: {i.get('verdict') or _DASH} \u00b7 {state['summaries'].get(i['nodeId'], '')}" for i in inputs
        )
        payload, fn = state.get("payload"), self.fn
        self.submit(node_id, iteration, "gate", lambda: _choose_arm(arms, inputs, context, payload, fn), inputs)

    def apply_gate(self, step: dict, iteration: int, inputs: list[dict], arm: dict | None) -> None:
        state = self.state
        node_id = step["id"]
        scenario = state["scenario"]
        route = None
        if arm is not None:
            targets = succs(scenario, node_id, arm.get("id"))
            route = targets[0] if targets else None
        culprit = next((i for i in inputs if i.get("verdict") == "FAIL"), None)
        title = title_of(by_id(scenario).get(route) or {"id": route or "", "title": "nowhere"})
        emit(state, "GateEvaluated", {
            "nodeId": node_id, "iteration": iteration, "inputs": inputs,
            "decision": "fail" if culprit else "pass", "route": route or "",
            "summary": f"{culprit['nodeId'] + ' FAIL' if culprit else 'group PASS'} \u2192 {title}",
        })
        state["ran"].append(node_id)
        state["take"][node_id] = iteration + 1
        state["verdicts"][node_id] = "FAIL" if culprit else "PASS"
        if not route:
            emit(state, "NodeFailed", {
                "nodeId": node_id, "iteration": iteration,
                "error": f'"{arm.get("label") or arm.get("id")}" isn\'t wired anywhere' if arm
                else "no arm matched, so the work has nowhere to go",
            })
            state["failed"] = True
            return
        if route in state["ran"]:
            cap = int(config_of(step).get("maxLoops") or 5)
            if state["loops"] >= cap:
                emit(state, "NodeFailed", {"nodeId": node_id, "iteration": iteration, "error": f"gave up after {cap} takes"})
                state["failed"] = True
                return
            state["loops"] += 1
            emit(state, "LoopAdvanced", {
                "loopId": node_id, "iteration": state["loops"], "to": route,
                "feedback": f"{culprit['nodeId']} feedback" if culprit else "another take",
            })
            for item in between(scenario, route, node_id):
                state["ran"] = [x for x in state["ran"] if x != item]
                if item not in {route, node_id} and state["verdicts"].get(item) == "PASS":
                    if item not in state["satisfied"]:
                        state["satisfied"].append(item)
                    emit(state, "NodeSkipped", {
                        "nodeId": item, "iteration": state["loops"],
                        "reason": f"satisfied \u00b7 PASS on take {state['take'].get(item) or 1}",
                    })
        self.route([route])

    # -- parks --

    def start_wait(self, step: dict, iteration: int) -> None:
        emit(self.state, "NodePending", {"nodeId": step["id"], "iteration": iteration})
        if park_wait(self.state, step, iteration) is None:
            self.end_wait(step["id"], iteration, "elapsed")

    def start_human(self, step: dict, iteration: int) -> None:
        emit(self.state, "NodePending", {"nodeId": step["id"], "iteration": iteration})
        park_human(self.state, step, iteration)

    def end_wait(self, node_id: str, iteration: int, by: str, payload: Any = None) -> None:
        state = self.state
        emit(state, "WaitResolved", {"nodeId": node_id, "iteration": iteration, "by": by})
        state["parks"].pop(node_id, None)
        if payload is not None:
            state["payload"] = payload
        state["ran"].append(node_id)
        state["take"][node_id] = iteration + 1
        state["verdicts"][node_id] = None
        self.route(succs(state["scenario"], node_id))

    def answer(self, node_id: str, decision: str, by: str | None) -> None:
        state = self.state
        park = state["parks"].get(node_id)
        if not park or park.get("kind") != "human":
            return  # answered already (another surface got there first)
        choice = "approved" if decision == "approved" else "denied"
        iteration = int(park.get("iteration") or 0)
        who = by or park.get("who") or "you"
        emit(state, "HumanResponded", {"nodeId": node_id, "iteration": iteration, "decision": choice, "by": who})
        state["parks"].pop(node_id, None)
        events.publish_if_wanted(events.APPROVAL_ANSWERED, {
            "workflowId": state["workflowId"], "runId": self.run_id, "nodeId": node_id, "decision": choice, "by": who,
        }, source="workflow")
        if choice == "approved":
            state["ran"].append(node_id)
            state["take"][node_id] = iteration + 1
            state["verdicts"][node_id] = "PASS"
            self.route(succs(state["scenario"], node_id))
        elif (park.get("onFail") or "halt") == "retry":
            state["queue"].append(node_id)
        else:
            state["ran"].append(node_id)
            state["take"][node_id] = iteration + 1
            state["verdicts"][node_id] = "FAIL"
            state["failed"] = True

    # -- mail --

    def fold_mail(self) -> None:
        state = self.state
        for mail in take_mail(self.run_id):
            kind = mail.get("kind")
            if state["status"] in FINAL:
                continue
            node_id = str(mail.get("nodeId") or "")
            if kind == "pause":
                state["pauseRequested"] = True
            elif kind == "resume":
                state["pauseRequested"] = False
                clear_stopping(self.run_id)
                if state["status"] == "paused":
                    state["status"] = "running"
            elif kind == "cancel":
                self.cancel()
            elif kind == "resolve":
                park = state["parks"].get(node_id)
                if park and park.get("kind") == "wait":
                    self.end_wait(node_id, int(park.get("iteration") or 0), str(mail.get("by") or "resolved"), mail.get("payload"))
            elif kind == "answer":
                self.answer(node_id, str(mail.get("decision") or ""), mail.get("by"))
            elif kind == "fire" and node_id in (state.get("dormant") or []) and node_id in self.steps:
                state["dormant"].remove(node_id)
                self.run_trigger(self.steps[node_id], int(state["take"].get(node_id) or 0), mail.get("payload"))
        if state["status"] in {"waiting_human", "waiting_world"}:
            state["status"] = "running"

    def cancel(self) -> None:
        state = self.state
        state["status"] = "cancelled"
        state["parks"] = {}
        state["queue"] = []
        state["inFlight"] = []
        clear_stopping(self.run_id)
        emit(state, "RunFinished", {"state": "failed"})
        save_run(state)


# ── step work (worker threads: no run state is written here) ───────────────────────────────────


def _compute_agent(job: dict, execute_fn: ExecuteFn | None) -> dict:
    if execute_fn is not None:
        return execute_fn(job["goal"], job["context"], job["payload"], job["cfg"])
    from workflow.agent import execute_agent_step

    return execute_agent_step(
        job["goal"], job["context"], job["payload"], job["cfg"],
        session_id=job["session_id"], parent_session_id=job["parent_session_id"], resume=job["resume"],
    )


def _arm_matches(arm: dict, inputs: list[dict], context: str, payload: Any, execute_fn: ExecuteFn | None) -> bool:
    when = arm.get("when") or {}
    if when.get("mode") != "prose":
        return holds(when, inputs)
    source = str(when.get("source") or "").strip() or "Should this arm be taken? Answer PASS or FAIL."
    if execute_fn is None:
        from workflow.agent import execute_agent_step

        result = execute_agent_step(source, context, payload, {"maxIterations": 8})
    else:
        result = execute_fn(source, context, payload, {"maxIterations": 8})
    if not result.get("ok", True):
        return False
    verdict = result.get("verdict")
    if verdict:
        return verdict == "PASS"
    text = str(result.get("summary") or "").upper()
    return "PASS" in text or text.startswith("YES")


def _choose_arm(arms: list[dict], inputs: list[dict], context: str, payload: Any, execute_fn: ExecuteFn | None) -> dict | None:
    return next((a for a in arms if _arm_matches(a, inputs, context, payload, execute_fn)), None)
