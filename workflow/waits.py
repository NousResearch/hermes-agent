"""Steps that stop: a human who has to answer, and a wait that has to elapse.

A park is durable on purpose — closing the app must not lose a run that is
sitting on an approval, so the state file records what it is waiting for and
one of the resume doors below picks it up again: ``respond`` for a human,
``resolve_event`` for the bus, and the two ticks here for a clock or a URL.

The ticks are also the boot path. A timer thread does not survive a restart, so
``rearm_parked`` re-arms what it finds and the ticks sweep anything already due.
"""

from __future__ import annotations

import time
import urllib.request

from workflow.runtime import arm, emit, lock_for, spawn
from workflow.store import list_runs, load_run, save_run
from workflow.topology import config_of, parse_poll, parse_wait_seconds, succs, title_of


def park_human(state: dict, step: dict, iteration: int) -> None:
    cfg = config_of(step)
    who = str(cfg.get("assignee") or "you").strip() or "you"
    prompt = str(cfg.get("goal") or "").strip() or f"{title_of(step)} — approve?"
    payload = {
        "nodeId": step["id"],
        "iteration": iteration,
        "prompt": prompt,
        "who": who,
        "onFail": cfg.get("onFail") or "halt",
    }
    emit(state, "HumanWaiting", payload)
    state["status"] = "waiting_human"
    state["park"] = {"kind": "human", **payload}


def park_wait(state: dict, step: dict, iteration: int) -> bool:
    """True when the run actually parked. A zero-length timer resolves on the
    spot and the loop carries straight on."""
    until = config_of(step).get("until") or {"type": "timer", "spec": ""}
    kind = str(until.get("type") or "timer")
    spec = str(until.get("spec") or "").strip()
    label = spec or kind
    emit(
        state,
        "WaitStarted",
        {"nodeId": step["id"], "iteration": iteration, "until": f"{kind} · {label}", "label": label},
    )
    if kind == "poll":
        parsed = parse_poll(spec)
        if parsed is not None:
            seconds, url = parsed
            state["status"] = "waiting_world"
            state["park"] = {
                "kind": "wait",
                "nodeId": step["id"],
                "iteration": iteration,
                "until": "poll",
                "url": url,
                "interval": seconds,
                "by": "poll matched",
            }
            arm_poll(state["runId"], seconds, url)
            return True
        # A bare name is still a bus park — something else has to tell us.
        state["status"] = "waiting_world"
        state["waitingEvent"] = spec or kind
        state["park"] = {
            "kind": "wait",
            "nodeId": step["id"],
            "iteration": iteration,
            "until": kind,
            "by": "event received",
        }
        return True
    if kind != "timer":
        state["status"] = "waiting_world"
        state["waitingEvent"] = spec or kind
        state["park"] = {
            "kind": "wait",
            "nodeId": step["id"],
            "iteration": iteration,
            "until": kind,
            "by": "event received",
        }
        return True
    seconds = parse_wait_seconds(spec)
    if seconds is None:
        seconds = 0
    if seconds <= 0:
        finish_wait(state, step["id"], iteration, "elapsed")
        return False
    state["status"] = "waiting_world"
    state["wakeAt"] = time.time() + seconds
    state["park"] = {"kind": "wait", "nodeId": step["id"], "iteration": iteration, "until": kind, "by": "elapsed"}
    arm_timer(state["runId"], seconds)
    return True


def finish_wait(state: dict, node_id: str, iteration: int, by: str) -> None:
    emit(state, "WaitResolved", {"nodeId": node_id, "iteration": iteration, "by": by})
    state["ran"].append(node_id)
    state["take"][node_id] = iteration + 1
    state["verdicts"][node_id] = None
    state["park"] = None
    state["wakeAt"] = None
    state["waitingEvent"] = None
    state["status"] = "running"


def arm_timer(run_id: str, seconds: float) -> None:
    arm(run_id, f"workflow-timer-{run_id}", seconds, lambda: tick_timers(run_id=run_id))


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except Exception:
        return False


def arm_poll(run_id: str, seconds: float, url: str) -> None:
    def fire() -> None:
        if http_ok(url):
            tick_polls(run_id=run_id)
            return
        # Still not up. Re-arm only while the run is genuinely still parked on
        # this URL, so a resumed or cancelled run stops the polling with it.
        live = load_run(run_id)
        park = (live or {}).get("park") or {}
        if live and live.get("status") == "waiting_world" and park.get("url") == url:
            arm_poll(run_id, float(park.get("interval") or seconds), url)

    arm(f"poll:{run_id}", f"workflow-poll-{run_id}", max(1.0, seconds), fire)


def _resume(state: dict, park: dict, by: str) -> None:
    """Finish the park under the run's lock and queue what comes next."""
    with lock_for(state["runId"]):
        live = load_run(state["runId"])
        if live is None or live.get("status") != "waiting_world":
            return
        node_id = park["nodeId"]
        finish_wait(live, node_id, int(park.get("iteration") or 0), by)
        for nxt in succs(live["scenario"], node_id):
            if nxt not in live["queue"]:
                live["queue"].append(nxt)
        save_run(live)
    spawn(state["runId"])


def tick_polls(run_id: str | None = None) -> list[str]:
    """Resume poll parks whose URL now answers. Called on a poll thread and at boot."""
    resumed: list[str] = []
    for state in [load_run(run_id)] if run_id else list_runs():
        if not state or state.get("status") != "waiting_world":
            continue
        park = state.get("park") or {}
        if park.get("until") != "poll" or not park.get("url"):
            continue
        if not http_ok(park["url"]):
            continue
        _resume(state, park, "poll matched")
        resumed.append(state["runId"])
    return resumed


def tick_timers(run_id: str | None = None) -> list[str]:
    """Resume timer parks whose wake time has passed. Called on a timer thread and at boot."""
    now = time.time()
    resumed: list[str] = []
    for state in [load_run(run_id)] if run_id else list_runs():
        if not state or state.get("status") != "waiting_world":
            continue
        wake = state.get("wakeAt")
        if wake is None or float(wake) > now:
            continue
        park = state.get("park") or {}
        if park.get("kind") != "wait":
            continue
        _resume(state, park, park.get("by") or "elapsed")
        resumed.append(state["runId"])
    return resumed


def rearm(state: dict) -> None:
    """Re-arm one parked run's clock after a restart — the thread that was
    counting it down died with the previous process."""
    park = state.get("park") or {}
    if park.get("until") == "poll" and park.get("url"):
        if not http_ok(park["url"]):
            arm_poll(state["runId"], float(park.get("interval") or 60), park["url"])
        return
    wake = state.get("wakeAt")
    if wake is None:
        return
    remaining = float(wake) - time.time()
    if remaining > 0:
        arm_timer(state["runId"], remaining)
