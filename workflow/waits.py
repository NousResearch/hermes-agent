"""Steps that stop: a person who has to answer, and a wait the world has to end.

A park belongs to its step, not to the run: a run can have an approval out, a timer counting and a
branch still working, all at once. Parks are durable on purpose — closing the app must not lose a
run that is sitting on an approval — so the state file records each one, and whatever ends it
(an answer, an event, a timer, a URL that came up) is mail to the run (``runtime.post``).

An approval is an event too. Parking a person publishes ``workflow.approval.requested`` with a
short code, sends the question to the step's ``notify`` targets, and accepts the answer from
anywhere: the canvas, ``hermes workflow approve <code>``, ``/workflow approve <code>`` in a chat.

Timers and polls are threads, and threads do not survive a restart: ``rearm_all`` re-arms whatever
the state files say is parked (a clock already past due fires at once).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any

from workflow.runtime import arm, emit, post
from workflow.store import list_runs, load_run
from workflow.topology import config_of, parse_poll, parse_wait_seconds, title_of

logger = logging.getLogger(__name__)


def notify_targets(cfg: dict) -> list[str]:
    raw = cfg.get("notify")
    items = raw if isinstance(raw, list) else str(raw or "").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _send(targets: list[str], text: str) -> None:
    from agent.memory_provider import spawn_context_thread
    from tools.send_message_tool import send_message_tool

    def work() -> None:
        for target in targets:
            try:
                send_message_tool({"action": "send", "target": target, "message": text})
            except Exception:
                logger.warning("workflow approval notice to %s failed", target, exc_info=True)

    spawn_context_thread(work, name="workflow-approval-notice").start()


def approval_text(state: dict, park: dict) -> str:
    return (
        f"Workflow \u201c{state.get('name') or state.get('workflowId')}\u201d is waiting on you:\n"
        f"{park['prompt']}\n\n"
        f"Reply /workflow approve {park['code']} or /workflow deny {park['code']}"
    )


def park_human(state: dict, step: dict, iteration: int) -> dict:
    from workflow import events

    cfg = config_of(step)
    park = {
        "kind": "human",
        "nodeId": step["id"],
        "iteration": iteration,
        "prompt": str(cfg.get("goal") or "").strip() or f"{title_of(step)} \u2014 approve?",
        "who": str(cfg.get("assignee") or "you").strip() or "you",
        "onFail": cfg.get("onFail") or "halt",
        "code": secrets.token_hex(3),
    }
    state.setdefault("parks", {})[step["id"]] = park
    emit(state, "HumanWaiting", {k: park[k] for k in ("nodeId", "iteration", "prompt", "who", "onFail", "code")})
    events.publish_if_wanted(
        events.APPROVAL_REQUESTED,
        {"workflowId": state["workflowId"], "runId": state["runId"], **{k: park[k] for k in ("nodeId", "prompt", "code")}},
        source="workflow",
    )
    targets = notify_targets(cfg)
    if targets:
        _send(targets, approval_text(state, park))
    return park


def park_wait(state: dict, step: dict, iteration: int) -> dict | None:
    """Park the step, or None when the wait is already over (a zero-length timer)."""
    until = config_of(step).get("until") or {"type": "timer", "spec": ""}
    kind = str(until.get("type") or "timer")
    spec = str(until.get("spec") or "").strip()
    label = spec or kind
    emit(state, "WaitStarted", {"nodeId": step["id"], "iteration": iteration, "until": f"{kind} \u00b7 {label}", "label": label})
    park: dict[str, Any] = {"kind": "wait", "nodeId": step["id"], "iteration": iteration}
    poll = parse_poll(spec) if kind == "poll" else None
    if poll is not None:
        park.update(until="poll", url=poll[1], interval=poll[0], by="poll matched")
    elif kind != "timer":
        # A named event (or a poll spec that names one): something else has to tell us.
        park.update(until="event", event=spec or kind, by="event received")
    else:
        seconds = parse_wait_seconds(spec) or 0
        if seconds <= 0:
            return None
        park.update(until="timer", wakeAt=time.time() + seconds, by="elapsed")
    state.setdefault("parks", {})[step["id"]] = park
    arm_park(state["runId"], park)
    return park


def arm_park(run_id: str, park: dict) -> None:
    if park.get("until") == "timer":
        arm_timer(run_id, park["nodeId"], max(0.0, float(park.get("wakeAt") or 0) - time.time()))
    elif park.get("until") == "poll":
        arm_poll(run_id, park["nodeId"], float(park.get("interval") or 60), str(park.get("url") or ""))


def _still_parked(run_id: str, node_id: str) -> dict | None:
    live = load_run(run_id) or {}
    return (live.get("parks") or {}).get(node_id)


def arm_timer(run_id: str, node_id: str, seconds: float) -> None:
    def fire() -> None:
        park = _still_parked(run_id, node_id)
        if park is not None and park.get("until") == "timer":
            post(run_id, {"kind": "resolve", "nodeId": node_id, "by": park.get("by") or "elapsed"})

    arm(f"workflow-timer-{run_id}-{node_id}", seconds, fire)


def http_ok(url: str) -> bool:
    """A poll wait's probe. The URL is authored in a model-editable workflow, so it goes through
    Hermes's one outbound policy: ``is_safe_url`` admission, then the SSRF-safe client that pins
    each connection (every redirect hop included) to a validated address. Private targets need
    ``security.allow_private_urls``; cloud metadata is refused regardless."""
    from tools.url_safety import create_ssrf_safe_client, is_safe_url

    if not is_safe_url(url):
        logger.warning("workflow poll refused an unsafe URL: %s", url)
        return False
    try:
        with create_ssrf_safe_client(timeout=10, follow_redirects=True) as client:
            return 200 <= client.get(url).status_code < 300
    except Exception as exc:
        logger.debug("workflow poll of %s failed: %s", url, exc)
        return False


def arm_poll(run_id: str, node_id: str, seconds: float, url: str) -> None:
    def fire() -> None:
        park = _still_parked(run_id, node_id)
        # A resumed or cancelled run stops the polling with it.
        if park is None or park.get("url") != url:
            return
        if http_ok(url):
            post(run_id, {"kind": "resolve", "nodeId": node_id, "by": "poll matched"})
        else:
            arm_poll(run_id, node_id, float(park.get("interval") or seconds), url)

    arm(f"workflow-poll-{run_id}-{node_id}", max(1.0, seconds), fire)


def parked_runs() -> list[tuple[dict, dict]]:
    """Every (run, park) on disk, newest run first."""
    out = []
    for state in sorted(list_runs(), key=lambda r: r.get("startedAt") or 0, reverse=True):
        for park in (state.get("parks") or {}).values():
            out.append((state, park))
    return out


def find_approval(code: str) -> tuple[dict, dict] | None:
    needle = str(code or "").strip().lower()
    for state, park in parked_runs():
        if park.get("kind") == "human" and str(park.get("code") or "").lower() == needle:
            return state, park
    return None


_rearmed: set[str] = set()
_rearm_lock = threading.Lock()


def rearm_all() -> None:
    """After a restart: re-arm every parked clock in this home, once per process."""
    from hermes_constants import get_hermes_home

    key = str(get_hermes_home())
    with _rearm_lock:
        if key in _rearmed:
            return
        _rearmed.add(key)
    for state, park in parked_runs():
        arm_park(state["runId"], park)
