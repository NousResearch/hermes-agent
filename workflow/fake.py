"""A stand-in agent step that calls no model.

Off unless a caller asks for it with the ``fake`` flag on ``workflow.run.start``
— tests and screen recordings do, the canvas does not. It emits the same trace
events a real step would, at a pace you can read, so a recording shows tools
scrolling past instead of a card that sits still and then flips to done.

Implement lingers for a beat; everything else lands in a few seconds. Pause and
cancel cut the linger short, so you can stop on that card without waiting the
clock out.
"""

from __future__ import annotations

import time

from workflow.runtime import absorb_signals, emit
from workflow.store import load_run

_TOOLS: dict[str, list[tuple[str, str]]] = {
    "implement": [
        ("read_file", "src/components/ui/button.tsx"),
        ("search_files", "Marketing Site v3"),
        ("write_file", "src/pages/Home.tsx"),
        ("patch", "src/styles.css"),
        ("read_file", "src/pages/Home.tsx"),
        ("search_files", "token --color-primary"),
        ("write_file", "src/components/Hero.tsx"),
        ("patch", "src/pages/Home.tsx"),
    ],
    "review": [
        ("read_file", "src/pages/Home.tsx"),
        ("search_files", "inline style"),
    ],
    "judge": [
        ("browser_navigate", "http://localhost:5173"),
        ("vision_analyze", "Figma · Marketing Site v3"),
    ],
    "ship": [
        ("terminal", "git status"),
        ("terminal", "gh pr create"),
    ],
}

_DONE: dict[str, dict] = {
    "implement": {
        "ok": True,
        "summary": "Built the header from the Figma tokens. diff +48 −0 · 3 files",
        "verdict": "PASS",
        "output": {"text": "header + hero from tokens", "files": 3},
    },
    "review": {
        "ok": True,
        "summary": "PASS · naming clean, no inline styles",
        "verdict": "PASS",
        "output": {"text": "review notes: none"},
    },
    "judge": {
        "ok": True,
        "summary": "PASS · H1 700 matches · pad 16=16px",
        "verdict": "PASS",
        "output": {"text": "visual match"},
    },
    "ship": {
        "ok": True,
        "summary": "PR #1241 opened",
        "verdict": "PASS",
        "output": {"text": "PR #1241", "href": "https://github.com/nousresearch/hermes-agent/pull/1241"},
    },
}


def _interrupted(state: dict) -> str | None:
    """"cancelled", "paused", or None — read from both the live dict and disk,
    because the signal can arrive on either side of a save."""
    absorb_signals(state)
    live = load_run(state["runId"]) or {}
    if live.get("status") in {"cancelled", "failed"}:
        state["status"] = live["status"]
        return "cancelled"
    if state.get("pauseRequested") or live.get("pauseRequested"):
        return "paused"
    return None


def play(state: dict, step: dict, iteration: int) -> dict:
    node_id = step["id"]
    tools = _TOOLS.get(node_id) or [("read_file", node_id)]
    hold = 28.0 if node_id == "implement" else 2.8
    tick = 1.6 if node_id == "implement" else 1.2
    started = time.time()
    i = 0

    while time.time() - started < hold:
        name, arg = tools[i % len(tools)]
        emit(state, "AgentTraceEvent", {"nodeId": node_id, "iteration": iteration, "tool": {"name": name, "arg": arg}})
        i += 1

        deadline = time.time() + tick
        while time.time() < deadline and _interrupted(state) is None:
            time.sleep(0.15)

        stopped = _interrupted(state)
        if stopped == "cancelled":
            return {"ok": False, "error": "cancelled"}
        if stopped == "paused":
            state["pauseRequested"] = True
            return {"ok": True, "_paused": True}

    return dict(_DONE.get(node_id) or {"ok": True, "summary": "done", "verdict": "PASS", "output": {"text": "done"}})
