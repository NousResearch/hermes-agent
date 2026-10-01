"""A stand-in agent step that calls no model.

Off unless a caller asks for it with the ``fake`` flag on ``workflow.run.start`` — tests and
screen recordings do, the canvas does not. It opens the same child Relay session a real step
would and records its tool calls there at a pace you can read, so a recording shows tools
scrolling past instead of a card that sits still and then flips to done.

Implement lingers for a beat; everything else lands in a few seconds. Pause and cancel cut the
linger short, so you can stop on that card without waiting the clock out.
"""

from __future__ import annotations

import time

from workflow import trace
from workflow.runtime import stopping

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
        ("vision_analyze", "Figma \u00b7 Marketing Site v3"),
    ],
    "ship": [
        ("terminal", "git status"),
        ("terminal", "gh pr create"),
    ],
}

_DONE: dict[str, dict] = {
    "implement": {
        "ok": True,
        "summary": "Built the header from the Figma tokens. diff +48 \u22120 \u00b7 3 files",
        "verdict": "PASS",
        "output": {"text": "header + hero from tokens", "files": 3},
    },
    "review": {
        "ok": True,
        "summary": "PASS \u00b7 naming clean, no inline styles",
        "verdict": "PASS",
        "output": {"text": "review notes: none"},
    },
    "judge": {
        "ok": True,
        "summary": "PASS \u00b7 H1 700 matches \u00b7 pad 16=16px",
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


def play(run_id: str, node_id: str, iteration: int) -> dict:
    del iteration
    tools = _TOOLS.get(node_id) or [("read_file", node_id)]
    hold = 28.0 if node_id == "implement" else 2.8
    tick = 1.6 if node_id == "implement" else 1.2
    started = time.time()
    with trace.step_session(run_id, node_id) as session:
        i = 0
        while time.time() - started < hold:
            name, arg = tools[i % len(tools)]
            trace.tool_span(session, name, arg)
            i += 1
            deadline = time.time() + tick
            while time.time() < deadline and stopping(run_id) is None:
                time.sleep(0.15)
            stopped = stopping(run_id)
            if stopped == "cancel":
                return {"ok": False, "error": "cancelled"}
            if stopped == "pause":
                return {"ok": True, "_paused": True}
    return dict(_DONE.get(node_id) or {"ok": True, "summary": "done", "verdict": "PASS", "output": {"text": "done"}})
