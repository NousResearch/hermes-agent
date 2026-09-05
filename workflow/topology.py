"""Reading a scenario: its steps, its wires, and what they say.

Pure functions over the authored dict — no run state, no I/O, no clock. The
runner walks a graph with these; anything else that has to understand a
scenario (a trigger sync, a CLI summary) can use them without importing the
loop.

A scenario is dicts all the way down because it arrives as JSON from the canvas
and from disk. These are the only place that knows the shapes: a step's kind
lives at ``kind`` or under a legacy ``def``, a config is either a ``config``
block or the leftover keys, and a wire's rework flag is ``loop``.
"""

from __future__ import annotations

import re


def scenario_of(doc_or_scenario: dict) -> dict:
    if "steps" in doc_or_scenario and "edges" in doc_or_scenario:
        return doc_or_scenario
    scenario = doc_or_scenario.get("scenario")
    return scenario if isinstance(scenario, dict) else {"steps": [], "edges": []}


def steps_of(scenario: dict) -> list[dict]:
    steps = scenario.get("steps") or []
    return [s for s in steps if isinstance(s, dict) and s.get("id")]


def edges_of(scenario: dict) -> list[dict]:
    edges = scenario.get("edges") or []
    return [e for e in edges if isinstance(e, dict) and e.get("source") and e.get("target")]


def by_id(scenario: dict) -> dict[str, dict]:
    return {s["id"]: s for s in steps_of(scenario)}


def is_loop(edge: dict) -> bool:
    return bool(edge.get("loop"))


def preds(scenario: dict, node_id: str, *, loops: bool = False) -> list[str]:
    """Incoming wires that must have run before ``node_id`` can. Rework loops
    are not inputs — they fire later, from a gate that already ran."""
    return [
        e["source"]
        for e in edges_of(scenario)
        if e["target"] == node_id and (loops or not is_loop(e))
    ]


def succs(scenario: dict, node_id: str, handle: str | None = None) -> list[str]:
    out = []
    for edge in edges_of(scenario):
        if edge["source"] != node_id:
            continue
        if handle is not None and (edge.get("sourceHandle") or "out") != handle:
            continue
        out.append(edge["target"])
    return out


def kind_of(step: dict) -> str:
    return str(step.get("kind") or step.get("def", {}).get("kind") or "agent")


def config_of(step: dict) -> dict:
    if isinstance(step.get("config"), dict):
        return step["config"]
    return {k: v for k, v in step.items() if k not in {"id", "kind", "def"}}


def title_of(step: dict) -> str:
    cfg = config_of(step)
    return str(cfg.get("title") or step.get("title") or step["id"])


def parse_poll(spec: str) -> tuple[float, str] | None:
    """Interval + URL, or None when the spec is an event name on the bus."""
    text = (spec or "").strip()
    every = re.match(
        r"^(?:every\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days)\s+(https?://\S+)",
        text,
        re.I,
    )
    if every:
        value = float(every.group(1))
        unit = every.group(2)[0].lower()
        seconds = value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return max(1.0, seconds), every.group(3)
    if re.match(r"^https?://", text, re.I):
        return 60.0, text
    return None


def parse_wait_seconds(spec: str) -> float | None:
    text = (spec or "").strip().lower()
    every = re.match(r"^every\s+(\d+(?:\.\d+)?)\s*([smhd])", text)
    match = every or re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days)\b", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)[0]
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def holds(when: dict, inputs: list[dict]) -> bool:
    """Does a gate arm's condition hold, given what fed the gate?"""
    mode = when.get("mode") or "always"
    if mode == "always":
        return True
    if mode == "all-pass":
        return bool(inputs) and all(i.get("verdict") == "PASS" for i in inputs)
    if mode == "any-fail":
        return any(i.get("verdict") == "FAIL" for i in inputs)
    if mode == "checks":
        checks = when.get("checks") or []
        hits = []
        for check in checks:
            got = next((i.get("verdict") for i in inputs if i.get("nodeId") == check.get("step")), None)
            is_match = str(got) == str(check.get("value"))
            hits.append(is_match if check.get("op", "is") == "is" else not is_match)
        join = when.get("join") or "all"
        return all(hits) if join == "all" else any(hits)
    if mode == "prose":
        return bool(inputs) and all(i.get("verdict") != "FAIL" for i in inputs)
    return False


def between(scenario: dict, start: str, end: str) -> list[str]:
    """Every step on a path from ``start`` to ``end`` — the body of a loop."""
    body: set[str] = set()

    def walk(node_id: str, path: list[str]) -> bool:
        if node_id == end:
            body.update([*path, node_id])
            return True
        if node_id in path:
            return False
        return any(walk(target, [*path, node_id]) for target in succs(scenario, node_id))

    walk(start, [])
    return list(body)
