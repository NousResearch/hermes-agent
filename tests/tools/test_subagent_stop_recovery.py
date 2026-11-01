"""Restart recovery fires ``subagent_stop`` for children whose owner died before their unit joined."""

from __future__ import annotations

import json
import os
import subprocess
import sys

_SETUP = r"""
import os, sys, threading
from unittest.mock import MagicMock
from hermes_cli import plugins
import tools.delegate_tool as dt
import tools.delegate_tool_dispatch as dd
parent = MagicMock(); parent._delegate_depth = 0; parent.session_id = "parent-sess"; parent._interrupt_requested = False
parent._active_children = []; parent._active_children_lock = None
def build(**kw):
    c = MagicMock(); i = kw["task_index"]
    c._delegate_role = "leaf"; c._subagent_id = f"sa-{i}"; c.session_id = f"child-sess-{i}"
    c._delegate_parent_tool_call_id = kw.get("parent_tool_call_id"); c.model = f"route-{i}"
    return c
def result(task_index, goal):
    return {"task_index": task_index, "status": "completed", "summary": f"done: {goal}", "api_calls": 2,
            "duration_seconds": 0.5, "model": "m", "exit_reason": "completed", "tokens": {"input": 10, "output": 3},
            "cost_usd": 0.0012, "cost_status": "estimated"}
creds = {"model": "m", "provider": None, "base_url": None, "api_key": None, "api_mode": None, "command": None, "args": None}
dt._build_child_agent = build; dt._resolve_delegation_credentials = lambda *a, **k: creds
manager = plugins.PluginManager(); manager._discovered = True
plugins._delivery_manager = lambda: manager
def dispatch():
    dt.delegate_task(tasks=[{"goal": "fast member of the group", "group": "g"},
                            {"goal": "slow member of the group", "group": "g"}], background=True,
                     parent_agent=parent, tool_call_id="call-delegate-1")
"""

_PRODUCER = _SETUP + r"""
recorded, parked = threading.Event(), threading.Event()
def child(task_index, goal, child=None, parent_agent=None, **kw):
    if task_index == 1:
        parked.wait()  # never set: the owner dies with this child still running
    return result(task_index, goal)
record = dd.record_unit_child
def record_then_signal(delegation_id, entry):
    record(delegation_id, entry)
    recorded.set()
dd.record_unit_child = record_then_signal
dt._run_single_child = child
dispatch()
# Die only once the fast child's result is committed to the ledger.
os._exit(0 if recorded.wait(20) else 3)
"""

# Both children finish; the owner dies while its live join is delivering the second child's stop.
_DIES_IN_LIVE_JOIN = _SETUP + r"""
fired = []
def stop(**kw):
    fired.append(kw["task_index"])
    if len(fired) == 2:
        os._exit(0)
manager._hooks["subagent_stop"] = [stop]
dt._run_single_child = lambda task_index, goal, child=None, parent_agent=None, **kw: result(task_index, goal)
dispatch()
threading.Event().wait(20)  # the hook exits first; reaching here fails the run
os._exit(3)
"""

# argv[1] == "crash": this recoverer dies inside its first recovered stop hook.
_CONSUMER = _SETUP + r"""
import json, queue
from tools import async_delegation as ad
seen = []
def stop(**kw):
    seen.append(kw)
    if sys.argv[1:] == ["crash"]:
        os._exit(0)
manager._hooks["subagent_stop"] = [stop]
ad.restore_undelivered_completions(queue.Queue())
ad.recover_abandoned_delegations()  # a second recovery pass must not replay the stops
print(json.dumps(seen, sort_keys=True))
"""


def _run(script: str, env: dict, *args: str) -> str:
    repo = env["PYTHONPATH"]
    return subprocess.run([sys.executable, "-c", script, *args], cwd=repo, env=env, text=True, encoding="utf-8",
                          capture_output=True, timeout=30, check=True, stdin=subprocess.DEVNULL).stdout


def _env(tmp_path) -> dict:
    return {**os.environ, "HERMES_HOME": str(tmp_path),
            "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}


def test_owner_death_replays_one_subagent_stop_per_started_child(tmp_path):
    """Real-import E2E: a background unit's owner dies with one child finished and one still running. The next
    process's recovery fires exactly one ``subagent_stop`` per child, naming the ``delegate_task`` call and task,
    with the finished child's recorded result and spend and the other ``unknown``."""
    env = _env(tmp_path)
    _run(_PRODUCER, env)
    out = _run(_CONSUMER, env)
    stops = json.loads(out.strip().splitlines()[-1])

    assert [s["task_index"] for s in stops] == [0, 1]
    for i, stop in enumerate(stops):
        assert stop["recovered"] is True
        assert stop["parent_session_id"] == "parent-sess"
        assert stop["parent_tool_call_id"] == "call-delegate-1"
        assert (stop["child_session_id"], stop["child_subagent_id"]) == (f"child-sess-{i}", f"sa-{i}")
    assert stops[0]["child_status"] == "completed" and stops[0]["child_summary"] == "done: fast member of the group"
    assert (stops[0]["cost_usd"], stops[0]["tokens"], stops[0]["api_calls"]) == (0.0012, {"input": 10, "output": 3}, 2)
    assert stops[1]["child_status"] == "unknown" and stops[1]["cost_status"] == "unknown"
    assert stops[1]["model"] == "route-1"  # the unfinished child's own route, not the unit's


def test_stops_survive_crashes_mid_delivery_without_replaying_delivered_ones(tmp_path):
    """Real-import E2E of the per-child delivery record: the owner dies inside its live join after delivering the
    first child's stop, then the first recoverer dies inside its first recovered stop. The next recovery delivers the
    one stop still unrecorded (at-least-once: the interrupted one repeats) and never the live-delivered one."""
    env = _env(tmp_path)
    _run(_DIES_IN_LIVE_JOIN, env)
    _run(_CONSUMER, env, "crash")
    stops = json.loads(_run(_CONSUMER, env).strip().splitlines()[-1])

    assert [(s["task_index"], s["child_status"], s["recovered"]) for s in stops] == [(1, "completed", True)]
