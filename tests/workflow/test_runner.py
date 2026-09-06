"""The runner walks the authored graph and parks for people and the world."""

import threading
import time

from workflow.runner import (
    advance,
    request_pause,
    resolve_event,
    respond,
    set_execute_fn,
    start_matching,
    start_run,
)
from workflow.store import load_events, load_run, save_documents, save_run
from workflow.topology import parse_poll, parse_wait_seconds


def _agent(_goal, context, payload, _config):
    return {
        "ok": True,
        "summary": f"did it · {payload}",
        "verdict": "PASS",
        "output": {"seen": payload, "context": context},
    }


def _scenario(*steps, edges=None):
    return {"steps": list(steps), "edges": list(edges or [])}


def _put(monkeypatch, tmp_path, doc):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents([doc], doc["id"])


def test_parse_wait_seconds():
    assert parse_wait_seconds("30s") == 30
    assert parse_wait_seconds("2h") == 7200
    assert parse_wait_seconds("every 5m") == 300
    assert parse_wait_seconds("github.pull_request.merged") is None


def test_parse_poll():
    assert parse_poll("deploy.green") is None
    assert parse_poll("https://status/ready") == (60.0, "https://status/ready")
    assert parse_poll("every 30s https://status/ready") == (30.0, "https://status/ready")


def test_agent_receives_trigger_payload(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "hooked",
            "name": "hooked",
            "scenario": _scenario(
                {"id": "start", "kind": "trigger", "config": {"title": "Hook", "on": {"type": "webhook", "spec": ""}}},
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "handle it"}},
                edges=[{"id": "start->work", "source": "start", "target": "work"}],
            ),
        },
    )
    state = start_run("hooked", payload={"pr": 12}, source="webhook", execute_fn=_agent, background=False)
    assert state["status"] == "succeeded"
    assert state["outputs"]["work"]["seen"] == {"pr": 12}
    events = load_events(state["runId"])
    types = [e["type"] for e in events]
    assert types[0] == "RunStarted"
    assert "NodeFinished" in types
    assert types[-1] == "RunFinished"
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs)))


def test_human_parks_and_survives_reload(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "approve",
            "name": "approve",
            "scenario": _scenario(
                {"id": "ask", "kind": "human", "config": {"title": "Ship?", "goal": "Ship it?"}},
                {"id": "ship", "kind": "agent", "config": {"title": "Ship", "goal": "open the PR"}},
                edges=[{"id": "ask->ship", "source": "ask", "target": "ship"}],
            ),
        },
    )
    parked = start_run("approve", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_human"
    assert parked["park"]["nodeId"] == "ask"
    done = respond(parked["runId"], "ask", "approved", execute_fn=_agent)
    assert done["status"] == "succeeded"
    assert "ship" in done["ran"]


def test_poll_url_resumes_when_the_world_answers(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    hits = {"n": 0}

    class Ready(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Ready)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/ready"
    try:
        _put(
            monkeypatch,
            tmp_path,
            {
                "id": "probe",
                "name": "probe",
                "scenario": _scenario(
                    {
                        "id": "hold",
                        "kind": "wait",
                        "config": {"title": "Green", "until": {"type": "poll", "spec": f"every 1s {url}"}},
                    },
                    {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "go"}},
                    edges=[{"id": "hold->work", "source": "hold", "target": "work"}],
                ),
            },
        )
        parked = start_run("probe", execute_fn=_agent, background=False)
        assert parked["status"] == "waiting_world"
        assert parked["park"]["url"] == url
        from workflow.runner import tick_polls
        from workflow.store import load_run

        set_execute_fn(_agent)
        tick_polls(run_id=parked["runId"])
        deadline = time.time() + 2
        done = load_run(parked["runId"])
        while time.time() < deadline and done and done.get("status") in {"running", "waiting_world"}:
            time.sleep(0.05)
            done = load_run(parked["runId"])
        assert done["status"] == "succeeded"
        assert "work" in done["ran"]
        assert hits["n"] >= 1
    finally:
        set_execute_fn(None)
        server.shutdown()


def test_poll_wait_parks_on_the_bus_not_a_timer(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "poll",
            "name": "poll",
            "scenario": _scenario(
                {
                    "id": "hold",
                    "kind": "wait",
                    "config": {"title": "Green", "until": {"type": "poll", "spec": "deploy.green"}},
                }
            ),
        },
    )
    parked = start_run("poll", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_world"
    assert parked["waitingEvent"] == "deploy.green"


def test_wait_event_resumes_on_the_same_bus(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "listen",
            "name": "listen",
            "scenario": _scenario(
                {
                    "id": "hold",
                    "kind": "wait",
                    "config": {"title": "PR", "until": {"type": "event", "spec": "github.pull_request.merged"}},
                },
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "continue"}},
                edges=[{"id": "hold->work", "source": "hold", "target": "work"}],
            ),
        },
    )
    parked = start_run("listen", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_world"
    resolve_event("github.pull_request.merged", {"merged": True}, background=False, execute_fn=_agent)
    from workflow.store import load_run

    done = load_run(parked["runId"])
    assert done["status"] == "succeeded"
    assert "work" in done["ran"]


def test_event_trigger_starts_matching_workflow(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "on-merge",
            "name": "on-merge",
            "scenario": _scenario(
                {
                    "id": "go",
                    "kind": "trigger",
                    "config": {"title": "Merged", "on": {"type": "event", "spec": "github.pull_request.merged"}},
                },
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "ship"}},
                edges=[{"id": "go->work", "source": "go", "target": "work"}],
            ),
        },
    )
    started = start_matching(
        event="github.pull_request.merged",
        payload={"n": 1},
        background=False,
        execute_fn=_agent,
    )
    assert len(started) == 1
    assert started[0]["status"] == "succeeded"
    assert started[0]["outputs"]["work"]["seen"] == {"n": 1}


def test_gate_routes_on_verdicts(tmp_path, monkeypatch):
    def judge(goal, _context, _payload, _config):
        return {"ok": True, "summary": "FAIL", "verdict": "FAIL", "output": {"verdict": "FAIL"}}

    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "gated",
            "name": "gated",
            "scenario": _scenario(
                {"id": "check", "kind": "agent", "config": {"title": "Check", "goal": "review"}},
                {
                    "id": "gate",
                    "kind": "gate",
                    "config": {
                        "title": "Gate",
                        "arms": [
                            {"id": "pass", "when": {"mode": "all-pass"}},
                            {"id": "loop", "when": {"mode": "any-fail"}},
                        ],
                    },
                },
                {"id": "ship", "kind": "agent", "config": {"title": "Ship", "goal": "open"}},
                {"id": "fix", "kind": "agent", "config": {"title": "Fix", "goal": "fix"}},
                edges=[
                    {"id": "check->gate", "source": "check", "target": "gate"},
                    {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
                    {"id": "gate->fix", "source": "gate", "target": "fix", "sourceHandle": "loop"},
                ],
            ),
        },
    )
    state = start_run("gated", execute_fn=judge, background=False)
    assert state["status"] == "succeeded"
    assert "fix" in state["ran"]
    assert "ship" not in state["ran"]


def test_user_fixable_error_stops_instead_of_shipping(tmp_path, monkeypatch):
    def boom(_goal, _context, _payload, _config):
        return {
            "ok": False,
            "error": "HTTP 404: Model 'claude-opus-4.8' not found. The requested model does not exist in our configuration or OpenRouter catalog.",
        }

    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "blocked",
            "name": "blocked",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "do it", "maxRetries": 2}},
                {"id": "ship", "kind": "agent", "config": {"title": "Ship", "goal": "open"}},
                edges=[{"id": "work->ship", "source": "work", "target": "ship"}],
            ),
        },
    )
    state = start_run("blocked", execute_fn=boom, background=False)
    assert state["status"] == "failed"
    assert "ship" not in state["ran"]
    assert any(e["type"] == "UserAsk" for e in load_events(state["runId"]))


def test_null_verdict_does_not_pass_the_gate(tmp_path, monkeypatch):
    """A step that never judged PASS/FAIL is not a pass. Shipping on a 404
    used to look like success because all-pass treated null as fine."""

    def mute(_goal, _context, _payload, _config):
        return {"ok": True, "summary": "HTTP 404: model missing", "verdict": None, "output": {}}

    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "gated",
            "name": "gated",
            "scenario": _scenario(
                {"id": "check", "kind": "agent", "config": {"title": "Check", "goal": "review"}},
                {
                    "id": "gate",
                    "kind": "gate",
                    "config": {
                        "title": "Gate",
                        "arms": [
                            {"id": "pass", "when": {"mode": "all-pass"}},
                            {"id": "loop", "when": {"mode": "any-fail"}},
                        ],
                    },
                },
                {"id": "ship", "kind": "agent", "config": {"title": "Ship", "goal": "open"}},
                edges=[
                    {"id": "check->gate", "source": "check", "target": "gate"},
                    {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
                ],
            ),
        },
    )
    state = start_run("gated", execute_fn=mute, background=False)
    assert "ship" not in state["ran"]
    assert state["status"] == "failed"


def test_ready_agents_run_together(tmp_path, monkeypatch):
    first = threading.Event()
    second = threading.Event()

    def pair(goal, _context, _payload, _config):
        if goal == "left":
            first.set()
            assert second.wait(2)
        else:
            assert first.wait(2)
            second.set()
        return {"ok": True, "summary": goal, "verdict": "PASS", "output": {"goal": goal}}

    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "fan",
            "name": "fan",
            "scenario": _scenario(
                {"id": "left", "kind": "agent", "config": {"title": "Left", "goal": "left"}},
                {"id": "right", "kind": "agent", "config": {"title": "Right", "goal": "right"}},
            ),
        },
    )
    state = start_run("fan", execute_fn=pair, background=False)
    assert state["status"] == "succeeded"
    assert set(state["ran"]) == {"left", "right"}


def test_prose_gate_takes_the_pass_arm(tmp_path, monkeypatch):
    def fn(goal, _context, _payload, _config):
        if "ship it" in goal.lower():
            return {"ok": True, "summary": "PASS", "verdict": "PASS", "output": {}}
        return {"ok": True, "summary": "drafted", "verdict": "PASS", "output": {}}

    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "prose",
            "name": "prose",
            "scenario": _scenario(
                {"id": "draft", "kind": "agent", "config": {"title": "Draft", "goal": "write"}},
                {
                    "id": "gate",
                    "kind": "gate",
                    "config": {
                        "title": "Ship?",
                        "arms": [
                            {"id": "yes", "when": {"mode": "prose", "source": "Should we ship it?"}},
                            {"id": "no", "when": {"mode": "any-fail"}},
                        ],
                    },
                },
                {"id": "open", "kind": "agent", "config": {"title": "Open", "goal": "pr"}},
                {"id": "hold", "kind": "agent", "config": {"title": "Hold", "goal": "wait"}},
                edges=[
                    {"id": "draft->gate", "source": "draft", "target": "gate"},
                    {"id": "gate->open", "source": "gate", "target": "open", "sourceHandle": "yes"},
                    {"id": "gate->hold", "source": "gate", "target": "hold", "sourceHandle": "no"},
                ],
            ),
        },
    )
    state = start_run("prose", execute_fn=fn, background=False)
    assert state["status"] == "succeeded"
    assert "open" in state["ran"]
    assert "hold" not in state["ran"]


def test_inflight_is_restored_on_advance(tmp_path, monkeypatch):
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "crash",
            "name": "crash",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "ship"}},
            ),
        },
    )
    save_run(
        {
            "runId": "crash-1",
            "workflowId": "crash",
            "name": "crash",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "ship"}},
            ),
            "payload": {"n": 7},
            "source": "manual",
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
            "startedAt": 1,
            "failed": False,
            "tries": {},
            "inFlight": ["work"],
            "sessions": {"work": "wf-crash-1-work"},
        }
    )
    state = advance("crash-1", execute_fn=_agent)
    assert state["status"] == "succeeded"
    assert "work" in state["ran"]
    assert state["outputs"]["work"]["seen"] == {"n": 7}
    assert state["sessions"]["work"] == "wf-crash-1-work"


def test_rework_loop_does_not_block_the_start_node(tmp_path, monkeypatch):
    """A loop-back is a rework wire, not an input. Counting it as a predecessor
    left the start node queued-but-unready and the run reported succeeded
    without running anything."""
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "looped",
            "name": "looped",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "do it"}},
                {
                    "id": "gate",
                    "kind": "gate",
                    "config": {
                        "title": "Gate",
                        "arms": [
                            {"id": "pass", "when": {"mode": "all-pass"}},
                            {"id": "loop", "when": {"mode": "any-fail"}},
                        ],
                    },
                },
                {"id": "ship", "kind": "agent", "config": {"title": "Ship", "goal": "open"}},
                edges=[
                    {"id": "work->gate", "source": "work", "target": "gate"},
                    {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
                    {
                        "id": "gate->work",
                        "source": "gate",
                        "target": "work",
                        "sourceHandle": "loop",
                        "targetHandle": "loopback",
                        "loop": True,
                    },
                ],
            ),
        },
    )
    state = start_run("looped", execute_fn=_agent, background=False)
    assert "work" in state["ran"]
    assert "ship" in state["ran"]
    assert state["status"] == "succeeded"


def test_manual_trigger_then_agent_ignores_rework_loop(tmp_path, monkeypatch):
    """Starter canvas: Play → Implement, plus Gate ↺ Implement. Play must
    dispatch Implement — the loop is not an input that Implement waits on."""
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "figma-pr",
            "name": "Figma → PR",
            "scenario": _scenario(
                {
                    "id": "start",
                    "kind": "trigger",
                    "config": {"title": "Play", "on": {"type": "manual", "spec": ""}},
                },
                {"id": "implement", "kind": "agent", "config": {"title": "Implement UI", "goal": "do it"}},
                {
                    "id": "gate",
                    "kind": "gate",
                    "config": {
                        "title": "Quality Gate",
                        "arms": [
                            {"id": "pass", "when": {"mode": "all-pass"}},
                            {"id": "loop", "when": {"mode": "any-fail"}},
                        ],
                    },
                },
                {"id": "approve", "kind": "human", "config": {"title": "Ship Approval", "goal": "ok?"}},
                edges=[
                    {"id": "start->implement", "source": "start", "target": "implement"},
                    {"id": "implement->gate", "source": "implement", "target": "gate"},
                    {"id": "gate->approve", "source": "gate", "target": "approve", "sourceHandle": "pass"},
                    {"id": "gate->implement", "source": "gate", "target": "implement", "loop": True},
                ],
            ),
        },
    )
    state = start_run("figma-pr", execute_fn=_agent, background=False)
    assert "start" in state["ran"]
    assert "implement" in state["ran"]
    assert "gate" in state["ran"]
    assert state["status"] == "waiting_human"


def test_pause_holds_a_live_fake_run(tmp_path, monkeypatch):
    """Pause used to write the flag to disk while the runner kept a stale
    copy — then save_run clobbered it. The in-flight step must freeze."""
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "held",
            "name": "held",
            "scenario": _scenario(
                {"id": "implement", "kind": "agent", "config": {"title": "Implement UI", "goal": "do it"}},
                {"id": "next", "kind": "agent", "config": {"title": "Next", "goal": "then"}},
                edges=[{"id": "implement->next", "source": "implement", "target": "next"}],
            ),
        },
    )
    state = start_run("held", fake=True, background=True)
    run_id = state["runId"]
    started = False
    for _ in range(80):
        if any(e["type"] == "NodeStarted" and e["payload"].get("nodeId") == "implement" for e in load_events(run_id)):
            started = True
            break
        time.sleep(0.05)
    assert started
    request_pause(run_id)
    parked = None
    for _ in range(80):
        parked = load_run(run_id)
        if parked and parked.get("status") == "paused":
            break
        time.sleep(0.05)
    assert parked is not None
    assert parked["status"] == "paused"
    assert "implement" not in parked.get("ran", [])
    assert "next" not in parked.get("ran", [])


def test_start_run_replaces_a_dead_running_run(tmp_path, monkeypatch):
    """A 'running' row with no live thread is leftover from a killed serve.
    Play must mint a new run instead of re-adopting the zombie."""
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "dead",
            "name": "dead",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "do it"}},
            ),
        },
    )
    save_run(
        {
            "runId": "zombie-1",
            "workflowId": "dead",
            "name": "dead",
            "scenario": _scenario(
                {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "do it"}},
            ),
            "payload": None,
            "source": "manual",
            "status": "running",
            "queue": ["work"],
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
            "startedAt": 1,
            "failed": False,
            "tries": {},
            "inFlight": [],
            "sessions": {},
        }
    )
    state = start_run("dead", execute_fn=_agent, background=False)
    assert state["runId"] != "zombie-1"
    assert state["status"] == "succeeded"
    assert "work" in state["ran"]


def test_unready_queue_fails_instead_of_succeeding(tmp_path, monkeypatch):
    """A cycle with no loop flag has no start. That is a stuck graph, not a
    successful empty run."""
    _put(
        monkeypatch,
        tmp_path,
        {
            "id": "cycle",
            "name": "cycle",
            "scenario": _scenario(
                {"id": "a", "kind": "agent", "config": {"title": "A", "goal": "a"}},
                {"id": "b", "kind": "agent", "config": {"title": "B", "goal": "b"}},
                edges=[
                    {"id": "a->b", "source": "a", "target": "b"},
                    {"id": "b->a", "source": "b", "target": "a"},
                ],
            ),
        },
    )
    state = start_run("cycle", execute_fn=_agent, background=False)
    assert state["status"] == "failed"
    assert state["ran"] == []
