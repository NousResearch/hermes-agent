"""The runner is a reactor over the authored graph: steps start when their inputs are in, parks
belong to steps, and events from outside reach a live run as mail."""

import threading
import time

from workflow.runner import (
    advance,
    deliver_event,
    request_pause,
    respond,
    set_execute_fn,
    start_run,
)
from workflow.store import load_run, save_documents, save_run
from workflow.topology import parse_poll, parse_wait_seconds
from workflow.trace import flush, recorded_events, runner_events


def _agent(_goal, context, payload, _config):
    return {
        "ok": True,
        "summary": f"did it · {payload}",
        "verdict": "PASS",
        "output": {"seen": payload, "context": context},
    }


def _scenario(*steps, edges=None):
    return {"steps": list(steps), "edges": list(edges or [])}


def _put(monkeypatch, tmp_path, *docs):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents(list(docs), docs[0]["id"])


def _types(run_id):
    flush()
    return [e["type"] for e in runner_events(recorded_events(run_id)[0])]


def _wait_for(run_id, done, timeout=3.0):
    deadline = time.time() + timeout
    state = load_run(run_id)
    while time.time() < deadline and not done(state):
        time.sleep(0.05)
        state = load_run(run_id)
    return state


def test_parse_wait_seconds():
    assert parse_wait_seconds("30s") == 30
    assert parse_wait_seconds("2h") == 7200
    assert parse_wait_seconds("every 5m") == 300
    assert parse_wait_seconds("github.pull_request.merged") is None


def test_parse_poll():
    assert parse_poll("deploy.green") is None
    assert parse_poll("https://status/ready") == (60.0, "https://status/ready")
    assert parse_poll("every 30s https://status/ready") == (30.0, "https://status/ready")


def test_a_run_is_recorded_as_its_relay_trace(tmp_path, monkeypatch):
    _put(monkeypatch, tmp_path, {
        "id": "hooked", "name": "hooked",
        "scenario": _scenario(
            {"id": "start", "kind": "trigger", "config": {"title": "Hook", "on": {"type": "webhook", "spec": ""}}},
            {"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "handle it"}},
            edges=[{"id": "start->work", "source": "start", "target": "work"}],
        ),
    })
    state = start_run("hooked", payload={"pr": 12}, source="webhook", execute_fn=_agent, background=False)
    assert state["status"] == "succeeded"
    assert state["outputs"]["work"]["seen"] == {"pr": 12}
    flush()
    events = runner_events(recorded_events(state["runId"])[0])
    types = [e["type"] for e in events]
    assert types[0] == "RunStarted" and "NodeFinished" in types and types[-1] == "RunFinished"
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_an_approval_parks_its_step_not_the_run(tmp_path, monkeypatch):
    """While a person is asked, an event fires a dormant trigger in the same run and its branch
    runs to the end; a ``join: any`` step takes the first arrival and the late one is not a
    second take."""
    _put(monkeypatch, tmp_path, {
        "id": "w", "name": "w",
        "scenario": _scenario(
            {"id": "play", "kind": "trigger", "config": {"on": {"type": "manual"}}},
            {"id": "deploy", "kind": "trigger", "config": {"on": {"type": "event", "spec": "deploy.*"}}},
            {"id": "draft", "kind": "agent", "config": {"goal": "draft"}},
            {"id": "ok", "kind": "human", "config": {"goal": "ship?"}},
            {"id": "verify", "kind": "agent", "config": {"goal": "verify"}},
            {"id": "post", "kind": "agent", "config": {"goal": "post", "join": "any"}},
            edges=[
                {"id": "1", "source": "play", "target": "draft"},
                {"id": "2", "source": "draft", "target": "ok"},
                {"id": "3", "source": "ok", "target": "post"},
                {"id": "4", "source": "deploy", "target": "verify"},
                {"id": "5", "source": "verify", "target": "post"},
            ],
        ),
    })
    parked = start_run("w", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_human"
    assert set(parked["parks"]) == {"ok"} and parked["dormant"] == ["deploy"]

    assert deliver_event("deploy.done", {"v": 2}, background=False, execute_fn=_agent) == [parked["runId"]]
    mid = load_run(parked["runId"])
    assert {"verify", "post"} <= set(mid["ran"]) and set(mid["parks"]) == {"ok"}

    done = respond(parked["runId"], "ok", "approved", background=False, execute_fn=_agent)
    assert done["status"] == "succeeded"
    assert done["ran"].count("post") == 1


def test_an_event_with_no_dormant_trigger_starts_its_own_run(tmp_path, monkeypatch):
    _put(monkeypatch, tmp_path, {
        "id": "on-merge", "name": "on-merge",
        "scenario": _scenario(
            {"id": "go", "kind": "trigger", "config": {"on": {"type": "event", "spec": "github.pull_request.*"}}},
            {"id": "work", "kind": "agent", "config": {"goal": "ship"}},
            edges=[{"id": "go->work", "source": "go", "target": "work"}],
        ),
    })
    first = deliver_event("github.pull_request.merged", {"n": 1}, background=False, execute_fn=_agent)
    second = deliver_event("github.pull_request.merged", {"n": 2}, background=False, execute_fn=_agent)
    assert len(first) == 1 and len(second) == 1 and first != second
    assert load_run(second[0])["outputs"]["work"]["seen"] == {"n": 2}
    assert deliver_event("github.issue.opened", {}, background=False, execute_fn=_agent) == []


def test_a_workflow_does_not_trigger_itself_from_its_own_events(tmp_path, monkeypatch):
    _put(monkeypatch, tmp_path, {
        "id": "loop", "name": "loop",
        "scenario": _scenario(
            {"id": "go", "kind": "trigger", "config": {"on": {"type": "event", "spec": "workflow.run.finished"}}},
            {"id": "work", "kind": "agent", "config": {"goal": "x"}},
            edges=[{"id": "1", "source": "go", "target": "work"}],
        ),
    })
    own = {"workflowId": "loop", "runId": "run-1-abc", "state": "succeeded"}
    assert deliver_event("workflow.run.finished", own, background=False, execute_fn=_agent) == []
    other = {**own, "workflowId": "someone-else"}
    assert len(deliver_event("workflow.run.finished", other, background=False, execute_fn=_agent)) == 1


def test_wait_event_resumes_on_a_glob(tmp_path, monkeypatch):
    _put(monkeypatch, tmp_path, {
        "id": "listen", "name": "listen",
        "scenario": _scenario(
            {"id": "hold", "kind": "wait", "config": {"until": {"type": "event", "spec": "github.pull_request.*"}}},
            {"id": "work", "kind": "agent", "config": {"goal": "continue"}},
            edges=[{"id": "hold->work", "source": "hold", "target": "work"}],
        ),
    })
    parked = start_run("listen", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_world"
    assert parked["parks"]["hold"]["event"] == "github.pull_request.*"
    deliver_event("github.pull_request.merged", {"merged": True}, background=False, execute_fn=_agent)
    done = load_run(parked["runId"])
    assert done["status"] == "succeeded"
    assert done["outputs"]["work"]["seen"] == {"merged": True}


def test_poll_url_resumes_when_the_world_answers(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from tools.url_safety import _reset_allow_private_cache

    class Ready(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Ready)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/ready"
    # Loopback is a private target: polling it is the explicit security opt-in, not the default.
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    _reset_allow_private_cache()
    set_execute_fn(_agent)
    try:
        _put(monkeypatch, tmp_path, {
            "id": "probe", "name": "probe",
            "scenario": _scenario(
                {"id": "hold", "kind": "wait", "config": {"until": {"type": "poll", "spec": f"every 1s {url}"}}},
                {"id": "work", "kind": "agent", "config": {"goal": "go"}},
                edges=[{"id": "hold->work", "source": "hold", "target": "work"}],
            ),
        })
        parked = start_run("probe", execute_fn=_agent, background=False)
        assert parked["status"] == "waiting_world" and parked["parks"]["hold"]["url"] == url
        done = _wait_for(parked["runId"], lambda s: s and s["status"] == "succeeded", timeout=5)
        assert done["status"] == "succeeded" and "work" in done["ran"]
    finally:
        set_execute_fn(None)
        server.shutdown()
        _reset_allow_private_cache()


def test_poll_spec_without_a_url_parks_on_the_event(tmp_path, monkeypatch):
    _put(monkeypatch, tmp_path, {
        "id": "poll", "name": "poll",
        "scenario": _scenario({"id": "hold", "kind": "wait", "config": {"until": {"type": "poll", "spec": "deploy.green"}}}),
    })
    parked = start_run("poll", execute_fn=_agent, background=False)
    assert parked["status"] == "waiting_world" and parked["parks"]["hold"]["event"] == "deploy.green"


def test_gate_routes_on_verdicts(tmp_path, monkeypatch):
    def judge(goal, _context, _payload, _config):
        return {"ok": True, "summary": "FAIL", "verdict": "FAIL", "output": {"verdict": "FAIL"}}

    _put(monkeypatch, tmp_path, {
        "id": "gated", "name": "gated",
        "scenario": _scenario(
            {"id": "check", "kind": "agent", "config": {"goal": "review"}},
            {"id": "gate", "kind": "gate", "config": {"arms": [
                {"id": "pass", "when": {"mode": "all-pass"}}, {"id": "loop", "when": {"mode": "any-fail"}},
            ]}},
            {"id": "ship", "kind": "agent", "config": {"goal": "open"}},
            {"id": "fix", "kind": "agent", "config": {"goal": "fix"}},
            edges=[
                {"id": "check->gate", "source": "check", "target": "gate"},
                {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
                {"id": "gate->fix", "source": "gate", "target": "fix", "sourceHandle": "loop"},
            ],
        ),
    })
    state = start_run("gated", execute_fn=judge, background=False)
    assert state["status"] == "succeeded"
    assert "fix" in state["ran"] and "ship" not in state["ran"]


def test_user_fixable_error_stops_instead_of_shipping(tmp_path, monkeypatch):
    def boom(_goal, _context, _payload, _config):
        return {"ok": False, "error": "HTTP 404: Model 'x' not found. The requested model does not exist."}

    _put(monkeypatch, tmp_path, {
        "id": "blocked", "name": "blocked",
        "scenario": _scenario(
            {"id": "work", "kind": "agent", "config": {"goal": "do it", "maxRetries": 2}},
            {"id": "ship", "kind": "agent", "config": {"goal": "open"}},
            edges=[{"id": "work->ship", "source": "work", "target": "ship"}],
        ),
    })
    state = start_run("blocked", execute_fn=boom, background=False)
    assert state["status"] == "failed" and "ship" not in state["ran"]
    assert "UserAsk" in _types(state["runId"])


def test_null_verdict_does_not_pass_the_gate(tmp_path, monkeypatch):
    """A step that never judged PASS/FAIL is not a pass."""

    def mute(_goal, _context, _payload, _config):
        return {"ok": True, "summary": "HTTP 404: model missing", "verdict": None, "output": {}}

    _put(monkeypatch, tmp_path, {
        "id": "gated", "name": "gated",
        "scenario": _scenario(
            {"id": "check", "kind": "agent", "config": {"goal": "review"}},
            {"id": "gate", "kind": "gate", "config": {"arms": [
                {"id": "pass", "when": {"mode": "all-pass"}}, {"id": "loop", "when": {"mode": "any-fail"}},
            ]}},
            {"id": "ship", "kind": "agent", "config": {"goal": "open"}},
            edges=[
                {"id": "check->gate", "source": "check", "target": "gate"},
                {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
            ],
        ),
    })
    state = start_run("gated", execute_fn=mute, background=False)
    assert "ship" not in state["ran"] and state["status"] == "failed"


def test_ready_agents_run_together(tmp_path, monkeypatch):
    first, second = threading.Event(), threading.Event()

    def pair(goal, _context, _payload, _config):
        if goal == "left":
            first.set()
            assert second.wait(2)
        else:
            assert first.wait(2)
            second.set()
        return {"ok": True, "summary": goal, "verdict": "PASS", "output": {"goal": goal}}

    _put(monkeypatch, tmp_path, {
        "id": "fan", "name": "fan",
        "scenario": _scenario(
            {"id": "left", "kind": "agent", "config": {"goal": "left"}},
            {"id": "right", "kind": "agent", "config": {"goal": "right"}},
        ),
    })
    state = start_run("fan", execute_fn=pair, background=False)
    assert state["status"] == "succeeded" and set(state["ran"]) == {"left", "right"}


def test_prose_gate_takes_the_pass_arm(tmp_path, monkeypatch):
    def fn(goal, _context, _payload, _config):
        if "ship it" in goal.lower():
            return {"ok": True, "summary": "PASS", "verdict": "PASS", "output": {}}
        return {"ok": True, "summary": "drafted", "verdict": "PASS", "output": {}}

    _put(monkeypatch, tmp_path, {
        "id": "prose", "name": "prose",
        "scenario": _scenario(
            {"id": "draft", "kind": "agent", "config": {"goal": "write"}},
            {"id": "gate", "kind": "gate", "config": {"arms": [
                {"id": "yes", "when": {"mode": "prose", "source": "Should we ship it?"}},
                {"id": "no", "when": {"mode": "any-fail"}},
            ]}},
            {"id": "open", "kind": "agent", "config": {"goal": "pr"}},
            {"id": "hold", "kind": "agent", "config": {"goal": "wait"}},
            edges=[
                {"id": "draft->gate", "source": "draft", "target": "gate"},
                {"id": "gate->open", "source": "gate", "target": "open", "sourceHandle": "yes"},
                {"id": "gate->hold", "source": "gate", "target": "hold", "sourceHandle": "no"},
            ],
        ),
    })
    state = start_run("prose", execute_fn=fn, background=False)
    assert state["status"] == "succeeded" and "open" in state["ran"] and "hold" not in state["ran"]


def test_inflight_is_restored_on_advance(tmp_path, monkeypatch):
    scenario = _scenario({"id": "work", "kind": "agent", "config": {"goal": "ship"}})
    _put(monkeypatch, tmp_path, {"id": "crash", "name": "crash", "scenario": scenario})
    save_run({
        "runId": "crash-1", "workflowId": "crash", "name": "crash", "scenario": scenario, "payload": {"n": 7},
        "source": "manual", "status": "running", "queue": [], "ran": [], "satisfied": [], "dormant": [],
        "verdicts": {}, "outputs": {}, "summaries": {}, "take": {}, "loops": 0, "parks": {},
        "pauseRequested": False, "seq": 0, "startedAt": 1, "failed": False, "tries": {},
        "inFlight": ["work"], "sessions": {"work": "wf-crash-1-work"},
    })
    state = advance("crash-1", execute_fn=_agent)
    assert state["status"] == "succeeded"
    assert state["outputs"]["work"]["seen"] == {"n": 7}
    assert state["sessions"]["work"] == "wf-crash-1-work"


def test_rework_loop_does_not_block_the_start_node(tmp_path, monkeypatch):
    """A loop-back is a rework wire, not an input."""
    _put(monkeypatch, tmp_path, {
        "id": "looped", "name": "looped",
        "scenario": _scenario(
            {"id": "work", "kind": "agent", "config": {"goal": "do it"}},
            {"id": "gate", "kind": "gate", "config": {"arms": [
                {"id": "pass", "when": {"mode": "all-pass"}}, {"id": "loop", "when": {"mode": "any-fail"}},
            ]}},
            {"id": "ship", "kind": "agent", "config": {"goal": "open"}},
            edges=[
                {"id": "work->gate", "source": "work", "target": "gate"},
                {"id": "gate->ship", "source": "gate", "target": "ship", "sourceHandle": "pass"},
                {"id": "gate->work", "source": "gate", "target": "work", "sourceHandle": "loop", "loop": True},
            ],
        ),
    })
    state = start_run("looped", execute_fn=_agent, background=False)
    assert "work" in state["ran"] and "ship" in state["ran"] and state["status"] == "succeeded"


def test_pause_holds_a_live_fake_run(tmp_path, monkeypatch):
    """The in-flight step stops at the pause and is neither done nor skipped."""
    _put(monkeypatch, tmp_path, {
        "id": "held", "name": "held",
        "scenario": _scenario(
            {"id": "implement", "kind": "agent", "config": {"goal": "do it"}},
            {"id": "next", "kind": "agent", "config": {"goal": "then"}},
            edges=[{"id": "implement->next", "source": "implement", "target": "next"}],
        ),
    })
    state = start_run("held", fake=True, background=True)
    run_id = state["runId"]
    assert _wait_for(run_id, lambda s: s and "implement" in (s.get("inFlight") or []))
    request_pause(run_id)
    parked = _wait_for(run_id, lambda s: s and s.get("status") == "paused")
    assert parked["status"] == "paused"
    assert "implement" not in parked["ran"] and "next" not in parked["ran"]


def test_play_replaces_a_dead_running_run(tmp_path, monkeypatch):
    """A 'running' row with no live loop is leftover from a killed process."""
    scenario = _scenario({"id": "work", "kind": "agent", "config": {"goal": "do it"}})
    _put(monkeypatch, tmp_path, {"id": "dead", "name": "dead", "scenario": scenario})
    save_run({
        "runId": "zombie-1", "workflowId": "dead", "name": "dead", "scenario": scenario, "payload": None,
        "source": "manual", "status": "running", "queue": ["work"], "ran": [], "satisfied": [], "dormant": [],
        "verdicts": {}, "outputs": {}, "summaries": {}, "take": {}, "loops": 0, "parks": {},
        "pauseRequested": False, "seq": 0, "startedAt": 1, "failed": False, "tries": {}, "inFlight": [], "sessions": {},
    })
    state = start_run("dead", execute_fn=_agent, background=False)
    assert state["runId"] != "zombie-1" and state["status"] == "succeeded"


def test_unready_queue_fails_instead_of_succeeding(tmp_path, monkeypatch):
    """A cycle with no loop flag has no start: a stuck graph, not a successful empty run."""
    _put(monkeypatch, tmp_path, {
        "id": "cycle", "name": "cycle",
        "scenario": _scenario(
            {"id": "a", "kind": "agent", "config": {"goal": "a"}},
            {"id": "b", "kind": "agent", "config": {"goal": "b"}},
            edges=[{"id": "a->b", "source": "a", "target": "b"}, {"id": "b->a", "source": "b", "target": "a"}],
        ),
    })
    state = start_run("cycle", execute_fn=_agent, background=False)
    assert state["status"] == "failed" and state["ran"] == []


def test_a_pinned_step_answers_without_running(tmp_path, monkeypatch):
    calls = []

    def counting(goal, context, payload, config):
        calls.append(goal)
        return _agent(goal, context, payload, config)

    _put(monkeypatch, tmp_path, {
        "id": "pinned", "name": "pinned",
        "scenario": _scenario(
            {"id": "go", "kind": "trigger", "config": {"on": {"type": "manual"}, "pinPayload": {"pr": 9}}},
            {"id": "fetch", "kind": "agent", "config": {"goal": "fetch", "pin": {
                "summary": "frozen", "verdict": "PASS", "output": {"rows": 3}}}},
            {"id": "use", "kind": "agent", "config": {"goal": "use"}},
            edges=[{"id": "1", "source": "go", "target": "fetch"}, {"id": "2", "source": "fetch", "target": "use"}],
        ),
    })
    state = start_run("pinned", execute_fn=counting, background=False)
    assert state["status"] == "succeeded"
    assert calls == ["use"]
    assert state["outputs"]["fetch"] == {"rows": 3}
    assert state["outputs"]["use"]["seen"] == {"pr": 9}
    assert "rows" in state["outputs"]["use"]["context"]
