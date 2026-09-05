"""workflow.store / workflow.run JSON-RPC — documents on disk, runs emit events."""

import tui_gateway.server as srv
from workflow.store import load_documents


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def test_store_put_then_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    put = _result(
        srv._methods["workflow.store.put"](
            1,
            {
                "docs": [{"id": "one", "name": "One", "scenario": {"steps": [], "edges": []}}],
                "currentId": "one",
            },
        )
    )
    assert put["currentId"] == "one"
    listed = _result(srv._methods["workflow.store.list"](2, {}))
    assert listed["docs"][0]["id"] == "one"
    assert load_documents()["docs"][0]["name"] == "One"


def test_run_start_walks_a_stub_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from workflow import runner

    monkeypatch.setattr(
        runner,
        "_execute_fn",
        lambda goal, context, payload, config: {
            "ok": True,
            "summary": "ok",
            "verdict": "PASS",
            "output": {"goal": goal},
        },
    )
    _result(
        srv._methods["workflow.store.put"](
            1,
            {
                "docs": [
                    {
                        "id": "job",
                        "name": "Job",
                        "scenario": {
                            "steps": [{"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "go"}}],
                            "edges": [],
                        },
                    }
                ],
                "currentId": "job",
            },
        )
    )
    started = _result(srv._methods["workflow.run.start"](2, {"workflowId": "job"}))
    assert started["runId"]
    # The method returns as soon as the thread is spawned; wait for the log.
    import time

    from workflow.store import load_events, load_run

    deadline = time.time() + 2
    state = load_run(started["runId"])
    while time.time() < deadline and state and state.get("status") in {"running", "paused"}:
        time.sleep(0.05)
        state = load_run(started["runId"])
    events = load_events(started["runId"])
    assert any(e["type"] == "RunStarted" for e in events)
