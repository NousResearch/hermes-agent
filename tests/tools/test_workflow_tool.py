"""The `workflow` tool's half of the canvas bridge.

Everything the graph itself means — which ops exist, what they accept, whether
an edit leaves a runnable workflow — lives in the plugin, deliberately, so it
can't drift from the canvas. What's testable here is the guard rail in front of
the round-trip: refuse the calls that can only waste a 30s block, forward the
rest verbatim, and turn every failure mode of the bridge into something the
model can act on.
"""

import json

import pytest

from tools import workflow_tools as wf


def call(**kwargs) -> dict:
    return json.loads(wf.workflow_tool(**kwargs))


def test_requires_callback():
    """Outside the desktop GUI there is no canvas — a clear error, no crash."""
    assert "error" in call(action="read", callback=None)


def test_run_does_not_need_the_canvas(tmp_path, monkeypatch):
    """A run is the gateway walking the stored graph, not a renderer round-trip."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from workflow.store import save_documents

    save_documents(
        [
            {
                "id": "job",
                "name": "Job",
                "scenario": {
                    "steps": [{"id": "work", "kind": "agent", "config": {"title": "Work", "goal": "go"}}],
                    "edges": [],
                },
            }
        ],
        "job",
    )
    from workflow import runner

    monkeypatch.setattr(
        runner,
        "_execute_fn",
        lambda goal, context, payload, config: {
            "ok": True,
            "summary": "ok",
            "verdict": "PASS",
            "output": {},
        },
    )
    result = call(action="run", workflow="job")
    assert "error" not in result
    assert result["runId"]
    assert result["workflow"] == "job"


def test_run_needs_a_workflow():
    result = call(action="run")
    assert "error" in result
    assert "run" in result["error"]


def test_rejects_an_unknown_action():
    result = call(action="redraw", callback=lambda _p: "{}")
    assert "error" in result
    assert "read" in result["error"]


def test_a_missing_action_is_named_not_guessed():
    """Defaulting to `read` would hide a malformed call behind a plausible one."""
    assert "error" in call(callback=lambda _p: "{}")


@pytest.mark.parametrize("verb", ["open", "create"])
def test_naming_verbs_need_a_workflow(verb):
    result = call(action=verb, callback=lambda _p: "{}")
    assert "error" in result
    assert verb in result["error"]


def test_edit_needs_ops():
    """An empty batch is a no-op the model should be told about, not sent."""
    for ops in (None, [], "graph_add_step"):
        result = call(action="edit", ops=ops, callback=lambda _p: "{}")
        assert "error" in result
        assert "graph_add_step" in result["error"], "the error should show the op shape"


def test_edit_ops_must_name_a_tool():
    result = call(action="edit", ops=[{"args": {"kind": "agent"}}], callback=lambda _p: "{}")
    assert "error" in result
    assert "tool" in result["error"]


def test_payload_forwards_only_what_was_given():
    """A `None` must not reach the renderer as a key — it reads as an argument
    the model supplied and can't be told apart from a deliberate null."""
    seen = {}
    wf.workflow_tool(action="read", callback=lambda p: seen.update(p) or "{}")
    assert seen == {"action": "read"}


def test_an_edit_forwards_its_batch_intact():
    seen = {}
    ops = [
        {"tool": "graph_add_step", "args": {"kind": "agent", "title": "Lint"}},
        {"tool": "graph_connect", "args": {"source": "review", "target": "lint"}},
    ]
    wf.workflow_tool(action="edit", ops=ops, callback=lambda p: seen.update(p) or "{}")
    assert seen == {"action": "edit", "ops": ops}


def test_the_action_is_normalized_before_the_round_trip():
    seen = {}
    wf.workflow_tool(action="  READ  ", callback=lambda p: seen.update(p) or "{}")
    assert seen["action"] == "read"


def test_passes_the_canvas_answer_straight_through():
    answer = {
        "workflow": {"id": "w1", "name": "Ship"},
        "scenario": {"steps": [{"id": "review", "kind": "agent"}], "edges": []},
        "problems": [],
    }
    assert call(action="read", callback=lambda _p: json.dumps(answer)) == answer


def test_wraps_non_json_text():
    assert call(action="read", callback=lambda _p: "plain words") == {"text": "plain words"}


def test_empty_answer_means_nobody_is_listening():
    """The block timed out or the plugin is off — say which door to open."""
    result = call(action="read", callback=lambda _p: "")
    assert "error" in result
    assert "Workflows" in result["error"]


def test_a_broken_bridge_is_reported_not_raised():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    result = call(action="read", callback=_boom)
    assert "error" in result
    assert "renderer went away" in result["error"]
