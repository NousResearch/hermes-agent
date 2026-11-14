"""Goal/context text echoed back to the parent by background delegate_task is capped by
``delegation.max_task_echo_chars``: the parent already holds the full text in its own call."""
import copy
import json
import logging
import re
from types import SimpleNamespace

from tools.delegate_tool_dispatch import _dispatched_payload
from tools.process_registry_notifications import format_process_notification

GOAL = "Review the parser\n\n" + "step " * 120 + "GOAL_TAIL"
CTX = "Repo at /src.\n" + "rule " * 120 + "CTX_TAIL"
POINTER = "Task text in this message is cut to "
# One-line form of the desktop batch splitter (apps/desktop/src/lib/chat-messages/hydration.ts).
HEADER_RE = re.compile(r"^--- [✓✗⚠] TASK \d+/\d+(?:: [^\n]*)?  \(status=[^\n]*\) ---$", re.MULTILINE)


def _set_config(monkeypatch, cfg):
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: cfg)


def _batch_event(goals, context=CTX):
    return {"type": "async_delegation", "delegation_id": "deleg_x", "is_batch": True, "goals": goals,
            "context": context, "completed_at": 1000.0, "dispatched_at": 900.0,
            "results": [{"task_index": i, "status": "completed", "summary": f"summary {i}", "duration_seconds": 1.5}
                        for i in range(len(goals))]}


def _single_event(goal=GOAL, context=CTX):
    return {"type": "async_delegation", "delegation_id": "deleg_s", "goal": goal, "context": context,
            "status": "completed", "summary": "ok", "completed_at": 1000.0}


def _failure_event(goals):
    return {"type": "async_delegation", "delegation_id": "deleg_f", "task_failure_notice": True, "goals": goals,
            "results": [{"task_index": 0, "status": "failed", "error": "boom"}]}


def _receipt(goals):
    batch = SimpleNamespace(task_list=[{"goal": g} for g in goals],
                            children=[(i, {}, object()) for i in range(len(goals))],
                            live_deleg_id="deleg_x", refused=[], live_paths=[])
    return _dispatched_payload(batch, [(None, "deleg_x")])


def _render(evt) -> str:
    text = format_process_notification(evt)
    assert text is not None
    return text


def _lines_starting(text, prefix):
    return [line for line in text.splitlines() if line.startswith(prefix)]


def test_long_goal_and_context_are_capped_with_pointer(monkeypatch):
    _set_config(monkeypatch, {})
    flat_goal = " ".join(GOAL.split())

    evt = _batch_event([GOAL, "Short goal"])
    before = copy.deepcopy(evt)
    text = _render(evt)
    assert evt == before and evt["goals"][0] == GOAL
    assert "GOAL_TAIL" not in text and "CTX_TAIL" not in text
    assert _lines_starting(text, "--- ✓ TASK 1/2: " + flat_goal[:200] + "...  (status=completed")
    assert _lines_starting(text, "--- ✓ TASK 2/2: Short goal  (status=")
    (context_line,) = _lines_starting(text, "Context you provided: ")
    assert context_line.endswith("...")
    (pointer,) = _lines_starting(text, POINTER + "200 chars")
    assert "delegate_task call" in pointer and "async_delegations.task_json" in pointer and "deleg_x" in pointer
    assert len(HEADER_RE.findall(text)) == 2

    text = _render(_single_event())
    (goal_line,) = _lines_starting(text, "Original goal: ")
    assert goal_line.endswith("...")
    assert "GOAL_TAIL" not in text and "CTX_TAIL" not in text
    assert len(_lines_starting(text, POINTER)) == 1

    text = _render(_failure_event([GOAL, "b"]))
    (task_line,) = _lines_starting(text, "Task: ")
    assert task_line.endswith("...") and "GOAL_TAIL" not in text
    assert len(_lines_starting(text, POINTER)) == 1

    payload = _receipt([GOAL, "Short goal"])
    assert payload["goals"] == [flat_goal[:200] + "...", "Short goal"]
    assert "GOAL_TAIL" not in json.dumps(payload)
    assert payload["status"] == "dispatched" and payload["delegation_id"] == "deleg_x" and payload["note"]


def test_short_text_has_no_pointer_and_matches_verbatim_output(monkeypatch):
    evt = _batch_event(["Short goal one", "Short goal two"], context="Short context")
    _set_config(monkeypatch, {})
    capped = _render(evt)
    _set_config(monkeypatch, {"max_task_echo_chars": 0})
    verbatim = _render(evt)
    assert capped == verbatim
    assert POINTER not in capped


def test_zero_cap_echoes_verbatim(monkeypatch):
    _set_config(monkeypatch, {"max_task_echo_chars": 0})
    text = _render(_batch_event([GOAL, "Short goal"]))
    assert f"TASK 1/2: {GOAL}  (status=" in text
    assert f"Context you provided: {CTX}" in text
    assert POINTER not in text
    text = _render(_single_event())
    assert f"Original goal: {GOAL}" in text and POINTER not in text
    text = _render(_failure_event([GOAL, "b"]))
    assert f"Task: {GOAL}" in text and POINTER not in text
    assert _receipt([GOAL, "Short goal"])["goals"] == [GOAL, "Short goal"]


def test_cap_is_read_from_config_yaml(monkeypatch, tmp_path, caplog):
    from tools.delegate_tool_config import _get_max_task_echo_chars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("delegation:\n  max_task_echo_chars: 50\n", encoding="utf-8")
    assert _get_max_task_echo_chars() == 50
    text = _render(_batch_event([GOAL, "Short goal"]))
    assert "TASK 1/2: " + " ".join(GOAL.split())[:50] + "...  (status=" in text

    (tmp_path / "config.yaml").write_text('delegation:\n  max_task_echo_chars: "lots"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert _get_max_task_echo_chars() == 200
    assert "not a valid integer" in caplog.text
