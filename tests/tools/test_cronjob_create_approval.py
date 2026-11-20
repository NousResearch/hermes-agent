"""A model-initiated ``cronjob_manage`` create waits for the user's approval.

The proactive-scheduling guidance lets the model create jobs nobody asked for, so the job is shown on the
shared approval gate (CLI panel, TUI prompt, Desktop card, messaging buttons) before it is saved. A context
with nobody to answer keeps creating jobs as before. Driven through the real registry handler.
"""

import json

import pytest

import model_tools  # discovery registers cronjob_manage
from tools import terminal_tool
from tools.registry import registry

_CREATE = {"action": "create", "schedule": "every day at 8am", "repeat": 3, "name": "Water",
           "prompt": "Remind the user to drink a glass of water."}


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    from cron import jobs
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_CRON_SESSION", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    with jobs.use_cron_store(tmp_path):
        yield jobs
    terminal_tool.set_approval_callback(None)


def _asking(monkeypatch, answer):
    asked = []
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(command, description, *, kind="command", **_):
        asked.append((kind, description))
        return answer
    terminal_tool.set_approval_callback(callback)
    return asked


@pytest.mark.parametrize("answer, saved", [("deny", 0), ("once", 1)])
def test_interactive_create_is_saved_only_after_approval(cron_store, monkeypatch, answer, saved):
    asked = _asking(monkeypatch, answer)
    result = json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))
    # The schedule card, not the command card, on every surface that draws one.
    assert len(asked) == 1 and asked[0][0] == "schedule"
    assert "Water" in asked[0][1] and "3 runs" in asked[0][1]
    assert result.get("success") is (saved == 1)
    assert len(cron_store.load_jobs()) == saved


def test_create_with_nobody_to_answer_never_asks(cron_store, monkeypatch):
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_approval_callback(lambda *a, **k: pytest.fail("asked with nobody present"))
    assert json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))["success"] is True
    assert len(cron_store.load_jobs()) == 1
