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


def _asking(monkeypatch, answer, late=False):
    asked = []
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(command, description, *, kind="command", **_):
        asked.append((kind, description))
        if late:  # the user answers an hour after the card appeared
            from datetime import timedelta
            from cron import jobs
            real_now = jobs._hermes_now
            monkeypatch.setattr(jobs, "_hermes_now", lambda: real_now() + timedelta(hours=1))
        return answer
    terminal_tool.set_approval_callback(callback)
    return asked


_GATE = 'import json\nprint(json.dumps({"wakeAgent": False}))\n'


@pytest.mark.parametrize("answer, saved, extra", [
    ("deny", 0, {}), ("once", 1, {}),
    # A script the call saves is consent text: the card shows its whole body, and a denied
    # create leaves no file behind.
    ("once", 1, {"script_content": _GATE}), ("deny", 0, {"script_content": _GATE}),
    # The card describes what create stores: a numeric-string repeat and the skills beside a prompt.
    ("once", 1, {"repeat": "3", "skill": "maps"}),
    # The job saves the first run the card showed, however long the answer takes (one-shot and recurring).
    ("once", 1, {"schedule": "in 2h", "repeat": None}), ("once", 1, {"schedule": "every 2h"}),
    # A create that would fail validation never asks: the user is not shown a job that cannot save.
    ("once", 0, {"timezone": "Not/AZone"}), ("once", 0, {"no_agent": True}),
])
def test_interactive_create_is_saved_only_after_approval(cron_store, monkeypatch, answer, saved, extra):
    asked = _asking(monkeypatch, answer, late="schedule" in extra)
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, **extra}))
    assert result.get("success") is (saved == 1)
    assert len(cron_store.load_jobs()) == saved
    if extra.keys() & {"timezone", "no_agent"}:
        assert asked == []
        return
    # The schedule card, not the command card, on every surface that draws one.
    assert len(asked) == 1 and asked[0][0] == "schedule"
    if "schedule" in extra:
        job = cron_store.load_jobs()[0]
        card_first = asked[0][1].split("first run ")[1].split(")")[0].split(",")[0]
        from datetime import datetime
        from hermes_time import now as hermes_now
        saved = datetime.fromisoformat(job["next_run_at"]).astimezone(hermes_now().tzinfo)
        assert saved.strftime("%a %b %d %H:%M %Z") == card_first
        return
    assert "Water" in asked[0][1] and "3 runs" in asked[0][1]
    if "skill" in extra:
        assert "skills: maps" in asked[0][1]
    elif extra:
        from hermes_constants import get_hermes_home
        assert "--- script cron_water_" in asked[0][1] and asked[0][1].endswith(_GATE)
        scripts = list((get_hermes_home() / "scripts").glob("cron_water_*.py"))
        assert len(scripts) == saved and all(p.read_text() == _GATE for p in scripts)


def test_create_with_nobody_to_answer_never_asks(cron_store, monkeypatch):
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_approval_callback(lambda *a, **k: pytest.fail("asked with nobody present"))
    assert json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))["success"] is True
    assert len(cron_store.load_jobs()) == 1


def test_denied_create_keeps_a_script_another_job_saved_meanwhile(cron_store, monkeypatch):
    """Chat A waits on its card while chat B saves a job on the same script; A's deny must not delete it."""
    from hermes_constants import get_hermes_home
    from tools.cronjob_tools import cronjob

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(command, description, **_):
        assert json.loads(cronjob(action="create", schedule="every 2h", name="B", prompt="p",
                                  script="shared.py", script_content=_GATE))["success"]
        return "deny"
    terminal_tool.set_approval_callback(callback)
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, "script": "shared.py", "script_content": _GATE}))
    assert result.get("success") is False
    assert [j["name"] for j in cron_store.load_jobs()] == ["B"]
    assert (get_hermes_home() / "scripts" / "shared.py").read_text() == _GATE
    # list never writes a script, even when handed script_content
    registry.dispatch("cronjob_manage", {"action": "list", "script_content": _GATE, "script": "listed.py"})
    assert not (get_hermes_home() / "scripts" / "listed.py").exists()
