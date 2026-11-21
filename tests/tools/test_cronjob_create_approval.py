"""A model-initiated ``cronjob_manage`` create waits for the user's approval.

The proactive-scheduling guidance lets the model create jobs nobody asked for, so the job is shown on the
shared approval gate (CLI panel, TUI prompt, Desktop card, messaging buttons) before it is saved. A context
where no card would be shown keeps creating jobs as before, and is never told to create unasked. Driven
through the real registry handler.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import model_tools  # discovery registers cronjob_manage
from tools import terminal_tool
from tools.registry import registry

_CREATE = {"action": "create", "schedule": "every day at 8am", "repeat": 3, "name": "Water",
           "prompt": "Remind the user to drink a glass of water."}


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    from cron import jobs
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_CRON_SESSION", "HERMES_SESSION_PLATFORM",
                 "HERMES_SINGLE_QUERY_SESSION", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    with jobs.use_cron_store(tmp_path):
        yield jobs
    terminal_tool.set_approval_callback(None)
    from tools.approval import clear_session
    from tools.approval_context import get_current_session_key
    clear_session(get_current_session_key())  # an "allow new jobs" grant must not outlive its test


def _asking(monkeypatch, answer, late=False):
    asked = []
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(command, description, *, kind="command", **_):
        asked.append((kind, description))
        if late:  # the user answers an hour after the card appeared
            from cron import jobs
            real_now = jobs._hermes_now
            monkeypatch.setattr(jobs, "_hermes_now", lambda: real_now() + timedelta(hours=1))
        return answer
    terminal_tool.set_approval_callback(callback)
    return asked


def _local(iso):
    from hermes_time import now as hermes_now
    return datetime.fromisoformat(iso).astimezone(hermes_now().tzinfo).strftime("%a %b %d %H:%M %Z")


@pytest.mark.parametrize("answer, saved, extra", [
    ("deny", 0, {}), ("once", 1, {}),
    # The card describes what create stores: a numeric-string repeat and the skills beside a prompt.
    ("once", 1, {"repeat": "3", "skill": "maps"}),
    # The job saves the first run the card showed, however long the answer takes (one-shot and recurring).
    ("once", 1, {"schedule": "in 2h", "repeat": None}), ("once", 1, {"schedule": "every 2h"}),
    # An answer arriving after the first run the card showed (salch-cred): stale, nothing saved, never
    # an error after consent and never a different time than the one approved.
    ("once", 0, {"schedule": "in 1m", "repeat": None}), ("once", 0, {"schedule": "every 2m"}),
])
def test_interactive_create_is_saved_only_after_approval(cron_store, monkeypatch, answer, saved, extra):
    asked = _asking(monkeypatch, answer, late="schedule" in extra)
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, **extra}))
    assert result.get("success") is (saved == 1)
    assert len(cron_store.load_jobs()) == saved
    # The schedule card, not the command card, on every surface that draws one.
    assert len(asked) == 1 and asked[0][0] == "schedule"
    if not saved:
        assert result.get("declined") is True
        return
    if "schedule" in extra:
        job = cron_store.load_jobs()[0]
        assert _local(job["next_run_at"]) in asked[0][1]
        return
    assert "Water" in asked[0][1] and "3 runs" in asked[0][1]
    if "skill" in extra:
        assert "skills: maps" in asked[0][1]


@pytest.mark.parametrize("extra", [
    {"timezone": "Not/AZone"}, {"no_agent": True},
    # Every rule create enforces runs before the card (GottZ 3.6): none of these may ask, then fail.
    {"workdir": "relative/dir"}, {"workdir": "/no/such/dir/anywhere"},
    {"paused_reason": "later"}, {"wrap_response": "sometimes"}, {"enabled_toolsets": "web"},
    {"schedule": "2020-01-01T00:00", "repeat": None},
])
def test_a_create_that_cannot_save_never_asks(cron_store, monkeypatch, extra):
    asked = _asking(monkeypatch, "once")
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, **extra}))
    assert result.get("success") is False
    assert asked == [] and cron_store.load_jobs() == []


@pytest.mark.parametrize("extra, shown", [
    # No name: the card names the job by the name it is saved under, not "new job".
    ({"name": None}, "Remind the user to drink a glass of water."),
    # Paused: the card says it will not run, never a first run.
    ({"paused": True}, "created paused"),
    ({"deliver": "telegram:12345"}, "results go to telegram:12345"),
    ({"monitor": "https://example.com/price"}, "checks URL https://example.com/price first"),
])
def test_the_card_shows_the_job_that_is_saved(cron_store, monkeypatch, extra, shown):
    asked = _asking(monkeypatch, "once")
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, **extra}))
    assert result["success"] is True, result
    assert shown in asked[0][1]
    if extra.get("paused"):
        assert "first run" not in asked[0][1]
    if "name" in extra:
        assert f'Schedule "{cron_store.load_jobs()[0]["name"]}"' in asked[0][1]


def test_the_card_says_who_sees_the_result(cron_store, monkeypatch):
    """Same job, different audience: the card text differs (ctaylor86), failures named when they diverge."""
    shown = []
    for extra in ({}, {"deliver": "telegram:-100123"}, {"deliver": "local", "failure_deliver": "telegram"}):
        asked = _asking(monkeypatch, "deny")
        registry.dispatch("cronjob_manage", {**_CREATE, **extra})
        shown.append(asked[0][1])
    assert "results go to " in shown[0]
    assert "results go to telegram:-100123" in shown[1]
    assert "failure notices go to telegram (home channel)" in shown[2]
    assert len(set(shown)) == 3


def test_an_iso_one_shot_with_a_zone_shows_when_it_actually_fires(cron_store, monkeypatch):
    asked = _asking(monkeypatch, "once")
    run_at = (datetime.now().astimezone() + timedelta(days=3)).replace(microsecond=0)
    args = {**_CREATE, "schedule": run_at.isoformat(), "repeat": None, "timezone": "Asia/Tokyo"}
    assert json.loads(registry.dispatch("cronjob_manage", args))["success"] is True
    saved = cron_store.load_jobs()[0]["next_run_at"]
    assert f"once, {_local(saved)}" in asked[0][1] and "Asia/Tokyo" not in asked[0][1]


def test_a_declined_job_is_an_answer_not_a_tool_failure(cron_store, monkeypatch):
    """Three denied jobs must not trip the loop guardrail's "keep using tools" warning (GottZ 3.5), and the
    model-facing text is the schedule wording, not "potentially dangerous" (askclaw-vesper)."""
    from agent.display import _detect_tool_failure
    from agent.tool_guardrails import ToolCallGuardrailController
    _asking(monkeypatch, "deny")
    guard = ToolCallGuardrailController()
    for i in range(4):
        args = {**_CREATE, "name": f"Job {i}"}
        raw = registry.dispatch("cronjob_manage", args)
        assert json.loads(raw)["declined"] is True
        assert _detect_tool_failure("cronjob_manage", raw)[0] is False
        assert guard.after_call("cronjob_manage", args, raw).action == "allow"
    assert "dangerous" not in json.loads(raw)["error"]


def _mode_off(monkeypatch):
    from tools import approval_context
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")


def _api_server_stream(monkeypatch):
    """A streaming /v1/chat/completions turn: the gateway's ask flag plus a registered notifier."""
    from tools import approval
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    monkeypatch.setattr(approval, "_gateway_notify_cb", lambda _k: (lambda *_a: None))


def _kanban_worker(monkeypatch):
    from gateway import session_policy
    monkeypatch.setattr(session_policy, "active_policy", lambda: SimpleNamespace(source="kanban"))


_NO_CARD_CONTEXTS = {
    "no_ui": lambda mp: mp.delenv("HERMES_INTERACTIVE"), "mode_off": _mode_off,
    "api_server": _api_server_stream, "kanban": _kanban_worker,
}


def _no_card_context(monkeypatch, how):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    _NO_CARD_CONTEXTS[how](monkeypatch)


@pytest.mark.parametrize("how", ["no_ui", "mode_off", "api_server", "kanban"])
def test_where_no_card_is_shown_the_job_saves_and_no_line_urges_one(cron_store, monkeypatch, how, caplog):
    """The gate and the proactive prompt line read one predicate (GottZ 3.1/3.2, salch-cred 3): wherever
    a create would save without a card, the line telling the model to create unasked is withheld too."""
    import logging
    from agent.system_prompt import _cron_guidance
    _no_card_context(monkeypatch, how)
    terminal_tool.set_approval_callback(lambda *a, **k: pytest.fail("asked where no card is shown"))
    with caplog.at_level(logging.INFO, logger="tools.cronjob_tools"):
        assert json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))["success"] is True
    assert len(cron_store.load_jobs()) == 1
    if how == "mode_off":
        assert "approvals are off" in caplog.text
    if how == "no_ui":  # a library embed: live presence is bound per turn, after the prompt is built
        return
    agent = SimpleNamespace(platform="api_server" if how == "api_server" else "cli",
                            valid_tool_names={"cronjob_manage"}, _oneshot_session=False)
    assert _cron_guidance(agent) is None


def test_an_interactive_session_gets_the_line_and_the_card(cron_store, monkeypatch):
    from agent.system_prompt import _cron_guidance
    asked = _asking(monkeypatch, "deny")
    agent = SimpleNamespace(platform="cli", valid_tool_names={"cronjob_manage"}, _oneshot_session=False)
    assert "create a job" in (_cron_guidance(agent) or "")
    registry.dispatch("cronjob_manage", dict(_CREATE))
    assert len(asked) == 1


def test_turning_the_guidance_off_drops_the_line_and_keeps_the_card(cron_store, monkeypatch):
    from agent.system_prompt import _cron_guidance
    from hermes_cli import config as hermes_config
    real = hermes_config.load_config_readonly
    monkeypatch.setattr(hermes_config, "load_config_readonly",
                        lambda *a, **k: {**real(*a, **k), "cron": {"proactive_guidance": False}})
    asked = _asking(monkeypatch, "deny")
    agent = SimpleNamespace(platform="cli", valid_tool_names={"cronjob_manage"}, _oneshot_session=False)
    assert _cron_guidance(agent) is None
    result = json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))
    assert len(asked) == 1 and result["declined"] is True
    assert cron_store.load_jobs() == []


# --- A model-initiated update asks on a "Change this job?" card -------------------------------------

def _existing_job(cron_store):
    return cron_store.create_job(prompt="Remind the user to drink a glass of water.", schedule="every day at 8am",
                                 name="Water", repeat=3)


def test_a_declined_edit_leaves_the_job_byte_identical(cron_store, monkeypatch):
    job = _existing_job(cron_store)
    before = cron_store.load_jobs()
    asked = _asking(monkeypatch, "deny")
    result = json.loads(registry.dispatch("cronjob_manage", {
        "action": "update", "job_id": job["id"], "schedule": "every day at 9am", "prompt": "Drink tea instead."}))
    assert result["success"] is False and result["declined"] is True
    assert cron_store.load_jobs() == before
    kind, card = asked[0]
    assert kind == "schedule_edit" and card.startswith('Change "Water": ')
    assert "prompt: Remind the user to drink a glass of water. -> Drink tea instead." in card
    assert "schedule: " in card and " -> " in card
    from agent.display import _detect_tool_failure
    assert _detect_tool_failure("cronjob_manage", json.dumps(result))[0] is False


def test_an_approved_edit_is_applied_and_offers_no_standing_grant(cron_store, monkeypatch):
    job = _existing_job(cron_store)
    seen = {}
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(command, description, *, kind="command", **kw):
        seen.update(kind=kind, **kw)
        return "once"
    terminal_tool.set_approval_callback(callback)
    result = json.loads(registry.dispatch("cronjob_manage", {"action": "update", "job_id": job["id"], "repeat": 5}))
    assert result["success"] is True
    assert cron_store.get_job(job["id"])["repeat"]["times"] == 5
    # Each change is its own consent: no "allow new jobs" scope on an edit card.
    assert seen["kind"] == "schedule_edit" and seen.get("allow_session") is False and seen["allow_permanent"] is False


def test_allowing_new_jobs_does_not_allow_edits(cron_store, monkeypatch):
    from tools.approval import approve_session
    from tools.approval_context import get_current_session_key
    job = _existing_job(cron_store)
    asked = _asking(monkeypatch, "deny")
    approve_session(get_current_session_key(), "cron_create")
    assert json.loads(registry.dispatch("cronjob_manage", dict(_CREATE)))["success"] is True  # new job: granted
    registry.dispatch("cronjob_manage", {"action": "update", "job_id": job["id"], "repeat": 9})
    assert [k for k, _ in asked] == ["schedule_edit"]
    assert cron_store.get_job(job["id"])["repeat"]["times"] == 3


@pytest.mark.parametrize("args", [
    {"action": "pause"}, {"action": "resume"},
    # Fields the card does not list are not asked about.
    {"action": "update", "name": "Hydrate"},
])
def test_lifecycle_actions_and_unlisted_fields_stay_unasked(cron_store, monkeypatch, args):
    job = _existing_job(cron_store)
    terminal_tool.set_approval_callback(lambda *a, **k: pytest.fail("asked for a lifecycle action"))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    assert json.loads(registry.dispatch("cronjob_manage", {**args, "job_id": job["id"]}))["success"] is True


def test_with_nobody_to_answer_an_edit_applies_as_before(cron_store, monkeypatch):
    job = _existing_job(cron_store)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_approval_callback(lambda *a, **k: pytest.fail("asked with nobody present"))
    result = json.loads(registry.dispatch("cronjob_manage", {"action": "update", "job_id": job["id"], "repeat": 5}))
    assert result["success"] is True and cron_store.get_job(job["id"])["repeat"]["times"] == 5


def test_an_invalid_edit_fails_before_asking(cron_store, monkeypatch):
    job = _existing_job(cron_store)
    asked = _asking(monkeypatch, "once")
    result = json.loads(registry.dispatch("cronjob_manage", {
        "action": "update", "job_id": job["id"], "workdir": "relative/dir"}))
    assert result["success"] is False and asked == []


@pytest.mark.parametrize("change, shown", [
    # Changes WHEN it runs without touching the schedule wording.
    ({"timezone": "Asia/Tokyo"}, "timezone: profile default -> Asia/Tokyo"),
    # Changes WHAT runs: an agent task becomes a bare script.
    ({"no_agent": True, "script": "check.py"}, "what runs: agent task -> script only"),
    ({"failure_deliver": "telegram"}, "failure notices to: (none) -> telegram"),
])
def test_an_edit_that_changes_when_what_or_where_asks(cron_store, monkeypatch, change, shown):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "scripts").mkdir(parents=True, exist_ok=True)
    (get_hermes_home() / "scripts" / "check.py").write_text("print('ok')\n")
    job = _existing_job(cron_store)
    asked = _asking(monkeypatch, "deny")
    result = json.loads(registry.dispatch("cronjob_manage", {"action": "update", "job_id": job["id"], **change}))
    assert result.get("declined") is True, result
    assert [k for k, _ in asked] == ["schedule_edit"] and shown in asked[0][1]


def test_a_run_finished_while_the_edit_card_waits_is_kept(cron_store, monkeypatch):
    """The card waits; a run completing meanwhile must survive the save, or the job gains extra runs."""
    job = _existing_job(cron_store)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def callback(*_a, **_k):
        cron_store.update_job(job["id"], {"repeat": {"times": 3, "completed": 2}})  # the scheduler ran twice
        return "once"
    terminal_tool.set_approval_callback(callback)
    assert json.loads(registry.dispatch("cronjob_manage", {"action": "update", "job_id": job["id"], "repeat": 5}))[
        "success"] is True
    assert cron_store.get_job(job["id"])["repeat"] == {"times": 5, "completed": 2}


@pytest.mark.parametrize("home, shown", [
    ("12345", "results go to the telegram home channel"),
    ("", "results go to nowhere (saved to the job's history only)"),
])
def test_a_job_with_no_chat_to_return_to_names_where_its_results_go(cron_store, monkeypatch, home, shown):
    """A CLI job has no captured origin: deliver=origin lands on a home channel, or nowhere, never "this chat"."""
    from cron import scheduler_delivery
    monkeypatch.setattr(scheduler_delivery, "_iter_home_target_platforms", lambda: iter(["telegram"]))
    monkeypatch.setattr(scheduler_delivery, "_get_home_target_chat_id", lambda _p: home)
    asked = _asking(monkeypatch, "deny")
    result = json.loads(registry.dispatch("cronjob_manage", {**_CREATE, "deliver": "origin"}))
    assert result.get("declined") is True, result
    assert shown in asked[0][1] and "this chat" not in asked[0][1]
