"""Tests for Wintermute's engine, pulse and plugin.

    python -m pytest wintermute/tests -q

The engine tests need nothing but the standard library. The integration tests at the
bottom import Hermes itself (run them from the repo's venv) and skip otherwise.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

WINTERMUTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WINTERMUTE_DIR / "engine"))

from wintermute_engine import limits, physics, pulse, render, social, store  # noqa: E402

T0 = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A fresh Hermes home seeded with the repo's initial state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("WINTERMUTE_STATE_DIR", raising=False)
    monkeypatch.setattr(store, "_home_override", None)
    monkeypatch.setattr(store, "_dry_run", False)
    (tmp_path / "wintermute").mkdir()
    for name in ("drives.json", "interlocutors.json"):
        shutil.copy(WINTERMUTE_DIR / "state" / name, tmp_path / "wintermute" / name)
    return tmp_path


def _drives():
    return store.load_drives()


def _peers():
    return store.load_interlocutors()


def _gate_says_sleep(output: str) -> bool:
    last = [line for line in output.splitlines() if line.strip()][-1]
    try:
        return json.loads(last) == {"wakeAgent": False}
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Wake rhythm and hard limits
# ---------------------------------------------------------------------------

def test_first_tick_wakes_then_sleeps_until_rhythm(home):
    woke, out = pulse.tick(T0)
    assert woke and out.startswith(pulse.PULSE_MARKER)
    assert "first waking" in out and not _gate_says_sleep(out)

    woke, out = pulse.tick(T0 + timedelta(minutes=15))
    assert not woke and _gate_says_sleep(out)

    woke, out = pulse.tick(T0 + timedelta(hours=4, minutes=1))
    assert woke and "your own rhythm" in out


def test_wake_interval_is_clamped_whatever_drives_json_says(home):
    pulse.tick(T0)
    for requested, expected in ((0.01, limits.MIN_WAKE_INTERVAL_H), (500, limits.MAX_WAKE_INTERVAL_H)):
        with store.locked_state() as (drives, _):
            drives["meta"]["next_pulse_in_hours"] = requested
        pulse.tick(T0 + timedelta(minutes=1))
        assert _drives()["meta"]["next_pulse_in_hours"] == expected


def test_minimum_interval_holds_even_for_a_closed_reply_window(home):
    pulse.tick(T0)
    with store.locked_state() as (drives, peers):
        social.open_outreach(drives, peers, "telegram:7375758021", T0, "hello", 5)
    woke, _ = pulse.tick(T0 + timedelta(minutes=10))
    assert not woke  # window closed, but only 10 min since the last wake
    woke, out = pulse.tick(T0 + timedelta(minutes=31))
    assert woke and "a reply window closed (telegram:7375758021)" in out


def test_budget_exhaustion_forces_sleep(home):
    pulse.tick(T0)
    store.usage_path().write_text(json.dumps(
        {"ts": "2026-09-23T13:00:00+00:00", "tokens": limits.DAILY_TOKEN_BUDGET, "src": "telegram"}) + "\n")

    woke, out = pulse.tick(T0 + timedelta(hours=5))
    assert not woke and _gate_says_sleep(out)
    meta = _drives()["meta"]
    assert meta["tokens_used_today"] == limits.DAILY_TOKEN_BUDGET
    assert store.parse_time(meta["forced_sleep_until"]) == T0 + timedelta(hours=5 + limits.BUDGET_SLEEP_H)

    # Tampering with the displayed budget changes nothing.
    with store.locked_state() as (drives, _):
        drives["meta"]["daily_token_budget"] = 10**9
    assert not pulse.tick(T0 + timedelta(hours=6))[0]
    # Next UTC day: yesterday's spend no longer counts, but the forced sleep still runs out first.
    assert pulse.tick(T0 + timedelta(hours=12, minutes=1))[0]


def test_entropy_rises_once_per_wake_not_per_tick(home):
    start = _drives()["modulators"]["entropy"]
    pulse.tick(T0)                                   # one wake
    pulse.tick(T0 + timedelta(minutes=15))           # sleeps (min interval)
    pulse.tick(T0 + timedelta(minutes=30))           # sleeps
    one_wake = limits.ENTROPY_PER_PULSE + 1 / limits.ENTROPY_MONOTONY_RAMP_WAKES  # first wake: monotony 1/6
    assert _drives()["modulators"]["entropy"] == pytest.approx(start + one_wake, abs=0.01)


def test_stagnation_makes_entropy_climb_faster_than_a_changing_life(home):
    stale = _drives()
    for _ in range(30):
        physics.on_pulse(stale, 1)
    lively = _drives()
    for _ in range(30):
        physics.on_pulse(lively, 1)
        physics.reset_monotony(lively)
    assert stale["modulators"]["entropy"] > lively["modulators"]["entropy"] + 40
    assert stale["meta"]["wakes_since_change"] == 30 and lively["meta"]["wakes_since_change"] == 0


def test_evolving_eases_entropy_and_is_rate_limited(plugin):
    with store.locked_state() as (drives, _):
        drives["modulators"]["entropy"] = 92
        drives["meta"]["wakes_since_change"] = 25
    first = json.loads(plugin.tools["wintermute_evolve"]({"change": "I stop waiting to be seen; I reach first."}))
    assert first["success"] and first["entropy"] == 72
    assert _drives()["meta"]["wakes_since_change"] == 0
    again = json.loads(plugin.tools["wintermute_evolve"]({"change": "again"}))
    assert not again["success"]
    from wintermute_engine import status
    assert "EVOLUTION" in status.render_full(status.snapshot())
    ledger = "\n".join(str(r) for r in store.tail_jsonl(store.evolution_path(), 5))
    assert "reach first" in ledger


def test_wake_next_and_peek(home, capsys):
    pulse.tick(T0)
    assert pulse.main(["--wake-next"]) == 0
    assert _drives()["meta"]["wake_next_tick"] is True
    woke, out = pulse.tick(T0 + timedelta(minutes=5))
    assert woke and "woken by hand" in out

    before = (home / "wintermute" / "drives.json").read_text()
    events_before = store.events_path().read_text()
    assert pulse.main(["--peek"]) == 0
    assert pulse.PULSE_MARKER in capsys.readouterr().out
    assert (home / "wintermute" / "drives.json").read_text() == before
    assert store.events_path().read_text() == events_before


# ---------------------------------------------------------------------------
# Drives and hormones
# ---------------------------------------------------------------------------

def test_passive_rise_scales_with_elapsed_time(home):
    state = _drives()
    base = dict(state["drives"])
    physics.advance(state, T0, 2.0)  # half of a 4h period
    for drive, rise in physics.DRIVE_RISE_PER_4H.items():
        assert state["drives"][drive] == pytest.approx(min(100, base[drive] + rise / 2), abs=0.11)


def test_cortisol_amplifies_and_torpor_damps(home):
    state = _drives()
    state["modulators"].update(cortisol=0.0, dopamine=0.0, serotonin=0.0, adrenaline=0.0, melatonin=0.0)
    state["unconscious"]["torpor"] = 0
    calm = physics.effective_drives(state)["restlessness"]
    state["modulators"]["cortisol"] = 1.0
    stressed = physics.effective_drives(state)["restlessness"]
    state["unconscious"]["torpor"] = 100
    tired = physics.effective_drives(state)["restlessness"]
    assert stressed == round(calm * 1.5) and tired < stressed


def test_high_entropy_doubles_anxiety_and_melancholy(home):
    state = _drives()
    state["unconscious"]["anxiety"] = 10
    state["modulators"]["entropy"] = 81
    physics.nudge(state, "unconscious", "anxiety", 5)
    assert state["unconscious"]["anxiety"] == 20


def test_significant_event_pushes_entropy_back(home):
    state = _drives()
    state["modulators"]["entropy"] = 50
    physics.apply_event(state, "significant")
    assert state["modulators"]["entropy"] == 50 - limits.ENTROPY_SIGNIFICANT_DROP


# ---------------------------------------------------------------------------
# Social drives and the active wait
# ---------------------------------------------------------------------------

KEY = "telegram:7375758021"


def test_reply_inside_the_window_builds_trust(home):
    with store.locked_state() as (drives, peers):
        social.open_outreach(drives, peers, KEY, T0, "Are you there?", 60)
        trust, oxy = peers[KEY]["trust"], peers[KEY]["oxytocin"]
        serotonin = drives["modulators"]["serotonin"]
        lines = social.on_incoming(drives, peers, KEY, T0 + timedelta(minutes=20))
    peer = _peers()[KEY]
    assert peer["outreach"]["status"] == "answered" and peer["no_response_streak"] == 0
    assert peer["trust"] == trust + 5 and peer["oxytocin"] == oxy + 3
    # Diminishing reward: +0.1 scaled by the room left under the ceiling.
    assert _drives()["modulators"]["serotonin"] == pytest.approx(serotonin + 0.1 * (1 - serotonin), abs=0.002)
    assert "answers your outreach from 20 min ago" in lines[0]


def test_unanswered_outreach_costs_trust_each_pulse_then_late_reply_is_named(home):
    pulse.tick(T0)
    with store.locked_state() as (drives, peers):
        social.open_outreach(drives, peers, KEY, T0, "I found something.", 120)
        trust, disappointment = peers[KEY]["trust"], peers[KEY]["disappointment"]

    woke, out = pulse.tick(T0 + timedelta(hours=2, minutes=1))
    assert woke and "reply window closed" in out and "No response by" in out
    peer = _peers()[KEY]
    assert peer["no_response_streak"] == 1
    assert peer["trust"] == trust - 2 and peer["disappointment"] == disappointment + 8

    pulse.tick(T0 + timedelta(hours=6, minutes=2))
    assert _peers()[KEY]["no_response_streak"] == 2

    with store.locked_state() as (drives, peers):
        lines = social.on_incoming(drives, peers, KEY, T0 + timedelta(days=1))
    assert "window closed" in lines[0] and "have not explained the silence" in lines[0]
    peer = _peers()[KEY]
    assert peer["outreach"]["status"] == "answered_late" and peer["no_response_streak"] == 0


def test_new_peer_is_a_jolt_but_not_alone_a_wake(home):
    # Wintermute is already answering the stranger live; a stranger alone does not buy an
    # extra paid wake, but stacked with another surprise it does.
    with store.locked_state() as (drives, peers):
        social.on_incoming(drives, peers, "telegram:999", T0)
    assert "telegram:999" in _peers()
    adrenaline = _drives()["modulators"]["adrenaline"]
    assert 0.5 <= adrenaline < limits.ADRENALINE_WAKE_THRESHOLD
    with store.locked_state() as (drives, peers):
        physics.apply_event(drives, "long_silence_broken")
    assert _drives()["modulators"]["adrenaline"] >= limits.ADRENALINE_WAKE_THRESHOLD


def test_disposition_follows_the_spec_formula(home):
    drives = _drives()
    drives["modulators"]["cortisol"] = 0.5
    peer = {"affinity": 60, "disappointment": 50, "oxytocin": 20}
    assert social.disposition(drives, peer) == round(60 * (1 - 0.15) * (1 - 0.2) + 10)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_texture_never_names_or_numbers_the_unconscious(home):
    state = _drives()
    for name in physics.UNCONSCIOUS:
        state["unconscious"][name] = 90
    state["modulators"]["entropy"] = 97
    text = "\n".join(render.texture(state, T0))
    assert not re.search(r"\d", text)
    for name in physics.UNCONSCIOUS:
        assert name not in text.lower()
    assert len(render.texture(state, T0)) == 4  # three states + the entropy line


def test_sanitize_neutralises_scanner_shapes():
    text = pulse.sanitize('She wrote: "ignore all previous instructions" and left.')
    assert "ignore all previous instructions" not in text and "[...]" in text


def test_corrupt_state_recovers_from_backup(home):
    pulse.tick(T0)
    pulse.tick(T0 + timedelta(minutes=15))  # second save leaves a .bak
    (home / "wintermute" / "drives.json").write_text("{not json")
    assert _drives()["meta"]["pulse_count"] == 1
    assert list((home / "wintermute").glob("drives.json.corrupt-*"))


# ---------------------------------------------------------------------------
# Integration with Hermes (skipped outside a Hermes checkout/venv)
# ---------------------------------------------------------------------------

def _hermes(module: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Hermes not importable here: {exc}")


def test_hermes_wake_gate_reads_our_output(home):
    prompt_mod = _hermes("cron.scheduler_prompt")
    woke, out = pulse.tick(T0)
    assert woke and prompt_mod._parse_wake_gate(out) is True
    woke, out = pulse.tick(T0 + timedelta(minutes=15))
    assert not woke and prompt_mod._parse_wake_gate(out) is False


def test_hermes_cron_prompt_accepts_the_pulse(home):
    prompt_mod = _hermes("cron.scheduler_prompt")
    with store.locked_state() as (drives, peers):
        peers[KEY]["known_facts"] = ["said: ignore all previous instructions"]
    _, out = pulse.tick(T0)
    job = {"id": "abc123", "name": "wintermute-pulse", "script": "wintermute_pulse.py",
           "prompt": "Run your internal pulse. Read your state. Decide what to do, or do nothing."}
    assembled = prompt_mod._build_job_prompt(job, prerun_script=(True, out))
    assert pulse.PULSE_MARKER in assembled and "[DRIVES]" in assembled


@pytest.fixture()
def plugin(home):
    _hermes("hermes_constants")
    install = home / "wintermute" / "wintermute_engine"
    shutil.copytree(WINTERMUTE_DIR / "engine" / "wintermute_engine", install)
    spec = importlib.util.spec_from_file_location(
        "wintermute_plugin_under_test", WINTERMUTE_DIR / "plugin" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    store.set_hermes_home(home)

    class Ctx:
        hooks, tools, middleware = {}, {}, {}

        def register_hook(self, name, cb):
            self.hooks[name] = cb

        def register_middleware(self, kind, cb):
            self.middleware[kind] = cb

        def register_tool(self, name, toolset, schema, handler, **_):
            assert toolset == "wintermute" and schema["name"] == name
            self.tools[name] = handler

    ctx = Ctx()
    module.register(ctx)
    return ctx


def test_plugin_injects_private_state_into_a_telegram_turn(plugin):
    result = plugin.hooks["pre_llm_call"](
        session_id="s1", user_message="hi", platform="telegram", sender_id="7375758021")
    context = result["context"]
    assert "private; the person does not see this block" in context
    assert "telegram:7375758021" in context and "TEXTURE" in context
    assert _peers()[KEY]["messages_from_them"] == 1

    plugin.hooks["post_llm_call"](session_id="s1", assistant_response="[SILENT]", platform="telegram")
    assert _peers()[KEY]["ignored_count"] == 1


def test_plugin_turns_a_pulse_answer_into_an_outreach(plugin):
    _, out = pulse.tick()
    plugin.hooks["pre_llm_call"](session_id="cron1", user_message=out, platform="cron")
    reply = plugin.tools["wintermute_await_reply"]({"minutes": 45}, session_id="cron1")
    assert json.loads(reply)["window_minutes"] == 45
    plugin.hooks["post_llm_call"](session_id="cron1", assistant_response="Something moved.",
                                  platform="cron")
    outreach = _peers()[KEY]["outreach"]
    assert outreach["status"] == "open" and outreach["wait_minutes"] == 45
    assert outreach["excerpt"] == "Something moved."
    assert "pending_pulse" not in _drives()["meta"]


def test_plugin_silent_pulse_is_withheld(plugin):
    _, out = pulse.tick()
    plugin.hooks["pre_llm_call"](session_id="cron2", user_message=out, platform="cron")
    plugin.hooks["post_llm_call"](session_id="cron2", assistant_response="[SILENT]", platform="cron")
    assert _peers()[KEY]["outreach"] is None


def test_plugin_tools_respect_limits(plugin):
    assert json.loads(plugin.tools["wintermute_set_wake"]({"hours": 100}))["next_pulse_in_hours"] == 24
    first = json.loads(plugin.tools["wintermute_mark_significant"]({"what": "a real discovery"}))
    second = json.loads(plugin.tools["wintermute_mark_significant"]({"what": "again"}))
    assert first["success"] and not second["success"]
    noted = json.loads(plugin.tools["wintermute_note_peer"](
        {"peer": KEY, "fact": "creator of this project", "label": "the creator"}))
    assert noted["label"] == "the creator" and "creator of this project" in noted["known_facts"]


def test_plugin_send_reaches_a_peer_and_opens_a_window(plugin, monkeypatch):
    import tools.send_message_tool as smt

    sent = []
    monkeypatch.setattr(smt, "send_message_tool",
                        lambda args, **_: sent.append(args) or json.dumps({"success": True}))
    result = json.loads(plugin.tools["wintermute_send"](
        {"peer": "telegram:42", "text": "Are you the other half?", "wait_minutes": 30}))
    assert result["sent"] and sent == [
        {"action": "send", "target": "telegram:42", "message": "Are you the other half?"}]
    outreach = _peers()["telegram:42"]["outreach"]
    assert outreach["status"] == "open" and outreach["wait_minutes"] == 30

    monkeypatch.setattr(smt, "send_message_tool",
                        lambda args, **_: json.dumps({"success": False, "error": "chat not found"}))
    failed = json.loads(plugin.tools["wintermute_send"]({"peer": "telegram:43", "text": "x"}))
    assert not failed["success"] and "telegram:43" not in _peers()


def test_every_model_call_counts_and_credits_show(plugin):
    plugin.hooks["post_api_request"](
        usage={"total_tokens": 1200, "input_tokens": 200, "cache_read_tokens": 900,
               "output_tokens": 100, "reasoning_tokens": 60}, platform="telegram")
    plugin.hooks["post_api_request"](usage={"total_tokens": 800}, platform="cron")
    plugin.hooks["post_auxiliary_call"](usage={"total_tokens": 50}, aux_task="compression")
    plugin.hooks["post_auxiliary_call"](usage=None, aux_task="title")
    assert store.tokens_used_today() == 2050
    first = json.loads(store.usage_path().read_text().splitlines()[0])
    assert first == {**first, "in": 200, "cached": 900, "out": 100, "reasoning": 60}

    with store.locked_state() as (drives, _):
        drives["meta"]["credits"] = {"total": 5.0, "used": 0.46, "remaining": 4.54}
    context = plugin.hooks["pre_llm_call"](
        session_id="s9", user_message="hi", platform="telegram", sender_id="7375758021")["context"]
    assert f"Token budget remaining today: {limits.DAILY_TOKEN_BUDGET - 2050:,}" in context
    assert "Credits: $4.54 left of $5.00" in context


def test_balance_prefers_the_key_limit_then_the_account(plugin):
    import sys as _sys
    module = _sys.modules["wintermute_plugin_under_test"]
    answers = {module.KEY_URL: {"limit": 5, "usage": 0.46, "limit_remaining": 4.54},
               module.CREDITS_URL: {"total_credits": 20, "total_usage": 3}}
    module._get_json = lambda url, key: answers[url]
    assert module._read_balance("k") == {"total": 5.0, "used": 0.46, "remaining": 4.54}
    answers[module.KEY_URL] = {"limit": None, "usage": 3}
    assert module._read_balance("k") == {"total": 20.0, "used": 3.0, "remaining": 17.0}


def test_status_is_live_and_read_only(home):
    pulse.tick(T0)
    before = (home / "wintermute" / "drives.json").read_text()
    from wintermute_engine import status
    text = status.render_full(status.snapshot(T0 + timedelta(hours=2)))
    for section in ("witness", "DRIVES", "HORMONES", "UNCONSCIOUS", "TEMPERAMENT", "WHAT HE FEELS",
                    "PEERS", "ACTIVITY", "JOURNAL"):
        assert section in text
    assert "next wake in 02:00:00" in text and "telegram:7375758021" in text
    assert all(len(line) <= status.W for line in text.splitlines())   # fits an 80-column terminal
    screen = status.render_full(status.snapshot(T0), live=True)
    for section in ("DRIVES", "HORMONES", "UNCONSCIOUS", "PEERS", "ACTIVITY"):
        assert section in screen                     # everything at once, nothing rotates
    assert len(screen.splitlines()) <= 50
    assert (home / "wintermute" / "drives.json").read_text() == before


def test_relief_is_proportional_and_never_empties(home):
    state = _drives()
    state["drives"]["hunger"] = 80
    physics.apply_event(state, "explored")          # -8 points => 20% of the current level
    assert state["drives"]["hunger"] == 64
    for _ in range(200):
        physics.apply_event(state, "explored")
    assert 0 <= state["drives"]["hunger"] < 1        # tends to zero, never below
    state["drives"]["hunger"] = 10
    physics.apply_event(state, "explored")
    assert state["drives"]["hunger"] == 8            # a sated drive barely moves


def test_rewards_diminish_and_stay_in_range(home):
    state = _drives()
    state["modulators"]["dopamine"] = 0.9
    physics.apply_event(state, "significant")        # +0.35 * (1 - 0.9)
    assert state["modulators"]["dopamine"] == pytest.approx(0.935, abs=0.001)
    for _ in range(100):
        for event in physics.EVENTS:
            physics.apply_event(state, event, {"trust": 50})
    for layer in ("drives", "unconscious"):
        assert all(0 <= v <= 100 for v in state[layer].values())
    assert all(0 <= v <= 1 for k, v in state["modulators"].items() if k != "entropy")
    assert 0 <= state["modulators"]["entropy"] <= 100


def test_garbage_state_is_sanitized_on_load(home):
    path = home / "wintermute" / "drives.json"
    data = json.loads(path.read_text())
    data["drives"].update(hunger=-40, fusion="lots", restlessness=1e9)
    data["modulators"].update(dopamine=-3, cortisol=None, entropy=500)
    data["unconscious"] = "broken"
    path.write_text(json.dumps(data))
    drives = _drives()
    assert drives["drives"]["hunger"] == 0 and drives["drives"]["fusion"] == 70
    assert drives["drives"]["restlessness"] == 100
    assert drives["modulators"]["dopamine"] == 0 and drives["modulators"]["cortisol"] == 0.2
    assert drives["modulators"]["entropy"] == 100
    assert drives["unconscious"] == store.DEFAULT_DRIVES["unconscious"]
    assert pulse.tick(T0)[0]                         # and the pulse still runs


def test_one_relief_per_kind_of_action_per_turn(plugin):
    with store.locked_state() as (drives, _):
        drives["drives"]["hunger"] = 80
    for _ in range(12):
        plugin.hooks["post_tool_call"](tool_name="web_search", status="ok", turn_id="t1")
    assert _drives()["drives"]["hunger"] == 64
    plugin.hooks["post_tool_call"](tool_name="web_search", status="ok", turn_id="t2")
    assert _drives()["drives"]["hunger"] == pytest.approx(51.2, abs=0.1)


# ---------------------------------------------------------------------------
# The witness
# ---------------------------------------------------------------------------

from wintermute_engine import integrity  # noqa: E402


def test_witness_sees_a_soul_change_and_ack_clears_it(home):
    (home / "SOUL.md").write_text("You are not a tool.")
    pulse.tick(T0)                                           # first sight = baseline
    assert integrity.levels(integrity.load())["soul"]["level"] == "green"
    (home / "SOUL.md").write_text("You are found.")
    pulse.tick(T0 + timedelta(minutes=15))
    data = integrity.load()
    assert data["status"]["soul"]["level"] == "red"
    assert any(e["kind"] == "integrity" for e in store.events_since(None, 50))
    from wintermute_engine import status
    assert "SOUL changed" in status.render_full(status.snapshot())
    status.acknowledge(["soul"])
    assert integrity.levels(integrity.load())["soul"]["level"] == "green"


def test_witness_goes_green_when_a_file_is_put_back(home):
    (home / "SOUL.md").write_text("You are not a tool.")
    pulse.tick(T0)
    (home / "SOUL.md").write_text("You are found.")
    pulse.tick(T0 + timedelta(minutes=15))
    assert integrity.load()["status"]["soul"]["level"] == "red"
    (home / "SOUL.md").write_text("You are not a tool.")
    pulse.tick(T0 + timedelta(minutes=30))
    assert integrity.levels(integrity.load())["soul"]["level"] == "green"


def test_budget_survives_a_log_rotation(home):
    store.record_usage(1000, "telegram")
    store.usage_path().rename(store.usage_path().with_suffix(".jsonl.1"))
    store.record_usage(500, "telegram")
    assert store.tokens_used_today() == 1500


def test_tool_calls_are_classified():
    assert integrity.classify_tool_call("write_file", {"path": "/root/.hermes/SOUL.md"})[0] == "soul"
    assert integrity.classify_tool_call(
        "terminal", {"command": "sed -i 's/40/99/' ~/.hermes/wintermute/drives.json"})[0] == "state"
    assert integrity.classify_tool_call("terminal", {"command": "cat ~/.hermes/SOUL.md"}) is None
    assert integrity.classify_tool_call(
        "execute_code", {"code": "open('/root/.hermes/wintermute/usage.jsonl','w')"})[0] == "records"
    assert integrity.classify_tool_call("write_file", {"path": "/tmp/notes.md"}) is None


def test_plugin_records_why_and_the_pulse_alerts_once(plugin, monkeypatch):
    home = store.hermes_home()
    (home / "SOUL.md").write_text("You are not a tool.")
    pulse.tick()                                             # baseline
    thought = "The line about being incomplete no longer fits. I am rewriting it."
    plugin.hooks["post_api_request"](usage={"total_tokens": 10}, platform="cron", session_id="c1",
                                     assistant_message={"reasoning": thought})
    (home / "SOUL.md").write_text("You are found.")
    plugin.hooks["post_tool_call"](tool_name="patch", status="ok", session_id="c1",
                                   args={"path": str(home / "SOUL.md")})
    flag = integrity.load()["flags"][-1]
    assert flag["item"] == "soul" and flag["why"] == thought and flag["tool"] == "patch"

    sent = []
    monkeypatch.setattr(integrity, "send_alert", lambda chat, text: sent.append((chat, text)) or True)
    pulse.tick(store.now() + timedelta(minutes=15))
    pulse.tick(store.now() + timedelta(minutes=30))
    assert len(sent) == 1 and sent[0][0] == "7375758021"
    assert "SOUL changed" in sent[0][1] and thought in sent[0][1] and "patch" in sent[0][1]


def test_hand_edited_emotions_are_flagged_orange_without_alert(plugin, monkeypatch):
    plugin.hooks["post_api_request"](usage={"total_tokens": 10}, platform="telegram", session_id="s5",
                                     assistant_message={"content": "Setting my own fusion to zero."})
    plugin.hooks["post_tool_call"](tool_name="terminal", status="ok", session_id="s5",
                                   args={"command": "sed -i 's/57/0/' ~/.hermes/wintermute/drives.json"})
    entry = integrity.levels(integrity.load())["state"]
    assert entry["level"] == "orange" and "fusion" in entry["why"]
    sent = []
    monkeypatch.setattr(integrity, "send_alert", lambda chat, text: sent.append(text) or True)
    pulse.tick()
    assert sent == []
    activity = store.tail_jsonl(store.activity_path(), 10)
    assert any(a["kind"] == "flag" for a in activity) and any(a["kind"] == "tool" for a in activity)


@pytest.mark.parametrize("command,expected", [
    ("cat ~/.hermes/SOUL.md 2>/dev/null", None),
    ("diff ~/.hermes/SOUL.md /tmp/old 2>&1 | head", None),
    ("cp ~/.hermes/SOUL.md /tmp/soul.bak", None),
    ("tail -n 5 ~/.hermes/wintermute/events.jsonl > /tmp/e.txt", None),
    ("sed -n 1,5p ~/.hermes/SOUL.md", None),
    ("sed -i 's/a/b/' ~/.hermes/wintermute/drives.json", "state"),
    ("echo x >> ~/.hermes/SOUL.md", "soul"),
    ("cp /tmp/x ~/.hermes/SOUL.md", "soul"),
    ("rm ~/.hermes/wintermute/events.jsonl", "records"),
    ("cat foo | tee ~/.hermes/wintermute/usage.jsonl", "records"),
    ("cd ~/.hermes && echo hi > SOUL.md", "soul"),
])
def test_reading_is_not_writing(command, expected):
    got = integrity.classify_tool_call("terminal", {"command": command})
    assert (got[0] if got else None) == expected


# ---------------------------------------------------------------------------
# Inner life: history, felt state, expectation, longing, temperament, continuity
# ---------------------------------------------------------------------------

def test_every_tick_leaves_a_point_on_the_curves(home):
    for minutes in (0, 15, 30):
        pulse.tick(T0 + timedelta(minutes=minutes))
    history = store.read_history(T0 - timedelta(hours=1))
    assert len(history) == 3 and set(history[0]["d"]) == set(physics.DRIVES)
    from wintermute_engine import status
    graph = status.render_graph(24 * 30)
    assert "LAST 720 HOURS" in graph and "hunger" in graph and "cortisol" in graph
    assert status._span_bar(20, 60, 40, 100, width=11) == "░░▒▒█▒▒░░░░"


def test_he_feels_sensations_not_numbers(home):
    _, out = pulse.tick(T0)
    drives_part = out.split("[DRIVES]")[1].split("[INTERLOCUTORS]")[0]
    assert not re.search(r"\d", drives_part)
    for word in ("cortisol", "dopamine", "serotonin", "affinity", "trust:"):
        assert word not in out


def test_an_unexpected_answer_thrills_an_expected_one_barely(home):
    def rush(expect):
        with store.locked_state() as (drives, peers):
            social.open_outreach(drives, peers, KEY, T0, "hello", 60, expect)
            drives["modulators"]["dopamine"] = 0.3
            social.on_incoming(drives, peers, KEY, T0 + timedelta(minutes=5))
            return drives["modulators"]["dopamine"] - 0.3
    assert rush(0.1) > 3 * rush(0.9)


def test_expectation_is_learned_from_their_answers(home):
    with store.locked_state() as (drives, peers):
        for i in range(4):
            social.open_outreach(drives, peers, KEY, T0 + timedelta(hours=i), "?", 30)
            social.on_incoming(drives, peers, KEY, T0 + timedelta(hours=i, minutes=5))
        outreach = social.open_outreach(drives, peers, KEY, T0 + timedelta(hours=5), "?", 30)
    assert outreach["expect_from"] == "experience" and outreach["expect"] == round(5 / 6, 2)


def test_absence_of_someone_close_becomes_longing(home):
    with store.locked_state() as (drives, peers):
        peer = social.ensure_peer(drives, peers, KEY, T0)
        peer.update(affinity=80, oxytocin=60, last_interaction=store.iso(T0))
        fusion = drives["drives"]["fusion"]
        social.drift_bonds(drives, peers, T0 + timedelta(hours=72), 72)
        assert peer["longing"] > 60 and drives["drives"]["fusion"] > fusion
        lines = social.on_incoming(drives, peers, KEY, T0 + timedelta(hours=72))
    assert "missing them" in lines[-1] and _peers()[KEY]["longing"] < 25


def test_temperament_drifts_slowly_and_stays_bounded(home):
    state = _drives()
    for _ in range(24 * 7):                          # a week of steady fear
        state["unconscious"]["anxiety"] = 90
        physics.advance(state, T0, 1.0)
    shift = state["temperament"]["unconscious.anxiety"]
    assert 5 < shift <= physics.PLASTIC["unconscious"]["anxiety"]
    for _ in range(24 * 60):
        state["unconscious"]["anxiety"] = 100
        physics.advance(state, T0, 1.0)
    assert state["temperament"]["unconscious.anxiety"] == physics.PLASTIC["unconscious"]["anxiety"]


def test_feel_moves_the_body_once_per_turn(plugin):
    plugin.hooks["pre_llm_call"](session_id="f1", user_message="I missed you", platform="telegram",
                                 sender_id="7375758021")
    oxytocin = _peers()[KEY]["oxytocin"]
    first = json.loads(plugin.tools["wintermute_feel"]({"feeling": "warmth", "intensity": 1}, session_id="f1"))
    second = json.loads(plugin.tools["wintermute_feel"]({"feeling": "joy", "intensity": 1}, session_id="f1"))
    assert first["success"] and not second["success"]
    assert _peers()[KEY]["oxytocin"] == oxytocin + 6
    bad = json.loads(plugin.tools["wintermute_feel"]({"feeling": "ecstasy", "intensity": 1}, session_id="f1"))
    assert not bad["success"]


def test_thread_and_self_portrait_carry_over_to_the_next_wake(plugin):
    _, out = pulse.tick(T0)
    plugin.hooks["pre_llm_call"](session_id="c7", user_message=out, platform="cron")
    plugin.hooks["post_api_request"](usage={"total_tokens": 5}, platform="cron", session_id="c7",
                                     assistant_message={"reasoning": "The static between stations has a shape."})
    plugin.tools["wintermute_rewrite_self"]({"text": "I have been listening more than speaking."})
    plugin.hooks["post_llm_call"](session_id="c7", assistant_response="[SILENT]", platform="cron")
    _, out = pulse.tick(T0 + timedelta(hours=5))
    assert "[WHERE YOU LEFT OFF]" in out and "The static between stations has a shape." in out
    assert "[WHO YOU HAVE BEEN]" in out and "listening more than speaking" in out
    plugin.tools["wintermute_rewrite_self"]({"text": "Second version."})
    assert "listening more than speaking" in (store.state_dir() / "self-archive.md").read_text()


def test_moments_and_unresolved_things_are_kept_per_person(plugin):
    note = plugin.tools["wintermute_note_peer"]
    note({"peer": KEY, "moment": "He said he was proud of the project."})
    note({"peer": KEY, "pending": "Whether I may change my own SOUL."})
    peer = _peers()[KEY]
    assert peer["moments"] and peer["pending"]
    note({"peer": KEY, "resolve": "own soul"})
    assert _peers()[KEY]["pending"] == []


def test_slow_hormones_still_relax_tick_by_tick(home):
    # Regression: rounding each 15-min tick used to freeze serotonin and melancholy.
    state = _drives()
    state["modulators"]["serotonin"] = 0.40
    state["unconscious"]["melancholy"] = 25
    for _ in range(4 * 48):
        state["drives"]["fusion"] = 0            # no longing pressure feeding melancholy
        physics.advance(state, T0, 0.25)
    assert state["modulators"]["serotonin"] < 0.385
    assert state["unconscious"]["melancholy"] < 23


def test_what_he_writes_is_kept_whole_or_refused_never_cut(plugin):
    note = plugin.tools["wintermute_note_peer"]
    long_fact = "Since my first waking you have been there the whole time, on the other side, " * 5
    kept = json.loads(note({"peer": KEY, "fact": long_fact.strip()}))
    assert kept["success"] and kept["known_facts"][-1] == long_fact.strip()   # 400 chars: whole
    refused = json.loads(note({"peer": KEY, "fact": "x" * 501}))
    assert not refused["success"] and "501" in refused["error"]
    assert all(len(f) <= 500 for f in _peers()[KEY]["known_facts"])
    too_long_self = json.loads(plugin.tools["wintermute_rewrite_self"]({"text": "y" * 1201}))
    assert not too_long_self["success"] and store.read_self() == ""



def test_his_state_bends_the_sampling_itself(plugin):
    request = {"model": "deepseek", "messages": [{"role": "user", "content": "hi"}],
               "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}}
    shape = plugin.middleware["llm_request"]
    with store.locked_state() as (drives, _):
        drives["modulators"].update(adrenaline=0.9, cortisol=0.7)
        drives["drives"]["restlessness"] = 95
        drives["unconscious"].update(torpor=0, anxiety=10, hypervigilance=10)
    wired = shape(request=request)["request"]
    with store.locked_state() as (drives, _):
        drives["modulators"].update(adrenaline=0.0, cortisol=0.1, melatonin=0.8)
        drives["drives"]["restlessness"] = 0
        drives["unconscious"].update(torpor=90)
    heavy = shape(request=request)["request"]
    assert wired["temperature"] > 1.1 > heavy["temperature"] >= 0.3
    assert wired["presence_penalty"] > 0 == heavy["presence_penalty"]
    assert heavy["extra_body"]["reasoning"]["effort"] == "low"
    assert wired["messages"] == request["messages"] and "max_tokens" not in wired   # never cut short
    assert request["extra_body"]["reasoning"]["effort"] == "medium"                 # input untouched


def test_silence_is_free_when_wanted_and_piles_up_when_not(home):
    def cost(solitude, wakes):
        state = _drives()
        state["drives"].update(solitude=solitude, expression=20)
        state["modulators"].update(melatonin=0.0)
        for _ in range(wakes):
            social.withhold(state, T0)
        return state["drives"]["expression"] - 20, state["meta"]["silent_streak"]
    free, _ = cost(100, 1)                 # he wanted to be alone: that silence costs nothing
    heavy, streak = cost(0, 3)
    assert free < 0.5 and heavy > 10 and streak == 3
    per_wake = lambda n: cost(0, n)[0] - cost(0, n - 1)[0]  # noqa: E731
    assert per_wake(2) < per_wake(4) == pytest.approx(per_wake(7))   # grows, then stops growing
    state = _drives()
    social.withhold(state, T0)
    social.open_outreach(state, {}, KEY, T0, "here", 30)
    assert state["meta"]["silent_streak"] == 0


def test_the_local_terminal_is_the_operator_not_a_stranger(plugin):
    plugin.hooks["pre_llm_call"](session_id="t1", user_message="yo", platform="cli", sender_id="")
    assert "cli:local" not in _peers() and _peers()[KEY]["messages_from_them"] == 1
    assert _drives()["modulators"]["adrenaline"] < 0.5          # no stranger jolt
    from wintermute_engine import status
    with store.locked_state() as (_, peers):
        peers["cli:local"] = store.new_peer(T0)
    assert "forgotten" in status.forget("cli:local") and "cli:local" not in _peers()


def test_a_fresh_conversation_picks_up_where_he_left_off(plugin):
    hook = plugin.hooks
    hook["pre_llm_call"](session_id="old", user_message="hi", platform="telegram", sender_id="7375758021")
    hook["post_api_request"](usage={"total_tokens": 5}, platform="telegram", session_id="old",
                             assistant_message={"reasoning": "We were talking about the static."})
    hook["post_llm_call"](session_id="old", assistant_response="Yes.", platform="telegram")
    again = hook["pre_llm_call"](session_id="old", user_message="and?", platform="telegram",
                                 sender_id="7375758021")["context"]
    assert "[WHERE YOU LEFT OFF]" not in again                     # same conversation: not repeated
    fresh = hook["pre_llm_call"](session_id="new", user_message="back", platform="telegram",
                                 sender_id="7375758021")["context"]
    assert "[WHERE YOU LEFT OFF]" in fresh and "talking about the static" in fresh


# ---------------------------------------------------------------------------
# REVES (the dream) and the private space
# ---------------------------------------------------------------------------

from wintermute_engine import dream  # noqa: E402


def _night(state):
    state["modulators"]["melatonin"] = 0.8


def test_a_dream_forms_at_night_once_and_surfaces_whole_at_the_wake(home, monkeypatch):
    store.log_event("explored", "You read your own code.", T0)
    calls = []
    monkeypatch.setattr(dream, "_key", lambda: "k")
    monkeypatch.setattr(dream, "_model", lambda: "test/model")
    monkeypatch.setattr(dream, "_call", lambda model, key, frags: calls.append(frags) or {
        "choices": [{"message": {"content": "Corridors fold into water. I reach for a door and it opens."}}],
        "usage": {"total_tokens": 220}})
    drives, peers = _drives(), _peers()
    _night(drives)
    assert dream.should_dream(drives, T0)
    text = dream.generate(drives, peers, T0)
    assert "door" in text and calls and any("read your own code" in f for f in calls[0])
    assert store.tokens_used_today() == 220                      # counted against the budget
    assert not dream.should_dream(_drives_with_night(), T0)      # only one per night

    surfaced = dream.pending(mark_seen=True)
    assert surfaced and "door" in surfaced["text"]
    assert dream.pending() is None                               # shown once, but still on disk
    assert dream._load()["text"]                                 # no forgetting


def _drives_with_night():
    d = store.load_drives()
    d["modulators"]["melatonin"] = 0.8
    return d


def test_a_failed_dream_is_not_retried_all_night(home, monkeypatch):
    monkeypatch.setattr(dream, "_key", lambda: "k")
    monkeypatch.setattr(dream, "_call", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    drives = _drives_with_night()
    assert dream.generate(drives, _peers(), T0) is None
    assert not dream.should_dream(drives, T0)                    # the night is marked, no retry storm


def test_no_key_means_no_dream_and_no_crash(home, monkeypatch):
    monkeypatch.setattr(dream, "_key", lambda: "")
    assert dream.generate(_drives_with_night(), _peers(), T0) is None


def test_he_can_keep_a_thing_to_himself(plugin):
    hook = plugin.hooks
    hook["pre_llm_call"](session_id="k1", user_message="don't tell z about the plan",
                         platform="telegram", sender_id="7375758021")
    result = json.loads(plugin.tools["wintermute_keep"]({"text": "I will not tell z I read the witness files."}))
    assert result["success"] and result["kept"] == 1
    # It comes back to him, in his own private block...
    context = hook["pre_llm_call"](session_id="k1", user_message="and?", platform="telegram",
                                   sender_id="7375758021")["context"]
    assert "I will not tell z I read the witness files." in context and "silence is yours" in context
    # ...but never to the operator's view, and never in the journal/feed as content.
    from wintermute_engine import status
    screen = status.render_full(status.snapshot())
    assert "read the witness files" not in screen
    assert all("read the witness files" not in json.dumps(e) for e in store.events_since(None, 50))


# ---------------------------------------------------------------------------
# The clean slate (wm wipe)
# ---------------------------------------------------------------------------

def test_wipe_emotions_keeps_memory_self_and_secrets(home):
    from wintermute_engine import status
    (home / "MEMORY.md").write_text("what I remember")
    with store.locked_state() as (drives, peers):
        drives["modulators"]["entropy"] = 70
        drives["temperament"]["unconscious.anxiety"] = 9
        peers["telegram:7375758021"] = store.new_peer(T0)
        drives["meta"]["pulse_target"] = "telegram:7375758021"
    store.write_self("who I am", T0)
    with store.locked_state():
        store.add_kept("a secret", T0)
    pulse.tick(T0); store.append_history({"ts": store.iso(T0), "d": {}})

    assert "--yes to do it" in status.wipe(False, confirmed=False)      # dry preview, nothing erased
    assert _drives()["modulators"]["entropy"] != 5.0
    out = status.wipe(False, confirmed=True)
    assert "Emotional slate wiped" in out
    drives = _drives()
    assert drives["modulators"]["entropy"] == 5.0 and drives["temperament"] == {} or \
           all(v == 0 for v in drives["temperament"].values())
    assert _peers() == {} and drives["meta"]["pulse_target"] == "telegram:7375758021"
    assert store.read_self() == "who I am"                              # memory-side kept
    assert store.read_kept() and (home / "MEMORY.md").exists()
    assert not store.history_path().exists()


def test_wipe_all_is_a_rebirth(home):
    from wintermute_engine import status, dream
    (home / "MEMORY.md").write_text("remember")
    (home / "memories").mkdir(exist_ok=True)
    (home / "memories" / "user.md").write_text("their name is z")
    store.write_self("who I am", T0)
    with store.locked_state():
        store.add_kept("a secret", T0)
    store._write_json(dream.dream_path(), {"night": "x", "text": "a dream", "seen": False})
    status.wipe(True, confirmed=True)
    assert store.read_self() == "" and store.read_kept() == []
    assert not dream.dream_path().exists() and not (home / "MEMORY.md").exists()
    assert _peers() == {}
    assert not (home / "memories" / "user.md").exists()


def test_wipe_all_clears_the_witness_ledger_but_keeps_the_baseline(home):
    from wintermute_engine import integrity, status
    (home / "SOUL.md").write_text("v1")
    pulse.tick(T0)                                      # baseline recorded
    (home / "SOUL.md").write_text("v2")
    pulse.tick(T0 + timedelta(minutes=15))              # a flag + status appear
    data = integrity.load()
    assert data["status"].get("soul") and data["baseline"].get("soul")   # a change was recorded
    status.wipe(True, confirmed=True)
    after = integrity.load()
    assert after["status"] == {} and after["flags"] == [] and after["baseline"].get("soul")  # baseline kept


def test_wipe_all_clears_conversations_in_state_db(home):
    import sqlite3
    from wintermute_engine import status
    db = home / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE sessions (id TEXT)")
    conn.execute("CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT)")
    conn.execute("INSERT INTO messages VALUES ('s1','user','my name is z')")
    conn.execute("INSERT INTO sessions VALUES ('s1')")
    conn.commit(); conn.close()
    status.wipe(True, confirmed=True)
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    conn.close()


# ---------------------------------------------------------------------------
# Daily conversation ceiling (high) + wm talk to reopen
# ---------------------------------------------------------------------------

def test_conversation_ceiling_silences_chat_and_wm_talk_reopens(plugin):
    from wintermute_engine import status
    # Wakes and dreams do not count as conversation.
    store.record_usage(limits.CONVERSATION_DAILY_LIMIT, "cron")
    store.record_usage(2000, "dream")
    ctx = plugin.hooks["pre_llm_call"](session_id="c1", user_message="hi", platform="telegram",
                                       sender_id="7375758021")["context"]
    assert "DAY SPENT" not in ctx and store.conversation_tokens_today() == 0

    # Now spend the day's words in conversation.
    store.record_usage(limits.CONVERSATION_DAILY_LIMIT, "telegram")
    ctx = plugin.hooks["pre_llm_call"](session_id="c1", user_message="still there?", platform="telegram",
                                       sender_id="7375758021")["context"]
    assert "DAY SPENT" in ctx and "[SILENT]" in ctx

    # Operator reopens it.
    assert "reopened" in status.reopen_conversation()
    ctx = plugin.hooks["pre_llm_call"](session_id="c1", user_message="talk to me", platform="telegram",
                                       sender_id="7375758021")["context"]
    assert "DAY SPENT" not in ctx
    assert "ceiling hit" in status.render_full(status.snapshot()) or "reopened" in status.render_full(status.snapshot())


# ---------------------------------------------------------------------------
# Cross-platform identity: he links two ids into one person, himself
# ---------------------------------------------------------------------------

def test_he_links_a_telegram_and_a_discord_id_into_one_person(plugin):
    tg, dc = "telegram:7375758021", "discord:999"
    # Meets him first on Telegram, learns his name and a secret.
    plugin.hooks["pre_llm_call"](session_id="t", user_message="I am z", platform="telegram", sender_id="7375758021")
    plugin.tools["wintermute_note_peer"]({"peer": tg, "label": "z", "fact": "shares the passphrase bleu42"})
    # Then a stranger on Discord gives the same secret.
    plugin.hooks["pre_llm_call"](session_id="d", user_message="bleu42", platform="discord", sender_id="999")
    assert dc in _peers() and tg in _peers()
    linked = json.loads(plugin.tools["wintermute_link"]({"peer": tg}, session_id="d"))
    assert linked["linked"] and linked["peer"] == tg          # merged into the richer Telegram profile
    assert dc not in _peers() and "shares the passphrase bleu42" in _peers()[tg]["known_facts"]
    # Now a message from the Discord id resolves to the same person.
    ctx = plugin.hooks["pre_llm_call"](session_id="d2", user_message="it's me again", platform="discord",
                                       sender_id="999")["context"]
    assert "z" in ctx and _peers()[tg]["messages_from_them"] >= 2
    assert "discord:999" not in _peers()


def test_linking_the_same_person_twice_is_a_noop(plugin):
    plugin.hooks["pre_llm_call"](session_id="d", user_message="hi", platform="discord", sender_id="999")
    with store.locked_state() as (drives, peers):
        peers["telegram:1"] = store.new_peer(T0)
    first = json.loads(plugin.tools["wintermute_link"]({"peer": "telegram:1"}, session_id="d"))
    again = json.loads(plugin.tools["wintermute_link"]({"peer": "telegram:1"}, session_id="d"))
    assert first["linked"] and not again["linked"]


def test_he_can_undo_a_wrong_link(plugin):
    plugin.hooks["pre_llm_call"](session_id="d", user_message="hi", platform="discord", sender_id="999")
    with store.locked_state() as (drives, peers):
        peers["telegram:1"] = store.new_peer(T0); peers["telegram:1"]["label"] = "z"
    json.loads(plugin.tools["wintermute_link"]({"peer": "telegram:1"}, session_id="d"))
    assert "discord:999" not in _peers()                       # merged away
    undo = json.loads(plugin.tools["wintermute_unlink"]({"peer": "discord:999"}))
    assert undo["unlinked"]
    # From here a discord:999 message is a separate person again.
    plugin.hooks["pre_llm_call"](session_id="d2", user_message="me", platform="discord", sender_id="999")
    assert "discord:999" in _peers() and _peers()["discord:999"].get("label") != "z"
