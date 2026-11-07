"""A busy-session follow-up must not redirect into a turn with a pending model switch.

#68876 (sky0eyes repro, Desktop 0.20.4): a mid-turn ``config.set model`` stashes
``session["pending_model_switch"]`` — the live swap is deferred to the next fresh
turn start (``_apply_pending_model_switch``). When a follow-up then arrived during
a long retry window (a 600s rate-limit backoff), the busy-submit path steered or
redirected it INTO the still-running old-provider turn, so the accepted switch was
never consumed: the composer pill showed the new model while every request kept
answering on the old provider, indefinitely.

The busy path now defers to the pending switch: a text-only follow-up queues (never
steers/redirects), ``interrupt`` mode ends the old turn so the next turn start applies
the stash and the drain runs the follow-up on the switched model, and the ``queue``
config keeps its explicit no-interrupt choice.
"""

import threading
import types

import tools.async_delegation as ad
from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


_PENDING_SWITCH = {"raw": "gpt-5.6-terra --provider openai-codex",
                   "confirm_expensive_model": True,
                   "display_model": "gpt-5.6-terra", "display_provider": "openai-codex"}


def test_pending_switch_forces_interrupt_mode_followup_to_queue(monkeypatch):
    """interrupt mode (the default): the follow-up queues instead of redirecting,
    and the hard interrupt ends the old turn so the stash applies at next turn start."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: (_ for _ in ()).throw(
            AssertionError("a follow-up must not redirect into a turn with a pending model switch")),
        steer=lambda text: (_ for _ in ()).throw(AssertionError("a follow-up must not steer")),
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True, pending_model_switch=dict(_PENDING_SWITCH))

    resp = server._handle_busy_submit("r1", "sid", session, "run the migration next", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "run the migration next"
    assert interrupted.wait(0.2), "interrupt mode must end the old turn so the switch stops deferring"
    # The stash survives for the next turn start to apply.
    assert session["pending_model_switch"]["raw"] == _PENDING_SWITCH["raw"]


def test_pending_switch_forces_steer_mode_followup_to_queue(monkeypatch):
    """steer mode escalates to interrupt under a pending switch: injecting into the
    old-provider turn would strand the accepted switch forever."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        steer=lambda text: (_ for _ in ()).throw(
            AssertionError("a pending switch must win over steer injection")),
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True, pending_model_switch=dict(_PENDING_SWITCH))

    resp = server._handle_busy_submit("r1", "sid", session, "and then deploy", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "and then deploy"
    assert interrupted.wait(0.2), "steer mode must escalate to interrupt so the old turn ends"


def test_pending_switch_keeps_queue_mode_without_interrupt(monkeypatch):
    """queue mode is already the no-live-arm policy: no interrupt fires, and the
    follow-up waits for the drain exactly as the config promises."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: interrupted.set())
    session = _session(agent=agent, running=True, pending_model_switch=dict(_PENDING_SWITCH))

    resp = server._handle_busy_submit("r1", "sid", session, "queued follow-up", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "queued follow-up"
    assert not interrupted.wait(0.2), "queue mode must keep its explicit no-interrupt choice"


def test_no_pending_switch_keeps_default_redirect(monkeypatch):
    """The ordinary busy path is untouched: without a pending switch, interrupt
    mode still redirects a live-correction into the running turn."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    redirected = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: redirected.append(text) or True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(AssertionError("redirect must not interrupt")),
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "make it shorter", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert redirected == ["make it shorter"]
    assert session.get("queued_prompt") is None
    with ad._records_lock:
        ad._records.clear()
