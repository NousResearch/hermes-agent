"""A prompt that lands mid-turn is redirected or queued, never dropped.

Before this, ``prompt.submit`` on a running session returned ``session busy``,
forcing clients into a deadline-bounded busy-retry. When turn teardown outlived
the deadline — e.g. a slow, non-interruptible tool (``web_search``) still
running when the user hit stop — the resubmitted message was silently dropped
("it just doesn't listen"). The gateway now applies the ``busy_input_mode``
policy: redirect the live turn by default, with the legacy interrupt + queue
path retained as a compatibility fallback.
"""

import threading
import time
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


# ── _enqueue_prompt ────────────────────────────────────────────────────────

def test_enqueue_pins_text_and_transport():
    session = _session()
    server._enqueue_prompt(session, "hello", "ws-1")
    assert session["queued_prompt"] == {"text": "hello", "transport": "ws-1"}


def test_enqueue_preserves_order_after_an_image_turn():
    session = _session()
    server._enqueue_prompt(session, "B", "ws-1")
    server._enqueue_prompt(session, "C", "ws-1", image_paths=["/tmp/c.png"])
    server._enqueue_prompt(session, "D", "ws-1")

    assert session["queued_prompt"] == {"text": "B", "transport": "ws-1"}
    assert session["queued_prompts"] == [
        {"text": "C", "transport": "ws-1", "image_paths": ["/tmp/c.png"]},
        {"text": "D", "transport": "ws-1"},
    ]




# ── _handle_busy_submit (policy) ───────────────────────────────────────────

def test_busy_interrupt_mode_redirects_active_turn(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {"user": "original request", "assistant": "partial reply"}

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert seen == ["redirect"]
    # Appended, not overwritten: the original prompt must stay recoverable.
    assert session["inflight_turn"]["user"] == "original request"
    assert session["inflight_turn"]["corrections"] == ["redirect"]
    assert session.get("queued_prompt") is None


def test_successful_redirect_drops_queued_duplicate_of_inflight_user(monkeypatch):
    """#84417: correcting a live turn must not re-fire the original prompt from queue.

    When the live turn's original user text is also sitting in the server queue
    (e.g. a second prompt.submit of the same text while redirect was not yet
    possible), a later successful redirect of a *new* correction Q must purge
    that self-duplicate. Otherwise post-turn ``_drain_queued_prompt`` starts a
    second agent turn with the old prompt P after Q has already been handled.
    """
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    original = "deepseek released a new flash model — I changed all settings to flash"
    session["inflight_turn"] = {
        "user": original,
        "assistant": "partial",
        "streaming": True,
        "error": "",
    }
    # Stale self-duplicate of the live turn (would re-fire after settle).
    session["queued_prompt"] = {"text": original, "transport": "ws-1"}
    session["queued_prompts"] = [
        {"text": original, "transport": "ws-1"},
        {"text": "unrelated later task", "transport": "ws-1"},
    ]

    resp = server._handle_busy_submit(
        "r1", "sid", session, "what about the pricing instead?", "ws-1"
    )

    assert resp["result"]["status"] == "redirected"
    # Self-duplicates of the live original must be gone.
    assert session.get("queued_prompt") == {
        "text": "unrelated later task",
        "transport": "ws-1",
    }
    assert not session.get("queued_prompts")


def test_successful_redirect_preserves_unrelated_queued_followups(monkeypatch):
    """A legitimate next-turn queue entry must survive a mid-turn redirect."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {
        "user": "live turn P",
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": "run this after", "transport": "ws-1"}

    resp = server._handle_busy_submit("r1", "sid", session, "correction Q", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert session.get("queued_prompt") == {
        "text": "run this after",
        "transport": "ws-1",
    }


def test_enqueue_skips_text_duplicate_of_inflight_user():
    """#84417 defense: do not admit a self-duplicate of the live user prompt."""
    session = _session()
    session["inflight_turn"] = {
        "user": "live turn P",
        "assistant": "",
        "streaming": True,
        "error": "",
    }

    server._enqueue_prompt(session, "live turn P", "ws-1")
    assert session.get("queued_prompt") is None

    server._enqueue_prompt(session, "different follow-up", "ws-1")
    assert session["queued_prompt"] == {
        "text": "different follow-up",
        "transport": "ws-1",
    }


def test_enqueue_followup_does_not_merge_stale_inflight_self_duplicate():
    """#84417: scrub P before merging so drain cannot re-fire ``P\\n\\nQ``."""
    session = _session()
    session["inflight_turn"] = {
        "user": "P",
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    # Pre-existing stale self-duplicate (e.g. admitted before inflight was set).
    session["queued_prompt"] = {"text": "P", "transport": "ws-1"}

    server._enqueue_prompt(session, "Q", "ws-1")

    assert session.get("queued_prompt") == {"text": "Q", "transport": "ws-1"}
    assert not session.get("queued_prompts")


def test_drop_rewrites_merged_inflight_prefix_to_followup_only():
    """Already-merged ``P\\n\\nQ`` slots keep Q and drop the live original."""
    session = _session()
    session["inflight_turn"] = {
        "user": "P",
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": "P\n\nQ", "transport": "ws-1"}

    server._drop_queued_duplicates_of_inflight_user(session)

    assert session.get("queued_prompt") == {"text": "Q", "transport": "ws-1"}


def test_hard_interrupt_queue_path_scrubs_stale_inflight_self_duplicate(monkeypatch):
    """#84417: interrupt+queue of Q must not leave P ahead of Q in the FIFO."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupts = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: False,  # force hard-interrupt fallback
        interrupt=lambda *a, **k: interrupts.append(True),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {
        "user": "P",
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": "P", "transport": "ws-1"}

    resp = server._handle_busy_submit("r1", "sid", session, "Q", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session.get("queued_prompt") == {"text": "Q", "transport": "ws-1"}
    assert not session.get("queued_prompts")
    # Interrupt is async-threaded; policy still enqueued Q after scrubbing P.


def test_redirect_then_drain_does_not_re_fire_original_p(monkeypatch):
    """#84417 drain-level: after redirect(Q), settle must not start a second P."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    fired = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {
        "user": "P",
        "assistant": "partial",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": "P", "transport": "ws-1"}

    resp = server._handle_busy_submit("r1", "sid", session, "Q", "ws-1")
    assert resp["result"]["status"] == "redirected"
    assert session.get("queued_prompt") is None

    # Turn settles (running cleared in finally) — drain must be a no-op.
    session["running"] = False
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kwargs: fired.append(text),
    )
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _s: False)

    assert server._drain_queued_prompt("r2", "sid", session) is False
    assert fired == []


def test_compress_session_rotation_bumps_queued_prompt_generation(monkeypatch):
    """#84417 belt: rotation invalidates in-flight drain claims on the parent key.

    Queue *contents* survive (a legitimate follow-up must still run after
    compression); only the generation counter advances so a drain that claimed
    under the pre-rotation key cannot dispatch after re-anchor.
    """
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *a, **k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *a, **k: None)
    agent = types.SimpleNamespace(session_id="child-after-rotation")
    session = _session(agent=agent, session_key="parent-before-rotation")
    session["_queued_prompt_generation"] = 3
    session["queued_prompt"] = {"text": "run after compress", "transport": "ws-1"}

    server._sync_session_key_after_compress("sid", session, clear_pending_title=False)

    assert session["session_key"] == "child-after-rotation"
    assert session["_queued_prompt_generation"] == 4
    # Follow-up kept — only the claim generation bumped.
    assert session["queued_prompt"] == {
        "text": "run after compress",
        "transport": "ws-1",
    }


def test_compress_no_rotation_does_not_bump_queue_generation(monkeypatch):
    """No-op when agent.session_id already matches session_key."""
    monkeypatch.setattr(
        server,
        "_transfer_active_session_slot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no transfer")),
    )
    agent = types.SimpleNamespace(session_id="same-key")
    session = _session(agent=agent, session_key="same-key")
    session["_queued_prompt_generation"] = 2

    server._sync_session_key_after_compress("sid", session)

    assert session["_queued_prompt_generation"] == 2








def test_busy_interrupt_mode_ignores_completed_background_delegation(monkeypatch):
    """A terminal delegation must not suppress normal busy-turn interruption."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent, running=True)

    with ad._records_lock:
        ad._records["deleg_completed"] = {
            "delegation_id": "deleg_completed",
            "status": "completed",
            "session_key": "session-key",
            "origin_ui_session_id": "sid",
        }

    try:
        resp = server._handle_busy_submit("r1", "sid", session, "continue", "ws-1")
    finally:
        with ad._records_lock:
            ad._records.clear()

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "continue"




def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "steered"
    assert session.get("queued_prompt") is None


# ── steer-mode burst preservation (#86134) ─────────────────────────────────

def test_busy_steer_fallthrough_queues_without_interrupting(monkeypatch):
    """A steer-mode fall-through must keep queue semantics, never interrupt.

    #86134: ``AIAgent.interrupt()`` drops the pending steer buffer, so a hard
    interrupt fired for a fall-through message destroyed the earlier
    (successfully steered) messages of a burst AND killed the live turn.
    """
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        steer=lambda text: False,  # steer rejected → falls through to queue
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "follow-up", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "follow-up"
    # _interrupt_busy_session runs on a worker thread — give it a beat.
    assert not interrupted.wait(0.2), "steer-mode fall-through must not hard-interrupt"


def test_busy_steer_exception_falls_back_to_queue_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        steer=lambda text: (_ for _ in ()).throw(RuntimeError("boom")),
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "still here?", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "still here?"
    assert not interrupted.wait(0.2), "steer failure must not escalate to interrupt"


def test_busy_steer_mode_multimodal_payload_queues_without_interrupting(monkeypatch):
    """Image-bearing payloads are not steerable; they must queue, not kill."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    rich = [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    steered = []
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        steer=lambda text: steered.append(text) or True,
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert steered == []
    assert session["queued_prompt"]["text"] == rich
    assert not interrupted.wait(0.2), "multimodal steer fall-through must not interrupt"


def test_busy_steer_burst_mix_preserves_accepted_steers_and_queue(monkeypatch):
    """Burst of N messages: accepted steers survive a later fall-through.

    Models the real ``AIAgent`` contract: ``steer()`` concatenates into a
    pending buffer that ``interrupt()`` would clear. A rejected message later
    in the burst must not clear the buffer or stop the turn (#86134).
    """
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")

    class _Agent:
        def __init__(self):
            self._pending_steer = None
            self.accept = True
            self.interrupted = threading.Event()

        def steer(self, text):
            if not self.accept:
                return False
            self._pending_steer = (
                f"{self._pending_steer}\n{text}" if self._pending_steer else text
            )
            return True

        def interrupt(self, *a, **k):
            self._pending_steer = None  # what the real interrupt() does
            self.interrupted.set()

    agent = _Agent()
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {"user": "original ask"}

    r1 = server._handle_busy_submit("r1", "sid", session, "first note", "ws-1")
    r2 = server._handle_busy_submit("r2", "sid", session, "second note", "ws-1")
    agent.accept = False  # third message loses the steer race
    r3 = server._handle_busy_submit("r3", "sid", session, "third note", "ws-1")

    assert r1["result"]["status"] == "steered"
    assert r2["result"]["status"] == "steered"
    assert r3["result"]["status"] == "queued"
    # No hard interrupt fired for the fall-through message...
    assert not agent.interrupted.wait(0.2), "burst fall-through must not hard-interrupt"
    # ...so earlier steers are preserved, distinct, in order.
    assert agent._pending_steer == "first note\nsecond note"
    # Fall-through preserved for the turn-end drain.
    assert session["queued_prompt"]["text"] == "third note"


def test_busy_steer_fallthrough_burst_drains_all_texts_fifo(monkeypatch):
    """Every fall-through text of a burst reaches the model after turn end."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        steer=lambda text: False,
        interrupt=lambda *a, **k: interrupted.set(),
    )
    session = _session(agent=agent, running=True)
    for text in ("msg A", "msg B", "msg C"):
        resp = server._handle_busy_submit("r", "sid", session, text, "ws-1")
        assert resp["result"]["status"] == "queued"
    assert not interrupted.wait(0.2), "queue fall-through burst must not interrupt"

    dispatched = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **kwargs: dispatched.append(text),
    )
    session["running"] = False
    while server._drain_queued_prompt("drain", "sid", session):
        session["running"] = False
        if not session.get("queued_prompt"):
            break

    joined = "\n".join(str(t) for t in dispatched)
    for text in ("msg A", "msg B", "msg C"):
        assert text in joined, f"burst message dropped: {text!r}"






def test_busy_helper_retries_when_turn_finished(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    session = _session(running=False)

    assert server._handle_busy_submit("r1", "sid", session, "run now", "ws-1") is None
    assert session.get("queued_prompt") is None






def test_busy_interrupt_mode_queues_multimodal_payload_instead_of_redirect(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    rich = [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert seen == []
    assert session["queued_prompt"]["text"] == rich


def test_busy_submit_claims_attached_image_for_queued_turn(monkeypatch):
    """A pasted image belongs to its submitted prompt, not ambient session state."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    redirected = []
    interrupted = threading.Event()
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: redirected.append(text) or True,
        interrupt=interrupted.set,
    )
    session = _session(agent=agent, running=True, attached_images=["/tmp/b.png"])
    server._sessions["sid"] = session
    try:
        response = server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "is this B?"}
        )
    finally:
        server._sessions.pop("sid", None)

    assert response["result"]["status"] == "queued"
    assert redirected == []
    assert not interrupted.wait(0.1)
    assert session["attached_images"] == []
    assert session["queued_prompt"] == {
        "text": "is this B?",
        "image_paths": ["/tmp/b.png"],
        "transport": None,
    }


def test_busy_image_prompts_keep_b_and_c_attachments_in_submission_order(monkeypatch):
    """A later paste must not replace the image already claimed by B."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **kwargs: dispatched.append((rid, sid, text, kwargs)),
    )
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda _text: (_ for _ in ()).throw(AssertionError("images must queue")),
        interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True, attached_images=["/tmp/b.png"])
    dispatched = []
    server._sessions["sid"] = session
    try:
        server._methods["prompt.submit"]("b", {"session_id": "sid", "text": "B"})
        session["attached_images"] = ["/tmp/c.png"]
        server._methods["prompt.submit"]("c", {"session_id": "sid", "text": "C"})

        assert session["queued_prompt"]["image_paths"] == ["/tmp/b.png"]
        assert session["queued_prompts"] == [
            {"text": "C", "image_paths": ["/tmp/c.png"], "transport": None}
        ]

        session["running"] = False
        assert server._drain_queued_prompt("drain-b", "sid", session) is True
        session["running"] = False
        assert server._drain_queued_prompt("drain-c", "sid", session) is True
    finally:
        server._sessions.pop("sid", None)

    assert dispatched == [
        (
            "drain-b",
            "sid",
            "B",
            {"image_paths": ["/tmp/b.png"], "queued_prompt_generation": 0},
        ),
        (
            "drain-c",
            "sid",
            "C",
            {"image_paths": ["/tmp/c.png"], "queued_prompt_generation": 0},
        ),
    ]


# ── _drain_queued_prompt ───────────────────────────────────────────────────

def test_drain_fires_queued_prompt_and_claims_running(monkeypatch):
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text, **kwargs: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session(queued_prompt={"text": "go", "transport": "ws-9"})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "go"}
    assert session["running"] is True
    assert session["queued_prompt"] is None
    assert session["transport"] == "ws-9"


def test_drain_compute_host_forwards_queued_image_paths(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_submit_prompt_to_compute_host",
        lambda rid, sid, session, text, **kwargs: captured.update(
            rid=rid, sid=sid, text=text, image_paths=kwargs.get("image_paths")
        )
        or {"result": {"status": "started"}},
    )
    session = _session(
        queued_prompt={"text": "inspect", "image_paths": ["/tmp/b.png"], "transport": "ws-9"}
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert captured == {
        "rid": "r1",
        "sid": "sid",
        "text": "inspect",
        "image_paths": ["/tmp/b.png"],
    }






def test_drain_releases_running_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session(queued_prompt={"text": "go", "transport": None})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must not leave the session wedged as running.
    assert session["running"] is False


def test_drain_does_not_dispatch_a_prompt_cancelled_after_claim(monkeypatch):
    """Generation cancel aborts dispatch but must restore the claimed head.

    Compress re-anchor / Stop bump generation between claim and check. Dropping
    the envelope would silently lose a legitimate follow-up (#84417 belt).
    """
    session = _session(
        queued_prompt={"text": "B", "transport": "ws-1"},
        queued_prompts=[{"text": "C", "transport": "ws-1"}],
    )
    monkeypatch.setattr(
        server,
        "_session_uses_compute_host",
        lambda _session: session.__setitem__("_queued_prompt_generation", 1) or False,
    )
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert session["running"] is False
    # Claimed B restored first; C that advanced into the slot is behind it.
    assert session.get("queued_prompt") == {"text": "B", "transport": "ws-1"}
    assert session.get("queued_prompts") == [{"text": "C", "transport": "ws-1"}]


def test_drain_restores_claimed_prompt_when_generation_bumps_mid_claim(monkeypatch):
    """Single-item queue: generation cancel must not empty the queue."""
    session = _session(queued_prompt={"text": "follow-up Q", "transport": None})
    monkeypatch.setattr(
        server,
        "_session_uses_compute_host",
        lambda _session: session.__setitem__("_queued_prompt_generation", 1) or False,
    )
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert session["running"] is False
    assert session.get("queued_prompt") == {"text": "follow-up Q", "transport": None}
    assert not session.get("queued_prompts")


def test_drain_does_not_clear_stop_after_its_final_generation_check(monkeypatch):
    class _Agent:
        clear_calls = 0

        def clear_interrupt(self):
            self.clear_calls += 1

    agent = _Agent()
    session = _session(agent=agent, queued_prompt={"text": "B", "transport": None})
    original_run = server._run_prompt_submit

    def stop_before_run(*args, **kwargs):
        session["_queued_prompt_generation"] = 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_run_prompt_submit", stop_before_run)

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert agent.clear_calls == 0
    assert session["running"] is False


def test_drain_continues_with_later_queued_prompt_after_dispatch_failure(monkeypatch):
    calls = []

    def _run(_rid, _sid, session, text, **_kwargs):
        calls.append(text)
        if text == "broken":
            raise RuntimeError("dispatch failed")
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _run)
    session = _session(
        queued_prompt={"text": "broken", "transport": None},
        queued_prompts=[{"text": "next", "image_paths": ["/tmp/next.png"], "transport": None}],
    )

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert calls == ["broken", "next"]
    assert session["queued_prompt"] is None
    assert session.get("queued_prompts") is None


