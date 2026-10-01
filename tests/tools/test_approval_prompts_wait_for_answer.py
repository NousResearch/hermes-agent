"""CLI / TUI / Desktop turns hold approval prompts open until answered; ``approvals.timeout`` is the
messaging-platform deadline only.

A surface that binds ``set_prompts_wait_for_answer()`` around its turn must get (1) an approval wait that
outlives ``approvals.timeout`` and (2) human-wait accounting that keeps excluding that wait from the batch
deadline even when the deadline loop reads it from a context that never saw the binding.
"""

import contextvars
import threading
import time

from tools import approval as mod
from tools import approval_context as ctx
from tools import approval_gateway_wait as wait_mod
from tools import approval_human_wait as human_wait


APPROVAL = {"command": "rm -rf build", "description": "d", "pattern_key": "dangerous", "pattern_keys": ["dangerous"]}


def _bound(fn):
    """Run *fn* in a fresh context that has the CLI/TUI/Desktop binding, like a surface turn."""
    def run():
        ctx.set_prompts_wait_for_answer()
        return fn()
    return lambda: contextvars.Context().run(run)


def _decide_in_thread(session_key, *, attended):
    box = {}
    wait = lambda: box.__setitem__(  # noqa: E731
        "decision", wait_mod._await_gateway_decision(session_key, lambda data: None, dict(APPROVAL)))
    thread = threading.Thread(target=_bound(wait) if attended else wait, daemon=True)
    thread.start()
    return thread, box


def test_attended_approval_outlives_approvals_timeout_and_messaging_still_times_out(monkeypatch):
    mod._gateway_queues.clear()
    monkeypatch.setattr(ctx, "_get_approval_config", lambda: {"timeout": 1})
    monkeypatch.setattr(ctx, "_fire_approval_hook", lambda name, **kw: None)

    attended, attended_box = _decide_in_thread("attended-session", attended=True)
    messaging, messaging_box = _decide_in_thread("messaging-session", attended=False)

    messaging.join(timeout=5)
    assert messaging_box["decision"] == {"resolved": False, "choice": None, "reason": None}

    attended.join(timeout=1.5)  # well past approvals.timeout
    assert attended.is_alive()
    assert mod.resolve_gateway_approval("attended-session", "once") == 1
    attended.join(timeout=5)
    assert attended_box["decision"] == {"resolved": True, "choice": "once", "reason": None}


def test_attended_wait_stays_excluded_when_read_from_an_unbound_context(monkeypatch):
    """The batch deadline loop may poll ``human_wait_seconds`` from a context without the binding; an open
    attended window must still count in full there, while a messaging window stays clamped."""
    monkeypatch.setattr(ctx, "_get_approval_config", lambda: {"timeout": 0})
    monkeypatch.setattr(human_wait, "HUMAN_WAIT_MARGIN_S", 0.0)
    release = threading.Event()

    def hold(session_key):
        with human_wait.human_wait_window(session_key):
            release.wait(5)

    threads = [threading.Thread(target=_bound(lambda: hold("attended-window")), daemon=True),
               threading.Thread(target=lambda: hold("messaging-window"), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        time.sleep(0.3)
        assert human_wait.human_wait_seconds("attended-window") >= 0.25
        assert human_wait.human_wait_seconds("messaging-window") == 0.0
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert human_wait.human_wait_seconds("attended-window") >= 0.25
