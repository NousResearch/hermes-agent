"""execute_code approval in a legacy tui_gateway turn its own client typed.

#139674 skips the whole-script ``execute_code`` approval for a local session's own turns, reading
``viewer_turn`` and the frozen launch policy that ``gateway.session_ingress.execute_admission`` binds.
A non-canonical Desktop or dashboard client submits with
``prompt.submit`` and the turn runs in tui_gateway, which never passes through that drain: every
``execute_code`` call there still raised the whole-script prompt (or, in smart mode, a guardian
assessment). tui_gateway's turn runner now sets its own marker for every turn it runs: a live
attachment probe for a turn its client typed, None for everything else.
"""

from __future__ import annotations

import threading
import types

import pytest

from tools import approval as A
from tools import approval_context

SESSION_KEY = "legacy-tui-session"


@pytest.fixture
def legacy_turn(monkeypatch):
    """The tui_gateway process environment (``_enable_gateway_prompts``), manual mode, the session's
    bound key and its registered notifier, which captures and denies every prompt."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    prompts: list[dict] = []

    def notify(approval_data):
        prompts.append(approval_data)
        with A._lock:
            entry = A._gateway_queues[SESSION_KEY][-1]
            entry.result = "deny"
            entry.event.set()

    key_token = approval_context.set_current_session_key(SESSION_KEY)
    A.register_gateway_notify(SESSION_KEY, notify)
    with A._lock:
        A._session_approved.pop(SESSION_KEY, None)
        had_permanent = "execute_code" in A._permanent_approved
        A._permanent_approved.discard("execute_code")
    try:
        yield prompts
    finally:
        A.unregister_gateway_notify(SESSION_KEY)
        approval_context.reset_current_session_key(key_token)
        with A._lock:
            A._session_approved.pop(SESSION_KEY, None)
            A._pending.pop(SESSION_KEY, None)
            A._gateway_queues.pop(SESSION_KEY, None)
            if had_permanent:
                A._permanent_approved.add("execute_code")


def _guard(attached: bool | None):
    """Run the real execute_code guard, inside the marker when ``attached`` is not None."""
    token = None if attached is None else approval_context.bind_tui_viewer_turn(lambda: attached)
    try:
        return A.check_execute_code_guard("print('hi')", "local")
    finally:
        if token is not None:
            approval_context.reset_tui_viewer_turn(token)


def test_legacy_viewer_turn_skips_whole_script_prompt(legacy_turn):
    result = _guard(attached=True)
    assert result["approved"] is True
    assert legacy_turn == []


def test_legacy_turn_without_marker_keeps_whole_script_prompt(legacy_turn):
    result = _guard(attached=None)
    assert result["approved"] is False
    assert [p["pattern_key"] for p in legacy_turn] == ["execute_code"]


def test_legacy_viewer_turn_with_no_approving_client_keeps_gate(legacy_turn):
    result = _guard(attached=False)
    assert result["approved"] is False
    assert [p["pattern_key"] for p in legacy_turn] == ["execute_code"]


def test_delegated_child_of_legacy_viewer_turn_keeps_gate(legacy_turn, monkeypatch):
    """A delegate_task child inherits the parent's context, marker included; it must keep the gate."""
    from agent import delegation_context
    monkeypatch.setattr(delegation_context, "is_delegated_child_process_context", lambda: True)
    result = _guard(attached=True)
    assert result["approved"] is False
    assert [p["pattern_key"] for p in legacy_turn] == ["execute_code"]


def test_side_task_key_without_notifier_keeps_gate(legacy_turn):
    """/bg and /btw bind their own task key with no notifier: the marker alone does not exempt them."""
    side_token = approval_context.set_current_session_key("bg_side_task")
    try:
        result = _guard(attached=True)
    finally:
        approval_context.reset_current_session_key(side_token)
        with A._lock:
            A._pending.pop("bg_side_task", None)
            A._gateway_queues.pop("bg_side_task", None)
    assert result["approved"] is False


def test_legacy_viewer_turn_skips_smart_script_assessment(legacy_turn, monkeypatch):
    from tools import approval_smart
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    assessed: list[str] = []
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda c, d: assessed.append(c) or "escalate")
    result = _guard(attached=True)
    assert result["approved"] is True
    assert assessed == [] and legacy_turn == []


# --- the real tui_gateway turn runner sets the marker per turn ------------------------------------------


@pytest.fixture
def real_turns(monkeypatch, tmp_path):
    """Run real turns on real threads with a fake agent. Each turn records what the approval gate sees inside
    it; the viewer's turn queues an injected prompt, which the same turn thread drains at turn end
    (``_run_post_turn_followups``) as a chained turn of its own."""
    from tui_gateway import server
    for name, value in {
        "_emit": lambda *a, **k: None, "_wire_callbacks": lambda sid: None,
        "_sync_agent_model_with_config": lambda sid, session: None, "_session_cwd": lambda session: str(tmp_path),
        "_register_session_cwd": lambda session: None, "_tts_stream_begin": lambda: None,
        "_sync_session_key_after_compress": lambda *a, **k: None, "_get_usage": lambda agent: {},
        "_session_has_approving_client": lambda _sid: True, "_wait_agent_for_prompt": lambda *_a, **_k: None,
    }.items():
        monkeypatch.setattr(server, name, value)
    seen: list[tuple[str, bool]] = []
    done = threading.Event()
    session: dict = {}

    def run_conversation(message, *_a, **_k):
        text = message if isinstance(message, str) else str(message)
        seen.append((text, approval_context.is_tui_viewer_turn_with_approver()))
        if "typed" in text:
            server._enqueue_prompt(session, "injected by a plugin", None)
        else:
            done.set()
        return {"final_response": "ok"}

    agent = types.SimpleNamespace(session_id="agent-sid", run_conversation=run_conversation,
                                  clear_interrupt=lambda: None)
    session.update({
        "agent": agent, "session_key": "legacy-real-turn", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
    })

    def settle():
        """Join every turn thread, so no turn outlives the monkeypatches."""
        assert done.wait(10), seen
        while (thread := session.get("_run_thread")) is not None and thread.is_alive():
            thread.join(10)

    def via_worker(text: str):
        """prompt.submit's real worker (``_run_after_agent_ready``) for a client-typed send."""
        server._run_after_agent_ready("rid", "ui-sid", session, text, None, None, None, None)
        settle()
        return seen

    def direct(text: str, **kwargs):
        server._run_prompt_submit("rid", "ui-sid", session, text, **kwargs)
        settle()
        return seen

    return types.SimpleNamespace(via_worker=via_worker, direct=direct)


def test_marker_covers_the_typed_turn_and_not_a_prompt_drained_after_it(real_turns):
    seen = real_turns.via_worker("typed by the viewer")
    assert [flag for _text, flag in seen] == [True, False], seen
    assert "injected by a plugin" in seen[1][0]


def test_every_turn_sets_its_own_marker_over_an_inherited_one(real_turns):
    """A turn spawned from a context that already holds a viewer marker (as a chained turn's spawner
    may) must not inherit it: the turn thread copies its spawner's context."""
    token = approval_context.bind_tui_viewer_turn(lambda: True)
    try:
        seen = real_turns.direct("a notification turn")
    finally:
        approval_context.reset_tui_viewer_turn(token)
    assert [flag for _text, flag in seen] == [False], seen


@pytest.fixture
def turn_body(monkeypatch):
    """``_run_after_agent_ready`` (the client's prompt.submit worker) with the agent ready; records the
    ``viewer_typed`` it passes to the turn runner."""
    from tui_gateway import server
    passed: list[bool] = []
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **k: passed.append(k["viewer_typed"]))

    def run(display_kind=None, hosted_terminal_callback=None, turn_author=None):
        session = {"history_lock": threading.RLock(), "running": True}
        server._run_after_agent_ready("rid", "sid", session, "hello", display_kind, None,
                                      hosted_terminal_callback, turn_author)
        return passed[-1]

    return run


def test_client_typed_submit_is_viewer_typed(turn_body):
    assert turn_body() is True


@pytest.mark.parametrize("overrides", [
    {"display_kind": "hidden"},
    {"turn_author": {"id": "another-sender"}},
    {"hosted_terminal_callback": lambda _result: None},
], ids=["hidden-send", "authored-turn", "hosted-room-task"])
def test_other_submits_are_not_viewer_typed(turn_body, overrides):
    assert turn_body(**overrides) is False


def test_no_attached_session_has_no_approving_client():
    from tui_gateway import server
    assert server._session_has_approving_client("no-such-session") is False
