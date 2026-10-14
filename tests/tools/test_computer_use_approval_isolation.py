"""computer_use approval is the shared ``tools.approval`` gate — no private grant store, no default-allow.

Two contracts:

* With nobody able to answer (no interactive CLI, no gateway, yolo off) a destructive action is REFUSED and
  never reaches the backend; under yolo it runs. Historically the tool default-allowed whenever no CLI callback
  was wired, which made every headless host (cron, api_server, tui_gateway, gateway turns) run desktop input
  ungated.
* A grant answered through computer_use lives in ``tools.approval``'s store under computer_use's own scope key,
  so ``is_approved`` sees it and ``clear_session`` retires it like any terminal pattern.

A leaked callback still poisons later tests (a raising one becomes deny, a blocking one hangs), so the autouse
reset in ``tests/conftest.py`` stays and the polluter/observer pair below keeps proving it.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _install_backend(cu_tool):
    class _RecordingBackend:
        def __init__(self):
            self.calls = []

        def start(self):
            pass

        def stop(self):
            pass

        def is_available(self):
            return True

        def click(self, **kw):
            self.calls.append(("click", kw))
            from tools.computer_use.backend import ActionResult

            return ActionResult(ok=True, action="click")

        def capture(self, mode="som", app=None):
            from tools.computer_use.backend import CaptureResult

            return CaptureResult(
                mode=mode, width=1, height=1, png_b64=None, elements=[],
                app="X", window_title="",
            )

    backend = _RecordingBackend()
    cu_tool.reset_backend_for_tests()
    cu_tool._backend = backend
    return backend


@pytest.fixture
def _nobody_to_ask(monkeypatch):
    """No interactive CLI, no gateway, no per-thread terminal callback, yolo off."""
    from tools import approval

    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None)
    yield


def test_no_callback_refuses_unless_yolo(_nobody_to_ask, monkeypatch):
    """Fail closed: with no human reachable the click is blocked and the backend sees nothing; yolo lets it run."""
    from tools import approval
    from tools.computer_use import tool as cu_tool

    backend = _install_backend(cu_tool)
    result = json.loads(cu_tool.handle_computer_use({"action": "click", "element": 3}))
    assert result["error"].startswith("BLOCKED"), result
    assert result["action"] == "click"
    assert backend.calls == []

    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    result = cu_tool.handle_computer_use({"action": "click", "element": 3})
    assert [name for name, _ in backend.calls] == ["click"], result


def test_always_grant_lands_in_the_shared_store(monkeypatch):
    """One grant store: an "always" answered through computer_use is what ``tools.approval.is_approved`` reports
    for the same session and ``cua:<action>:<mode>`` key, and the next call is served from that store."""
    from tools import approval
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "save_permanent_allowlist", lambda patterns: None)
    prompts = []
    cu_tool.set_approval_callback(lambda command, description, **kw: prompts.append(command) or "always")
    token = set_current_session_key("cua-grant-session")
    try:
        assert not approval.is_approved("cua-grant-session", "cua:click:background")
        assert cu_tool._request_approval("click", {"element": 3}) is None
        assert approval.is_approved("cua-grant-session", "cua:click:background")
        assert cu_tool._request_approval("click", {"element": 3}) is None
        assert len(prompts) == 1
    finally:
        cu_tool.set_approval_callback(None)
        reset_current_session_key(token)
        approval.clear_session("cua-grant-session")
        with approval._lock:
            approval._permanent_set().discard("cua:click:background")


def test_a_forgets_a_poisoned_approval_callback():
    """Simulates the polluter: installs a raising callback and deliberately does not reset it."""
    from tools.computer_use import tool as cu_tool

    def poisoned(command, description, **kw):
        raise RuntimeError("dead UI")

    cu_tool.set_approval_callback(poisoned)
    # no reset — the autouse fixture must clean this up


def test_b_still_dispatches_after_the_polluter(monkeypatch):
    """Answers through the per-thread terminal callback only. The explicit computer_use callback takes precedence
    in the shared gate, so if the polluter's raising one had leaked, this click would be denied."""
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: lambda command, description, **kw: "once")
    backend = _install_backend(cu_tool)
    result = cu_tool.handle_computer_use({"action": "click", "element": 3})
    assert [name for name, _ in backend.calls] == ["click"], f"leaked approval callback poisoned this test: {result!r}"


def test_colliding_profile_session_ids_keep_computer_use_authority_and_cleanup_separate(tmp_path, monkeypatch):
    """A → B → A with one raw id: prompts, session grants, YOLO daemon mode and close stay with their owner."""
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools import approval, approval_context
    from tools.bot_desktop import lease as desktop_lease
    from tools.bot_desktop import runtime as desktop_runtime
    from tools.computer_use import cua_backend
    from tools.computer_use import tool as cu_tool
    from tools.computer_use.backend import ActionResult

    session_id = "shared-session"
    homes = [tmp_path / "profiles" / name for name in ("a", "b")]
    for home in homes:
        home.mkdir(parents=True)

    @contextmanager
    def owner(home):
        home_token = set_hermes_home_override(home)
        session_token = approval_context.set_current_session_key(session_id)
        try:
            yield
        finally:
            approval_context.reset_current_session_key(session_token)
            reset_hermes_home_override(home_token)

    class _Backend:
        def __init__(self, permission_mode):
            self.permission_mode = permission_mode
            self.calls = 0
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def click(self, **_kwargs):
            self.calls += 1
            return ActionResult(ok=True, action="click")

    prompts = []
    lease = SimpleNamespace(epoch=0)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(desktop_runtime, "ensure_started_for_tool", lambda: None)
    monkeypatch.setattr(desktop_lease, "assert_agent_may_act", lambda: lease)
    monkeypatch.setattr(desktop_lease, "get", lambda: lease)
    monkeypatch.setattr(cua_backend, "desktop_identity", lambda: "")
    monkeypatch.setattr(cua_backend, "backend_display_stale", lambda recorded, current: recorded != current)
    monkeypatch.setattr(cu_tool, "_new_backend", lambda mode: _Backend(mode))
    cu_tool.set_approval_callback(
        lambda command, description, **kwargs: prompts.append(hermes_home_key()) or "session"
    )
    cu_tool.reset_backend_for_tests()
    approval.clear_session(session_id)

    def click():
        result = json.loads(cu_tool.handle_computer_use(
            {"action": "click", "element": 3}, session_id=session_id
        ))
        assert "error" not in result, result

    try:
        with owner(homes[0]):
            click()
            first_a = cu_tool._get_backend(session_id)
            click()  # A's session grant suppresses only A's second prompt.
            approval.enable_session_yolo(session_id)
            assert getattr(first_a, "stopped")
            click()
            yolo_a = cu_tool._get_backend(session_id)
            assert getattr(yolo_a, "permission_mode") == "unrestricted"

        with owner(homes[1]):
            click()  # B must prompt despite A's grant and YOLO state.
            backend_b = cu_tool._get_backend(session_id)
            click()
            assert getattr(backend_b, "permission_mode") == "standard"

        with owner(homes[0]):
            click()  # A still owns its unrestricted backend and needs no prompt.
            assert cu_tool._get_backend(session_id) is yolo_a
            approval.clear_session(session_id)
            assert getattr(yolo_a, "stopped")
            assert not approval.is_session_yolo_enabled(session_id)
            a_grant_after_close = approval.is_approved(session_id, "cua:click:background")

        with owner(homes[1]):
            click()  # Closing A cannot revoke B's grant or stop B's backend.
            b_grant_after_a_close = approval.is_approved(session_id, "cua:click:background")
            backend_b_after_a_close = cu_tool._get_backend(session_id)

        with owner(homes[0]):
            click()  # Reopened A prompts again and gets a fresh standard backend.
            reopened_a = cu_tool._get_backend(session_id)

        assert prompts == [hermes_home_key(homes[0]), hermes_home_key(homes[1]), hermes_home_key(homes[0])]
        assert a_grant_after_close is False
        assert b_grant_after_a_close is True
        assert backend_b_after_a_close is backend_b and not getattr(backend_b, "stopped")
        assert reopened_a not in {first_a, yolo_a, backend_b}
        assert getattr(reopened_a, "permission_mode") == "standard"
    finally:
        for home in homes:
            with owner(home):
                approval.clear_session(session_id)
        approval.clear_session(session_id)
        cu_tool.reset_backend_for_tests()
