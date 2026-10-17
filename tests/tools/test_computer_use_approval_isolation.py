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
