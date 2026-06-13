import contextvars
import threading

import pytest

import tools.thread_context as thread_context
from tools.terminal_tool import (
    _get_approval_callback,
    _get_sudo_password_callback,
    set_approval_callback,
    set_sudo_password_callback,
)


@pytest.fixture(autouse=True)
def clear_terminal_callbacks():
    set_approval_callback(None)
    set_sudo_password_callback(None)
    yield
    set_approval_callback(None)
    set_sudo_password_callback(None)


def test_propagates_contextvars_snapshot_to_worker_thread():
    marker = contextvars.ContextVar("marker", default="unset")
    marker.set("parent-value")

    def target(prefix):
        return f"{prefix}:{marker.get()}"

    wrapped = thread_context.propagate_context_to_thread(target)
    marker.set("changed-after-wrap")

    result = {}

    def worker():
        result["value"] = wrapped("seen")

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert result["value"] == "seen:parent-value"


def test_installs_and_clears_terminal_callbacks_on_worker_thread():
    def approval_cb(*_args, **_kwargs):
        return True

    def sudo_cb(*_args, **_kwargs):
        return "secret"

    set_approval_callback(approval_cb)
    set_sudo_password_callback(sudo_cb)

    def target():
        return (_get_approval_callback(), _get_sudo_password_callback())

    wrapped = thread_context.propagate_context_to_thread(target)
    result = {}

    def worker():
        result["during"] = wrapped()
        result["after"] = (_get_approval_callback(), _get_sudo_password_callback())

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert result["during"] == (approval_cb, sudo_cb)
    assert result["after"] == (None, None)
    assert _get_approval_callback() is approval_cb
    assert _get_sudo_password_callback() is sudo_cb


def test_clears_callbacks_when_target_raises():
    set_approval_callback(lambda *_args, **_kwargs: True)
    set_sudo_password_callback(lambda *_args, **_kwargs: "secret")

    def target():
        raise RuntimeError("boom")

    wrapped = thread_context.propagate_context_to_thread(target)
    result = {}

    def worker():
        with pytest.raises(RuntimeError, match="boom"):
            wrapped()
        result["after"] = (_get_approval_callback(), _get_sudo_password_callback())

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert result["after"] == (None, None)


def test_callback_capture_failure_still_runs_target(monkeypatch):
    marker = contextvars.ContextVar("capture_failure_marker", default="unset")
    marker.set("captured")

    def broken_api():
        raise RuntimeError("terminal_tool unavailable")

    monkeypatch.setattr(thread_context, "_callback_api", broken_api)

    wrapped = thread_context.propagate_context_to_thread(lambda: marker.get())

    assert wrapped() == "captured"


def test_callback_install_and_clear_fail_closed(monkeypatch):
    calls = []

    def get_approval():
        return object()

    def get_sudo():
        return None

    def set_approval(value):
        calls.append(value)
        raise RuntimeError("setter failed")

    def set_sudo(value):
        calls.append(value)

    monkeypatch.setattr(
        thread_context,
        "_callback_api",
        lambda: (get_approval, get_sudo, set_approval, set_sudo),
    )

    wrapped = thread_context.propagate_context_to_thread(lambda: "ran")

    assert wrapped() == "ran"
    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is None
