"""Tests for the sudo password cache and interactive prompt helpers.

Covers ``tools/terminal_tool.py`` functions:

* ``_get_sudo_password_cache_scope`` / get/set/reset of the scoped cache
* ``_sudo_wrong_password_failure`` / ``_invalidate_cached_sudo_on_auth_failure``
* ``_prompt_for_sudo_password`` (callback delegation + non-interactive fallback)
* ``_sudo_nopasswd_works``
"""

import threading

import pytest

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


# ---------------------------------------------------------------------------
# _get_sudo_password_cache_scope
# ---------------------------------------------------------------------------

def test_sudo_scope_prefers_session_key(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-abc")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    assert terminal_tool._get_sudo_password_cache_scope() == "session:sess-abc"


def test_sudo_scope_session_key_takes_priority_over_callback(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-abc")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: object())
    assert terminal_tool._get_sudo_password_cache_scope() == "session:sess-abc"


def test_sudo_scope_plain_callback_uses_callback_id(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")

    def plain_cb():
        pass

    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: plain_cb)
    assert terminal_tool._get_sudo_password_cache_scope() == f"callback:{id(plain_cb)}"


def test_sudo_scope_bound_method_callback_uses_owner_and_func(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")

    class _Owner:
        def prompt(self):
            pass

    owner = _Owner()
    bound = owner.prompt
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: bound)
    assert (
        terminal_tool._get_sudo_password_cache_scope()
        == f"callback-owner:{id(owner)}:{id(bound.__func__)}"
    )


def test_sudo_scope_falls_back_to_thread_id(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    assert terminal_tool._get_sudo_password_cache_scope() == f"thread:{threading.get_ident()}"


# ---------------------------------------------------------------------------
# _get_cached_sudo_password / _set_cached_sudo_password / _reset
# ---------------------------------------------------------------------------

def test_get_cached_sudo_password_empty_by_default(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    assert terminal_tool._get_cached_sudo_password() == ""


def test_set_then_get_cached_sudo_password(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    terminal_tool._set_cached_sudo_password("s3cret")
    assert terminal_tool._get_cached_sudo_password() == "s3cret"


def test_set_cached_sudo_password_empty_removes_entry(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    terminal_tool._set_cached_sudo_password("s3cret")
    assert terminal_tool._get_cached_sudo_password() == "s3cret"
    terminal_tool._set_cached_sudo_password("")
    assert terminal_tool._get_cached_sudo_password() == ""


def test_set_cached_sudo_password_without_setting_empty_is_absent(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)
    terminal_tool._set_cached_sudo_password("")
    # Empty string never creates a cache entry.
    assert terminal_tool._get_cached_sudo_password() == ""


def test_cached_sudo_password_scoped_by_session(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-1")
    terminal_tool._set_cached_sudo_password("pw1")
    assert terminal_tool._get_cached_sudo_password() == "pw1"
    # A different session key reads a different (empty) slot.
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-2")
    assert terminal_tool._get_cached_sudo_password() == ""


def test_reset_cached_sudo_passwords_clears_all(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-1")
    terminal_tool._set_cached_sudo_password("pw1")
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-2")
    terminal_tool._set_cached_sudo_password("pw2")
    terminal_tool._reset_cached_sudo_passwords()
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-1")
    assert terminal_tool._get_cached_sudo_password() == ""
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "sess-2")
    assert terminal_tool._get_cached_sudo_password() == ""


# ---------------------------------------------------------------------------
# _sudo_wrong_password_failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output",
    [
        "sudo: authentication failed",
        "something before\nsudo: incorrect password attempt\nafter",
        "sudo: maximum 3 incorrect authentication attempts",
        "sudo: 3 incorrect password attempts",
        "SUDO: AUTHENTICATION FAILED",  # case-insensitive
    ],
)
def test_sudo_wrong_password_failure_true_on_markers(output):
    assert terminal_tool._sudo_wrong_password_failure(output) is True


@pytest.mark.parametrize(
    "output",
    ["", None, "sudo succeeded", "permission denied", "command not found"],
)
def test_sudo_wrong_password_failure_false_otherwise(output):
    assert terminal_tool._sudo_wrong_password_failure(output) is False


# ---------------------------------------------------------------------------
# _invalidate_cached_sudo_on_auth_failure
# ---------------------------------------------------------------------------

def _thread_scope(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: "")
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)


def test_invalidate_auth_failure_returns_false_when_sudo_password_in_env(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "operator-set")
    _thread_scope(monkeypatch)
    terminal_tool._set_cached_sudo_password("cached")
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure(
            "sudo true", "sudo: authentication failed"
        )
        is False
    )
    # The env-configured password path is left alone: cache preserved.
    assert terminal_tool._get_cached_sudo_password() == "cached"


def test_invalidate_auth_failure_returns_false_without_sudo_failure(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    _thread_scope(monkeypatch)
    terminal_tool._set_cached_sudo_password("cached")
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure("sudo true", "done")
        is False
    )
    assert terminal_tool._get_cached_sudo_password() == "cached"


def test_invalidate_auth_failure_returns_false_without_real_sudo(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    _thread_scope(monkeypatch)
    terminal_tool._set_cached_sudo_password("cached")
    # "sudo" is an argument (grep pattern), not a real sudo invocation.
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure(
            "grep -r sudo .", "sudo: authentication failed"
        )
        is False
    )
    assert terminal_tool._get_cached_sudo_password() == "cached"


def test_invalidate_auth_failure_returns_false_when_no_cached_password(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    _thread_scope(monkeypatch)
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure(
            "sudo true", "sudo: authentication failed"
        )
        is False
    )


def test_invalidate_auth_failure_clears_cache_when_all_conditions_met(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    _thread_scope(monkeypatch)
    terminal_tool._set_cached_sudo_password("bad-pw")
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure(
            "sudo apt install -y curl", "sudo: authentication failed"
        )
        is True
    )
    assert terminal_tool._get_cached_sudo_password() == ""


def test_invalidate_auth_failure_short_circuits_on_command_none(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    _thread_scope(monkeypatch)
    terminal_tool._set_cached_sudo_password("bad-pw")
    # None command coerces to "" -> zero real sudo invocations -> False.
    assert (
        terminal_tool._invalidate_cached_sudo_on_auth_failure(None, "sudo: authentication failed")
        is False
    )


# ---------------------------------------------------------------------------
# _prompt_for_sudo_password
# ---------------------------------------------------------------------------

def test_prompt_delegates_to_registered_callback(monkeypatch):
    calls = []

    def mock_cb():
        calls.append(True)
        return "typed-pw"

    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: mock_cb)
    assert terminal_tool._prompt_for_sudo_password(timeout_seconds=1) == "typed-pw"
    assert calls == [True]


def test_prompt_returns_empty_when_callback_returns_none(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: lambda: None)
    assert terminal_tool._prompt_for_sudo_password(timeout_seconds=1) == ""


def test_prompt_returns_empty_when_callback_returns_empty_string(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: lambda: "")
    assert terminal_tool._prompt_for_sudo_password(timeout_seconds=1) == ""


def test_prompt_returns_empty_when_callback_raises(monkeypatch):
    def exploding_cb():
        raise RuntimeError("boom")

    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: exploding_cb)
    assert terminal_tool._prompt_for_sudo_password(timeout_seconds=1) == ""


def test_prompt_non_interactive_no_tty_returns_empty(monkeypatch):
    """Without a callback, a headless/uninteractive context never prompts.

    The /dev/tty read path is exercised with ``os.open`` forced to fail, as it
    does when no controlling terminal (or raw non-interactive child) is
    attached; the function must degrade to an empty string instead of hanging.
    """
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: None)

    def _no_tty(*_args, **_kwargs):
        raise OSError("no controlling terminal")

    monkeypatch.setattr(terminal_tool.os, "open", _no_tty)
    monkeypatch.setattr(terminal_tool.os.environ, "__contains__", lambda k: False)

    assert terminal_tool._prompt_for_sudo_password(timeout_seconds=1) == ""


# ---------------------------------------------------------------------------
# _sudo_nopasswd_works
# ---------------------------------------------------------------------------

def test_sudo_nopasswd_returns_false_when_terminal_env_not_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(terminal_tool, "subprocess", _never_called_subprocess())
    assert terminal_tool._sudo_nopasswd_works() is False


def test_sudo_nopasswd_returns_true_on_subprocess_success(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(terminal_tool, "subprocess", _stub_subprocess(rc=0))
    assert terminal_tool._sudo_nopasswd_works() is True


def test_sudo_nopasswd_returns_false_on_subprocess_failure(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(terminal_tool, "subprocess", _stub_subprocess(rc=1))
    assert terminal_tool._sudo_nopasswd_works() is False


def test_sudo_nopasswd_returns_false_when_subprocess_raises(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(terminal_tool, "subprocess", _raising_subprocess())
    assert terminal_tool._sudo_nopasswd_works() is False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stub_subprocess(rc):
    class _Stub:
        DEVNULL = object()

        @staticmethod
        def run(*_args, **_kwargs):
            return type("Probe", (), {"returncode": rc})()

    return _Stub()


def _raising_subprocess():
    class _Stub:
        DEVNULL = object()

        @staticmethod
        def run(*_args, **_kwargs):
            raise OSError("no sudo")

    return _Stub()


def _never_called_subprocess():
    class _Stub:
        DEVNULL = object()

        @staticmethod
        def run(*_args, **_kwargs):
            raise AssertionError("subprocess.run should not be called for non-local TERMINAL_ENV")

    return _Stub()
