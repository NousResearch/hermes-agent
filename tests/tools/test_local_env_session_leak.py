"""Cross-session HERMES_SESSION_* leak guard for the local terminal backend.

Regression coverage for the bug where a terminal subprocess could observe a
*different concurrent session's* ``HERMES_SESSION_KEY`` (and the other
``HERMES_SESSION_*`` vars).

Root cause: the session vars have a process-global ``os.environ`` mirror (written
last-writer-wins as a CLI/cron fallback, never cleared), while the
concurrency-safe source of truth is a task-local ``ContextVar``. The subprocess
env was built from ``os.environ`` and only *overrode* the session vars when the
ContextVar was set+truthy. When the subprocess was spawned from a thread/context
that never inherited the agent's copied context (ContextVar ``_UNSET``), the
override no-op'd and the stale, foreign ``os.environ`` value leaked into the
child — so e.g. ``bug_thread.py whoami`` read another session's thread id.

The fix: once the session-context machinery is engaged in this process (any
concurrent host — gateway, ACP, API server, TUI, cron — has called
``set_session_vars``), the session vars are ContextVar-authoritative. The
subprocess-env bridge resolves each ``HERMES_SESSION_*`` from the ContextVar and,
when it is ``_UNSET``, STRIPS the var from the child env rather than inheriting
the process-global value that may belong to another session. A pure
single-process CLI/one-shot that never engaged the session-context system keeps
the ``os.environ`` fallback.
"""

import os

import pytest

import gateway.session_context as sc
from gateway.session_context import _VAR_MAP, clear_session_vars, set_session_vars
from tools.environments.local import _make_run_env, _sanitize_subprocess_env, hermes_subprocess_env

# The full set of session vars the bridge owns.
SESSION_VARS = list(_VAR_MAP.keys())


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + os.environ + engaged-latch slate per test, restored."""
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(sc._UNSET)
    sc._session_context_engaged = False
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _engage():
    """Mark the session-context machinery engaged, like a concurrent host would."""
    sc._session_context_engaged = True


# --------------------------------------------------------------------------- #
# Foreground path (_make_run_env)
# --------------------------------------------------------------------------- #

def test_engaged_unset_contextvar_strips_foreign_session_key(monkeypatch):
    """Engaged host + UNSET ContextVar must NOT inherit a foreign global.

    This is the production hijack: a concurrent session wrote
    os.environ["HERMES_SESSION_KEY"], this task's ContextVar is unset, and the
    subprocess must see NO key rather than the foreign one.
    """
    _engage()
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN_CONCURRENT:FOREIGN_CONCURRENT",
    )

    env = _make_run_env({})

    assert "HERMES_SESSION_KEY" not in env, (
        "Foreign concurrent session key leaked into subprocess env: "
        f"{env.get('HERMES_SESSION_KEY')!r}"
    )


def test_set_session_vars_engages_and_overrides_foreign_global(monkeypatch):
    """set_session_vars itself engages the latch and the bound value wins.

    Mirrors a real host: calling set_session_vars both marks the process engaged
    and binds the ContextVar, so the bound value overrides the foreign global.
    """
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN:FOREIGN",
    )

    tokens = set_session_vars(
        session_key="agent:main:discord:group:MY_BUGS_ROOT:111",
        platform="discord",
        chat_id="MY_BUGS_ROOT",
    )
    try:
        assert sc.session_context_engaged() is True
        env = _make_run_env({})
    finally:
        clear_session_vars(tokens)

    assert env.get("HERMES_SESSION_KEY") == "agent:main:discord:group:MY_BUGS_ROOT:111"


def test_engaged_strips_all_session_vars_when_unset(monkeypatch):
    """The strip covers every HERMES_SESSION_* mirror, not just the key."""
    _engage()
    monkeypatch.setenv("HERMES_SESSION_KEY", "foreign-key")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "foreign-thread")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "foreign-chat")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "foreign-user")

    env = _make_run_env({})

    for var in (
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_USER_ID",
    ):
        assert var not in env, f"{var} leaked from a foreign global: {env.get(var)!r}"


def test_unengaged_process_preserves_os_environ_fallback(monkeypatch):
    """A process that never engaged the session-context system keeps the fallback.

    Pure single-process CLI/one-shot sets HERMES_SESSION_* directly in os.environ
    and relies on the subprocess inheriting them; there is no concurrency to leak
    across, so the strip must NOT apply.
    """
    # _isolate_session_context already forced engaged=False.
    monkeypatch.setenv("HERMES_SESSION_KEY", "cli-session-key")
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-id")

    env = _make_run_env({})

    assert env.get("HERMES_SESSION_KEY") == "cli-session-key"
    assert env.get("HERMES_SESSION_ID") == "cli-session-id"


def test_engaged_explicit_empty_contextvar_clears(monkeypatch):
    """An explicitly-cleared ContextVar ("" via clear_session_vars) clears the var.

    After a handler finishes it calls clear_session_vars which sets each var to
    "" (distinct from _UNSET). A subprocess spawned in that window must see the
    empty value (which overrides the foreign global), NOT the foreign global —
    an empty key is safe (whoami reads "" → no thread).
    """
    monkeypatch.setenv("HERMES_SESSION_KEY", "foreign-after-clear")

    tokens = set_session_vars(session_key="real-key", platform="discord", chat_id="c")
    clear_session_vars(tokens)  # sets vars to "" (explicitly cleared); stays engaged

    env = _make_run_env({})

    # Explicit-empty wins over the foreign global: either stripped or "" — never
    # the foreign value. Both outcomes are safe for the consumer.
    assert env.get("HERMES_SESSION_KEY", "") == "", (
        f"Foreign key survived an explicit clear: {env.get('HERMES_SESSION_KEY')!r}"
    )


def test_explicit_empty_thread_id_overrides_stale_value(monkeypatch):
    """A bound-but-empty thread id must override a stale inherited value.

    This is the complementary case (the #38507 scenario): a top-level post with
    no thread id binds HERMES_SESSION_THREAD_ID="" and that empty value must win
    over an older non-empty value left in os.environ.
    """
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "stale-thread-from-prior-turn")

    tokens = set_session_vars(
        session_key="mm:chan",
        platform="mattermost",
        chat_id="chan",
        thread_id="",  # explicitly no thread
    )
    try:
        env = _make_run_env({})
    finally:
        clear_session_vars(tokens)

    assert env.get("HERMES_SESSION_THREAD_ID") == "", (
        "Bound-empty thread id did not override the stale value: "
        f"{env.get('HERMES_SESSION_THREAD_ID')!r}"
    )
    assert env.get("HERMES_SESSION_KEY") == "mm:chan"


# --------------------------------------------------------------------------- #
# Background / PTY path (_sanitize_subprocess_env via process_registry)
# --------------------------------------------------------------------------- #

def test_sanitize_subprocess_env_strips_foreign_session_key_when_engaged(monkeypatch):
    """The background/PTY spawn path gets the same cross-session strip.

    process_registry.spawn_local() builds its env via _sanitize_subprocess_env(
    os.environ, env_vars). A background subprocess spawned with an UNSET
    ContextVar in an engaged process must not inherit a foreign session key.
    """
    _engage()
    stale_base = {
        "PATH": "/usr/bin:/bin",
        "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN_BG:FOREIGN_BG",
        "HERMES_SESSION_THREAD_ID": "FOREIGN_BG",
    }

    sanitized = _sanitize_subprocess_env(stale_base)

    assert "HERMES_SESSION_KEY" not in sanitized, (
        f"Background subprocess inherited foreign key: {sanitized.get('HERMES_SESSION_KEY')!r}"
    )
    assert "HERMES_SESSION_THREAD_ID" not in sanitized


def test_sanitize_subprocess_env_set_contextvar_wins_when_engaged():
    """Background path: a SET ContextVar overrides the foreign global base."""
    stale_base = {
        "PATH": "/usr/bin:/bin",
        "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN_BG:FOREIGN_BG",
    }
    tokens = set_session_vars(
        session_key="agent:main:discord:group:REAL_BG:222",
        platform="discord",
        chat_id="REAL_BG",
    )
    try:
        sanitized = _sanitize_subprocess_env(stale_base)
    finally:
        clear_session_vars(tokens)

    assert sanitized.get("HERMES_SESSION_KEY") == "agent:main:discord:group:REAL_BG:222"


def test_sanitize_subprocess_env_unengaged_preserves_fallback(monkeypatch):
    """Background path in an unengaged process keeps the inherited value."""
    stale_base = {
        "PATH": "/usr/bin:/bin",
        "HERMES_SESSION_KEY": "cli-bg-key",
    }

    sanitized = _sanitize_subprocess_env(stale_base)

    assert sanitized.get("HERMES_SESSION_KEY") == "cli-bg-key"


# --------------------------------------------------------------------------- #
# Non-terminal spawn surface (hermes_subprocess_env) — sibling path
# --------------------------------------------------------------------------- #

def test_hermes_subprocess_env_strips_foreign_session_key_when_engaged(monkeypatch):
    """hermes_subprocess_env (browser/ACP/CLI/TUI-host spawns) must not leak a
    foreign session key either. cli.exec spawns via this helper WITHOUT re-binding
    the session identity, so an UNSET ContextVar under an engaged host must strip
    the inherited global rather than hand the child another session's identity.
    """
    _engage()
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN_CONCURRENT:FOREIGN_CONCURRENT",
    )

    env = hermes_subprocess_env()

    assert "HERMES_SESSION_KEY" not in env, (
        "Foreign concurrent session key leaked into non-terminal spawn env: "
        f"{env.get('HERMES_SESSION_KEY')!r}"
    )


def test_hermes_subprocess_env_bound_contextvar_wins(monkeypatch):
    """A caller that binds the session identity keeps it through this helper."""
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN:FOREIGN",
    )
    tokens = set_session_vars(
        session_key="agent:main:discord:group:MINE:111",
        platform="discord",
        chat_id="MINE",
    )
    try:
        env = hermes_subprocess_env()
        assert env.get("HERMES_SESSION_KEY") == "agent:main:discord:group:MINE:111"
    finally:
        clear_session_vars(tokens)


def test_hermes_subprocess_env_unengaged_preserves_fallback(monkeypatch):
    """A pure single-process CLI (never engaged) keeps the inherited fallback."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "cli-fallback-key")
    # not engaged (autouse fixture leaves _session_context_engaged False)
    env = hermes_subprocess_env()
    assert env.get("HERMES_SESSION_KEY") == "cli-fallback-key"
