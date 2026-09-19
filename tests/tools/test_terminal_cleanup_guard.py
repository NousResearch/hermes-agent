"""Issue #114362: a reaped sandbox must fail loudly, never be reused silently.

_cleanup_inactive_envs expires envs purely on _last_activity (plus a
background-process refresh), so an env whose owning conversation is mid
provider-call can be reaped. The reaped handle must then be marked, and the
next foreground use must refuse to run against the dead sandbox with an
explicit error instead of silently executing on it.
"""

import json
import time
from types import SimpleNamespace

import tools.terminal_tool as tt
from tools.terminal_tool_lifecycle import _cleanup_inactive_envs, is_reaped_env


TASK = "t-cleanup-guard-114362"


def setup_function(_func):
    tt._active_environments.pop(TASK, None)
    tt._last_activity.pop(TASK, None)


def teardown_function(_func):
    tt._active_environments.pop(TASK, None)
    tt._last_activity.pop(TASK, None)


def _register_idle_fake(**ns_kwargs):
    fake = SimpleNamespace(**ns_kwargs)
    tt._active_environments[TASK] = fake
    tt._last_activity[TASK] = time.time() - 3600  # idle past any test lifetime
    return fake


def test_reaped_env_is_marked():
    """An env torn down by the idle reaper carries the reaped marker."""
    fake = _register_idle_fake()
    assert not is_reaped_env(fake)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert TASK not in tt._active_environments
    assert is_reaped_env(fake)


def _foreground_plan():
    return tt._ExecPlan(
        config={}, env_type="local", effective_task_id=TASK,
        image="", cwd="/tmp", host_cwd=None, effective_timeout=30,
    )


def test_run_foreground_fails_loudly_on_reaped_env():
    """A reaped handle is refused with an explicit error; execute never runs."""
    calls = []
    fake = _register_idle_fake(
        execute=lambda *a, **k: (calls.append(1), {"output": "", "exit_code": 0})[1],
    )
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert TASK not in tt._active_environments  # reap happened
    assert is_reaped_env(fake)

    try:
        raw = tt._run_foreground(
            "echo hi", fake, _foreground_plan(), task_id=TASK,
            session_id=None, session_key=TASK, workdir=None,
            approval_note=None, clear_interrupt=False,
        )
    except Exception:
        raw = None  # any raise still proves the point if execute ran
    assert calls == [], "reaped sandbox handle was executed against"
    assert raw is not None
    payload = json.loads(raw)
    assert payload.get("error"), f"expected explicit error, got: {raw!r}"
    assert "reclaim" in payload["error"].lower() or "reap" in payload["error"].lower(), raw
