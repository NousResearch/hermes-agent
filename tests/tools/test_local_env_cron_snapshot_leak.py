"""Regression test for issue #64199: terminal snapshot leaks cron delivery context.

A local terminal environment snapshot can retain ``HERMES_CRON_AUTO_DELIVER_*``
values from one cron run and override the correct task-local values in a later,
unrelated cron run. The shell snapshot (``export -p``) is sourced *after* the
authoritative task-local ContextVars are bridged into the child env, so a stale
snapshot copy wins and a nested ``hermes send`` gets the wrong dedupe target.

Fix (tools/environments/base.py): runtime-scoped Hermes routing vars
(``HERMES_CRON_AUTO_DELIVER_*``) are excluded from the persisted snapshot and
unset from the child shell after the snapshot is sourced, so the per-call
ContextVar values (re-injected via ``_inject_session_context_env``) remain
authoritative.
"""

import json
import os

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools.environments.local import LocalEnvironment

_CRON_VARS = (
    "HERMES_CRON_AUTO_DELIVER_PLATFORM",
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
)


@pytest.fixture(autouse=True)
def _isolate_cron_context():
    """Snapshot and restore the three cron ContextVars around each test."""
    saved = {name: _VAR_MAP[name].get() for name in _CRON_VARS}
    for name in _CRON_VARS:
        _VAR_MAP[name].set(_UNSET)
    try:
        yield
    finally:
        for name in _CRON_VARS:
            _VAR_MAP[name].set(saved[name])


def _set_context(platform, chat_id, thread_id):
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set(platform)
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set(chat_id)
    _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set(thread_id)


def _read_cron_env(env: LocalEnvironment) -> dict:
    """Run a command that dumps the cron delivery vars from the child env."""
    code = (
        "import json, os; "
        "print(json.dumps({k: os.environ.get(k) for k in "
        + repr(list(_CRON_VARS))
        + "}, sort_keys=True))"
    )
    result = env.execute(f"python3 -c {code!r}", timeout=30)
    return json.loads(result["output"].strip().splitlines()[-1])


def test_snapshot_does_not_leak_cron_context_between_jobs():
    """Reusing one LocalEnvironment across two cron contexts must see the 2nd.

    Mirrors the issue's reproduction: job A sets Telegram delivery, the
    snapshot captures it, then job B switches to ntfy and must see ntfy — not
    the stale Telegram values.
    """
    env = LocalEnvironment(cwd="/tmp", timeout=30)
    try:
        # Cron job A: auto-deliver to Telegram.
        _set_context("telegram", "-100111", "")
        _read_cron_env(env)  # forces the snapshot to capture context A

        # Unrelated cron job B: auto-deliver to ntfy.
        _set_context("ntfy", "alerts", "")

        observed = _read_cron_env(env)
        assert observed == {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "ntfy",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "alerts",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "",
        }, f"stale snapshot leaked: {observed}"
    finally:
        env.cleanup()


def test_snapshot_redump_excludes_cron_vars():
    """The persisted snapshot file must not contain HERMES_CRON_AUTO_DELIVER_*."""
    _set_context("telegram", "-100111", "9")

    env = LocalEnvironment(cwd="/tmp", timeout=30)
    try:
        _read_cron_env(env)  # populate + redump the snapshot
        with open(env._snapshot_path, "r", encoding="utf-8", errors="replace") as fh:
            snapshot = fh.read()
        for var in _CRON_VARS:
            assert f'HERMES_CRON_AUTO_DELIVER' not in snapshot or \
                f'"{var}"' not in snapshot, \
                f"{var} leaked into the persisted snapshot"
    finally:
        env.cleanup()
