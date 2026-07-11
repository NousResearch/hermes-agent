"""Session-identity vars must never persist in the shared env snapshot.

2026-07-10 misbinding mechanism (finally pinned): the persistent env snapshot
(`export -p` dump, re-sourced by every subsequent command) outlives a turn,
and the gateway's "default" LocalEnvironment is SHARED across sessions.
Session B's turn re-dumped its injected HERMES_SESSION_* into the snapshot;
session A's next command sourced it AFTER Popen env injection, so A's
subprocess ran with B's complete coherent identity — self-perpetuating via
the end-of-command re-dump. Same vector leaked a hermetic test's
HERMES_HOME=/tmp/dw_e2e/home into a live interactive shell.

The fix filters HERMES_SESSION_* and HERMES_HOME out of both snapshot dumps
(init_session bootstrap + per-command re-dump); those vars are per-spawn
injected and the injection must stay authoritative.
"""
from __future__ import annotations

import os
import re

import pytest

from tools.environments.local import LocalEnvironment
from tools.environments.base import _SNAPSHOT_EXCLUDE_PATTERN


pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX shell semantics")


@pytest.fixture()
def env(tmp_path):
    e = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    try:
        e.init_session()
        yield e
    finally:
        try:
            e.cleanup()
        except Exception:
            pass


def _snapshot_text(e) -> str:
    with open(e._snapshot_path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


class TestSnapshotExcludesSessionIdentity:
    def test_incident_sequence_foreign_identity_does_not_stick(self, env):
        """THE incident: session B's turn exports its identity; session A's
        later command must NOT see B's identity from the snapshot."""
        # Session B's turn: identity lands in the shell env (as the per-spawn
        # injection does) and the end-of-command re-dump runs.
        res = env.execute(
            "export HERMES_SESSION_KEY='agent:main:discord:thread:B:B' "
            "HERMES_SESSION_CHAT_ID=B HERMES_HOME=/tmp/foreign_home; true"
        )
        assert res["returncode"] == 0
        snap = _snapshot_text(env)
        assert "HERMES_SESSION_KEY" not in snap, "identity leaked into snapshot"
        assert "HERMES_SESSION_CHAT_ID" not in snap
        assert "/tmp/foreign_home" not in snap

        # Session A's command: no injection here (plain execute), so if the
        # snapshot carried B's identity it would surface now.
        res = env.execute(
            "echo KEY=${HERMES_SESSION_KEY:-EMPTY} HOME_=${HERMES_HOME:-EMPTY}"
        )
        out = res["output"]
        assert res["returncode"] == 0
        assert "KEY=EMPTY" in out, f"foreign identity persisted: {out!r}"
        # HERMES_HOME may legitimately be inherited from the *process* env
        # (Popen injection is authoritative); the contract is that B's
        # foreign value must never win.  Assert equality with the expected
        # inherited value — not a weak "foreign string absent" disjunction.
        expected_home = os.environ.get("HERMES_HOME") or "EMPTY"
        assert f"HOME_={expected_home}" in out, (
            f"HERMES_HOME not the process-inherited value: {out!r}"
        )
        assert "/tmp/foreign_home" not in out

    def test_non_session_vars_still_persist(self, env):
        """The snapshot's PURPOSE (env persistence across calls) must survive
        the filter — only session-identity vars are excluded."""
        res = env.execute("export MY_PROJECT_VAR=hello_world; true")
        assert res["returncode"] == 0
        res = env.execute("echo VAL=${MY_PROJECT_VAR:-MISSING}")
        out = res["output"]
        assert res["returncode"] == 0
        assert "VAL=hello_world" in out

    def test_bootstrap_dump_also_filtered(self, tmp_path, monkeypatch):
        """init_session's FIRST dump (login-shell bootstrap) is filtered too —
        a gateway process env carrying session vars must not seed them."""
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:bootleak:X:X")
        e = LocalEnvironment(cwd=str(tmp_path), timeout=60)
        try:
            e.init_session()
            snap = _snapshot_text(e)
            assert "bootleak" not in snap
        finally:
            e.cleanup()

    def test_exclude_pattern_shape(self):
        """Contract for the filter regex itself: matches both bash and plain
        export forms for every session var + HERMES_HOME; does NOT match
        lookalikes that legitimately persist."""
        rx = re.compile(_SNAPSHOT_EXCLUDE_PATTERN)
        excluded = [
            'declare -x HERMES_SESSION_KEY="x"',
            'declare -x HERMES_SESSION_CHAT_ID="1"',
            'declare -x HERMES_SESSION_THREAD_ID="1"',
            'declare -x HERMES_SESSION_MESSAGE_ID="1"',
            'declare -x HERMES_SESSION_USER_NAME="Ace"',
            'declare -x HERMES_SESSION_CHANNEL_1="x"',  # digit suffix (Greptile P2)
            'declare -x HERMES_HOME="/tmp/dw_e2e/home"',
            'export HERMES_SESSION_KEY="x"',
            'export HERMES_HOME="/x"',
        ]
        kept = [
            'declare -x HERMES_HOME_BACKUP="/x"',   # different var
            'declare -x MY_HERMES_SESSION_KEY="x"', # prefixed
            'declare -x HERMES_SESSIONS_DIR="/x"',  # not SESSION_
            'declare -x PATH="/usr/bin"',
        ]
        for line in excluded:
            assert rx.search(line), f"should exclude: {line}"
        for line in kept:
            assert not rx.search(line), f"should keep: {line}"
