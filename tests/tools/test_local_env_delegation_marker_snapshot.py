"""Regression: the delegate_task lineage marker must not persist in the shared
terminal snapshot.

``HERMES_DELEGATED_CHILD_CONTEXT=1`` is invocation-scoped metadata describing the
currently executing subprocess lineage: a terminal command run by a
``delegate_task`` child should see it, an ordinary later command must not. But the
local backend collapses parent and delegated child onto one cached
``LocalEnvironment`` (the ``"default"`` key) whose ``export -p`` snapshot is
re-sourced before every command. Before the fix the child's marker was serialized
into that shared snapshot, so a subsequent *non-delegated* command sourced the
stale value and was misidentified as a delegated child — e.g. Kanban mutation
guards denying otherwise legitimate ops.

Regression coverage for
https://github.com/NousResearch/hermes-agent/issues/71941.
"""

import subprocess
from pathlib import Path

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    delegated_child_context,
)
from tools.environments.base import _export_dump_excluding_session_vars
from tools.environments.local import LocalEnvironment

_MARKER = DELEGATED_CHILD_ENV_MARKER


def _probe(name: str) -> str:
    # presence=set/unset, value=<value>/unset — same shape as the issue repro.
    return (
        f'printf "presence=%s value=%s\\n" '
        f'"${{{name}+set}}" "${{{name}-unset}}"'
    )


def test_delegated_marker_not_persisted_to_shared_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv(_MARKER, raising=False)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        snapshot = Path(env._snapshot_path)
        assert _MARKER not in snapshot.read_text(encoding="utf-8")

        # A delegated child command must still SEE the marker...
        with delegated_child_context():
            child = env.execute(_probe(_MARKER), timeout=15)
        assert child["returncode"] == 0
        assert "presence=set value=1" in child["output"]

        # ...but it must NOT leak into the reusable snapshot.
        assert _MARKER not in snapshot.read_text(encoding="utf-8")

        # A subsequent ordinary (non-delegated) command sees it as unset,
        # and re-dumping keeps the snapshot clean.
        parent = env.execute(_probe(_MARKER), timeout=15)
        assert parent["returncode"] == 0
        assert "presence= value=unset" in parent["output"]
        assert _MARKER not in snapshot.read_text(encoding="utf-8")
    finally:
        env.cleanup()


def test_kanban_ownership_env_not_persisted_to_shared_snapshot(tmp_path, monkeypatch):
    # Kanban dispatcher-ownership vars are invocation-scoped too; a delegated
    # child that inherits one in its process env must not persist it into the
    # snapshot where a later turn would recover stale ownership.
    key = KANBAN_ENV_KEYS[0]
    monkeypatch.setenv(key, "board-owner-token")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        env.execute("true", timeout=15)
        assert key not in Path(env._snapshot_path).read_text(encoding="utf-8")
    finally:
        env.cleanup()


def test_user_kanban_config_survives_snapshot(tmp_path, monkeypatch):
    # HERMES_KANBAN_HOME / HERMES_KANBAN_DISPATCH_IN_GATEWAY are user-authored
    # config, NOT dispatcher-owned lineage (they are absent from KANBAN_ENV_KEYS),
    # so a broad HERMES_KANBAN_* exclusion must not strip them: the snapshot has
    # to preserve the user's own exports.
    monkeypatch.setenv("HERMES_KANBAN_HOME", "/home/u/.hermes")
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "0")
    assert "HERMES_KANBAN_HOME" not in KANBAN_ENV_KEYS
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        env.execute("true", timeout=15)
        probe = env.execute(_probe("HERMES_KANBAN_HOME"), timeout=15)
        assert probe["returncode"] == 0
        assert "presence=set value=/home/u/.hermes" in probe["output"]
        dispatch = env.execute(_probe("HERMES_KANBAN_DISPATCH_IN_GATEWAY"), timeout=15)
        assert "presence=set value=0" in dispatch["output"]
    finally:
        env.cleanup()


def test_precontaminated_snapshot_marker_stripped_before_command(tmp_path, monkeypatch):
    # topbronson's case: a snapshot persisted by an OLDER build still carries
    # ``export HERMES_DELEGATED_CHILD_CONTEXT=1``. Sourcing it before a non-
    # delegated command would leak the marker for that first post-upgrade
    # command; the post-source cleanup must strip it so the command is not
    # misidentified as a delegated child.
    monkeypatch.delenv(_MARKER, raising=False)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        snapshot = Path(env._snapshot_path)
        # Simulate the contaminated on-disk snapshot from before the fix.
        snapshot.write_text(
            snapshot.read_text(encoding="utf-8") + f'\nexport {_MARKER}=1\n',
            encoding="utf-8",
        )
        parent = env.execute(_probe(_MARKER), timeout=15)
        assert parent["returncode"] == 0
        assert "presence= value=unset" in parent["output"]

        # A genuine delegated child on the SAME contaminated snapshot must still
        # see the marker — the cleanup restores the authoritative process-env
        # state, it does not blanket-drop it.
        with delegated_child_context():
            child = env.execute(_probe(_MARKER), timeout=15)
        assert child["returncode"] == 0
        assert "presence=set value=1" in child["output"]
    finally:
        env.cleanup()


def test_export_dump_filter_drops_marker_and_kanban_keeps_user_exports():
    # Faithful check of the real ``unset``-before-``export -p`` filter, independent
    # of the backend plumbing: invocation-scoped Hermes identity/lineage vars are
    # dropped, ordinary user exports (incl. unrelated HERMES_HOME) survive. The
    # dump unsets by name/prefix, so the vars must be genuinely exported in the
    # shell the snippet runs in (a mocked ``export -p`` would never see the unset).
    exports = "; ".join(
        [
            'export PATH="/usr/bin"',
            'export HERMES_HOME="/home/u/.hermes"',
            f'export {_MARKER}="1"',
            f'export {KANBAN_ENV_KEYS[0]}="tok"',
            'export HERMES_SESSION_ID="s1"',
            'export MYVAR="keep"',
        ]
    )
    snippet = _export_dump_excluding_session_vars("/dev/stdout")
    out = subprocess.run(
        ["bash", "-c", f"{exports}; {snippet}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert f'declare -x {_MARKER}=' not in out
    assert f'declare -x {KANBAN_ENV_KEYS[0]}=' not in out
    assert 'declare -x HERMES_SESSION_ID=' not in out
    assert 'declare -x PATH="/usr/bin"' in out
    assert 'declare -x HERMES_HOME="/home/u/.hermes"' in out
    assert 'declare -x MYVAR="keep"' in out
