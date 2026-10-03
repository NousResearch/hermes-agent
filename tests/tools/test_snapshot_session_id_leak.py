"""Cross-session HERMES_SESSION_ID leak via the shared bash snapshot.

Regression coverage for the bug where a single long-lived backend serves many
sessions through ONE ``_active_environments["default"]`` LocalEnvironment (the
messaging gateway, TUI, and desktop/web dashboard all collapse the terminal to
"default"). That environment persists a bash *session snapshot* file and
``source``s it before every command. ``export -p`` dumped the FIRST session's
``HERMES_SESSION_ID`` into the snapshot, so every LATER session ``source``d that
stale value and its ``echo $HERMES_SESSION_ID`` reported a FOREIGN session's id
— overriding the correct per-command Popen env injected by
``_inject_session_context_env``.

The fix strips the per-session bridged vars (HERMES_SESSION_* / UI /
CRON_AUTO_DELIVER_) from the snapshot at both dump sites in
``tools/environments/base_session_env.py``; they are re-injected fresh on every
command. The same dump must drop the scope markers a delegate_task child / cron
run stamps per command (HERMES_DELEGATED_CHILD_CONTEXT, HERMES_CRON_SESSION), or
the parent's next command is misread as that child (#90782, #71941).
"""

import os
import re
import subprocess
import sys

import pytest

from tools.environments.base_session_env import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


# ---------------------------------------------------------------------------
# Unit: the exclusion regex matches exactly the bridged vars, nothing else.
# ---------------------------------------------------------------------------

def test_regex_matches_bridged_session_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    # Every var the gateway bridges, and the delegate_task marker, must be excluded.
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    from gateway.session_context import _VAR_MAP

    for name in (*_VAR_MAP, DELEGATED_CHILD_ENV_MARKER):
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"




# ---------------------------------------------------------------------------
# Integration: real LocalEnvironment, two sessions, no cross-contamination.
# ---------------------------------------------------------------------------

@pytest.mark.platforms("posix")  # POSIX bash snapshot path
def test_shared_snapshot_no_cross_session_leak(tmp_path):
    import threading

    from gateway.session_context import _VAR_MAP, _UNSET, set_session_vars
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        def run_as(sid):
            out = {}

            def worker():
                for v in _VAR_MAP.values():
                    v.set(_UNSET)
                set_session_vars(session_key="k" + sid, session_id=sid, source="desktop")
                out["r"] = env.execute('echo "[$HERMES_SESSION_ID]"')

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            return out["r"].get("output", "")

        out_a = run_as("SIDAAA")
        out_b = run_as("SIDBBB")

        assert "SIDAAA" in out_a, f"session A saw {out_a!r}"
        # The core assertion: B must see its OWN id, not A's leaked via snapshot.
        assert "SIDBBB" in out_b, f"session B saw {out_b!r}"
        assert "SIDAAA" not in out_b, f"session B leaked A's id: {out_b!r}"

        # And the snapshot file must not carry the session id at all.
        snap = env._snapshot_path
        if os.path.exists(snap):
            with open(snap) as f:
                assert "HERMES_SESSION_ID" not in f.read()
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# #90782 / #71941: scope markers (delegate_task child, cron run) must not
# persist into the snapshot either.
# ---------------------------------------------------------------------------

@pytest.mark.platforms("posix")  # POSIX bash snapshot path
def test_export_dump_drops_every_bridged_var_and_the_delegation_marker():
    """Run the real dump: nothing the gateway bridges per command, nor the
    delegate_task marker, may survive ``export -p``; ordinary exports must."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    from gateway.session_context import _VAR_MAP

    scoped = [*_VAR_MAP, DELEGATED_CHILD_ENV_MARKER]
    exports = "; ".join([f'export {n}="x"' for n in scoped] + ['export HERMES_HOME="/h"', 'export MYVAR="keep"'])
    out = subprocess.run(
        ["bash", "-c", f"{exports}; {_export_dump_excluding_session_vars('/dev/stdout')}"],
        capture_output=True, text=True, check=True).stdout
    leaked = [n for n in scoped if f"declare -x {n}=" in out]
    assert not leaked, f"persisted into the snapshot: {leaked}"
    assert 'declare -x HERMES_HOME="/h"' in out
    assert 'declare -x MYVAR="keep"' in out


@pytest.mark.platforms("posix")  # POSIX bash snapshot path
def test_snapshot_does_not_turn_later_commands_into_delegated_children(tmp_path):
    """A snapshot re-dumped during a delegated child's command must not re-export
    the marker into the parent's next ``source`` (#90782)."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
    from tools.environments.local import LocalEnvironment

    probe = f'printf "[${{{DELEGATED_CHILD_ENV_MARKER}+set}}]"'
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        with delegated_child_context():
            child = env.execute(probe)
        assert "[set]" in child["output"], f"delegated child lost its marker: {child!r}"
        parent = env.execute(probe)
        assert "[]" in parent["output"], f"parent command inherited the marker: {parent!r}"
    finally:
        env.cleanup()
