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
``tools/environments/base.py``; they are re-injected fresh on every command.
"""

import os
import re
import sys

import pytest

from tools.environments.base import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


# ---------------------------------------------------------------------------
# Unit: the exclusion regex matches exactly the bridged vars, nothing else.
# ---------------------------------------------------------------------------

def test_regex_matches_bridged_session_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    # Every var the gateway bridges must be excluded.
    from gateway.session_context import _VAR_MAP

    for name in _VAR_MAP:
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"


def test_export_snippet_shape():
    snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    assert "export -p" in snippet
    # Unset-by-name (not line-grep): multi-line declare values must not leave
    # continuation lines in the snapshot (issue #71296).
    assert "unset" in snippet
    assert "${!HERMES_SESSION_*}" in snippet
    assert "${!HERMES_CRON_AUTO_DELIVER_*}" in snippet
    assert "${!HERMES_BROWSER_CONTROL_*}" in snippet
    assert "HERMES_UI_SESSION_ID" in snippet
    assert "grep -vE" not in snippet
    assert '"$__hermes_snap_tmp"' in snippet
    # The redirection must be attached to a brace group wrapping the dump,
    # NOT to a pipeline segment: a redirect on a pipeline segment expands the
    # temp-path variable inside that segment's subshell (potentially
    # inconsistently with the parent that expands the follow-up ``mv``
    # operand), silently orphaning the dump and breaking snapshot env
    # persistence entirely.
    assert snippet.lstrip().startswith("{ ")
    assert "|| true; }" in snippet
    assert snippet.rstrip().endswith('> "$__hermes_snap_tmp"')


# ---------------------------------------------------------------------------
# Integration: real LocalEnvironment, two sessions, no cross-contamination.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_delegated_marker_survives_snapshot_helper_collision(tmp_path, monkeypatch):
    """Snapshot bookkeeping cannot override a delegated-child marker."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
    from tools.environments.local import LocalEnvironment

    marker = DELEGATED_CHILD_ENV_MARKER
    present = f"_HERMES_RUNTIME_PASSTHROUGH_{marker}_PRESENT"
    value = f"_HERMES_RUNTIME_PASSTHROUGH_{marker}_VALUE"
    marker_command = f'printf "[%s]" "${{{marker}:-absent}}"'
    monkeypatch.delenv(marker, raising=False)

    def run_case(name, *, delegated=False, parent_value=None):
        cwd = tmp_path / name
        cwd.mkdir()
        kwargs = {"cwd": str(cwd), "timeout": 30}
        if parent_value is not None:
            kwargs["env"] = {marker: parent_value}
        env = LocalEnvironment(**kwargs)
        env.init_session()
        try:
            seeded = env.execute(
                f"export {present}=x; export {value}=forged-from-snapshot-helper; printf '[seeded]'"
            )
            assert seeded["returncode"] == 0
            if delegated:
                with delegated_child_context():
                    result = env.execute(marker_command)
            else:
                result = env.execute(marker_command)
            with open(env._snapshot_path, encoding="utf-8") as stream:
                snapshot = stream.read()
            return result, snapshot
        finally:
            env.cleanup()

    child, child_snapshot = run_case("child", delegated=True)
    ordinary, ordinary_snapshot = run_case("ordinary")
    parent, parent_snapshot = run_case("parent", parent_value="parent-marker")

    assert child["returncode"] == 0
    assert child["output"].strip() == "[1]"
    assert ordinary["returncode"] == 0
    assert ordinary["output"].strip() == "[absent]"
    assert parent["returncode"] == 0
    assert parent["output"].strip() == "[parent-marker]"
    for snapshot in (child_snapshot, ordinary_snapshot, parent_snapshot):
        assert marker not in snapshot
        assert present not in snapshot
        assert value not in snapshot
