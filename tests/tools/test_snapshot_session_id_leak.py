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


def test_regex_matches_profile_home_only():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)

    assert rx.search('declare -x HERMES_HOME="/profiles/profile-a"')
    assert not rx.search('declare -x HERMES_HOME_BACKUP="/profiles/profile-a"')


def test_export_snippet_shape():
    snippet = _export_dump_excluding_session_vars("/tmp/snap.tmp.$BASHPID")
    assert "export -p" in snippet
    # Unset-by-name (not line-grep): multi-line declare values must not leave
    # continuation lines in the snapshot (issue #71296).
    assert "unset" in snippet
    assert "${!HERMES_SESSION_*}" in snippet
    assert "${!HERMES_CRON_AUTO_DELIVER_*}" in snippet
    assert "HERMES_UI_SESSION_ID" in snippet
    assert "HERMES_HOME" in snippet
    assert "grep -vE" not in snippet
    assert "/tmp/snap.tmp.$BASHPID" in snippet
    # The redirection must be attached to a brace group wrapping the dump,
    # NOT to a pipeline segment: a redirect on a pipeline segment expands
    # $BASHPID inside that segment's subshell (a different PID than the parent
    # that expands the follow-up ``mv`` operand), silently orphaning the dump
    # and breaking snapshot env persistence entirely.
    assert snippet.lstrip().startswith("{ ")
    assert "|| true; }" in snippet
    assert snippet.rstrip().endswith("> /tmp/snap.tmp.$BASHPID")


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
            with open(snap, encoding="utf-8") as f:
                assert "HERMES_SESSION_ID" not in f.read()
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_shared_snapshot_preserves_context_profile_home(tmp_path):
    """Each command must receive its profile HERMES_HOME, never its predecessor's."""
    import threading

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.environments.local import LocalEnvironment

    profile_a = tmp_path / "profiles" / "profile-a"
    profile_b = tmp_path / "profiles" / "profile-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        def run_as(profile_home):
            out = {}

            def worker():
                token = set_hermes_home_override(profile_home)
                try:
                    out["result"] = env.execute('printf "[HOME=%s]" "$HERMES_HOME"')
                finally:
                    reset_hermes_home_override(token)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            return out["result"].get("output", "")

        out_a = run_as(profile_a)
        out_b = run_as(profile_b)
        assert f"[HOME={profile_a}]" in out_a
        assert f"[HOME={profile_b}]" in out_b
        assert f"[HOME={profile_a}]" not in out_b

        with open(env._snapshot_path, encoding="utf-8") as snapshot:
            assert "HERMES_HOME" not in snapshot.read()
    finally:
        env.cleanup()
