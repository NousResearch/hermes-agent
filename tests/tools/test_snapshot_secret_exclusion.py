"""Snapshotter secret-exclusion: INFISICAL_* never persists to the bash snapshot.

Regression coverage for AGENTS-104: the terminal session snapshotter
(``export -p`` dump in ``tools/environments/base_session_env.py``) previously
persisted the profile's Infisical universal-auth pair
(``INFISICAL_UNIVERSAL_AUTH_CLIENT_ID`` / ``INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET``,
loaded from the profile .env at launch) as plaintext ``declare -x`` lines in
``profiles/<p>/cache/terminal/hermes-snap-<session>.sh`` — real storage, 0600,
refreshed on every command while a session lives. Purge alone was insufficient:
the next terminal call regenerated the file.

The fix unsets ``${!INFISICAL_*}`` before the dump (same mechanism as the
bridged HERMES_SESSION_* vars) and extends the Python-side contract regex
``_SNAPSHOT_EXCLUDED_ENV_REGEX``. Exclusion is snapshot-only: the pair is
re-injected onto every command's process env from the launch environment, so
live-session credential usability is unaffected.
"""

import os
import re

from tools.environments.base_session_env import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


# ---------------------------------------------------------------------------
# Unit: the Python-side exclusion contract covers INFISICAL_*.
# ---------------------------------------------------------------------------

def test_regex_matches_infisical_lines():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    for name in (
        "INFISICAL_UNIVERSAL_AUTH_CLIENT_ID",
        "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET",
        "INFISICAL_TOKEN",
        "INFISICAL_DOMAIN",
        "INFISICAL_ANYTHING_ELSE",
    ):
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"


def test_regex_still_matches_bridged_session_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    for name in ("HERMES_SESSION_ID", "HERMES_UI_SESSION_ID",
                 "HERMES_CRON_SESSION", "HERMES_BROWSER_CONTROL_TOKEN"):
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} must stay excluded (existing contract)"


def test_regex_does_not_match_unrelated_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    assert not rx.search('declare -x PATH="/usr/bin:/bin"')
    assert not rx.search('declare -x FOO="bar"')
    # Non-export declarations are not the dump shape either.
    assert not rx.search('FOO="bar"')


def test_export_snippet_unsets_infisical_prefix():
    snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    assert "${!INFISICAL_*}" in snippet
    assert "export -p" in snippet
    # Unset-by-prefix (not line-grep): multi-line declare values must not
    # leave continuation lines in the snapshot.
    assert "grep -vE" not in snippet


# ---------------------------------------------------------------------------
# Behavioral: real LocalEnvironment — secret excluded from the snapshot file,
# control var persists, credential still usable in live commands.
# ---------------------------------------------------------------------------

import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_snapshot_excludes_infisical_but_keeps_control_var(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET", "dummy-secret-value")
    monkeypatch.setenv("FOO", "bar")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        # Terminal still functions with the fix in place.
        out = env.execute('echo "runnable"')
        assert "runnable" in out.get("output", "")

        snap = env._snapshot_path
        assert os.path.exists(snap), f"snapshot missing at {snap}"
        with open(snap) as f:
            content = f.read()
        assert "INFISICAL_" not in content, \
            "snapshot must contain zero INFISICAL_* declare lines (AGENTS-104)"
        assert "dummy-secret-value" not in content
        # Control var persists — the dump is not over-stripped.
        assert "FOO" in content
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_live_session_credential_still_usable(tmp_path, monkeypatch):
    """Exclusion is snapshot-only: the process env of the running command must
    still carry the pair (re-injected from the launch environment), so the
    mint-in-subshell pattern keeps working."""
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET", "dummy-secret-value")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        out = env.execute('printf "%s" "$INFISICAL_UNIVERSAL_AUTH_CLIENT_ID"')
        assert out.get("output", "").strip() == "dummy-client-id", \
            "live command must still see the pair from its process env"
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_regenerated_snapshot_stays_clean_after_repeated_commands(tmp_path, monkeypatch):
    """The leak refreshed on every command (~30s cadence). After N commands the
    snapshot must STILL contain zero INFISICAL_ lines (non-regeneration)."""
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET", "dummy-secret-value")
    monkeypatch.setenv("FOO", "bar")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        for i in range(3):
            env.execute(f"echo cycle-{i}")
        with open(env._snapshot_path) as f:
            content = f.read()
        assert "INFISICAL_" not in content, \
            "snapshot re-dumped after commands must stay free of INFISICAL_*"
        assert "FOO" in content
    finally:
        env.cleanup()