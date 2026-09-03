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


def _bash() -> str:
    return "/bin/bash" if os.path.exists("/bin/bash") else "bash"


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
# Credential scrub: env vars whose NAME contains credential markers
# (KEY/TOKEN/SECRET/PASSWORD/PASSWD/CREDENTIAL, any case) must not land in
# the snapshot dump.
# ---------------------------------------------------------------------------

def test_export_snippet_contains_case_insensitive_credential_scrub():
    snippet = _export_dump_excluding_session_vars('"$__hermes_snap_tmp"')
    # bash 3.2-compatible case-insensitive matching: nocasematch (3.1+) +
    # word-boundary globs, NOT ${__v^^} (bash 4.0+ only).
    assert "shopt -s nocasematch" in snippet
    assert "compgen -e" in snippet
    assert 'case "_${__v}_" in' in snippet
    for marker in (
        "*_key_*",
        "*_token_*",
        "*_secret_*",
        "*_password_*",
        "*_passwd_*",
        "*_credential_*",
    ):
        assert marker in snippet, f"{marker} should be in the scrub pattern"
    # camelCase suffixes (ApiKey) match via the *key_ tail.
    for tail in ("*key_", "*token_", "*secret_", "*password_", "*passwd_", "*credential_"):
        assert tail in snippet, f"{tail} should be in the scrub pattern"
    # export -n (not unset) so readonly exported vars are scrubbed too.
    assert "export -n" in snippet
    # bash-4-only case fold must NOT be present (macOS bash 3.2 compat).
    assert "${__v^^}" not in snippet
    # Lowercase-only patterns are gone: nocasematch covers case.
    assert "*key*" not in snippet
    assert "*password*" not in snippet


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_export_snapshot_scrubs_credential_names_any_case(tmp_path):
    import shlex
    import subprocess

    snap = tmp_path / "snap.sh"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    script = "set -e\n"
    script += "export db_password=leak1\n"
    script += "export ApiKey=leak2\n"
    script += "export MY_SECRET=leak3\n"
    script += "export ACCESS_TOKEN=leak4\n"
    script += "export normal_var=keepme\n"
    script += "export APPLE=keepme\n"
    script += dump + "\n"
    script += "if grep -qE 'db_password|ApiKey|MY_SECRET|ACCESS_TOKEN' " + q_snap + "; then echo 'LEAKED_INTO_SNAPSHOT' >&2; exit 2; fi\n"
    script += "source " + q_snap + "\n"
    script += "if grep -qE 'db_password|ApiKey|MY_SECRET|ACCESS_TOKEN' " + q_snap + "; then echo 'LEAKED_INTO_SNAPSHOT' >&2; exit 2; fi\n"
    script += "if ! grep -qE '^declare -x normal_var=' " + q_snap + "; then echo 'NORMAL_VAR_MISSING' >&2; exit 3; fi\n"
    script += "if ! grep -qE '^declare -x APPLE=' " + q_snap + "; then echo 'APPLE_MISSING' >&2; exit 4; fi\n"
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_export_snapshot_keeps_tokenizers_var(tmp_path):
    """TOKENIZERS_PARALLELISM (benign huggingface knob) must NOT be scrubbed.

    Regression for the earlier substring match (*TOKEN*) which also deleted
    TOKENIZERS_* vars. Word-boundary matching keeps it.
    """
    import shlex
    import subprocess

    snap = tmp_path / "snap.sh"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    script = "set -e\n"
    script += "export TOKENIZERS_PARALLELISM=true\n"
    script += "export ACCESS_TOKEN=leak\n"
    script += dump + "\n"
    script += "if ! grep -qE '^declare -x TOKENIZERS_PARALLELISM=' " + q_snap + "; then echo 'TOKENIZERS_SCRUBBED' >&2; exit 2; fi\n"
    script += "if grep -qE 'ACCESS_TOKEN' " + q_snap + "; then echo 'LEAKED_INTO_SNAPSHOT' >&2; exit 3; fi\n"
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_export_snapshot_scrubs_readonly_exported_creds(tmp_path):
    """readonly exported credential vars are scrubbed via export -n."""
    import shlex
    import subprocess

    snap = tmp_path / "snap.sh"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    script = "set -e\n"
    script += "readonly RO_API_KEY=leakro\n"
    script += "export RO_API_KEY\n"
    script += "export normal_ro_var=keepme\n"
    script += dump + "\n"
    script += "if grep -qE 'RO_API_KEY' " + q_snap + "; then echo 'READONLY_LEAKED' >&2; exit 2; fi\n"
    script += "if ! grep -qE '^declare -x normal_ro_var=' " + q_snap + "; then echo 'NORMAL_VAR_MISSING' >&2; exit 3; fi\n"
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_export_snapshot_scrubs_credentials_under_nonwhitespace_ifs(tmp_path):
    """The scrub must survive a non-whitespace IFS (e.g. ``IFS=:``).

    Regression: ``for __v in $(compgen -e)`` splits on IFS; under ``IFS=:``
    the whole name list is one word, the case match never fires, and the
    swallowed ``export -n`` error hides the failure — every credential
    leaks.  A ``while IFS= read -r`` loop is IFS-independent.
    """
    import shlex
    import subprocess

    snap = tmp_path / "snap.sh"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    script = "set -e\n"
    script += "IFS=:\n"
    script += "export API_KEY=leak1\n"
    script += "export db_password=leak2\n"
    script += "export normal_var=keepme\n"
    script += dump + "\n"
    script += "if grep -qE 'API_KEY|db_password' " + q_snap + "; then echo 'LEAKED_UNDER_IFS' >&2; exit 2; fi\n"
    script += "if ! grep -qE '^declare -x normal_var=' " + q_snap + "; then echo 'NORMAL_VAR_MISSING' >&2; exit 3; fi\n"
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


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
