"""The at-rest terminal snapshot must never carry a secret-shaped export.

Regression matrix for the credential-exposure work (Viktor review
``hermes-snapshot-secret-scrub-20261102``, findings 1-4, CWE-312/CWE-755):

* the scrub runs on **bash 3.2**, the module's stated baseline, so it may not use the
  bash-4-only ``${var^^}`` case operator (finding 1). Sabotaging ``tr`` in a test
  reproduces exactly what that operator does on 3.2: every iteration fails, no
  ``unset`` runs, and the pre-patch full dump would be published;
* a **readonly** exported secret cannot be unset at all, so the post-dump net — not the
  loop — has to withhold it (finding 2);
* the name set has to cover the shapes real tooling uses (finding 3);
* every failure resolves to a published no-op stub, never to a secret-bearing file
  (fail closed, finding 4).

The snapshot is sourced by later commands, so a leak here is a leak in every subsequent
command on that backend. All values in this module are synthetic and every test runs
with a deliberately minimal environment, so no real process secret can reach a snapshot
even if the control regresses.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from tools.environments.base_session_env import (
    _SNAP_TMP_SUFFIX,
    _export_dump_excluding_session_vars,
    _snapshot_bootstrap_script,
    _wrap_command_script,
)

# POSIX bash snapshot path (the Windows environment has no shell snapshot).
pytestmark = pytest.mark.platforms("posix")

_CWD_MARKER = "__HERMES_CWD_scrubtest__"

# Nothing in this environment may be inherited from the test runner: a real secret in
# the parent process would end up in the very snapshot this file inspects.
_BASE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "SHELL": "/bin/bash",
    "EDITOR": "vim",
    "PAGER": "less",
}

# Names whose value carries credentials today, on bash 5, with the pre-fix scrub.
# Covered suffixes, in the three casings that matter (``tr`` upper-cases the name, so
# a lower- or mixed-case name must be caught by the same rule).
_COVERED = {
    "GITHUB_TOKEN": "SYNTHETIC-token-upper",
    "github_token": "SYNTHETIC-token-lower",
    "GitHub_Token": "SYNTHETIC-token-mixed",
    "OPENAI_API_KEY": "SYNTHETIC-key",
    "BUZZ_PRIVATE_KEY": "SYNTHETIC-private-key",
    "APP_SECRET": "SYNTHETIC-secret",
    "DB_PASSWORD": "SYNTHETIC-password",
}

# Finding 3 gaps: names that survived every pre-fix run.
_GAPS = {
    "PASSWORD": "SYNTHETIC-bare-password",
    "DB_PASSWD": "SYNTHETIC-passwd",
    "APP_PASS": "SYNTHETIC-pass",
    "API_AUTH": "SYNTHETIC-auth",
    "HTTP_BEARER": "SYNTHETIC-bearer",
    "AWS_ACCESS_KEY_ID": "SYNTHETIC-key-id",
    "GCP_CREDENTIALS": "SYNTHETIC-service-account",
    "STRIPE_APIKEY": "SYNTHETIC-apikey",
    "GITHUB_AUTHTOKEN": "SYNTHETIC-authtoken",
    "SSH_PRIVATEKEY": "SYNTHETIC-ssh-key",
    "PG_DSN": "SYNTHETIC-dsn",
}

# Credentials embedded in a connection string: no name pattern can classify these, so a
# value-shape rule does. ``DOCS_URL`` carries no credentials and must survive, or the
# rule is just eating ordinary config.
_URL_SECRET = {"DATABASE_URL": "postgres://user:pass@db.internal:5432/app"}
_URL_SAFE = {"DOCS_URL": "https://docs.example.com/guide", "KEEP_ME": "ordinary-value"}

# Ordinary state a developer's shell actually needs after the snapshot is sourced.
_ORDINARY = {
    "EDITOR": "vim",
    "PAGER": "less",
    "SHELL": "/bin/bash",
    "HOME": None,  # filled with the tmp dir per test
    "KEEP_ME": "ordinary-value",
    "DOCS_URL": "https://docs.example.com/guide",
    # Positive control for the underscore in ``*_KEY``: a scrub that eats MY_MONKEY is
    # a scrub someone will switch off. Same for the ssh-agent socket path, which
    # ``*_AUTH`` must not match (it would break ssh).
    "MY_MONKEY": "banana",
    "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
}

_DECL_RE = re.compile(
    r"^(?:declare -[A-Za-z]*x[A-Za-z]*|export) ([A-Za-z_][A-Za-z0-9_]*)=", re.MULTILINE)


def _bash() -> str:
    return "/bin/bash" if os.path.exists("/bin/bash") else "bash"


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = dict(_BASE_ENV, HOME=str(tmp_path))
    env.update(overrides)
    return env


def _declared(snapshot: str) -> set[str]:
    """Names ``source``-ing this snapshot would (re-)export."""
    return set(_DECL_RE.findall(snapshot))


def _prelude(**vars_: str) -> str:
    return "".join(f"export {name}={shlex.quote(value)}\n" for name, value in vars_.items())


def _run(script: str, tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_bash(), "-c", script], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, check=False)


def _bootstrap_script(tmp_path: Path) -> tuple[str, Path]:
    snap = tmp_path / "hermes-snap.sh"
    script = _snapshot_bootstrap_script(
        quoted_cwd=shlex.quote(str(tmp_path)),
        quoted_snap=shlex.quote(str(snap)),
        snap_tmp_template=shlex.quote(str(snap) + _SNAP_TMP_SUFFIX),
        excluded_names=[], cwd_marker=_CWD_MARKER)
    return script, snap


def _run_bootstrap(tmp_path: Path, env: dict[str, str], prelude: str = "") -> tuple[str, subprocess.CompletedProcess]:
    """Execute the real login-time bootstrap; return (snapshot text, completed proc)."""
    script, snap = _bootstrap_script(tmp_path)
    proc = _run(prelude + "\n" + script, tmp_path, env)
    assert proc.returncode == 0, f"bootstrap failed rc={proc.returncode}\n{proc.stderr}"
    assert snap.exists(), "bootstrap published no snapshot at all"
    return snap.read_text(), proc


# ---------------------------------------------------------------------------
# 1. Name matrix: covered shapes (any casing) and the finding-3 gaps are omitted,
#    ordinary shell state is preserved.
# ---------------------------------------------------------------------------

def test_secret_shaped_exports_are_never_snapshotted(tmp_path: Path):
    variables = {**_COVERED, **_GAPS, **_URL_SECRET, **_ORDINARY, "HOME": str(tmp_path)}
    env = _env(tmp_path, **{k: v for k, v in variables.items() if k != "PATH"})

    snapshot, _ = _run_bootstrap(tmp_path, env, _prelude(**variables))

    declared = _declared(snapshot)
    secret_names = set(_COVERED) | set(_GAPS) | set(_URL_SECRET)
    leaked = declared & secret_names
    assert leaked == set(), f"secret-shaped names reached the snapshot: {sorted(leaked)}"

    # Not one synthetic value may survive anywhere in the file (a name-based check
    # alone would miss a value smuggled onto a continuation line).
    assert "SYNTHETIC-" not in snapshot
    assert "postgres://user:pass@" not in snapshot


def test_ordinary_shell_state_survives_the_scrub(tmp_path: Path):
    """The control must not eat the user's environment — that gets it disabled."""
    variables = {**_ORDINARY, "HOME": str(tmp_path)}
    env = _env(tmp_path, **variables)

    snapshot, _ = _run_bootstrap(tmp_path, env, _prelude(**variables))

    declared = _declared(snapshot)
    for name in ("PATH", "EDITOR", "PAGER", "SHELL", "HOME",
                 "MY_MONKEY", "SSH_AUTH_SOCK", "KEEP_ME", "DOCS_URL"):
        assert name in declared, f"{name} was scrubbed from the snapshot"
    assert 'declare -x MY_MONKEY="banana"' in snapshot
    assert "docs.example.com" in snapshot


def test_connection_string_with_embedded_credentials_is_dropped(tmp_path: Path):
    """Creds-in-URL is a value shape, not a name shape (``DATABASE_URL``, ``PROXY`` …)."""
    variables = {"DATABASE_URL": "postgres://user:pass@db.internal:5432/app",
                 "HTTPS_PROXY": "http://user:pass@proxy.internal:3128",
                 "DOCS_URL": "https://docs.example.com/guide"}
    env = _env(tmp_path, **variables)

    snapshot, _ = _run_bootstrap(tmp_path, env, _prelude(**variables))

    declared = _declared(snapshot)
    assert "DATABASE_URL" not in declared
    assert "HTTPS_PROXY" not in declared
    assert "DOCS_URL" in declared, "a credential-free URL must stay"


def test_multiline_secret_value_leaves_no_continuation_lines(tmp_path: Path):
    """Issue #71296 shape: a newline in a value becomes extra lines in ``export -p``."""
    variables = {"MULTILINE_TOKEN": "SYNTHETIC-line1\nSYNTHETIC-line2\nSYNTHETIC-line3"}
    env = _env(tmp_path, **variables)

    snapshot, _ = _run_bootstrap(tmp_path, env, _prelude(**variables))

    assert "MULTILINE_TOKEN" not in _declared(snapshot)
    assert "SYNTHETIC-line" not in snapshot


# ---------------------------------------------------------------------------
# 2. Fail-closed: every degraded scrub publishes a no-op stub, never a secret.
# ---------------------------------------------------------------------------

def test_readonly_secret_is_withheld_not_published(tmp_path: Path):
    """Finding 2: ``unset`` and ``declare +r`` are both refused on a readonly export."""
    readonly = "RO_TOKEN='SYNTHETIC-readonly-token'; readonly RO_TOKEN; export RO_TOKEN\n"
    env = _env(tmp_path, RO_TOKEN="SYNTHETIC-readonly-token")

    script, snap = _bootstrap_script(tmp_path)
    proc = _run(f"{readonly}{_prelude(GITHUB_TOKEN='SYNTHETIC-token')}\n{script}", tmp_path, env)

    assert proc.returncode == 0
    snapshot = snap.read_text()
    # Names may appear in the "why withheld" comment; no declaration and no value may.
    assert "RO_TOKEN" not in _declared(snapshot)
    assert "SYNTHETIC-" not in snapshot
    # Withheld, not silently emptied: the stub records why, with names only.
    assert "snapshot withheld" in snapshot
    assert "RO_TOKEN" in proc.stderr and "SYNTHETIC-" not in proc.stderr


def test_broken_case_normalisation_fails_closed(tmp_path: Path):
    """Finding 1 injection: exactly what ``${var^^}`` does on bash 3.2.

    A ``tr`` that fails makes the case normalisation yield an empty string, so no name
    rule matches, no ``unset`` runs and the dump is the full pre-patch leak. The
    post-dump net is the only thing standing between that and an at-rest credential.
    """
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "tr").write_text("#!/bin/sh\nexit 1\n")
    (shim / "tr").chmod(0o755)
    env = _env(tmp_path, PATH=f"{shim}:{_BASE_ENV['PATH']}")

    variables = {"GITHUB_TOKEN": "SYNTHETIC-token", "DB_PASSWORD": "SYNTHETIC-password"}
    snapshot, proc = _run_bootstrap(tmp_path, env, _prelude(**variables))

    assert "SYNTHETIC-" not in snapshot, "a failed scrub published the full dump"
    assert not _declared(snapshot) & set(variables)
    assert "snapshot withheld" in snapshot
    assert "snapshot withheld" in proc.stderr
    # The diagnostic names what survived, so an operator can fix the real cause.
    assert "GITHUB_TOKEN" in proc.stderr


def test_broken_grep_withholds_instead_of_publishing(tmp_path: Path):
    """The net's own failure is a hit, not a pass: rc > 1 must never publish."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "grep").write_text("#!/bin/sh\nexit 2\n")
    (shim / "grep").chmod(0o755)
    env = _env(tmp_path, PATH=f"{shim}:{_BASE_ENV['PATH']}")

    snapshot, _proc = _run_bootstrap(tmp_path, env, _prelude(GITHUB_TOKEN="SYNTHETIC-token"))

    assert "SYNTHETIC-" not in snapshot
    assert "snapshot withheld" in snapshot


def test_per_command_redump_is_also_guarded(tmp_path: Path):
    """The wrapper re-dumps after every command; that publish needs the same net."""
    readonly = "RO_TOKEN='SYNTHETIC-readonly-token'; readonly RO_TOKEN; export RO_TOKEN\n"
    env = _env(tmp_path, RO_TOKEN="SYNTHETIC-readonly-token")
    snap = tmp_path / "hermes-snap.sh"
    snap.write_text("")  # a previous, clean snapshot the wrapper would replace

    script = _wrap_command_script(
        "true", quoted_cwd=shlex.quote(str(tmp_path)), quoted_snap=shlex.quote(str(snap)),
        snap_tmp_template=shlex.quote(str(snap) + _SNAP_TMP_SUFFIX), passthrough_names=(),
        snapshot_ready=True, cwd_marker=_CWD_MARKER)

    proc = _run(readonly + _prelude(GITHUB_TOKEN="SYNTHETIC-token") + "\n" + script, tmp_path, env)

    assert proc.returncode == 0, f"wrapped command failed rc={proc.returncode}\n{proc.stderr}"
    snapshot = snap.read_text()
    assert "SYNTHETIC-" not in snapshot
    assert "snapshot withheld" in snapshot


def test_clean_run_does_not_withhold(tmp_path: Path):
    """Positive control: the net must not fire on a scrubbed snapshot.

    Without this, a net that always truncates would satisfy every assertion above and
    silently destroy the user's persisted shell state.
    """
    variables = {**_ORDINARY, **_COVERED, "HOME": str(tmp_path)}
    env = _env(tmp_path, **variables)

    snapshot, proc = _run_bootstrap(tmp_path, env, _prelude(**variables))

    assert "snapshot withheld" not in snapshot
    assert "withheld" not in proc.stderr
    assert "PATH" in _declared(snapshot)


def test_snapshot_still_publishes_under_a_set_e_profile(tmp_path: Path):
    """A profile may ``set -e``; a bare ``grep -q`` returning 1 must not kill the shell.

    "No secret-shaped declaration" *is* exit 1, so the net's probe has to sit in an ``if``
    condition — otherwise every clean session loses its snapshot and the guard becomes a
    control that destroys the feature it protects.
    """
    variables = {**_ORDINARY, "HOME": str(tmp_path)}
    env = _env(tmp_path, **variables)

    snapshot, proc = _run_bootstrap(tmp_path, env, "set -e\n" + _prelude(**variables))

    assert proc.returncode == 0, f"bootstrap died under set -e: {proc.stderr}"
    assert "snapshot withheld" not in snapshot
    assert "PATH" in _declared(snapshot)


# ---------------------------------------------------------------------------
# 3. Portability guard: the generated shell must stay bash-3.2-parseable.
# ---------------------------------------------------------------------------

def test_generated_snippets_use_no_bash4_only_expansions():
    """Finding 1 root cause, asserted statically.

    ``${var^^}``/``${var,,}`` are bash 4.0+; on the supported macOS 3.2 baseline they
    abort every loop iteration and the scrub silently degrades to a no-op.
    """
    dump = _export_dump_excluding_session_vars(shlex.quote("/tmp/hermes-snap-test"))
    assert "^^" not in dump
    assert ",," not in dump
    # The 3.2-safe normalisation is what replaced it.
    assert "tr \"[:lower:]\" \"[:upper:]\"" in dump
