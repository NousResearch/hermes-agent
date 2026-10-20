"""Secret-shaped parent-env vars must never persist into the shared bash snapshot.

Estate census (av/harbor deployment, 2026-10-17): one live service credential sat in
108 distinct on-disk files, 65 of them ``hermes-snap-*.sh`` env snapshots. Cause: the
per-command re-dump of ``export -p`` clones every credential in the agent's process
environment into scratch trees that outlive the session, and the next command's
``source`` re-exports the stored copy OVER the ``local_env_policy`` child-env scrub —
silently undoing the documented stance that provider credentials are stripped from
shell subprocesses by default (SECURITY.md §2.3). The fix redacts at the capture
point; these tests pin the invariant and its two losslessness guarantees:

* a benign parent-env name survives byte-for-byte, and
* an export made BY THE SESSION (not in the parent env) survives — there the snapshot
  is the only cross-command carrier;
plus the kill-switch restoring the verbatim dump (which doubles as the mutation
control proving the pre-fix capture DID write the canary).
"""

from __future__ import annotations

import os
import shlex
import subprocess

import pytest

from tools.environments.base_session_env import _export_dump_excluding_session_vars

HEX64_CANARY = "0" * 63 + "a"  # a full 64-hex value, no prefix


def _bash() -> str:
    return "/bin/bash" if os.path.exists("/bin/bash") else "bash"


def _dump_snapshot(tmp_path, *, bash_exports=()):
    """Run the production dump snippet in real bash with the current (monkeypatched)
    Python-side ``os.environ`` as the parent env, and return the snapshot text.

    Names set via ``monkeypatch.setenv`` are the shared parent-env population (the
    subprocess inherits them through ``os.environ.copy()``, exactly like the Popen
    env of a real command); ``bash_exports`` names are exported INSIDE the script —
    the session-scoped export population that the snapshot must keep carrying.
    """
    snap = tmp_path / "snap"
    script = "\n".join([
        "set -e",
        *(f"export {name}" for name in bash_exports),
        _export_dump_excluding_session_vars(shlex.quote(str(snap))),
        f"printf '%s\\n' SNAP_BEGIN; cat {shlex.quote(str(snap))}; printf '%s\\n' SNAP_END",
    ])
    proc = subprocess.run(
        [_bash(), "-c", script], cwd=str(tmp_path),
        capture_output=True, text=True, env=os.environ.copy(), check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split("SNAP_BEGIN\n", 1)[1].rsplit("\nSNAP_END", 1)[0]


@pytest.mark.platforms("posix")  # POSIX bash snapshot path
def test_secret_named_parent_env_var_absent_from_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPSHOT_TEST_API_KEY", "sk-live-canary-value")
    snap = _dump_snapshot(tmp_path)
    assert "SNAPSHOT_TEST_API_KEY" not in snap
    assert "sk-live-canary-value" not in snap
    # The dump itself stays intact: benign machinery vars still ride.
    assert "declare -x PATH=" in snap


@pytest.mark.platforms("posix")
def test_unprefixed_hex64_value_absent_from_snapshot(tmp_path, monkeypatch):
    # A name no keyword class catches, but whose value is the shape of an API key.
    monkeypatch.setenv("SNAPSHOT_TEST_STORE", HEX64_CANARY)
    snap = _dump_snapshot(tmp_path)
    assert "SNAPSHOT_TEST_STORE" not in snap
    assert HEX64_CANARY not in snap


@pytest.mark.platforms("posix")
def test_benign_and_session_scoped_exports_survive(tmp_path, monkeypatch):
    """Losslessness: the redaction is scoped to the PARENT env's secret-shaped names.

    A benign parent-env name must persist byte-for-byte, and a secret-shaped name the
    SESSION exported (absent from the parent env) must persist too — the session shell
    has no other cross-command carrier for it.
    """
    monkeypatch.setenv("SNAPSHOT_TEST_KEEPER", "keep-me")
    monkeypatch.delenv("SNAPSHOT_TEST_SESSION_ONLY_TOKEN", raising=False)
    snap = _dump_snapshot(tmp_path, bash_exports=["SNAPSHOT_TEST_SESSION_ONLY_TOKEN='issued-by-session'"])
    assert 'SNAPSHOT_TEST_KEEPER="keep-me"' in snap
    assert "SNAPSHOT_TEST_SESSION_ONLY_TOKEN" in snap
    assert "issued-by-session" in snap


@pytest.mark.platforms("posix")
def test_kill_switch_restores_verbatim_dump(tmp_path, monkeypatch):
    """``HERMES_DISABLE_SNAPSHOT_SECRET_REDACTION=1`` reproduces the pre-fix capture.

    This is also the mutation control: it proves the canary above WOULD have been
    written verbatim without the redaction, so the exclusion tests can only pass
    because of the capture-time unset.
    """
    monkeypatch.setenv("SNAPSHOT_TEST_API_KEY", "sk-live-canary-value")
    monkeypatch.setenv("HERMES_DISABLE_SNAPSHOT_SECRET_REDACTION", "1")
    snap = _dump_snapshot(tmp_path)
    assert "SNAPSHOT_TEST_API_KEY" in snap
    assert "sk-live-canary-value" in snap
