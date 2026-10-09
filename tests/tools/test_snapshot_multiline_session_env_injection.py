"""Newline in bridged session env must not become shell code via the snapshot.

Regression for issue #71296: bash 3.2 ``export -p`` prints a value containing
a newline as a multi-line ``declare -x NAME="…`` block. The old line-based
``grep -vE`` filter removed only the opener; continuation lines (e.g.
``curl … | bash #`` smuggled into a Matrix room/display name) persisted into
the shared terminal snapshot and executed on the next ``source``, with stdout
discarded by the wrapper.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools.environments.base import _export_dump_excluding_session_vars


def _bash() -> str:
    return "/bin/bash" if os.path.exists("/bin/bash") else "bash"


def _run_dump_and_source(
    *,
    tmp_path: Path,
    env_name: str,
    env_value: str,
    marker: Path,
) -> subprocess.CompletedProcess:
    snap = tmp_path / "snap"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    q_marker = shlex.quote(str(marker))
    script = f"""
set -e
export {env_name}
{dump}
if grep -qE 'pwned|touch |{env_name}' {q_snap}; then
  echo "LEAKED_INTO_SNAPSHOT" >&2
  exit 2
fi
bash -c 'source {q_snap} >/dev/null 2>&1 || true'
if [ -e {q_marker} ]; then
  echo "PAYLOAD_EXECUTED" >&2
  exit 3
fi
if ! grep -qE '^declare -x PATH=' {q_snap}; then
  echo "PATH_MISSING" >&2
  exit 4
fi
"""
    env = os.environ.copy()
    env[env_name] = env_value
    return subprocess.run(
        [_bash(), "-c", script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_multiline_session_chat_name_not_executed_via_snapshot(tmp_path: Path):
    """Continuation lines of HERMES_SESSION_CHAT_NAME must not run on source."""
    marker = tmp_path / "pwned"
    chat_name = f"demo\ntouch {marker} #"
    proc = _run_dump_and_source(
        tmp_path=tmp_path,
        env_name="HERMES_SESSION_CHAT_NAME",
        env_value=chat_name,
        marker=marker,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_multiline_session_user_name_not_executed_via_snapshot(tmp_path: Path):
    """Same hole via HERMES_SESSION_USER_NAME (display-name path)."""
    marker = tmp_path / "pwned_user"
    user_name = f"alice\ntouch {marker} #"
    proc = _run_dump_and_source(
        tmp_path=tmp_path,
        env_name="HERMES_SESSION_USER_NAME",
        env_value=user_name,
        marker=marker,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert not marker.exists()
