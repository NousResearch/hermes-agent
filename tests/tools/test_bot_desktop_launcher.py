"""Bot Desktop launcher: the X server it execs is resolved by capability, not by the name `Xvnc`.

RealVNC Server ships its own `/usr/bin/Xvnc`, and a second package claiming a name the
update-alternatives link normally owns takes it over silently — its X server then rejects the
RFB options the launcher passes (`-rfbport`, `-rfbunixpath`, `-rfbunixmode`) and answers with its
own several-hundred-line parameter help. The launcher must run TigerVNC's own binary when the
name exists, and must refuse a same-named server that lacks the options instead of exec'ing it.

The script itself is the unit under test: the runtime's Python tests stub the launcher out, so
nothing else exercises this resolution.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"

# Tools the launcher needs BEFORE the X server section (directory setup + the xauth cookie). The
# fake PATH still reaches them, so a failure here is the resolution under test and not a missing
# tool: `xtigervnc`-style option probing, argv recording and the exec all happen after this point.
_PREAMBLE_TOOLS = ("bash", "mkdir", "rm", "tr", "od", "xauth", "chmod", "sed", "dirname",
                   "grep", "seq", "sleep", "cat", "mv", "cut")

# `-help` output decides the probe. The negative fixture must not contain the token anywhere —
# a stub that spells it out in a disclaimer would satisfy the very check it is meant to fail.
_STUB = """#!/bin/sh
echo "$0 $@" >> "$RECORD"
if [ "$1" = "-help" ]; then
  {help_body}
  exit 0
fi
echo "stub X server $(basename "$0"): refusing to start" >&2
exit 1
"""


def _write_stub(directory: Path, name: str, *, tiger_vnc: bool) -> None:
    path = directory / name
    path.write_text(
        _STUB.format(
            help_body='echo "  rfbunixpath    - Unix socket to listen for RFB protocol"'
            if tiger_vnc
            else 'echo "Vendor parameter list without Unix-socket support"',
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_launcher(tmp_path: Path, *, display: int) -> tuple[int, str, list[str]]:
    """Run the real launcher on a fake PATH; return (exit code, output, recorded server invocations)."""
    tools_dir = tmp_path / "tools-bin"
    tools_dir.mkdir(exist_ok=True)
    for tool in _PREAMBLE_TOOLS:
        # A host without the tool cannot exercise the launcher at all; skip rather than fail.
        real = shutil.which(tool) or pytest.skip(f"{tool} is not on this host's PATH")
        (tools_dir / tool).symlink_to(real)
    state = tmp_path / "state"
    (state / "xdg").mkdir(parents=True, exist_ok=True)
    record = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'stubs'}:{tools_dir}",
        "RECORD": str(record),
        "HERMES_BD_PROFILE": "launcher-test",
        "HERMES_BD_DISPLAY_NUM": str(display),
        "HERMES_BD_SOCKET": str(state / "rfb.sock"),
        "HERMES_BD_XAUTH": str(state / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(state / "env"),
        "HERMES_BD_CONFIG_HOME": str(state / "xdg"),
        "HERMES_BD_GEOMETRY": "1024x768",
    }
    proc = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=60)
    calls = record.read_text(encoding="utf-8").splitlines() if record.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, calls


def test_a_same_named_x_server_without_the_rfb_options_is_refused(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(stubs, "Xvnc", tiger_vnc=False)

    code, out, calls = _run_launcher(tmp_path, display=71)

    assert code != 0
    assert "not TigerVNC's X server" in out
    assert [c for c in calls if " -help" not in c] == [], (
        "the launcher started a non-TigerVNC X server with its RFB options"
    )


def test_tigervnc_binary_wins_when_both_names_exist(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(stubs, "Xvnc", tiger_vnc=False)  # the name another package took over
    _write_stub(stubs, "Xtigervnc", tiger_vnc=True)

    _code, _out, calls = _run_launcher(tmp_path, display=72)

    assert any(re.search(r"/Xtigervnc :72 .*-rfbunixpath", c) for c in calls), (
        "TigerVNC's binary was not the one started with the RFB options"
    )
    assert not [c for c in calls if c.split()[0].endswith("/Xvnc")], (
        "the same-named server was invoked"
    )
