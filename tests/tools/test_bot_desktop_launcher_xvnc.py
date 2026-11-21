"""The Bot Desktop's Xvnc is started with the RFB options the lease design relies on."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def _stub_bin(tmp_path: Path, xvnc_body: str, xdpyinfo_body: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "xvnc-argv"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\n{xvnc_body}', encoding="utf-8")
    (bindir / "xdpyinfo").write_text(f"#!/bin/sh\n{xdpyinfo_body}", encoding="utf-8")
    for stub in ("setxkbmap", "xsetroot", "xset", "dbus-run-session", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep", "seq", "sleep", "kill", "timeout"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    return bindir


def _launcher_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": str(tmp_path / "bin"), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }


def test_xvnc_never_sends_the_holders_clipboard_to_watchers(tmp_path):
    """Whoever holds control may paste INTO the screen (AcceptCutText), but the screen's clipboard must
    not be pushed to every connected viewer (SendCutText off): watchers are not the holder."""
    _stub_bin(tmp_path, "exec sleep 3\n", "exit 0\n")
    env = _launcher_env(tmp_path)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    argv = (tmp_path / "xvnc-argv").read_text(encoding="utf-8-sig").split("\n")
    assert "-SendCutText=0" in argv, argv
    assert not any(a.startswith("-AcceptCutText") for a in argv), "paste into the screen must keep working"
    # Xvnc's own cut-text cap and the bridge filter's must agree, or one side drops a paste the other admits.
    from tools.bot_desktop.rfb_filter import _MAX_CUT_TEXT
    assert int(argv[argv.index("-MaxCutText") + 1]) == _MAX_CUT_TEXT, argv
    assert not os.path.exists(tmp_path / "rfb.sock")  # stub never bound it; nothing leaked


def test_x_readiness_timeout_bounds_blocking_probe(tmp_path):
    """xdpyinfo can block in connect() if it reaches the X socket before Xvnc answers.
    The launcher must give up well before runtime.start()'s 15s budget and report that
    the screen never became ready. Without the bounded probe this test hangs.

    A 30-second stub sleep outlasts the 13-second worst-case budget (8-second loop
    plus 5-second final probe) so the launcher must report that Xvnc never became
    ready."""
    _stub_bin(tmp_path, "exec sleep 30\n", "sleep 30\nexit 0\n")
    env = _launcher_env(tmp_path)
    start = time.monotonic()
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=20, check=False,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"launcher took {elapsed:.1f}s, must stay under runtime.start()'s 15s budget"
    assert result.returncode != 0, result
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert "Xvnc did not become ready" in stderr, stderr


def test_readiness_probes_are_timeout_wrapped():
    """Every xdpyinfo/xprop readiness probe in launcher.sh must be wrapped in timeout
    so a future edit cannot quietly reintroduce a blocking check."""
    script = LAUNCHER.read_text(encoding="utf-8").splitlines()
    probe_patterns = (re.compile(r"\bxdpyinfo\s+-display\b"), re.compile(r"\bxprop\s+-root\b"))
    for i, line in enumerate(script, start=1):
        if any(p.search(line) for p in probe_patterns):
            assert re.search(r"\btimeout\s+\d+\s+(?:xdpyinfo|xprop)\b", line), (
                f"probe line {i} not wrapped in timeout: {line.strip()}"
            )
