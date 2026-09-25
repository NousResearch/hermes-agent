"""The Bot Desktop's Xvnc is started with the RFB options the lease design relies on."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")


def test_xvnc_never_sends_the_holders_clipboard_to_watchers(tmp_path):
    """Whoever holds control may paste INTO the screen (AcceptCutText), but the screen's clipboard must
    not be pushed to every connected viewer (SendCutText off): watchers are not the holder."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "xvnc-argv"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\nexec sleep 3\n', encoding="utf-8")
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "dbus-run-session", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    argv = argv_log.read_text(encoding="utf-8-sig").split("\n")
    assert "-SendCutText=0" in argv, argv
    assert not any(a.startswith("-AcceptCutText") for a in argv), "paste into the screen must keep working"
    # Xvnc's own cut-text cap and the bridge filter's must agree, or one side drops a paste the other admits.
    from tools.bot_desktop.rfb_filter import _MAX_CUT_TEXT
    assert int(argv[argv.index("-MaxCutText") + 1]) == _MAX_CUT_TEXT, argv
    assert not os.path.exists(tmp_path / "rfb.sock")  # stub never bound it; nothing leaked


def test_readiness_probe_stays_on_the_unix_socket(tmp_path):
    """Regression for #123130: Xvnc runs -nolisten tcp, yet the readiness probe addressed it as a bare
    ":N", which lets libxcb fall back to TCP 127.0.0.1:60NN when the unix socket is not bound yet. That
    fallback can only ever hit a dead port — refused instantly on normal kernels, but silently dropped
    (SYN-SENT for ~2 minutes) under WSL2 mirrored networking, outliving runtime.start()'s 15 s window.
    The probe must use the "unix:N" form so libxcb only ever tries the unix sockets."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    probe_log = tmp_path / "xdpyinfo-calls"
    (bindir / "Xvnc").write_text("#!/bin/sh\nexec sleep 3\n", encoding="utf-8")
    # A stand-in for "only the unix socket leads to the server": a bare ":N" (the TCP-fallback shape)
    # must fail, "unix:N" must succeed. If the launcher ever probes with the bare form again, the
    # 100-iteration loop exhausts and the launcher exits 1 ("Xvnc did not become ready").
    (bindir / "xdpyinfo").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{probe_log}"\ncase "$2" in unix:*) exit 0 ;; *) exit 1 ;; esac\n',
        encoding="utf-8",
    )
    for stub in ("setxkbmap", "xsetroot", "xset", "dbus-run-session", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env = {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": "99",
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    calls = probe_log.read_text(encoding="utf-8-sig").split("\n")
    assert "unix:99" in calls, calls
    assert ":99" not in calls, "the readiness probe must never hand libxcb a bare display (TCP fallback)"
