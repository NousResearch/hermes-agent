"""The Bot Desktop's Xvnc is started with the RFB options the lease design relies on."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"
pytestmark = pytest.mark.platforms("linux")

# The stub Xvnc binds this display's socket for real: the X protocol fixes display sockets under
# /tmp/.X11-unix, so the readiness wait has something to observe. Far from the :0..:2 a host X
# server actually takes, so a bind never collides with one.
_DISPLAY_NUM = "137"
_XSOCK = Path(f"/tmp/.X11-unix/X{_DISPLAY_NUM}")


@pytest.fixture
def xvnc_x_socket():
    """Hands the stub Xvnc a clean display-socket path and removes it again — no residue on hosts."""
    _XSOCK.parent.mkdir(parents=True, exist_ok=True)
    yield _XSOCK
    _XSOCK.unlink(missing_ok=True)
    try:
        _XSOCK.parent.rmdir()  # leave no empty dir behind on hosts without X
    except OSError:
        pass  # a host X server (or another test) still owns the directory


def _bind_x_socket_line() -> str:
    real_python = shutil.which("python3")
    return f"'{real_python}' -c 'import socket; socket.socket(socket.AF_UNIX).bind(\"{_XSOCK}\")'\n"


def _write_common_stubs(bindir: Path) -> None:
    for stub in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "dbus-run-session", "xauth"):
        (bindir / stub).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for exe in bindir.iterdir():
        exe.chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "xauth", "od", "tr", "awk", "grep", "seq", "sleep", "kill"):
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)


def _launcher_env(bindir: Path, tmp_path: Path) -> dict:
    return {
        "PATH": str(bindir), "HOME": str(tmp_path),
        "HERMES_BD_PROFILE": "t", "HERMES_BD_DISPLAY_NUM": _DISPLAY_NUM,
        "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"), "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(tmp_path / "env"), "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg"),
    }


def test_xvnc_never_sends_the_holders_clipboard_to_watchers(tmp_path, xvnc_x_socket):
    """Whoever holds control may paste INTO the screen (AcceptCutText), but the screen's clipboard must
    not be pushed to every connected viewer (SendCutText off): watchers are not the holder."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "xvnc-argv"
    (bindir / "Xvnc").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\nsleep 0.3\n{_bind_x_socket_line()}exec sleep 5\n',
        encoding="utf-8",
    )
    _write_common_stubs(bindir)
    subprocess.run(["bash", str(LAUNCHER)], env=_launcher_env(bindir, tmp_path), check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    argv = argv_log.read_text(encoding="utf-8-sig").split("\n")
    assert "-SendCutText=0" in argv, argv
    assert not any(a.startswith("-AcceptCutText") for a in argv), "paste into the screen must keep working"
    # Xvnc's own cut-text cap and the bridge filter's must agree, or one side drops a paste the other admits.
    from tools.bot_desktop.rfb_filter import _MAX_CUT_TEXT
    assert int(argv[argv.index("-MaxCutText") + 1]) == _MAX_CUT_TEXT, argv
    assert not os.path.exists(tmp_path / "rfb.sock")  # stub never bound it; nothing leaked


def test_readiness_probe_waits_for_the_x_socket_itself(tmp_path, xvnc_x_socket):
    """Regression for #123130: Xvnc runs -nolisten tcp, yet the readiness probe drove an X client at a
    bare ":N", which lets libxcb fall back to TCP 127.0.0.1:60NN while the unix socket is not bound
    yet — refused instantly on normal kernels, but silently dropped (SYN-SENT for ~2 minutes) under
    WSL2 mirrored networking, outliving runtime.start()'s 15 s window. The probe therefore waits for
    the display socket Xvnc itself binds instead of driving xdpyinfo at all: every display string a
    client could use is either the TCP-fallback shape (":N") or a parse error on libxcb >= 1.16
    ("unix:N" parses as a socket path there — commit 09525553 — and only ever worked by the accident
    of "unix" reading as a hostname)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    probe_log = tmp_path / "xdpyinfo-calls"
    (bindir / "Xvnc").write_text(f'#!/bin/sh\nsleep 0.3\n{_bind_x_socket_line()}exec sleep 5\n', encoding="utf-8")
    _write_common_stubs(bindir)
    # No real X server backs the stub socket, so any X client the launcher drives at the display
    # fails — matching libxcb, which has no display string that is both unix-only and parseable.
    # If the readiness path ever grows an xdpyinfo call again, the launcher exits 1 ("Xvnc did not
    # become ready") and this test fails with it.
    (bindir / "xdpyinfo").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{probe_log}"\nexit 1\n', encoding="utf-8")
    (bindir / "xdpyinfo").chmod(0o755)
    subprocess.run(["bash", str(LAUNCHER)], env=_launcher_env(bindir, tmp_path), check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    assert not probe_log.exists(), "readiness must not drive an X client at the display"
