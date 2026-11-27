"""Copy out of a bot's screen: the X ``CLIPBOARD`` text, read for the human who holds the lease.

Xvnc runs ``-SendCutText=0`` and stays that way: the display bridge filters only the client→server
stream, so a ServerCutText would reach every viewer on the screen, watchers included (TigerVNC pushes
the text unsolicited to a client whose clipboard capabilities the bridge dropped, which it does for
every watcher). This read is the gated way out instead. ``display.clipboard.read`` calls it only for
the viewer that holds the lease, the text goes back on that viewer's own RPC reply, and nothing is
logged.

The selection is read with ``xclip`` on the screen's own display (inside the sandbox for a
sandbox-hosted screen). A screen without xclip reports ``unavailable`` rather than failing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from tools.bot_desktop import runtime

#: Same bound as paste-in (launcher.sh -MaxCutText, rfb_filter._MAX_CUT_TEXT): one cap both ways.
MAX_TEXT = 256 * 1024
_ARGV = ["xclip", "-o", "-selection", "clipboard"]
_TIMEOUT = 5.0


@dataclass(frozen=True)
class ClipboardRead:
    text: Optional[str]
    reason: Optional[str] = None  # "empty" | "too_large" | "unavailable" | "not_running"


def _from_bytes(raw: bytes) -> ClipboardRead:
    if not raw:
        return ClipboardRead(None, "empty")
    if len(raw) > MAX_TEXT:
        return ClipboardRead(None, "too_large")
    return ClipboardRead(raw.decode("utf-8", "replace"))


def read_text() -> ClipboardRead:
    env = runtime.published_env()
    display = env.get("DISPLAY")
    if not display:
        return ClipboardRead(None, "not_running")
    child_env = {k: v for k, v in env.items() if k in ("DISPLAY", "XAUTHORITY")}
    if runtime.sandbox_screen_running():
        from tools.bot_desktop import sandbox_host
        sandbox = runtime._owned_sandbox_env()
        if sandbox is None:
            return ClipboardRead(None, "not_running")
        return sandbox_host.read_clipboard(sandbox, child_env, _ARGV, _TIMEOUT) or ClipboardRead(None, "unavailable")
    if shutil.which("xclip") is None:
        return ClipboardRead(None, "unavailable")
    try:
        proc = subprocess.run(_ARGV, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **child_env},
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return ClipboardRead(None, "unavailable")
    # xclip exits 1 when nothing owns CLIPBOARD or the owner offers no text target: that is "empty".
    return _from_bytes(proc.stdout if proc.returncode == 0 else b"")
