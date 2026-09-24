"""Bot Desktop hosted INSIDE the configured terminal backend (docker / ssh / singularity).

Same launcher, same lease, same pane: only WHERE Xvnc + Xfce run changes. ``runtime`` decides between the
gateway host and this module via ``placement()``; callers keep using ``runtime.start/stop/status/desktop_env``.

Layout inside the sandbox, under ``<sandbox tmp>/hermes-bot-desktop/<profile>/``: ``launcher.sh`` (copied in
at start), ``env`` (published DISPLAY/XAUTHORITY/DBUS), ``rfb.sock`` (Xvnc's unix socket), ``launcher.pid``.
Host-side state under ``<HERMES_HOME>/bot-desktop/``: ``sandbox.json`` = ``{"env_key": ..., "display": ...}``
so status() knows which terminal environment owns the screen without re-deriving it.

The Desktop pane never sees a socket path: ``open_rfb_stream()`` returns a Popen whose stdin/stdout ARE the
RFB byte stream, relayed by a 15-line Python script inside the sandbox (python3 is in every sandbox image;
socat is not). cua-driver runs the same way: ``cua_mcp_argv()`` is the docker/ssh prefix + ``cua-driver mcp``.
"""
from __future__ import annotations

import base64
import json
import logging
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from tools.environments import streams

logger = logging.getLogger(__name__)

# The user desktop processes run as inside the sandbox image. nikolaik (and hermes-sandbox:desktop on top of
# it) ships uid 1000 `pn`; a root Xvnc/Chromium/cua-driver is the wrong shape (Chromium refuses the sandbox,
# AT-SPI wants a user session). Falls back to the exec default when the image has no such user.
DESKTOP_USER = "pn"
_REQUIRED = ("Xvnc", "xfwm4", "xfce4-panel", "xfdesktop", "xfsettingsd", "dbus-run-session", "xauth", "xdpyinfo", "xprop")
SANDBOX_IMAGE_HINT = "nousresearch/hermes-sandbox:desktop"

_RELAY = (
    "import os,socket,sys,threading\n"
    "s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1])\n"
    "def up():\n"
    "  while True:\n"
    "    d=os.read(0,65536)\n"
    "    if not d: break\n"
    "    s.sendall(d)\n"
    "  s.shutdown(socket.SHUT_WR)\n"
    "threading.Thread(target=up,daemon=True).start()\n"
    "while True:\n"
    "  d=s.recv(65536)\n"
    "  if not d: break\n"
    "  os.write(1,d)\n"
)


def _marker() -> Path:
    return get_hermes_home() / "bot-desktop" / "sandbox.json"


def _read_marker() -> Dict[str, Any]:
    try:
        return json.loads(_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remote_dir(env: Any, profile: str) -> str:
    return f"{env.get_temp_dir().rstrip('/')}/hermes-bot-desktop/{profile}"


def _user_for(env: Any) -> Optional[str]:
    """DESKTOP_USER when the sandbox has it, else None (exec default). Cached on the env object."""
    cached = getattr(env, "_bd_desktop_user", "unset")
    if cached != "unset":
        return cached
    probe = streams.run_in(env, ["id", "-u", DESKTOP_USER], timeout=15)
    user = DESKTOP_USER if probe.returncode == 0 else None
    env._bd_desktop_user = user
    return user


def missing_binaries(env: Any) -> list[str]:
    script = "for b in " + " ".join(_REQUIRED) + '; do command -v "$b" >/dev/null 2>&1 || echo "$b"; done'
    proc = streams.run_in(env, ["bash", "-c", script], user=_user_for(env), timeout=20)
    return [b for b in proc.stdout.decode("utf-8", "replace").split() if b]


def _published(env: Any, rdir: str) -> Dict[str, str]:
    proc = streams.run_in(env, ["bash", "-c", f"kill -0 $(cat {shlex.quote(rdir)}/launcher.pid 2>/dev/null) 2>/dev/null "
                                             f"&& test -S {shlex.quote(rdir)}/rfb.sock && cat {shlex.quote(rdir)}/env"],
                          user=_user_for(env), timeout=15)
    if proc.returncode != 0:
        return {}
    out: Dict[str, str] = {}
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def published_env(env: Any, profile: str) -> Dict[str, str]:
    return _published(env, _remote_dir(env, profile))


def start(env: Any, profile: str, *, geometry: str, wait_seconds: float = 20.0,
          browser_exec: Optional[str] = None, browser_exec_line: Optional[str] = None) -> Dict[str, str]:
    """Bring the screen up inside ``env`` (idempotent); returns the published env. Raises RuntimeError naming
    the blocker."""
    rdir = _remote_dir(env, profile)
    live = _published(env, rdir)
    if live.get("DISPLAY"):
        return live
    stop(env, profile)  # a dead launcher may have left Xvnc holding :20; the relaunch needs it gone
    missing = missing_binaries(env)
    if missing:
        raise RuntimeError(
            f"Bot Desktop needs {', '.join(missing)} inside the terminal backend's sandbox. The configured image "
            f"does not have them; use {SANDBOX_IMAGE_HINT} (the default sandbox base plus the desktop stack) as "
            f"terminal.docker_image / modal_image / singularity_image, or set bot_desktop.placement: gateway.")
    user = _user_for(env)
    launcher = (Path(__file__).with_name("launcher.sh").read_bytes())
    wallpaper = Path(__file__).with_name("wallpaper.png").read_bytes()
    seed = (
        f"set -e; rm -rf {shlex.quote(rdir)}; mkdir -p {shlex.quote(rdir)}; cd {shlex.quote(rdir)};"
        f" echo {shlex.quote(base64.b64encode(launcher).decode())} | base64 -d > launcher.sh;"
        f" echo {shlex.quote(base64.b64encode(wallpaper).decode())} | base64 -d > wallpaper.png;"
        " chmod 0700 . launcher.sh"
    )
    proc = streams.run_in(env, ["bash", "-c", seed], user=user, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"could not seed the sandbox desktop dir: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    num = 20  # one profile per sandbox; the launcher's stale-lock logic handles a leftover :20
    child_env = {
        "HERMES_BD_PROFILE": profile, "HERMES_BD_DISPLAY_NUM": str(num), "HERMES_BD_SOCKET": f"{rdir}/rfb.sock",
        "HERMES_BD_XAUTH": f"{rdir}/Xauthority", "HERMES_BD_ENV_FILE": f"{rdir}/env",
        "HERMES_BD_CONFIG_HOME": f"{rdir}/xdg", "HERMES_BD_GEOMETRY": geometry, "HERMES_BD_WALLPAPER": f"{rdir}/wallpaper.png",
    }
    if browser_exec and browser_exec_line:
        child_env["HERMES_BD_BROWSER_EXEC"] = browser_exec
        child_env["HERMES_BD_BROWSER_EXEC_LINE"] = browser_exec_line
    # The launcher must outlive this exec (docker exec / ssh) and be its own session so stop() can take the
    # whole group (Xvnc, dbus, Xfce) with one kill. `setsid -f` forks, so `$!` would be the wrong pid: the
    # launcher records ITS OWN pid (= its session id) before doing anything else.
    spawn = (f"cd {shlex.quote(rdir)} && setsid -f bash -c "
             f"'echo $$ > launcher.pid; exec bash launcher.sh' > launcher.log 2>&1 < /dev/null; "
             f"for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s launcher.pid ] && break; sleep 0.1; done; [ -s launcher.pid ]")
    proc = streams.run_in(env, ["bash", "-c", spawn], child_env=child_env, user=user, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"sandbox desktop launcher did not start: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        live = _published(env, rdir)
        if live.get("DISPLAY"):
            _marker().parent.mkdir(parents=True, exist_ok=True)
            _marker().write_text(json.dumps({"display": live["DISPLAY"], "dir": rdir, "profile": profile}), encoding="utf-8")
            logger.info("Bot Desktop for profile %s up inside %s on %s", profile, type(env).__name__, live["DISPLAY"])
            return live
        time.sleep(0.25)
    tail = streams.run_in(env, ["tail", "-c", "2000", f"{rdir}/launcher.log"], user=user, timeout=10).stdout
    stop(env, profile)
    raise RuntimeError(f"sandbox desktop did not publish its display within {wait_seconds:.0f}s:\n"
                       f"{tail.decode('utf-8', 'replace')}")


def stop(env: Any, profile: str) -> bool:
    """Kill the launcher's session (Xvnc, dbus, Xfce, anything the desktop spawned) and, as a backstop, every
    process still holding this profile's rfb.sock or Xauthority path (a launcher that lost its pid file left
    an Xvnc that 'Server is already active for display 20' on the next start). True when something was live."""
    rdir = _remote_dir(env, profile)
    q = shlex.quote(rdir)
    script = f"""
live=0
p=$(cat {q}/launcher.pid 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then live=1; kill -TERM -- -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; fi
for pid in $(pgrep -f -- {shlex.quote(rdir + "/")} 2>/dev/null); do [ "$pid" != "$$" ] && {{ live=1; kill -TERM "$pid" 2>/dev/null; }}; done
sleep 0.5
for pid in $(pgrep -f -- {shlex.quote(rdir + "/")} 2>/dev/null); do [ "$pid" != "$$" ] && kill -KILL "$pid" 2>/dev/null; done
rm -f {q}/env {q}/launcher.pid {q}/rfb.sock
[ "$live" = 1 ]
"""
    proc = streams.run_in(env, ["bash", "-c", script], user=_user_for(env), timeout=30)
    _marker().unlink(missing_ok=True)
    return proc.returncode == 0


def open_rfb_stream(env: Any, profile: str) -> subprocess.Popen:
    """Popen whose stdin/stdout carry the RFB bytes of the sandbox's Xvnc."""
    rdir = _remote_dir(env, profile)
    return streams.open_stream(env, ["python3", "-c", _RELAY, f"{rdir}/rfb.sock"], user=_user_for(env))


def cua_mcp_invocation(env: Any, profile: str, published: Dict[str, str]) -> tuple[str, list[str]]:
    """``(command, args)`` for ``StdioServerParameters``: the backend's exec prefix running ``cua-driver mcp`` on
    the sandbox display."""
    prefix = streams.exec_prefix(env, user=_user_for(env), interactive=True)
    if prefix is None:
        raise RuntimeError(f"{type(env).__name__} cannot host cua-driver")
    argv = streams.remote_argv(prefix, ["cua-driver", "mcp", "--no-overlay"], env=published)
    return argv[0], argv[1:]


def exec_prefix_for_tools(env: Any) -> Optional[list[str]]:
    """The prefix agent-browser invocations are wrapped in when the browser follows the terminal backend."""
    return streams.exec_prefix(env, user=_user_for(env), interactive=True)
