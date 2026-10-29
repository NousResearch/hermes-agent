"""Per-profile runtime and tmp dirs for browser-harness daemons (Browser Use mode).

A multiplexed gateway serves every profile from one process. The harness names a daemon by
``BU_NAME`` alone and reuses any live one, so each served profile gets its own runtime dir (daemon
sockets) and tmp dir (screenshots, debug overlays, logs). A runtime dir too long for a 64-character
session's AF_UNIX socket moves to a private short dir under ``$XDG_RUNTIME_DIR`` or the socket temp
root; outside a served profile, so does a default runtime dir whose socket for the call's own
``BU_NAME`` would overflow (a deep HOME).

``browser_use_cli._base_subprocess_env`` applies the result through ``namespace_harness_dirs``, and
``browser_exec`` calls ``fit_default_socket_dir`` once the call's ``BU_NAME`` is final.
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Optional

# browser_harness/_ipc.py: a shared runtime dir names a daemon socket ``<dir>/bu-<BU_NAME>.sock`` and
# BU_NAME may be 64 characters, so the dir gets sun_path (macOS 104 bytes, Linux 108, with the NUL)
# minus 73 (`/bu-` + 64 + `.sock`).
_MACOS_SOCKET_DIR_BUDGET = 103 - len("/bu-.sock") - 64
_LINUX_SOCKET_DIR_BUDGET = 107 - len("/bu-.sock") - 64


# The env keys that, with BU_NAME, identify one harness daemon (a served profile's dirs are its own).
HARNESS_DIR_KEYS = ("BH_RUNTIME_DIR", "BH_RUNTIME_DIR_SHARED", "BH_TMP_DIR", "BH_TMP_DIR_SHARED")


def daemon_identity(env: dict) -> tuple[str, ...]:
    """``(*HARNESS_DIR_KEYS values, BU_NAME)`` of the daemon a call with ``env`` drives."""
    return (*(env.get(k, "") for k in HARNESS_DIR_KEYS), env.get("BU_NAME", "default"))


def daemon_env(env: dict, identity: tuple[str, ...]) -> dict:
    """``env`` re-pointed at the daemon ``identity`` names (its own dirs and BU_NAME)."""
    *dirs, name = identity
    target = {k: v for k, v in env.items() if k not in HARNESS_DIR_KEYS}
    target.update((k, v) for k, v in zip(HARNESS_DIR_KEYS, dirs) if v)
    return {**target, "BU_NAME": name}


def socket_dir_budget(platform: str) -> Optional[int]:
    """The runtime-dir byte budget on ``platform`` (a ``sys.platform`` value) where the harness binds
    AF_UNIX sockets, else None."""
    if platform == "darwin":
        return _MACOS_SOCKET_DIR_BUDGET
    if platform.startswith("linux"):
        return _LINUX_SOCKET_DIR_BUDGET
    return None  # Windows: the harness serves over loopback TCP (port files), no sun_path


def fits_socket_budget(resolved_dir: str, budget: Optional[int]) -> bool:
    """Whether a 64-character session's socket under ``resolved_dir`` (the absolute, symlink-resolved
    path the harness binds) fits ``budget``; no budget means no sun_path limit."""
    return budget is None or len(resolved_dir.encode()) <= budget


def namespace_harness_dirs(env: dict, tag: str) -> None:
    """Give served profile ``tag`` its own harness runtime and tmp dirs in the child ``env``. An empty
    ``tag`` (outside a served-profile scope) leaves ``env`` unchanged."""
    if not tag:
        return
    root = _harness_home(env)
    leaf = f"p-{hashlib.sha256(tag.encode()).hexdigest()[:12]}"
    base = env.get("BH_RUNTIME_DIR") or env.get("BH_TMP_DIR") or str(Path(root) / "runtime")
    runtime = str(Path(base) / leaf)
    env["BH_RUNTIME_DIR"] = socket_fit_runtime_dir(runtime, base, tag)
    env["BH_RUNTIME_DIR_SHARED"] = "1"  # several BU_NAMEs share the dir: keep bu-<name> stems
    # Screenshots, debug overlays and daemon logs (paths.tmp_dir): the profile's own, never shared.
    env["BH_TMP_DIR"] = str(Path(env.get("BH_TMP_DIR") or str(Path(root) / "tmp")) / leaf)
    env["BH_TMP_DIR_SHARED"] = "1"


def _harness_home(env: dict) -> str:
    """browser-harness ``paths.home_dir()``, resolved against the child's env."""
    root = env.get("BH_HOME") or env.get("BROWSER_HARNESS_HOME")
    if not root:
        xdg = env.get("XDG_CONFIG_HOME")
        root = str(Path(xdg) / "browser-harness") if xdg else str(
            Path(env.get("HOME") or Path.home()) / ".config" / "browser-harness")
    return str(Path(root).expanduser())


def fit_default_socket_dir(env: dict) -> None:
    """Outside a served profile the harness binds ``<home>/runtime/bu-<BU_NAME>.sock``. Under a deep
    HOME (a container's profile home ``$HERMES_HOME/home``, a long account home) that path overflows
    sun_path and every call dies with ``AF_UNIX path too long``; such a call's runtime dir moves to a
    private short dir. Judged on the call's own BU_NAME so a socket that already binds keeps its
    path (and its live daemon). A runtime/tmp dir already set (a served profile's, or exported by
    the user) is left alone."""
    if env.get("BH_RUNTIME_DIR") or env.get("BH_TMP_DIR"):
        return
    budget = socket_dir_budget(sys.platform)
    if budget is None:
        return
    root = _harness_home(env)
    stem_budget = budget + 64 - len(env.get("BU_NAME") or "default")  # budget assumes a 64-char name
    if fits_socket_budget(str((Path(root) / "runtime").resolve()), stem_budget):
        return
    short = _claim_short_dir(f"{root}\0", stem_budget)
    if short:
        env["BH_RUNTIME_DIR"] = short
        env["BH_RUNTIME_DIR_SHARED"] = "1"  # several BU_NAMEs share the dir: keep bu-<name> stems


def socket_fit_runtime_dir(runtime: str, base: str, tag: str) -> str:
    """``runtime`` unless the socket the harness binds under it would be too long for a 64-character
    session. The harness binds ``Path(BH_RUNTIME_DIR).expanduser().resolve()``, so the length is
    judged on that path (a short relative or symlinked spelling can still resolve deep); a dir that
    fits is returned as given. Otherwise the served profile gets a private short dir
    (``_claim_short_dir``) keyed by base and profile; if none fits or can be claimed safely the long
    path stays (shorter names only)."""
    budget = socket_dir_budget(sys.platform)
    if fits_socket_budget(str(Path(runtime).expanduser().resolve()), budget):
        return runtime
    return _claim_short_dir(f"{base}\0{tag}", budget) or runtime


def _claim_short_dir(key: str, budget: Optional[int]) -> Optional[str]:
    """A private ``hbh-<12 hex>`` dir (keyed by uid and ``key``) that fits ``budget``: under the
    user's private ``$XDG_RUNTIME_DIR`` first, else the short socket temp root; None when no
    candidate fits or can be claimed safely."""
    from hermes_constants import socket_safe_tmpdir
    getuid = getattr(os, "getuid", lambda: 0)
    digest = hashlib.sha256(f"{getuid()}\0{key}".encode()).hexdigest()[:12]
    # socket_safe_tmpdir() caps the root at 50 bytes for Chrome's longer socket names; ours needs a
    # root of at most 17-21, which a Hermes scratch TMPDIR exceeds, so the sticky system root follows.
    # A container or macOS has no XDG_RUNTIME_DIR, which is why the shared roots stay as fallbacks.
    xdg = [xdg_dir] if (xdg_dir := private_user_runtime_dir()) else []
    roots = dict.fromkeys(os.path.realpath(r) for r in (*xdg, socket_safe_tmpdir(), "/tmp"))  # no-tmp: ok — AF_UNIX sun_path budget; dir claimed 0700 and ownership-checked
    for root in roots:
        short = os.path.join(root, f"hbh-{digest}")
        if fits_socket_budget(short, budget) and claim_private_dir(short):
            return short
    return None


def private_user_runtime_dir() -> Optional[str]:
    """``$XDG_RUNTIME_DIR`` (a system-unit gateway has none: ``/run/user/<uid>`` then) when it is
    what the spec promises: an absolute real directory owned by this uid with no group/other access,
    so no other user can pre-create a name inside it."""
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return None
    raw = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{getuid()}"
    if not os.path.isabs(raw):
        return None
    try:
        st = os.lstat(raw)
    except OSError:
        return None
    ours = stat.S_ISDIR(st.st_mode) and st.st_uid == getuid()
    return raw if ours and not st.st_mode & 0o077 else None


def claim_private_dir(path: str) -> bool:
    """Create ``path`` 0700, or accept an existing one only when it is a real directory (not a
    symlink) owned by this uid and closed to group/other: the temp root is world-writable, and the
    harness resolves symlinks and only chmods a dir it created itself."""
    try:
        os.mkdir(path, 0o700)
        return True
    except FileExistsError:
        pass
    except OSError:
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return (stat.S_ISDIR(st.st_mode) and st.st_uid == getattr(os, "getuid", lambda: -1)()
            and not st.st_mode & 0o077)
