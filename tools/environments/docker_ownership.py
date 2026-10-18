"""Hand root-written sandbox files back to the host user when a root Docker container is removed.

The Docker backend runs the image's default user (root) unless ``docker_run_as_host_user`` is set,
so everything the container writes into the Hermes-created bind mounts (``<sandbox>/home`` at
``/root``, ``<sandbox>/workspace`` at ``/workspace``) lands on the host owned by uid 0, and the user
cannot delete it. Until the container is removed, a root container with its volumes can ``chown``
those trees back; once it is gone only ``sudo`` can. The handback runs after it stopped, so files a
process writes on shutdown are included.

Persistent sandbox dirs outlive their container, and root refuses some host-owned files: git stops
at "dubious ownership", ssh at "Bad owner" on ``~/.ssh/config``. So the handback records the exact
paths it changes beside the dirs (on the host, outside every mount), and the next container that
mounts them gives exactly those paths back to root before its first command. Files the user added
on the host in between keep their owner. Every walk is ``find -P`` (no symlinks followed) and prunes
each mount Docker reports nested under the sandbox paths (read-only skills, credentials, caches,
``docker_volumes`` and ``docker_extra_args`` binds), so host data outside the sandbox dir is never touched.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from tools.environments.remote_common import run_capture

logger = logging.getLogger(__name__)

# Bounded so a huge workspace cannot hold teardown past the atexit drain (15s). Paths are recorded
# before any chown, so a timeout mid-chown never leaves a changed path unrecorded.
_CHOWN_TIMEOUT_S = 10
HANDBACK_RECORD = ".hermes-owner-handback"
# Root runs agent-writable trees: name tools by absolute path (an image without them skips the pass)
# and reset PATH and loader env so nothing planted under /root or /workspace runs instead.
_SAFE_ENV = ["-e", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "-e", "LD_PRELOAD=", "-e", "LD_LIBRARY_PATH=", "-w", "/"]
# The throwaway gets only what find and chown need, so even a planted binary gains no host power.
_THROWAWAY_CONFINEMENT = ["--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "DAC_OVERRIDE", "--cap-add", "FOWNER",
                          "--security-opt", "no-new-privileges", "--pids-limit", "64", "--label", "hermes-handback=1"]

_remapping_cache: dict[str, bool] = {}


def _daemon_remaps_users(docker_exe: str) -> bool:
    """True when container uid 0 is not host uid 0 (rootless Docker, userns-remap) or when that
    cannot be told. Chowning to the host uid there maps to a subordinate id and makes the files
    less reachable, so the handback is skipped. Cached per executable for the process."""
    if docker_exe not in _remapping_cache:
        try:
            result = run_capture([docker_exe, "info", "--format", "{{.SecurityOptions}}"], timeout=10)
            options = result.stdout.lower() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker info for ownership handback failed: %s", e)
            options = ""
        _remapping_cache[docker_exe] = not options.strip() or "rootless" in options or "userns" in options
    return _remapping_cache[docker_exe]


def _handback_ids(docker_exe: str) -> Optional[tuple[int, int]]:
    """Host ``(uid, gid)`` to hand sandbox files to, or ``None`` when no handback applies: the
    host is not Linux (Docker Desktop bind mounts are not root-owned on the host), Hermes itself
    runs as root, or the daemon remaps users."""
    if sys.platform != "linux" or os.geteuid() == 0:  # windows-footgun: ok — linux-gated
        return None
    return None if _daemon_remaps_users(docker_exe) else (os.geteuid(), os.getegid())  # windows-footgun: ok


def _record_path(mounts: list[tuple[str, str]]) -> Optional[Path]:
    """The record lives in the per-task sandbox dir, the parent of every Hermes bind source."""
    return Path(mounts[0][0]).parent / HANDBACK_RECORD if mounts else None


def _inspect(docker_exe: str, container_id: str) -> Optional[dict]:
    """The container's ``docker inspect`` record (its real user, image id and every mount,
    ``docker_extra_args`` binds and named volumes included), or ``None`` when Docker cannot say
    (then nothing is walked: a same-filesystem bind is invisible to ``find -xdev``)."""
    try:
        result = run_capture([docker_exe, "inspect", container_id], timeout=10)
        if result.returncode == 0:
            return json.loads(result.stdout)[0]
    except (subprocess.TimeoutExpired, OSError, ValueError, IndexError, KeyError) as e:
        logger.debug("docker inspect of %s failed: %s", container_id[:12], e)
    return None


def _runs_as_root(info: dict) -> bool:
    """The container's effective user (``--user`` from any source, else the image ``USER``) is root."""
    return ((info.get("Config") or {}).get("User") or "").split(":")[0] in ("", "root", "0")


def walk_argv(mounts: list[tuple[str, str]], mount_points: list[str], owner: int) -> list[str]:
    """``find`` argv printing, NUL-terminated, every entry owned by *owner* under the sandbox
    mounts. It never follows symlinks (``-P``) and prunes each mount point nested below them."""
    targets = [container for _, container in mounts]
    prune: list[str] = []
    for point in sorted({p.rstrip("/") for p in mount_points}):
        if any(point.startswith(t.rstrip("/") + "/") for t in targets):
            prune += [*(["-o"] if prune else []), "-path", re.sub(r"([][*?\\])", r"\\\1", point)]
    return ["/usr/bin/find", "-P", *targets, "-xdev", *(["(", *prune, ")", "-prune", "-o"] if prune else []),
            # Device nodes and FIFOs stay root's: a host-owned block device is raw disk access.
            "(", "-type", "f", "-o", "-type", "d", "-o", "-type", "l", ")", "-user", str(owner), "-print0"]


def _walk(docker_exe: str, target: list[str], mounts: list[tuple[str, str]], info: dict, owner: int,
          deadline: float) -> Optional[list[bytes]]:
    """Entries owned by *owner*, or ``None`` when the walk failed (its output may be partial)."""
    points = [m["Destination"] for m in info.get("Mounts") or []]
    result = _docker(docker_exe, target, walk_argv(mounts, points, owner), deadline)
    if result.returncode != 0:
        logger.debug("ownership walk exited %d: %s", result.returncode,
                     result.stderr.decode("utf-8", "replace").strip()[:500])
        return None
    return [p for p in result.stdout.split(b"\0") if p]


def _docker(docker_exe: str, target: list[str], argv: list[str], deadline: float, **kwargs):
    """Run *argv* in *target*. A timeout only kills the CLI, so a named throwaway is removed then
    instead of running on with the sandbox volumes."""
    try:
        stdin = kwargs.pop("stdin", None if "input" in kwargs else subprocess.DEVNULL)
        return subprocess.run([docker_exe, *target, *argv], capture_output=True, stdin=stdin,
                              timeout=max(1.0, deadline - time.monotonic()), check=False, **kwargs)
    except subprocess.TimeoutExpired:
        if "--name" in target:
            run_capture([docker_exe, "rm", "-f", target[target.index("--name") + 1]], timeout=10)
        raise


def _chown(docker_exe: str, target: list[str], paths: list[bytes], owner: str, deadline: float) -> bool:
    result = _docker(docker_exe, target, ["/usr/bin/xargs", "-0", "-r", "/bin/chown", "-h", owner, "--"], deadline,
                     input=b"".join(p + b"\0" for p in paths))
    if result.returncode != 0:
        logger.debug("ownership chown to %s exited %d", owner, result.returncode)
    return result.returncode == 0


def hand_back_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]]) -> None:
    """After a root container has stopped (so nothing writes any more) and before it is removed:
    give its root-owned sandbox files (``mounts`` are ``(host_dir, container_path)``) to the host
    user from a throwaway container of the same local image with the same volumes, and record
    which. The paths are recorded before any of them changes owner, so a timeout mid-chown still
    leaves a complete record. Best-effort: a gone container, an image without ``find`` or a
    timeout logs at debug."""
    record = _record_path(mounts)
    ids = _handback_ids(docker_exe) if record else None
    info = _inspect(docker_exe, container_id) if ids else None
    if not info or not _runs_as_root(info):
        return
    target = ["run", "--rm", "-i", "--pull", "never", "--network", "none", "--user", "0:0", *_THROWAWAY_CONFINEMENT,
              *_SAFE_ENV, "--name", f"hermes-handback-{container_id[:12]}-{secrets.token_hex(4)}",
              "--volumes-from", container_id, "--entrypoint", "", info["Image"]]
    deadline = time.monotonic() + _CHOWN_TIMEOUT_S
    try:
        paths = _walk(docker_exe, target, mounts, info, 0, deadline)
        if not paths:
            return
        known = set(record.read_bytes().split(b"\0")) if record.is_file() else set()
        with record.open("ab") as fh:  # append: an unreclaimed earlier record still applies
            fh.write(b"".join(p + b"\0" for p in paths if p not in known))
        _chown(docker_exe, target, paths, f"{ids[0]}:{ids[1]}", deadline)
    except (subprocess.TimeoutExpired, OSError, KeyError) as e:
        logger.debug("ownership handback of %s failed: %s", container_id[:12], e)


def reclaim_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]]) -> None:
    """After a container mounts sandbox dirs an earlier teardown handed back: when it runs as root,
    return the recorded paths to root (so git/ssh ownership checks pass again), then drop the
    record. A container running as any other user keeps the handed-back owner and just drops the
    record. Only recorded paths a fresh no-symlink walk of this container's sandbox mounts still
    finds (outside every nested mount) are touched, so a symlinked directory or a real host dir now
    bound inside is never chowned. A failed walk or chown keeps the record for the next container."""
    record = _record_path(mounts)
    if record is None or not record.is_file():
        return
    info = _inspect(docker_exe, container_id) if sys.platform == "linux" else {}
    if info is None:
        return
    if not info or not _runs_as_root(info):
        record.unlink(missing_ok=True)
        return
    target = ["exec", "-i", "-u", "0:0", *_SAFE_ENV, container_id]
    deadline = time.monotonic() + _CHOWN_TIMEOUT_S
    try:
        recorded = set(record.read_bytes().split(b"\0"))
        owned = _walk(docker_exe, target, mounts, info, os.geteuid(), deadline)  # windows-footgun: ok — linux-gated
        paths = [p for p in owned or [] if p in recorded]
        if owned is None or (paths and not _chown(docker_exe, target, paths, "0:0", deadline)):
            return
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ownership reclaim in %s failed: %s", container_id[:12], e)
        return
    record.unlink(missing_ok=True)
