"""Hand root-written sandbox files back to the host user when a root Docker container is removed.

The Docker backend runs the image's default user (root) unless ``docker_run_as_host_user`` is set,
so everything the container writes into the Hermes-created bind mounts (``<sandbox>/home`` at
``/root``, ``<sandbox>/workspace`` at ``/workspace``) lands on the host owned by uid 0, and the user
cannot delete it. While the container still runs, root inside it can ``chown`` those trees back;
once it is gone only ``sudo`` can.

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


def handback_ids(docker_exe: str, runs_as_host_user: bool) -> Optional[tuple[int, int]]:
    """Host ``(uid, gid)`` to hand sandbox files to, or ``None`` when no handback applies: the
    container already writes as the host user, the host is not Linux (Docker Desktop bind mounts
    are not root-owned on the host), Hermes itself runs as root, or the daemon remaps users."""
    if runs_as_host_user or sys.platform != "linux" or os.geteuid() == 0:  # windows-footgun: ok — linux-gated
        return None
    return None if _daemon_remaps_users(docker_exe) else (os.geteuid(), os.getegid())  # windows-footgun: ok


def _record_path(mounts: list[tuple[str, str]]) -> Optional[Path]:
    """The record lives in the per-task sandbox dir, the parent of every Hermes bind source."""
    return Path(mounts[0][0]).parent / HANDBACK_RECORD if mounts else None


def _container_mount_points(docker_exe: str, container_id: str) -> Optional[list[str]]:
    """Destination of every mount the container really has, including ``docker_extra_args``
    ``-v``/``--mount`` binds and named volumes, or ``None`` when Docker cannot say (then the
    walk is skipped: a same-filesystem bind is invisible to ``find -xdev``)."""
    try:
        result = run_capture([docker_exe, "inspect", "--format", "{{json .Mounts}}", container_id], timeout=10)
        if result.returncode == 0:
            return [m["Destination"] for m in json.loads(result.stdout)]
    except (subprocess.TimeoutExpired, OSError, ValueError, TypeError, KeyError) as e:
        logger.debug("docker inspect mounts of %s failed: %s", container_id[:12], e)
    return None


def walk_argv(mounts: list[tuple[str, str]], mount_points: list[str], owner: int) -> list[str]:
    """``find`` argv printing, NUL-terminated, every entry owned by *owner* under the sandbox
    mounts. It never follows symlinks (``-P``) and prunes each mount point nested below them."""
    targets = [container for _, container in mounts]
    prune: list[str] = []
    for point in sorted({p.rstrip("/") for p in mount_points}):
        if any(point.startswith(t.rstrip("/") + "/") for t in targets):
            prune += [*(["-o"] if prune else []), "-path", re.sub(r"([][*?\\])", r"\\\1", point)]
    return ["find", "-P", *targets, "-xdev", *(["(", *prune, ")", "-prune", "-o"] if prune else []),
            "-user", str(owner), "-print0"]


def _walk(docker_exe: str, container_id: str, mounts: list[tuple[str, str]], owner: int,
          deadline: float) -> Optional[list[bytes]]:
    points = _container_mount_points(docker_exe, container_id)
    if points is None:
        return None
    result = subprocess.run([docker_exe, "exec", "-u", "0:0", container_id, *walk_argv(mounts, points, owner)],
                            capture_output=True, timeout=max(1.0, deadline - time.monotonic()),
                            stdin=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        logger.debug("ownership walk in %s exited %d: %s", container_id[:12], result.returncode,
                     result.stderr.decode("utf-8", "replace").strip()[:500])
    return [p for p in result.stdout.split(b"\0") if p]


def _chown(docker_exe: str, container_id: str, paths: list[bytes], owner: str, deadline: float) -> int:
    result = subprocess.run(
        [docker_exe, "exec", "-i", "-u", "0:0", container_id, "xargs", "-0", "-r", "chown", "-h", owner, "--"],
        input=b"".join(p + b"\0" for p in paths), capture_output=True,
        timeout=max(1.0, deadline - time.monotonic()), check=False)
    return result.returncode


def hand_back_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]],
                            ids: tuple[int, int]) -> None:
    """Before the container is stopped: give its root-owned sandbox files (``mounts`` are
    ``(host_dir, container_path)``) to the host user and record which. The paths are recorded
    before any of them changes owner, so a timeout mid-chown still leaves a complete record.
    Best-effort: a gone container, an image without ``find`` or a timeout logs at debug."""
    record = _record_path(mounts)
    if record is None:
        return
    deadline = time.monotonic() + _CHOWN_TIMEOUT_S
    try:
        paths = _walk(docker_exe, container_id, mounts, 0, deadline)
        if not paths:
            return
        with record.open("ab") as fh:  # append: an unreclaimed earlier record still applies
            fh.write(b"".join(p + b"\0" for p in paths))
        _chown(docker_exe, container_id, paths, f"{ids[0]}:{ids[1]}", deadline)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ownership handback in %s failed: %s", container_id[:12], e)


def reclaim_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]],
                          runs_as_host_user: bool) -> None:
    """After a container mounts sandbox dirs an earlier teardown handed back: return the recorded
    paths to root (so git/ssh ownership checks pass again), then drop the record. A container that
    runs as the host user already owns them, so the record is just dropped. Only recorded paths a
    fresh no-symlink walk of this container's sandbox mounts still finds (outside every nested
    mount) are touched, so a symlinked directory or a real host dir now bound inside is never chowned."""
    record = _record_path(mounts)
    if record is None or not record.is_file():
        return
    if runs_as_host_user or sys.platform != "linux":
        record.unlink(missing_ok=True)
        return
    try:
        recorded = set(record.read_bytes().split(b"\0"))
        owned = _walk(docker_exe, container_id, mounts, os.geteuid(),  # windows-footgun: ok — linux-gated
                      time.monotonic() + _CHOWN_TIMEOUT_S)
        if owned is None:
            return  # keep the record for the next container
        paths = [p for p in owned if p in recorded]
        if paths:
            rc = _chown(docker_exe, container_id, paths, "0:0", time.monotonic() + _CHOWN_TIMEOUT_S)
            logger.debug("ownership reclaim in %s exited %d", container_id[:12], rc)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ownership reclaim in %s failed: %s", container_id[:12], e)
        return
    record.unlink(missing_ok=True)
