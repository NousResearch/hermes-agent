"""Hand root-written sandbox files back to the host user when a root Docker container is removed.

The Docker backend runs the image's default user (root) unless ``docker_run_as_host_user`` is set,
so everything the container writes into the Hermes-created bind mounts (``<sandbox>/home`` at
``/root``, ``<sandbox>/workspace`` at ``/workspace``) lands on the host owned by uid 0, and the user
cannot delete it. While the container still runs, root inside it can ``chown`` those trees back;
once it is gone only ``sudo`` can.

Persistent sandbox dirs outlive their container, and root refuses some host-owned files: git stops
at "dubious ownership", ssh at "Bad owner" on ``~/.ssh/config``. So the handback records the exact
paths it changed beside the dirs (on the host, outside every mount), and the next container that
mounts them gives exactly those paths back to root before its first command. Files the user added
on the host in between keep their owner. Mounts nested under the sandbox paths (read-only skills,
credentials, caches, user volumes) are pruned, so host data outside the sandbox dir is never touched.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from tools.environments.remote_common import run_capture

logger = logging.getLogger(__name__)

# Bounded so a huge workspace cannot hold teardown past the atexit drain (15s); a partial walk
# still records and changes everything it reached.
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


def handback_argv(mounts: list[tuple[str, str]], other_mounts: list[str], uid: int, gid: int) -> list[str]:
    """``find`` argv that chowns every root-owned entry under the sandbox mounts to ``uid:gid`` and
    prints each changed path NUL-terminated, pruning the *other_mounts* nested below them."""
    targets = [container for _, container in mounts]
    prune: list[str] = []
    for mount in (m.rstrip("/") for m in other_mounts):
        if any(mount.startswith(t.rstrip("/") + "/") for t in targets):
            prune += [*(["-o"] if prune else []), "-path", mount]
    return ["find", *targets, "-xdev", *(["(", *prune, ")", "-prune", "-o"] if prune else []),
            "-user", "0", "-print0", "-exec", "chown", "-h", f"{uid}:{gid}", "{}", "+"]


def hand_back_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]],
                            other_mounts: list[str], ids: tuple[int, int]) -> None:
    """Before the container is stopped: give its root-owned sandbox files (``mounts`` are
    ``(host_dir, container_path)``) to the host user and record which. Best-effort: a gone
    container, an image without ``find`` or a timeout logs at debug and never raises."""
    record = _record_path(mounts)
    if record is None:
        return
    argv = [docker_exe, "exec", "-u", "0:0", container_id, *handback_argv(mounts, other_mounts, *ids)]
    try:
        result = subprocess.run(argv, capture_output=True, timeout=_CHOWN_TIMEOUT_S,
                                stdin=subprocess.DEVNULL, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ownership handback in %s failed: %s", container_id[:12], e)
        return
    if result.returncode != 0:
        logger.debug("ownership handback in %s exited %d: %s", container_id[:12], result.returncode,
                     result.stderr.decode("utf-8", "replace").strip()[:500])
    if result.stdout:
        try:
            with record.open("ab") as fh:  # append: an unreclaimed earlier record still applies
                fh.write(result.stdout)
        except OSError as e:
            logger.debug("could not record ownership handback at %s: %s", record, e)


def _under(path: bytes, roots: list[str]) -> bool:
    encoded = [r.rstrip("/").encode() for r in roots]
    return any(path == r or path.startswith(r + b"/") for r in encoded)


def reclaim_sandbox_files(docker_exe: str, container_id: str, mounts: list[tuple[str, str]],
                          other_mounts: list[str]) -> None:
    """After a container mounts sandbox dirs an earlier teardown handed back: return the recorded
    paths to root (so git/ssh ownership checks pass again), then drop the record. Only paths still
    under this container's sandbox mounts and outside its other mounts are touched, so a config
    change that now binds a real host directory at ``/workspace`` never gets it chowned."""
    record = _record_path(mounts)
    if record is None or not record.is_file():
        return
    try:
        sandbox_roots = [container for _, container in mounts]
        paths = b"".join(
            p + b"\0" for p in record.read_bytes().split(b"\0")
            if p and _under(p, sandbox_roots) and not _under(p, other_mounts))
        result = subprocess.run(
            [docker_exe, "exec", "-i", "-u", "0:0", container_id, "xargs", "-0", "-r", "chown", "-h", "0:0", "--"],
            input=paths, capture_output=True, timeout=_CHOWN_TIMEOUT_S, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ownership reclaim in %s failed: %s", container_id[:12], e)
        return
    # Nonzero is expected when the user deleted some recorded files; everything else was restored.
    logger.debug("ownership reclaim in %s exited %d", container_id[:12], result.returncode)
    record.unlink(missing_ok=True)
