"""Systemd worker scope launch, manager targeting and verified lifecycle.

Shared by terminal, Kanban and cron. Unsupported hosts keep their caller-owned
launch policy; indeterminate stop evidence never becomes verified quiescence.
"""
from contextlib import suppress
import logging
import os
import platform
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional

logger = logging.getLogger("tools.process_registry")
_IS_LINUX = platform.system() == "Linux"

# --- systemd cgroup isolation for gateway-spawned local executors ------------------
# Under a systemd gateway with MemoryMax, local background commands inherit the gateway's
# cgroup, so a memory-heavy executor can get the ENTIRE gateway killed by systemd-oomd;
# ``systemd-run --user --scope`` gives the worker its own transient cgroup. Usability is
# probed once (binary present but user D-Bus absent in system services/containers).
# A memory-heavy executor (Codex, tests, Node) can push the whole cgroup past MemoryMax and trigger
# systemd-oomd to kill the ENTIRE gateway — taking down the messaging control plane and silently losing the
# active turn. We probe *once* whether ``systemd-run --user --scope`` is actually usable (the binary can
# exist on the PATH while the user D-Bus session is unavailable — common for system services and
# containers), and cache the result for the process lifetime. See #70716.
_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _worker_memory_max_bytes(
    self_cgroup: str = "/proc/self/cgroup",
    mountinfo: str = "/proc/self/mountinfo",
) -> int:
    """Return a finite per-worker cgroup limit without widening host risk.

    The proposed local-memory-guard environment override is honored when it
    tightens the safe bound, so this isolation composes with PR #57121 instead
    of inventing a second knob.  An oversized override cannot widen host risk.
    Otherwise retain the tighter of the gateway's current cgroup memory limit
    (v2 ``memory.max`` on the unified hierarchy, v1
    ``memory.limit_in_bytes`` on the memory controller hierarchy) and half of
    physical RAM, capped at 4 GiB.  This keeps the sibling worker outside the
    gateway cgroup while ensuring the worker cannot consume memory up to the
    enclosing user slice or host limit.  ``self_cgroup`` / ``mountinfo`` are
    injectable so tests can feed synthetic procfs.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    candidates: List[int] = []
    try:
        for line in Path(self_cgroup).read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (
                    Path(_cgroup2_mount_point(mountinfo)) / relative / "memory.max"
                ).read_text(encoding="utf-8").strip()
                if raw_limit.isdigit():
                    cgroup_limit = int(raw_limit)
                    if cgroup_limit >= _MIN_WORKER_MEMORY_MAX_BYTES:
                        candidates.append(cgroup_limit)
                break
            # cgroup v1 hosts have no 0:: line; the memory controller's
            # entry looks like "10:memory:/user.slice/…" and lives in
            # its own hierarchy, mounted per controller (Gate B pass 4,
            # S — previously v1 hosts silently fell through to the
            # physical-RAM bound with a hardcoded /sys/fs/cgroup probe).
            hierarchy, controllers, relative = line.split(":", 2)
            if "memory" in controllers.split(","):
                mount = _cgroup_v1_controller_mount(
                    mountinfo, preferred=("memory",),
                )
                if mount:
                    raw_limit = (
                        Path(mount) / relative.lstrip("/")
                        / "memory.limit_in_bytes"
                    ).read_text(encoding="utf-8").strip()
                    if raw_limit.isdigit():
                        cgroup_limit = int(raw_limit)
                        if cgroup_limit >= _MIN_WORKER_MEMORY_MAX_BYTES:
                            candidates.append(cgroup_limit)
                break
    except (OSError, ValueError):
        pass

    try:
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
        physical_bound = min(
            _WORKER_MEMORY_MAX_CAP_BYTES,
            max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2),
        )
        candidates.append(physical_bound)
    except (OSError, ValueError, TypeError):
        pass

    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_scope_argv(
    binary: str, unit_name: str, *command: str,
    collect: bool = True, description: Optional[str] = None,
    memory_max_bytes: Optional[int] = None,
    memory_swap_max_bytes: Optional[int] = None,
) -> List[str]:
    """One scope argv contract for probes and launches on old/new systemd.

    Transient scopes reject OOMPolicy on systemd <253. Positive, caller-resolved
    memory bounds are optional: zero must never silently mean unlimited memory.
    """
    argv = [binary, "--user", "--scope", "--quiet", "--unit", unit_name]
    if collect:
        argv.append("--collect")
    if description:
        argv += ["--description", description]
    argv += ["--property", "MemoryAccounting=yes"]
    for name, bound in (("MemoryMax", memory_max_bytes), ("MemorySwapMax", memory_swap_max_bytes)):
        if bound is not None and bound > 0:
            argv += ["--property", f"{name}={bound}"]
    return [*argv, "--", *command]


def _default_user_runtime_dir() -> Path:
    """``/run/user/<uid>``; a function so tests can point it at a temp dir with a real socket."""
    return Path(f"/run/user/{os.getuid()}")  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env


def _secure_user_runtime_dir(path: Path) -> bool:
    """Accept only an absolute, owned, non-writable real directory."""
    try:
        metadata = path.lstat()
        return (
            path.is_absolute()
            and stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env
            and metadata.st_mode & 0o022 == 0
        )
    except OSError:
        return False





def _systemd_scope_cached() -> Optional[bool]:
    """Cached probe verdict, or None when a (re)probe is due. True is permanent; False
    expires after ``_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS`` so a D-Bus blip isn't sticky."""
    if _SYSTEMD_SCOPE_AVAILABLE is True:
        return True
    stale = time.monotonic() - _SYSTEMD_SCOPE_PROBED_AT >= _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    return None if _SYSTEMD_SCOPE_AVAILABLE is None or stale else False


def _systemd_run_user_scope_available() -> bool:
    """True if ``systemd-run --user --scope`` can create a cgroup.
    ``shutil.which`` alone is insufficient: system services and containers may lack
    the user D-Bus bus even with the binary on PATH (every spawn would fail with
    ``Failed to connect to user bus``), so a cheap probe is run and cached.

    Use ``/bin/sh -c 'exit 0'``: NixOS provides ``/bin/sh`` but not ``/bin/true``
    (#105365), regardless of the gateway service's PATH."""
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    verdict = _systemd_scope_cached()
    if verdict is not None:
        return verdict
    # Double-checked locking: a concurrent first-use spawn must not observe a temporary
    # False mid-probe, or it would launch back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        verdict = _systemd_scope_cached()
        if verdict is not None:
            return verdict
        available = False
        if _IS_LINUX:
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # Unique unit avoids collisions; the timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        _systemd_scope_argv(binary, probe_unit, "/bin/sh", "-c", "exit 0",
                                            memory_max_bytes=_worker_memory_max_bytes()),
                        capture_output=True,
                        timeout=3,
                        env=_registry.systemd_user_bus_env(),
                                 stdin=subprocess.DEVNULL,
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode, (result.stderr or b"").decode("utf-8", "replace").strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)
        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Whether this process is the live, supervised Hermes gateway itself.
    Supervisor markers and ``_HERMES_GATEWAY`` are inherited by every descendant (and
    importing ``gateway.run`` sets the latter), so also require ownership of the live
    gateway PID file — scopes are for the gateway, not terminal children or CLIs."""
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False
    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return is_gateway_supervisor_process() and get_running_pid(cleanup_stale=False) == os.getpid()
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        return False


def _build_systemd_scope_argv(
    shell_argv: List[str], unit_suffix: str, *,
    unit_name: Optional[str] = None, description: Optional[str] = None,
    memory_max_bytes: Optional[int] = None,
    memory_swap_max_bytes: Optional[int] = None, collect: bool = True,
) -> List[str]:
    """Build the shared wrapper; an unavailable binary preserves caller policy.

    Kanban passes collect=False so a failed unit remains inspectable: collecting
    it early would confuse a fast worker failure with a refused launch.
    """
    import shutil

    binary = shutil.which("systemd-run")
    if binary is None:
        return shell_argv
    return _systemd_scope_argv(
        binary, unit_name or f"hermes-worker-{unit_suffix}", *shell_argv,
        collect=collect, description=description,
        memory_max_bytes=memory_max_bytes, memory_swap_max_bytes=memory_swap_max_bytes,
    )


_scope_degraded_warned = False


def _warn_scope_degraded_once(detail: str) -> None:
    """Warn once per process: the condition is host-level and the probe verdict
    is cached, so this would otherwise fire on every cron dispatch."""
    global _scope_degraded_warned
    if _scope_degraded_warned:
        return
    _scope_degraded_warned = True
    logger.warning(
        "managed gateway: %s; cron children are dispatched as direct external subprocesses "
        "without restart-safe cgroup isolation (killed if the gateway restarts mid-job). "
        "Set cron.require_restart_safe_scope=true in config.yaml to fail closed instead.",
        detail,
    )








# Upper bound on how long a systemctl helper may sit between
# cancellation checks inside ``_run_systemctl_cancellable`` (pass 10,
# AM): a cancelled in-flight stop must return promptly, so a caller
# passing a pathological poll interval is clamped rather than trusted.
_HELPER_CANCEL_POLL_MAX_SECONDS = 0.5


def _reap_killed_systemctl_helper(proc) -> None:
    """Unconditionally reap a killed helper subprocess (pass 10, AM).

    ``communicate`` after ``kill`` can itself raise — the helper exits
    during the kill race, or inherited pipe handles keep its read loop
    from completing — and the old swallow-everything handler left the
    child unreaped: one zombie per cancelled or timed-out helper.
    ``wait`` is the backstop that always reaps (a no-op when
    ``communicate`` already did); if even the bounded wait expires, one
    last ``kill`` + ``wait`` closes it out. The pipes are closed
    explicitly because the reaping path that skipped a successful
    ``communicate`` leaves them open.
    """
    try:
        proc.wait(timeout=5.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            pass
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass


def _run_systemctl_cancellable(
    argv: List[str],
    *,
    timeout: float,
    cancel_event=None,
    deadline=None,
    poll_interval: float = 0.1,
):
    """Run a systemctl helper with mid-flight cancellation (pass 9, AH).

    ``subprocess.run(..., timeout=N)`` cannot observe a cancel event: it
    blocks for the full client timeout even when the caller — a gateway
    shutdown whose budget is gone, a dispatcher lock already released —
    has moved on, so the old process kept signalling (stop, SIGKILL)
    after its output could no longer matter. This polls the helper and
    KILLS it the moment either cancel signal fires; the poll interval is
    clamped to ``_HELPER_CANCEL_POLL_MAX_SECONDS`` so cancellation is
    observed promptly regardless of what the caller asked for (pass 10,
    AM).

    Returns ``(returncode, stdout, stderr)`` on completion (the shape
    callers used to read off ``subprocess.run``), or ``None`` when
    cancelled — callers treat ``None`` exactly like an unconfirmed
    outcome. On the plain client timeout it mirrors ``subprocess.run``:
    kill, reap, raise ``TimeoutExpired``. Both kill paths reap the
    helper unconditionally — a killed helper must never linger as a
    zombie (pass 10, AM).
    """

    def _cancelled() -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return True
        return deadline is not None and time.monotonic() >= deadline

    if _cancelled():
        return None
    poll_interval = max(0.01, min(poll_interval, _HELPER_CANCEL_POLL_MAX_SECONDS))
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, env=_registry.systemd_user_bus_env(),
    )
    wall = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            out, err = proc.communicate(timeout=poll_interval)
            return (proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            # Repeated communicate-after-timeout is the documented
            # pattern (subprocess.run is built on it); no output is lost.
            pass
        if _cancelled():
            logger.info(
                "systemctl helper for %s cancelled mid-flight — killing "
                "the helper; the unit keeps whatever state it is in for "
                "re-adoption", argv[-1],
            )
            try:
                proc.kill()
                try:
                    proc.communicate(timeout=5.0)
                except Exception:
                    pass
            finally:
                _reap_killed_systemctl_helper(proc)
            return None
        if time.monotonic() >= wall:
            try:
                proc.kill()
                try:
                    proc.communicate(timeout=5.0)
                except Exception:
                    pass
            finally:
                _reap_killed_systemctl_helper(proc)
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)


def _stop_systemd_unit(unit_name: str, *, cancel_event=None, deadline=None) -> bool:
    """Stop a transient systemd user scope by unit name.
    Reaps the *entire* cgroup — catching double-forked descendants reparented to init
    inside the scope that survive a plain PID signal (SIGTERM all, SIGKILL after
    ``TimeoutStopSec``). True if stopped or already gone; False if ``systemctl`` is
    unavailable or the stop failed.

    This reaps the *entire* cgroup — catching double-forked descendants that
    survive a plain PID signal because they were reparented to init inside the
    scope (issue #70716, reviewer gap #2).  ``systemctl --user stop`` sends
    SIGTERM to every process in the unit's cgroup and escalates to SIGKILL
    after the unit's ``TimeoutStopSec``.

    ``cancel_event``/``deadline`` (pass 9, AH) make the client call itself
    cancellable: the helper subprocess is killed the moment either fires
    instead of blocking for its full 15 s client timeout, and the unit is
    left in whatever state it is in (a cancelled stop is reported like an
    unconfirmed one — False).

    Returns True if the unit was successfully stopped (or was already gone),
    False if ``systemctl`` is unavailable, the stop command failed, or the
    call was cancelled.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = _run_systemctl_cancellable(
            [binary, "--user", "stop", unit_name],
            timeout=15,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if result is None:
            logger.info(
                "systemctl --user stop %s cancelled mid-flight — treating "
                "it as still stopping", unit_name,
            )
            return False
        returncode, _stdout, stderr_bytes = result
        if returncode != 0:
            stderr = (stderr_bytes or b"").decode(errors="replace").strip()
            stderr_lower = stderr.lower()
            if any(
                marker in stderr_lower
                for marker in ("not loaded", "not found", "does not exist")
            ):
                return True
            logger.debug(
                "systemctl --user stop %s exited %d: %s",
                unit_name, returncode,
                stderr,
            )
            return False
        return True
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        return False


# ---------------------------------------------------------------------------
# Verified scope lifecycle helpers.
#
# ``systemd-run --scope`` units have no main process: a scope stays active
# while ANY process in its cgroup lives (dev servers, browsers, databases a
# worker double-forked), and ``--collect`` only unloads the unit AFTER it
# becomes empty.  "Is this worker really gone?" therefore cannot be answered
# from a PID — and not from ``ActiveState`` alone either: a ``systemctl
# show`` failure yields no answer at all, and ``deactivating`` means the
# stop job is still draining the cgroup.  The kernel truth is the unit's
# ``cgroup.procs`` file: dead means the unit is not loaded AND that file is
# absent or empty.  Everything else ("unknown", "deactivating", a failed
# query) is NOT dead — callers keep their claim and retry.
# ---------------------------------------------------------------------------

_SCOPE_ACTIVE_STATES = frozenset({"active", "activating", "reloading"})


def _collect_dead_systemd_unit(unit_name: str) -> None:
    """Best-effort unload of a transient unit that is verified empty.

    ``systemctl --user reset-failed`` clears the failed state, which is
    what keeps a dead transient scope loaded on the bus.  Scopes spawned
    WITHOUT ``--collect`` (the kanban dispatcher's, so a fast nonzero
    worker stays inspectable — see ``_build_systemd_scope_argv``) rely on
    this call to be unloaded once their run is terminal; scopes that
    already unloaded are a no-op.  Never raises and never returns
    anything: this is garbage collection, not verification — liveness is
    answered by ``_scope_unit_liveness`` before this is called."""
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return
    try:
        subprocess.run(
            [binary, "--user", "reset-failed", unit_name],
            capture_output=True,
            timeout=10,
            env=_registry.systemd_user_bus_env(), stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("systemctl --user reset-failed %s failed: %s", unit_name, exc)

# How long ``_stop_systemd_unit_verified`` waits after its SIGKILL
# escalation before giving up (liveness re-probes every 0.2 s).  A module
# constant rather than a hardcoded literal so tests can shrink it.
_SCOPE_STOP_VERIFY_TIMEOUT = 6.0

# Worst-case wall clock of one ``_stop_systemd_unit_verified`` call: stop
# (15 s) + SIGKILL escalation (10 s) + verify loop (6 s).  Callers that
# must bound their own waits (dispatcher ticks, shutdown joins) use this
# as the per-unit deadline instead of inventing their own.
SCOPE_STOP_VERIFY_BOUND_SECONDS = 31.0


_CGROUP_V2_MOUNT_FALLBACK = "/sys/fs/cgroup"

# cgroup v1 mounts one hierarchy per controller set and lists the
# controllers in the mount's superoptions (``cpu,net_cls`` or the named
# hierarchy ``name=systemd``).  Whitelisted so unknown mount flags
# (relatime, release_agent=…, xattr) can never masquerade as one.
_CGROUP_V1_CONTROLLERS = frozenset({
    "blkio", "cpu", "cpuacct", "cpuset", "devices", "freezer",
    "hugetlb", "memory", "net_cls", "net_prio", "perf_event", "pids",
})


def _parse_cgroup_mounts(
    mountinfo_path: str = "/proc/self/mountinfo",
) -> tuple[list[str], list[tuple[str, frozenset[str]]], bool]:
    """(v2_mounts, v1_mounts, readable) from ``/proc/self/mountinfo``.

    Each v1 entry is ``(mount_point, controllers)`` where controllers
    comes from the superoptions, with ``name=systemd`` normalised to
    ``systemd`` — the hierarchy ``systemctl show -p ControlGroup`` paths
    are relative to.  ``readable`` is False when mountinfo cannot be
    opened at all (non-Linux); callers decide their own fallback for
    that case.
    """
    v2: list[str] = []
    v1: list[tuple[str, frozenset[str]]] = []
    try:
        with open(mountinfo_path, "r", encoding="utf-8") as f:
            for line in f:
                # ... mount-ID parent-ID major:minor root MOUNT-POINT
                # options [optional...] "-" FSTYPE source superoptions
                # — fstype is the third field from the end, superoptions
                # the last.
                cols = line.split()
                if len(cols) <= 9:
                    continue
                fstype = cols[-3]
                if fstype == "cgroup2":
                    v2.append(cols[4])
                elif fstype == "cgroup":
                    controllers: set[str] = set()
                    for token in cols[-1].split(","):
                        if token in _CGROUP_V1_CONTROLLERS:
                            controllers.add(token)
                        elif token == "name=systemd":
                            controllers.add("systemd")
                    v1.append((cols[4], frozenset(controllers)))
    except OSError:
        return [], [], False
    return v2, v1, True


def _cgroup2_mount_point(
    mountinfo_path: str = "/proc/self/mountinfo",
) -> str:
    """Mount point of the unified (v2) cgroup hierarchy.

    Read from ``/proc/self/mountinfo`` rather than hardcoded
    ``/sys/fs/cgroup``: containers and alternative layouts mount the
    hierarchy elsewhere, and joining a systemd-reported ControlGroup
    onto the wrong prefix yields a path that cannot exist — which must
    surface as "unknown", not as a false verified death.  Falls back to
    the canonical mount when mountinfo is unreadable (non-Linux) or has
    no cgroup2 entry.
    """
    v2, _v1, _readable = _parse_cgroup_mounts(mountinfo_path)
    if not v2:
        return _CGROUP_V2_MOUNT_FALLBACK
    # Prefer the canonical mount when present; otherwise the first
    # cgroup2 mount wins (a container usually exposes exactly one).
    for mount in v2:
        if mount == _CGROUP_V2_MOUNT_FALLBACK:
            return mount
    return v2[0]


def _cgroup_v1_controller_mount(
    mountinfo_path: str = "/proc/self/mountinfo",
    preferred: tuple[str, ...] = ("systemd", "pids"),
) -> Optional[str]:
    """Mount point of the cgroup v1 hierarchy carrying *preferred*
    controllers, or None when the host has no cgroup v1 mounts.

    v1 mounts one hierarchy per controller set; ``preferred`` is walked
    in order because the caller cares which hierarchy it lands on —
    ``systemd`` for ControlGroup-relative paths, ``memory`` for the
    memory limit file.  ``cgroup.procs`` lists the same processes in
    every hierarchy, so the first v1 mount is an acceptable last
    resort for procs lookups.
    """
    _v2, v1, _readable = _parse_cgroup_mounts(mountinfo_path)
    for want in preferred:
        for mount, controllers in v1:
            if want in controllers:
                return mount
    if v1:
        return v1[0][0]
    return None


def _cgroup_mount_point(
    mountinfo_path: str = "/proc/self/mountinfo",
) -> tuple[int, Optional[str]]:
    """(version, mount point) of the hierarchy backing cgroup.procs.

    ``2`` — unified hierarchy, canonical mount preferred exactly as
    :func:`_cgroup2_mount_point` chooses it.  ``1`` — cgroup v1: the
    best controller mount for process truth (systemd hierarchy first,
    then pids, then any v1 mount).  ``(0, None)`` is DEFINITE: mountinfo
    is readable and exposes no cgroup hierarchy at all — a host where
    cgroup-based verification is impossible by construction, not merely
    failing right now.  Callers surface that as ``"unsupported"`` and
    fall back to PID semantics instead of retrying ``"unknown"``
    forever (Gate B pass 4, S).  An UNREADABLE mountinfo keeps the
    legacy assumption (canonical v2 mount), so that case stays
    ``"unknown"`` rather than inventing a definite verdict.
    """
    v2, v1, readable = _parse_cgroup_mounts(mountinfo_path)
    if v2:
        return 2, _cgroup2_mount_point(mountinfo_path)
    v1_mount = _cgroup_v1_controller_mount(mountinfo_path)
    if v1_mount is not None:
        return 1, v1_mount
    if not readable:
        return 2, _CGROUP_V2_MOUNT_FALLBACK
    return 0, None


def _scope_cgroup_procs_path(
    unit_name: str, control_group: str = ""
) -> Optional[str]:
    """Filesystem path of the unit's ``cgroup.procs`` file.

    ``control_group`` is the ``systemctl show -p ControlGroup`` value
    (cgroupfs-relative, e.g. ``/user.slice/user-1000.slice/...``),
    joined onto the REAL cgroup mount — v2's unified hierarchy or, on a
    v1 host, the systemd controller hierarchy (see
    :func:`_cgroup_mount_point`; the slice layout of the reported path
    is the same on both).  An already-absolute path outside the
    canonical cgroup roots (a real cgroupfs mount spelled out by
    systemd, or a test shim's state dir) is honoured as-is.  When
    empty, fall back to the canonical user-scope location so a unit
    systemd forgot to report on is still verified.  ``None`` when no
    path can be formed (non-Linux without a reported ControlGroup, or a
    host with no readable cgroup hierarchy at all) — callers treat
    "no hierarchy" as a definite ``"unsupported"``, never "dead".
    """
    control_group = (control_group or "").strip()
    if control_group:
        if control_group.startswith((
            "/user.slice", "/machine.slice", "/system.slice", "/init.scope",
            "/user-0.slice",
        )):
            # cgroupfs-relative path as systemd reports it.  On a v1
            # host the same path is relative to the systemd controller
            # hierarchy's mount, not the v1 root.
            _version, mount = _cgroup_mount_point()
            if mount is None:
                return None
            return mount + control_group.rstrip("/") + "/cgroup.procs"
        return control_group.rstrip("/") + "/cgroup.procs"
    if platform.system() != "Linux" or not hasattr(os, "getuid"):
        return None
    _version, mount = _cgroup_mount_point()
    if mount is None:
        return None
    uid = os.getuid()
    return (
        f"{mount}/user.slice/user-{uid}.slice/"
        f"user@{uid}.service/app.slice/{unit_name}/cgroup.procs"
    )


def _scope_unit_liveness(unit_name: str) -> str:
    """Return alive/dead/unknown/unsupported for a manager-owned scope.

    Recursive kernel population includes nested descendants even if the
    manager forgot the unit. Deactivation, malformed state, unreadable or
    absent membership, and failed manager queries are unknown. A loaded
    unit on a host without a cgroup hierarchy is unsupported.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return "unknown"
    try:
        result = subprocess.run(
            [
                binary, "--user", "show", unit_name,
                "--property", "LoadState",
                "--property", "ActiveState",
                "--property", "ControlGroup",
            ],
            capture_output=True,
            timeout=5,
                     env=_registry.systemd_user_bus_env(), stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("systemctl --user show %s failed: %s", unit_name, exc)
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    props: Dict[str, str] = {}
    for line in (result.stdout or b"").decode(errors="replace").splitlines():
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in props:
            props[key] = value.strip()
    load_state = props.get("LoadState", "").lower()
    active_state = props.get("ActiveState", "").lower()
    if load_state == "not-found":
        # Manager absence does not prove that a persisted scope's processes
        # stopped: the manager may have restarted while descendants survived.
        # Consult the deterministic kernel path independently. A missing path
        # is also unknown without persisted manager/boot provenance. Fresh
        # launch refusal is a separate authority in _scope_unit_was_created.
        procs_path = _scope_cgroup_procs_path(unit_name)
        if procs_path is None:
            return "unknown"
        return _cgroup_tree_liveness(Path(procs_path).parent)
    if load_state != "loaded" or not active_state:
        # masked/error/garbled — we cannot verify this unit.
        return "unknown"
    if active_state == "deactivating":
        # A stop job is mid-flight: the cgroup is being drained but the
        # unit is not gone.  NOT dead.
        return "unknown"
    procs_path = _scope_cgroup_procs_path(
        unit_name, props.get("ControlGroup", "")
    )
    if procs_path is None:
        # No path could be formed.  When the host has no cgroup
        # hierarchy AT ALL that is a definite verdict, not a transient
        # failure — surface it so callers stop retrying and fall back
        # to PID semantics.
        _version, mount = _cgroup_mount_point()
        if mount is None:
            return "unsupported"
        return "unknown"
    return _cgroup_tree_liveness(Path(procs_path).parent)


def _cgroup_tree_liveness(root: Path) -> str:
    """Read recursive kernel population; legacy cgroup v1 needs a bounded walk.

    An empty root cgroup.procs excludes neither nested cgroups nor surviving
    grandchildren. Invalid/inaccessible evidence always remains unknown.
    """
    try:
        events = (root / "cgroup.events").read_text(encoding="utf-8")
    except FileNotFoundError:
        events = None  # cgroup v1 has no recursive populated indicator.
    except (OSError, UnicodeError):
        return "unknown"
    if events is not None:
        populated = [line.split() for line in events.splitlines() if line.split()[:1] == ["populated"]]
        if len(populated) != 1 or len(populated[0]) != 2:
            return "unknown"
        return {"0": "dead", "1": "alive"}.get(populated[0][1], "unknown")

    pending = [root]
    scanned = 0
    try:
        while pending:
            current = pending.pop()
            scanned += 1
            if scanned > 1024:
                return "unknown"
            if current.is_symlink():
                return "unknown"
            with (current / "cgroup.procs").open(encoding="utf-8") as membership:
                if membership.read().strip():
                    return "alive"
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        return "unknown"
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
    except (OSError, UnicodeError):
        return "unknown"
    return "dead"


def _scope_unit_active_state(unit_name: str) -> str:
    """Map recursive scope liveness into the Kanban lifecycle vocabulary.

    Unknown and unsupported remain distinct; neither is proof of death.
    """
    liveness = _scope_unit_liveness(unit_name)
    return {
        "alive": "active",
        "dead": "dead",
        "unsupported": "unsupported",
    }.get(liveness, "unknown")


def _scope_unit_was_created(unit_name: str) -> bool:
    """True when the transient unit exists or existed on the bus.

    ``systemctl show -p LoadState``: ``not-found`` means systemd never
    created the unit (a refused ``systemd-run`` launch); ``loaded``
    means it exists.  Kanban scopes are spawned WITHOUT ``--collect``,
    so a fast NONZERO worker exit leaves its failed unit loaded (failed
    transient units are kept for inspection unless collected) and this
    stays True — exactly the signal the spawn probe needs to tell
    "launch refused" from "worker ran and exited"; the unit is unloaded
    explicitly (``_collect_dead_systemd_unit``) once the run is
    terminal.  A failed query also returns True: when we cannot tell,
    assume the launch happened (the cost of wrongly assuming a refused
    launch is a duplicate plain-spawn beside a live scope).
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return True
    try:
        result = subprocess.run(
            [binary, "--user", "show", unit_name, "--property", "LoadState"],
            capture_output=True,
            timeout=5,
                     env=_registry.systemd_user_bus_env(), stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("systemctl --user show %s failed: %s", unit_name, exc)
        return True
    if result.returncode != 0:
        return True
    for line in (result.stdout or b"").decode(errors="replace").splitlines():
        if line.startswith("LoadState="):
            return line.partition("=")[2].strip().lower() != "not-found"
    return True


def _scope_unit_bus_inactive(unit_name: str) -> bool:
    """True when the BUS itself reports the unit not running.

    Fallback confirmation for hosts where no cgroup hierarchy is
    readable (liveness ``"unsupported"``): after we fired a stop or a
    SIGKILL ourselves, systemd moving the unit out of its active states
    is the strongest verification that host can give — the cgroup
    cannot be read to say more.  Conservative: only not-found, inactive
    and failed count; ``deactivating`` and anything unreadable do not.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [
                binary, "--user", "show", unit_name,
                "--property", "LoadState",
                "--property", "ActiveState",
            ],
            capture_output=True,
            timeout=5,
                     env=_registry.systemd_user_bus_env(), stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("systemctl --user show %s failed: %s", unit_name, exc)
        return False
    if result.returncode != 0:
        return False
    props: Dict[str, str] = {}
    for line in (result.stdout or b"").decode(errors="replace").splitlines():
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in props:
            props[key] = value.strip().lower()
    load_state = props.get("LoadState", "")
    if load_state == "not-found":
        return True
    return load_state == "loaded" and props.get("ActiveState", "") in (
        "inactive", "failed",
    )


def _stop_systemd_unit_verified(
    unit_name: str,
    *,
    verify_timeout: Optional[float] = None,
    cancel_event: Optional["threading.Event"] = None,
    deadline: Optional[float] = None,
) -> bool:
    """Stop a user unit AND verify its cgroup is actually empty.

    Unlike :func:`_stop_systemd_unit` (fire-and-forget, used where the
    caller cannot act on the difference), this escalates to
    ``systemctl kill --signal=SIGKILL`` when the unit is not verified
    dead after the stop — including a ``systemctl stop`` client timeout,
    which does NOT mean the unit died (the stop job may still be running
    server-side) — and returns True only once liveness is verified dead.
    On a cgroup-less host (liveness ``"unsupported"``) verification
    falls back to the bus reporting the unit inactive/failed, so a stop
    still terminates instead of burning the escalation budget forever.
    Callers must treat a False return as "still stopping": retry next
    tick, never assume success and release bookkeeping that guards
    against a duplicate spawn.

    Worst case one call takes about ``SCOPE_STOP_VERIFY_BOUND_SECONDS``
    (stop + SIGKILL + verify loop); anything that cannot afford that
    must hand the unit to a background stopper instead of calling this
    synchronously (the kanban dispatcher's scope-stop service does).

    ``cancel_event``/``deadline`` (pass 8, Y) make that worst case
    cancellable mid-flight: an in-progress stop checks them after the
    TERM wait, before the SIGKILL wait, and before each final-verify
    probe, returning False ("still stopping") the moment either fires —
    so a shutdown whose budget expired stops paying for escalation a
    caller that already moved on will not read.
    """
    import shutil

    def _cancelled() -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return True
        return deadline is not None and time.monotonic() >= deadline

    def confirmed_dead(unit: str) -> bool:
        state = _scope_unit_liveness(unit)
        if state == "dead":
            return True
        if state == "unsupported":
            return _scope_unit_bus_inactive(unit)
        return False

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    # Fire the stop; its return value is only a hint.  A non-zero exit
    # with "not loaded" means already gone, and liveness below confirms
    # that; anything else (including a client timeout) leaves the real
    # verdict to the liveness checks that follow.  The cancel plumbing
    # reaches INTO the client call (pass 9, AH): a cancelled caller's
    # systemctl helper is killed instead of blocking out its timeout.
    _stop_systemd_unit(
        unit_name, cancel_event=cancel_event, deadline=deadline,
    )
    if _cancelled():
        logger.info(
            "stop of systemd unit %s cancelled after the TERM wait; "
            "treating it as still stopping", unit_name,
        )
        return False
    if confirmed_dead(unit_name):
        _collect_dead_systemd_unit(unit_name)
        return True
    # Not verified dead (slow descendants, a wedged stop job draining
    # the cgroup, or an unreachable bus).  Escalate: SIGKILL every
    # process in the unit's cgroup, then re-verify until the deadline.
    if _cancelled():
        logger.info(
            "stop of systemd unit %s cancelled before the SIGKILL "
            "escalation; treating it as still stopping", unit_name,
        )
        return False
    try:
        kill_result = _run_systemctl_cancellable(
            [binary, "--user", "kill", "--signal=SIGKILL", unit_name],
            timeout=10,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        if kill_result is None:
            logger.info(
                "stop of systemd unit %s cancelled during the SIGKILL "
                "escalation — killing the helper and treating the unit "
                "as still stopping", unit_name,
            )
            return False
    except Exception as exc:
        logger.debug("systemctl --user kill %s failed: %s", unit_name, exc)
        return False
    timeout = _SCOPE_STOP_VERIFY_TIMEOUT if verify_timeout is None else verify_timeout
    deadline_local = time.monotonic() + timeout
    while time.monotonic() < deadline_local:
        if _cancelled():
            logger.info(
                "stop of systemd unit %s cancelled before the final "
                "verify; treating it as still stopping", unit_name,
            )
            return False
        if confirmed_dead(unit_name):
            _collect_dead_systemd_unit(unit_name)
            return True
        time.sleep(0.2)
    logger.warning(
        "systemd unit %s still active after SIGTERM+SIGKILL escalation",
        unit_name,
    )
    return False


def _list_systemd_scope_units(pattern: str) -> Dict[str, str]:
    """List user scope units matching *pattern* with their active state.

    Returns ``{unit_name: state}`` where state is ``"active"`` or the raw
    systemd sub/active state string (only ``"active"`` rows matter to
    callers).  Empty dict when systemctl is unavailable or fails — a
    listing problem must never look like "no orphaned units".
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return {}
    try:
        result = subprocess.run(
            [
                binary, "--user", "list-units", "--type=scope", "--all",
                "--no-legend", "--plain", pattern,
            ],
            capture_output=True,
            timeout=10,
                     env=_registry.systemd_user_bus_env(), stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("systemctl --user list-units %s failed: %s", pattern, exc)
        return {}
    if result.returncode != 0:
        return {}
    units: Dict[str, str] = {}
    for line in (result.stdout or b"").decode(errors="replace").splitlines():
        cols = line.split()
        # <unit> <load> <active> <sub> [<description>...]
        if len(cols) >= 3 and cols[0].endswith(".scope"):
            units[cols[0]] = cols[2].lower()
    return units


from tools import process_registry as _registry
