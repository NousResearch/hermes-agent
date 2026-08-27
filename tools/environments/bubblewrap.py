"""Bubblewrap (bwrap) terminal backend: the pure argv builder.

Every terminal command under ``terminal.backend: bubblewrap`` runs inside a
bwrap sandbox. This module builds the bwrap argv prefix from configuration
and construction-time inputs only. Nothing produced inside a sandbox (the
tracked cwd, the shell snapshot, command output) feeds a mount or a flag
argument; the tracked cwd is used for ``--chdir`` alone.

Layout of the argv (later mounts overlay earlier ones):

1. namespace and process-safety flags
2. read-only root, fresh /dev, /proc and a tmpfs /tmp
3. the initial cwd, read-write for workspace and network, read-only for
   restricted
4. operator binds from terminal.bubblewrap_binds, minus sensitive sources
5. the sensitive overlays: a tmpfs over each sensitive directory and an
   empty file over each sensitive file that exists on the host, then the
   same for HERMES_HOME
6. under terminal.home_mode=profile, HERMES_HOME/home read-write on top of
   that overlay (it is the subprocess HOME then)
7. the per-environment state dir read-write at the same path
8. ``--chdir`` to the tracked cwd, then ``--`` so the caller can append the
   shell argv
"""

from __future__ import annotations

import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping

from hermes_constants import get_hermes_home
from tools.environments.base import EnvironmentConnectionError, get_sandbox_dir
from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd

logger = logging.getLogger(__name__)

# Paths under HOME whose contents must never be visible inside a sandbox
# and that operator binds may not expose.
SENSITIVE_HOME_PATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".gpg",
    ".config/gcloud",
    ".azure",
    ".docker",
    ".kube",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".env",
)

DEFAULT_PROFILE = "network"
DEFAULT_HOME_MODE = "auto"
DEFAULT_MEMORY_MB = 256
DEFAULT_CPU_SECONDS = 30
DEFAULT_MAX_PROCS = 256

ENV_PROFILE = "TERMINAL_BUBBLEWRAP_PROFILE"
ENV_BINDS = "TERMINAL_BUBBLEWRAP_BINDS"
ENV_MEMORY_MB = "TERMINAL_BUBBLEWRAP_MEMORY_MB"
ENV_CPU_SECONDS = "TERMINAL_BUBBLEWRAP_CPU_SECONDS"
ENV_MAX_PROCS = "TERMINAL_BUBBLEWRAP_MAX_PROCS"
# terminal.home_mode, bridged like the keys above. The spellings that
# hermes_constants.get_subprocess_home treats as "profile".
ENV_HOME_MODE = "TERMINAL_HOME_MODE"
PROFILE_HOME_MODES: frozenset[str] = frozenset({"profile", "isolated", "profile_home", "profile-home"})


@dataclass(frozen=True)
class Profile:
    """What a named profile allows: a writable cwd and/or host networking."""

    name: str
    writable_cwd: bool
    share_net: bool


PROFILES: dict[str, Profile] = {
    "restricted": Profile("restricted", writable_cwd=False, share_net=False),
    "workspace": Profile("workspace", writable_cwd=True, share_net=False),
    "network": Profile("network", writable_cwd=True, share_net=True),
}
PROFILE_NAMES: tuple[str, ...] = tuple(PROFILES)


def resolve_profile(name: str) -> Profile:
    """Return the profile for *name* or raise ValueError listing the valid names."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown terminal.bubblewrap_profile {name!r}. "
            f"Valid profiles: {', '.join(PROFILE_NAMES)}"
        ) from None


@dataclass(frozen=True)
class BindMount:
    """One operator-supplied bind: host *src* mounted at *dest* in the sandbox."""

    src: str
    dest: str
    readonly: bool = True


@dataclass(frozen=True)
class BubblewrapConfig:
    """The terminal.bubblewrap_* settings, with the documented defaults."""

    profile: str = DEFAULT_PROFILE
    binds: tuple[BindMount, ...] = ()
    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_seconds: int = DEFAULT_CPU_SECONDS
    max_procs: int = DEFAULT_MAX_PROCS
    # terminal.home_mode rides along because it decides whether HERMES_HOME/home
    # is the subprocess HOME and so must be bound back over the overlay.
    home_mode: str = DEFAULT_HOME_MODE


def _parse_binds(raw: str) -> tuple[BindMount, ...]:
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ENV_BINDS} must be a JSON list of {{src, dest, readonly}} objects: {exc}") from None
    if not isinstance(entries, list):
        raise ValueError(f"{ENV_BINDS} must be a JSON list, got {type(entries).__name__}")
    binds: list[BindMount] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("src"), str) or not entry["src"]:
            raise ValueError(f"{ENV_BINDS} entries need a non-empty 'src' string, got {entry!r}")
        src = entry["src"]
        dest = entry.get("dest") or src
        if not isinstance(dest, str):
            raise ValueError(f"{ENV_BINDS} 'dest' must be a string, got {dest!r}")
        binds.append(BindMount(src=src, dest=dest, readonly=bool(entry.get("readonly", True))))
    return tuple(binds)


def _parse_limit(name: str, raw: str, default: int) -> int:
    value = raw.strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer (0 disables the limit), got {raw!r}") from None
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer (0 disables the limit), got {raw!r}")
    return parsed


def load_bubblewrap_config(environ: Mapping[str, str] | None = None) -> BubblewrapConfig:
    """Read the terminal.bubblewrap_* settings from their TERMINAL_BUBBLEWRAP_* env names.

    Blank or missing values take the documented defaults. Malformed values
    raise ValueError naming the offending variable. The profile name is not
    validated here; :func:`resolve_profile` rejects unknown names at
    environment construction.
    """
    env = os.environ if environ is None else environ
    profile = env.get(ENV_PROFILE, "").strip().lower() or DEFAULT_PROFILE
    raw_binds = env.get(ENV_BINDS, "").strip()
    binds = _parse_binds(raw_binds) if raw_binds else ()
    return BubblewrapConfig(
        profile=profile,
        binds=binds,
        memory_mb=_parse_limit(ENV_MEMORY_MB, env.get(ENV_MEMORY_MB, ""), DEFAULT_MEMORY_MB),
        cpu_seconds=_parse_limit(ENV_CPU_SECONDS, env.get(ENV_CPU_SECONDS, ""), DEFAULT_CPU_SECONDS),
        max_procs=_parse_limit(ENV_MAX_PROCS, env.get(ENV_MAX_PROCS, ""), DEFAULT_MAX_PROCS),
        home_mode=env.get(ENV_HOME_MODE, "").strip().lower() or DEFAULT_HOME_MODE,
    )


def sensitive_paths(home: str, hermes_home: str) -> tuple[str, ...]:
    """Absolute host paths that must stay hidden: the HOME set plus HERMES_HOME."""
    home = os.path.abspath(os.path.expanduser(home))
    return tuple(os.path.join(home, rel) for rel in SENSITIVE_HOME_PATHS) + (
        os.path.abspath(os.path.expanduser(hermes_home)),
    )


def empty_file_path(state_dir: str) -> str:
    """Host path of the zero-length file bound over sensitive files.

    It sits beside the state dir, not inside it: the state dir is bound
    read-write into every spawn, so a file kept there could be rewritten
    from inside the sandbox and would then show at the hidden paths.
    """
    return state_dir.rstrip(os.sep) + ".empty"


def sensitive_overlay_args(home: str, hermes_home: str, state_dir: str) -> list[str]:
    """Mount directives that hide the sensitive set and HERMES_HOME.

    A directory gets a fresh tmpfs, a file gets the empty file bound over
    it, and a path missing on the host gets nothing so bwrap never fails
    on an absent mount target. On bwrap 0.9.0, ``--tmpfs`` on a
    file path fails with "Not a directory", and a ro-bind of /dev/null
    mounts but reads fail with EACCES because bwrap remounts binds nodev
    inside the user namespace; only the empty-file bind works for files.
    """
    empty = empty_file_path(state_dir)
    argv: list[str] = []
    for path in sensitive_paths(home, hermes_home):
        if os.path.isdir(path):
            argv += ["--tmpfs", path]
        elif os.path.exists(path):
            argv += ["--ro-bind", empty, path]
    return argv


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def is_sensitive_source(src: str, home: str, hermes_home: str) -> bool:
    """True when *src* (or what it symlinks to) is at or under a sensitive path."""
    abs_src = os.path.abspath(os.path.expanduser(src))
    candidates = {abs_src, os.path.realpath(abs_src)}
    for root in sensitive_paths(home, hermes_home):
        roots = {root, os.path.realpath(root)}
        if any(_is_within(c, r) for c in candidates for r in roots):
            return True
    return False


def filter_binds(binds: tuple[BindMount, ...], home: str, hermes_home: str) -> list[BindMount]:
    """Drop binds whose source is sensitive, logging a warning for each."""
    kept: list[BindMount] = []
    for bind in binds:
        if is_sensitive_source(bind.src, home, hermes_home):
            logger.warning(
                "Ignoring terminal.bubblewrap_binds entry %s: source is under a sensitive path",
                bind.src,
            )
            continue
        kept.append(bind)
    return kept


def build_bwrap_args(
    config: BubblewrapConfig,
    initial_cwd: str,
    state_dir: str,
    home: str,
    hermes_home: str,
    tracked_cwd: str,
    *,
    bwrap_path: str = "bwrap",
) -> list[str]:
    """Build the bwrap argv prefix; the caller appends the shell argv after the trailing ``--``.

    All arguments are fixed at environment construction except *tracked_cwd*,
    which only sets ``--chdir``.
    """
    profile = resolve_profile(config.profile)

    argv: list[str] = [
        bwrap_path,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--unshare-cgroup-try",
    ]
    if profile.share_net:
        argv.append("--share-net")

    argv += [
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]
    # The cwd is always bound at its own path so --chdir resolves even when
    # it sits under the masked /tmp; the profile decides whether it is
    # writable.
    argv += ["--bind" if profile.writable_cwd else "--ro-bind", initial_cwd, initial_cwd]

    for bind in filter_binds(config.binds, home, hermes_home):
        argv += ["--ro-bind" if bind.readonly else "--bind", bind.src, bind.dest]

    # The overlays come after the cwd and operator binds so a bind of HOME
    # itself still hides what sits under it, and before the state dir so
    # that stays reachable under a hidden HERMES_HOME.
    argv += sensitive_overlay_args(home, hermes_home, state_dir)

    # Under home_mode=profile the subprocess HOME is HERMES_HOME/home
    # (hermes_constants.get_subprocess_home), so bind it back read-write on
    # top of the overlay; the rest of HERMES_HOME stays hidden.
    if config.home_mode in PROFILE_HOME_MODES:
        profile_home = os.path.join(os.path.abspath(os.path.expanduser(hermes_home)), "home")
        if os.path.isdir(profile_home):
            argv += ["--bind", profile_home, profile_home]

    # The state dir holds the shell snapshot and cwd file; it is bound after
    # the sensitive overlays so it stays writable at the same path in every
    # spawn.
    argv += ["--bind", state_dir, state_dir]

    argv += ["--chdir", tracked_cwd, "--"]
    return argv


PROBE_ARGS: tuple[str, ...] = ("--unshare-user", "--ro-bind", "/", "/", "true")
PROBE_TIMEOUT_SECONDS = 5
INSTALL_HINT = (
    "Install the bubblewrap package (apt, dnf or pacman: bubblewrap) and make "
    "sure unprivileged user namespaces are allowed on this host, then retry; "
    "or set terminal.backend to another backend."
)

# Path of the bwrap that passed the runtime probe, kept for the life of the
# process. A failed probe is not cached: the next construction probes again,
# so installing or fixing bwrap needs no restart.
_probed_bwrap_path: str | None = None


def run_probe() -> tuple[str | None, str | None]:
    """Run the bwrap probe once: ``(path, None)`` on success, ``(path, failure)`` otherwise.

    The probe is ``bwrap --unshare-user --ro-bind / / true`` with a
    5 s timeout: it fails where user namespaces are disabled, where bwrap
    is not setuid on a kernel that needs it, or where bwrap is missing.
    """
    path = shutil.which("bwrap")
    if path is None:
        return None, "bubblewrap (bwrap) is not on PATH"
    try:
        result = subprocess.run(
            [path, *PROBE_ARGS],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return path, f"bwrap probe timed out after {PROBE_TIMEOUT_SECONDS} s: {path} {' '.join(PROBE_ARGS)}"
    except OSError as exc:
        return path, f"bwrap probe could not start: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return path, f"bwrap probe failed (exit {result.returncode}): {detail}"
    return path, None


def probe_bwrap() -> str:
    """Return the path of a bwrap that passed the probe, probing once per process.

    Raises EnvironmentConnectionError, with a retry hint naming the
    bubblewrap package, when bwrap is missing from PATH or the probe fails.
    """
    global _probed_bwrap_path
    if _probed_bwrap_path is None:
        path, failure = run_probe()
        if failure is not None:
            raise EnvironmentConnectionError(
                f"bubblewrap backend unavailable: {failure}",
                retry_hint=INSTALL_HINT,
            )
        _probed_bwrap_path = path
    return _probed_bwrap_path


def uid_thread_count(uid: int) -> int:
    """Threads owned by *uid* host-wide, counted from /proc.

    RLIMIT_NPROC counts threads, not processes (getrlimit(2)), and a desktop
    uid runs several threads per process.
    """
    count = 0
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        proc = os.path.join("/proc", name)
        try:
            if os.stat(proc).st_uid == uid:
                count += len(os.listdir(os.path.join(proc, "task")))
        except OSError:
            continue
    return count


def rlimit_values(config: BubblewrapConfig, *, uid_threads: int) -> dict[int, int]:
    """The rlimits a spawn gets from the three terminal.bubblewrap_* keys.

    A key at 0 leaves its limit out. RLIMIT_NPROC is counted per uid
    host-wide, and a bwrap user namespace does not change that (kernel
    6.8: with the limit below the uid's thread count bwrap cannot even
    create its namespace), so max_procs is applied on top of the uid's
    current thread count: it bounds what the sandbox may add, and the
    documented default of 256 keeps working on a desktop that already runs
    more.
    """
    limits: dict[int, int] = {}
    if config.memory_mb:
        limits[resource.RLIMIT_AS] = config.memory_mb * 1024 * 1024
    if config.cpu_seconds:
        limits[resource.RLIMIT_CPU] = config.cpu_seconds
    if config.max_procs:
        limits[resource.RLIMIT_NPROC] = uid_threads + config.max_procs
    return limits


def make_preexec(limits: Mapping[int, int]) -> Callable[[], None] | None:
    """A Popen preexec_fn applying *limits* as soft and hard, or None when empty.

    A value above the inherited hard limit is clamped to it: raising a
    hard limit needs CAP_SYS_RESOURCE and would fail the spawn.
    """
    if not limits:
        return None
    pairs = tuple(limits.items())

    def _apply_rlimits() -> None:
        for res, value in pairs:
            _, hard = resource.getrlimit(res)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(res, (value, value))

    return _apply_rlimits


class BubblewrapEnvironment(LocalEnvironment):
    """LocalEnvironment whose every spawn runs inside a bwrap sandbox.

    Bash resolution, the run env, missing-cwd recovery and process-group
    kill come from LocalEnvironment. This class adds the argv prefix, the
    rlimit preexec, a per-instance state dir for the shell snapshot and cwd
    file, the empty file bound over sensitive files, and their removal on
    cleanup.
    """

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 60,
        env: dict | None = None,
        *,
        config: BubblewrapConfig | None = None,
    ):
        self._config = load_bubblewrap_config() if config is None else config
        # Reject an unknown profile and an unusable bwrap before anything is
        # created on disk; the probe raises EnvironmentConnectionError, which
        # the terminal tool turns into its degraded or error result.
        resolve_profile(self._config.profile)
        self._bwrap_path = probe_bwrap()
        self._home = os.path.expanduser("~")
        self._hermes_home = str(get_hermes_home())
        # The mount set is fixed here; only --chdir follows the tracked cwd.
        self._initial_cwd = _resolve_local_initial_cwd(cwd)
        # BaseEnvironment.__init__ derives the snapshot and cwd file paths
        # from get_temp_dir() and LocalEnvironment.__init__ runs the login
        # bootstrap straight away, so the state dir must exist first.
        self._state_dir = str(get_sandbox_dir() / f"bwrap-{uuid.uuid4().hex[:12]}")
        os.makedirs(self._state_dir, mode=0o700)
        # Read-only and outside the state dir: nothing in a sandbox can
        # write to what shows at the hidden file paths.
        self._empty_file = empty_file_path(self._state_dir)
        os.close(os.open(self._empty_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400))
        super().__init__(cwd=self._initial_cwd, timeout=timeout, env=env)

    def get_temp_dir(self) -> str:
        return self._state_dir

    def _wrap_popen_args(self, args: list[str]) -> list[str]:
        prefix = build_bwrap_args(
            self._config,
            self._initial_cwd,
            self._state_dir,
            self._home,
            self._hermes_home,
            self.cwd,
            bwrap_path=self._bwrap_path,
        )
        return prefix + list(args)

    def _popen_preexec(self):
        uid_threads = uid_thread_count(os.getuid()) if self._config.max_procs else 0
        return make_preexec(rlimit_values(self._config, uid_threads=uid_threads))

    def _live_sandbox_pids(self) -> list[int]:
        """PIDs of this instance's bwrap wrappers still running.

        A wrapper is a direct child of this process whose argv is the bwrap
        path with this instance's state dir bound; the state dir is unique
        per instance and fixed at construction, so nothing from inside a
        sandbox can forge it. Zombies are left for the thread that spawned
        them to reap.
        """
        me = os.getpid()
        bwrap = self._bwrap_path.encode()
        state_dir = self._state_dir.encode()
        pids: list[int] = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as fh:
                    fields = fh.read().rsplit(b")", 1)[1].split()
                if fields[0] == b"Z" or int(fields[1]) != me:
                    continue
                with open(f"/proc/{name}/cmdline", "rb") as fh:
                    argv = fh.read().split(b"\0")
            except (OSError, IndexError, ValueError):
                continue
            if argv and argv[0] == bwrap and state_dir in argv:
                pids.append(int(name))
        return pids

    def _kill_live_sandboxes(self, wait: float = 2.0) -> None:
        """SIGKILL this instance's running bwrap wrappers.

        --die-with-parent takes the sandboxed tree down with each wrapper,
        background children included, since the wrapper's death kills the
        pid namespace's init.
        """
        for pid in self._live_sandbox_pids():
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
        deadline = time.monotonic() + wait
        while self._live_sandbox_pids() and time.monotonic() < deadline:
            time.sleep(0.05)

    def cleanup(self):
        self._kill_live_sandboxes()
        super().cleanup()
        shutil.rmtree(self._state_dir, ignore_errors=True)
        try:
            os.unlink(self._empty_file)
        except OSError:
            pass
