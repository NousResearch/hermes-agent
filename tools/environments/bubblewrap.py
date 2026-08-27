"""Bubblewrap (bwrap) terminal backend: the pure argv builder.

Every terminal command under ``terminal.backend: bubblewrap`` runs inside a
bwrap sandbox. This module builds the bwrap argv prefix from configuration
and construction-time inputs only. Nothing produced inside a sandbox (the
tracked cwd, the shell snapshot, command output) feeds a mount or a flag
argument; the tracked cwd is used for ``--chdir`` alone.

Layout of the argv (later mounts overlay earlier ones):

1. namespace and process-safety flags
2. read-only root, fresh /dev, /proc and a tmpfs /tmp, then a tmpfs over
   the user's runtime dir (/run/user/<uid>: agent and bus sockets) and an
   empty file over the docker socket
3. the initial cwd, read-write for workspace and network, read-only for
   restricted; a ``-try`` bind, so a cwd deleted on the host (the agent's
   own ``rm -rf``) skips the bind and the command runs in the recovered
   cwd instead of wedging every later spawn
4. operator binds from terminal.bubblewrap_binds, minus sensitive sources
5. the pins: each ancestor of a sensitive path or of HERMES_HOME that lies
   strictly inside a writable bind (the cwd under the workspace and
   network profiles, a read-write operator bind) is bound over itself, so
   a command cannot rename the parent of a hidden path out from under its
   overlay
6. the sensitive overlays: a tmpfs over each sensitive directory and an
   empty file over each sensitive file that exists on the host, then the
   same for HERMES_HOME
7. under terminal.home_mode=profile, HERMES_HOME/home read-write on top of
   that overlay (it is the subprocess HOME then)
8. the per-environment state dir read-write at the same path; between
   commands it holds the shell snapshot and the cwd file, and for the
   duration of an execute_code call the hermes_exec_<id> dir with the
   script, the tools module and the rpc files
9. ``--chdir`` to the tracked cwd, then ``--`` so the caller can append the
   shell argv

The paths in the argv are fixed at construction: the hidden set is
resolved through realpath once, so a symlinked entry is hidden at its
target, and the cwd, state dir and operator bind destinations use their
real paths (bwrap resolves a mount destination inside the sandbox root,
where an absolute symlink points nowhere), so the pins are computed in
the real tree of each bind. What varies per spawn is presence only: an
overlay is emitted for a sensitive path that exists on the host at spawn
time and never for one that does not, a pin for an ancestor directory that
exists inside a bind that is writable anyway, and nothing at all for a
path that has become a symlink since construction. So host changes (or a
sandbox with a writable HOME planting a symlink) can only add hiding
mounts and pins, never expose anything. The pins are what
keep that true: a hidden entry is a mount point and cannot be renamed
from inside the sandbox, but without the pins a writable cwd covering its
parent, or a writable bind whose destination resolves into the real tree
above it, lets a command rename the parent, and the next spawn then finds
nothing to hide at the old path while the secret is readable under the
new one.

Resource limits are applied through Popen's preexec_fn, as the spec asks.
CPython documents preexec_fn as unsafe in a threaded process; the callable
here only calls getrlimit and setrlimit, which take no locks.
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
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

from hermes_constants import get_hermes_home, get_real_home
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

# Variables that name host agent or bus sockets. LocalEnvironment passes them
# through; the bwrap prefix unsets them so the sandbox env is local's minus
# exactly these, with no change to LocalEnvironment.
HOST_SOCKET_VARS: tuple[str, ...] = ("SSH_AUTH_SOCK", "GPG_AGENT_INFO", "DBUS_SESSION_BUS_ADDRESS")

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
    """Real host paths that must stay hidden: the HOME set plus HERMES_HOME.

    Each path goes through os.path.realpath, so an entry that is a symlink
    (a dotfiles repository linking ~/.ssh to ~/dotfiles/ssh) or that sits
    under a symlinked component names the directory or file holding the
    secret. That is also the only path bwrap can mount over: it resolves a
    mount destination inside the sandbox root, where an absolute symlink
    points nowhere. BubblewrapEnvironment resolves the set once at
    construction and keeps it for its life. Resolving per spawn would let
    a sandbox that can replace the symlink (HOME in its writable set)
    point the next overlay elsewhere and leave the secret bare.
    """
    home = os.path.abspath(os.path.expanduser(home))
    nominal = [os.path.join(home, rel) for rel in SENSITIVE_HOME_PATHS]
    nominal.append(os.path.abspath(os.path.expanduser(hermes_home)))
    resolved: list[str] = []
    for path in nominal:
        real = os.path.realpath(path)
        if real not in resolved:
            resolved.append(real)
    return tuple(resolved)


def empty_file_path(state_dir: str) -> str:
    """Host path of the zero-length file bound over sensitive files.

    It sits beside the state dir, not inside it: the state dir is bound
    read-write into every spawn, so a file kept there could be rewritten
    from inside the sandbox and would then show at the hidden paths. That
    holds only while the parent of the state dir lies outside every
    writable bind, since bwrap follows a symlink put in the file's place
    when it resolves the bind source; BubblewrapEnvironment therefore
    refuses a sandbox dir inside the cwd or a read-write operator bind
    unless a hidden path covers it (_check_sandbox_root).
    """
    return state_dir.rstrip(os.sep) + ".empty"


def sensitive_overlay_args(hidden_paths: Sequence[str], state_dir: str) -> list[str]:
    """Mount directives that hide *hidden_paths*, the set from sensitive_paths.

    A directory gets a fresh tmpfs, a file gets the empty file bound over
    it, and a path missing on the host gets nothing so bwrap never fails
    on an absent mount target. A path that is a symlink at spawn time gets
    nothing either: the set holds real paths, so a symlink there was
    planted after construction where nothing hidden existed (by a sandbox
    whose writable set covers it), and a mount on it would follow it. On
    bwrap 0.9.0: ``--tmpfs`` on a file path fails with "Not a
    directory", and a ro-bind of /dev/null mounts but reads fail with
    EACCES because bwrap remounts binds nodev inside the user namespace;
    only the empty-file bind works for files.
    """
    empty = empty_file_path(state_dir)
    argv: list[str] = []
    for path in hidden_paths:
        if os.path.islink(path):
            continue
        if os.path.isdir(path):
            argv += ["--tmpfs", path]
        elif os.path.exists(path):
            argv += ["--ro-bind", empty, path]
    return argv


DOCKER_SOCKETS: tuple[str, ...] = ("/var/run/docker.sock", "/run/docker.sock")


def runtime_overlay_args(state_dir: str, uid: int) -> list[str]:
    """Hide the user's runtime dir and the docker socket.

    A read-only bind of / still lets a command connect() to unix sockets:
    the gpg-agent, keyring and ssh-agent sockets under /run/user/<uid>
    would sign and decrypt with the user's loaded keys, and the docker
    socket is a root-equivalent escape for a user in the docker group. The
    runtime dir gets a tmpfs (nothing a command needs lives there) and each
    docker socket that exists gets the empty file bound over it, which
    makes it a plain file.
    """
    argv: list[str] = []
    runtime_dir = f"/run/user/{uid}"
    if os.path.isdir(runtime_dir):
        argv += ["--tmpfs", runtime_dir]
    empty = empty_file_path(state_dir)
    seen: set[str] = set()
    for sock in DOCKER_SOCKETS:
        if not os.path.exists(sock):
            continue
        # Mount at the real path: bwrap cannot create a mount point through
        # the /var/run -> /run symlink, and the symlink resolves to it anyway.
        real = os.path.realpath(sock)
        if real in seen:
            continue
        seen.add(real)
        argv += ["--ro-bind", empty, real]
    return argv


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def is_sensitive_source(src: str, hidden_paths: Sequence[str]) -> bool:
    """True when *src* (or what it symlinks to) is at or under a hidden path."""
    abs_src = os.path.abspath(os.path.expanduser(src))
    candidates = {abs_src, os.path.realpath(abs_src)}
    return any(_is_within(c, root) for c in candidates for root in hidden_paths)


def hidden_path_under(src: str, hidden_paths: Sequence[str]) -> str | None:
    """The first hidden path strictly under *src* (or under what it symlinks to), else None."""
    abs_src = os.path.abspath(os.path.expanduser(src))
    for root in (abs_src, os.path.realpath(abs_src)):
        for hidden in hidden_paths:
            if hidden != root and _is_within(hidden, root):
                return hidden
    return None


def _real_host_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _same_host_path(a: str, b: str) -> bool:
    return _real_host_path(a) == _real_host_path(b)


def _swappable_link(path: str, roots: Iterable[str]) -> str | None:
    """The outermost symlink component of *path* whose parent resolves inside a writable root, else None.

    Such a link sits in a directory a sandbox can write to, so a command can
    replace it; a link outside the writable set is on the read-only root.
    """
    roots = list(roots)
    found: str | None = None
    while True:
        parent = os.path.dirname(path)
        if parent == path:
            return found
        if os.path.islink(path) and any(_is_within(os.path.realpath(parent), root) for root in roots):
            found = path
        path = parent


def filter_binds(binds: tuple[BindMount, ...], hidden_paths: Sequence[str]) -> list[BindMount]:
    """Drop binds that would expose a hidden path, logging a warning for each.

    A source at or under a hidden path would mount the secret itself. A
    source that contains a hidden path and lands at another destination
    would show the secret there: the overlays cover a hidden path only at
    its real location, and a mirror of an ancestor is a second view of the
    same host tree. With dest equal to src the bind
    is the cwd=HOME shape, and the overlays and pins land on top of it.
    """
    kept: list[BindMount] = []
    for bind in binds:
        if is_sensitive_source(bind.src, hidden_paths):
            logger.warning(
                "Ignoring terminal.bubblewrap_binds entry %s: source is under a sensitive path",
                bind.src,
            )
            continue
        hidden = hidden_path_under(bind.src, hidden_paths)
        if hidden is not None and not _same_host_path(bind.src, bind.dest):
            logger.warning(
                "Ignoring terminal.bubblewrap_binds entry %s -> %s: the source contains the "
                "hidden path %s, which would be readable at the destination. Bind it at its "
                "own path (dest equal to src) instead.",
                bind.src, bind.dest, hidden,
            )
            continue
        kept.append(bind)
    return kept


def expand_bind_srcs(binds: Iterable[BindMount]) -> list[BindMount]:
    """Return *binds* with each src taken through expanduser and abspath.

    bwrap does not expand a tilde, so a source written as ~/data made every
    spawn fail on a missing source path. The
    sensitivity checks expand the same way, so they and the emitted argv
    see one path. BubblewrapEnvironment applies this once at construction;
    build_bwrap_args emits a src as given and resolves nothing per spawn.
    """
    return [replace(bind, src=os.path.abspath(os.path.expanduser(bind.src))) for bind in binds]


def resolve_bind_dests(binds: Iterable[BindMount]) -> list[BindMount]:
    """Return *binds* with each dest taken through realpath on the host.

    bwrap resolves a mount destination inside the sandbox root: a relative
    symlink lands the mount on its target, an absolute one aborts the
    spawn. Naming the real path up front gives both the same mount, and
    the ancestor pins are then computed against the real tree the bind
    makes writable. BubblewrapEnvironment applies this once at
    construction; build_bwrap_args does not, so a symlink planted under
    a dest between spawns cannot move the mount.
    """
    return [
        replace(bind, dest=os.path.realpath(os.path.abspath(os.path.expanduser(bind.dest))))
        for bind in binds
    ]


def _ancestors_within(path: str, root: str) -> list[str]:
    """Ancestors of *path* (not *path* itself) strictly inside *root*, outermost first."""
    found: list[str] = []
    parent = os.path.dirname(path)
    while parent != root and _is_within(parent, root):
        found.append(parent)
        parent = os.path.dirname(parent)
    found.reverse()
    return found


def ancestor_pin_args(
    writable_binds: Sequence[tuple[str, str]],
    mount_points: Iterable[str],
    hidden_paths: Sequence[str],
) -> list[str]:
    """Bind over itself each ancestor of a hidden path that lies strictly inside a writable bind.

    A hidden entry is a mount point, so a command cannot rename or remove
    it, but a writable bind covering its parent (the cwd at HOME or above
    it, a read-write operator bind of the same) lets a command rename the
    parent. The next spawn then finds nothing at the hidden path, emits no
    overlay, and the secret is readable under the new name. Binding each
    such ancestor over itself makes it a mount point too: rename and rmdir
    fail with EBUSY while it stays writable, so the bind loses nothing.

    *writable_binds* are (src, dest) pairs in argv order; a pin's source is
    the host path the ancestor maps to through its bind, which is the
    ancestor itself when src and dest agree. No pin is emitted for an
    ancestor that is a mount point already (*mount_points*: the cwd and the
    operator bind destinations), nor for one missing on the host or a
    symlink there: a mount cannot pin a symlink and would bind its target
    instead. Nor for one reached through a symlinked component between
    the bind root and the ancestor: the pin must land inside the real tree
    of the bind (realpath of the host path equals realpath of the bind
    source plus the relative path), or a link leaving the bind would carry
    the pin, and write access, outside it. Presence is the only per-spawn
    input, and a pin never grants more than the bind around it already
    did.
    """
    normalize = lambda p: os.path.abspath(os.path.expanduser(p))
    seen: set[str] = {normalize(p) for p in mount_points}
    argv: list[str] = []
    for src, dest in writable_binds:
        root = normalize(dest)
        real_src = os.path.realpath(normalize(src))
        for path in hidden_paths:
            for ancestor in _ancestors_within(path, root):
                if ancestor in seen:
                    continue
                seen.add(ancestor)
                rel = os.path.relpath(ancestor, root)
                host = os.path.join(normalize(src), rel)
                if not os.path.isdir(host) or os.path.islink(host):
                    continue
                if os.path.realpath(host) != os.path.join(real_src, rel):
                    continue
                argv += ["--bind", host, ancestor]
    return argv


def build_bwrap_args(
    config: BubblewrapConfig,
    initial_cwd: str,
    state_dir: str,
    home: str,
    hermes_home: str,
    tracked_cwd: str,
    *,
    bwrap_path: str = "bwrap",
    hidden_paths: Sequence[str] | None = None,
) -> list[str]:
    """Build the bwrap argv prefix; the caller appends the shell argv after the trailing ``--``.

    All arguments are fixed at environment construction except *tracked_cwd*,
    which only sets ``--chdir``. *hidden_paths* is the set
    BubblewrapEnvironment resolved at construction; when omitted it is
    resolved from *home* and *hermes_home* on this call, which suits tests
    of the pure builder only.
    """
    profile = resolve_profile(config.profile)
    if hidden_paths is None:
        hidden_paths = sensitive_paths(home, hermes_home)

    argv: list[str] = [
        bwrap_path,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--unshare-cgroup-try",
    ]
    if profile.share_net:
        argv.append("--share-net")
    for name in HOST_SOCKET_VARS:
        argv += ["--unsetenv", name]

    argv += [
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]
    argv += runtime_overlay_args(state_dir, os.getuid())

    # The cwd is always bound at its own path so --chdir resolves even when
    # it sits under the masked /tmp; the profile decides whether it is
    # writable. The -try form skips the bind when the directory is gone
    # from the host, so LocalEnvironment's cwd recovery (a parent dir on
    # the read-only root) keeps commands running instead of every spawn
    # failing on a missing bind source.
    argv += ["--bind-try" if profile.writable_cwd else "--ro-bind-try", initial_cwd, initial_cwd]

    # Dests are emitted as given: BubblewrapEnvironment resolved them at
    # construction, and a realpath here would let a symlink planted under a
    # dest between spawns move the mount.
    binds = filter_binds(config.binds, hidden_paths)
    for bind in binds:
        argv += ["--ro-bind" if bind.readonly else "--bind", bind.src, bind.dest]

    # Pin the parents of the hidden paths that sit inside a writable bind
    # (see ancestor_pin_args) after those binds, so the pins land on top of
    # them, and before the overlays, so the overlays land on the pins.
    writable = [(initial_cwd, initial_cwd)] if profile.writable_cwd else []
    writable += [(bind.src, bind.dest) for bind in binds if not bind.readonly]
    argv += ancestor_pin_args(writable, [initial_cwd, *(bind.dest for bind in binds)], hidden_paths)

    # The overlays come after the cwd, operator binds and pins so a bind of
    # HOME itself still hides what sits under it, and before the state dir
    # so that stays reachable under a hidden HERMES_HOME.
    argv += sensitive_overlay_args(hidden_paths, state_dir)

    # Under home_mode=profile the subprocess HOME is HERMES_HOME/home
    # (hermes_constants.get_subprocess_home), so bind it back read-write on
    # top of the overlay; the rest of HERMES_HOME stays hidden.
    if config.home_mode in PROFILE_HOME_MODES:
        profile_home = os.path.join(os.path.abspath(os.path.expanduser(hermes_home)), "home")
        if os.path.isdir(profile_home):
            argv += ["--bind", profile_home, profile_home]

    # The state dir holds the shell snapshot and the cwd file between
    # commands (and the execute_code sandbox dir during a call); it is
    # bound after the sensitive overlays so it stays writable at the same
    # path in every spawn.
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


# Processes the wrapper itself forks before the command runs: bwrap, its
# pid-1 init, the bash that runs the command, and that shell's command
# substitutions and mktemp for the snapshot; a python3 under execute_code
# counts too. RLIMIT_NPROC is checked against the uid's live thread count
# at each fork, so without headroom a tight max_procs (or host threads
# started between the /proc scan and the fork) fails the spawn before the
# command starts: "bwrap: Can't fork for pid 1: Resource temporarily
# unavailable".
WRAPPER_PROCESS_ALLOWANCE = 16


def rlimit_values(config: BubblewrapConfig, *, uid_threads: int) -> dict[int, int]:
    """The rlimits a spawn gets from the three terminal.bubblewrap_* keys.

    A key at 0 leaves its limit out. RLIMIT_NPROC is counted per uid
    host-wide, and a bwrap user namespace does not change that (kernel
    6.8: with the limit below the uid's thread count bwrap cannot even
    create its namespace), so max_procs is applied on top of the uid's
    current thread count: it bounds what the sandbox may add, and the
    documented default of 256 keeps working on a desktop that already runs
    more. WRAPPER_PROCESS_ALLOWANCE is added on top of that for the
    wrapper's own processes, so max_procs is what the command may add,
    not what the command and the wrapper share.
    """
    limits: dict[int, int] = {}
    if config.memory_mb:
        limits[resource.RLIMIT_AS] = config.memory_mb * 1024 * 1024
    if config.cpu_seconds:
        limits[resource.RLIMIT_CPU] = config.cpu_seconds
    if config.max_procs:
        limits[resource.RLIMIT_NPROC] = uid_threads + config.max_procs + WRAPPER_PROCESS_ALLOWANCE
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


_MOUNT_ARITY: dict[str, int] = {
    "--bind": 2, "--ro-bind": 2, "--bind-try": 2, "--ro-bind-try": 2,
    "--tmpfs": 1, "--dev": 1, "--proc": 1,
}


def masked_inside(argv: Sequence[str], path: str) -> bool:
    """True when host directory *path* is hidden inside the sandbox *argv* builds.

    Mounts stack in argv order, so the last directive whose destination is
    *path* or an ancestor of it decides what shows there: a mount root is
    visible (bwrap creates the mount point), a fresh --tmpfs, --dev or
    --proc hides everything below its root, and a bind shows what its
    source holds at the same relative path (the root bind of / shows the
    host itself). A -try bind whose source is missing is skipped, as bwrap
    skips it. Reads only the host presence of directories, as the pins do.
    """
    visible = True
    i = 0
    while i < len(argv):
        arity = _MOUNT_ARITY.get(argv[i])
        if arity is None:
            i += 1
            continue
        operands = argv[i + 1:i + 1 + arity]
        i += 1 + arity
        dest = operands[-1]
        if dest == path:
            visible = True
        elif _is_within(path, dest):
            if arity == 1:
                visible = False
                continue
            src = operands[0]
            if argv[i - 1 - arity].endswith("-try") and not os.path.exists(src):
                continue
            visible = os.path.isdir(os.path.join(src, os.path.relpath(path, dest)))
    return not visible


def chdir_failed(result: Mapping[str, object], tracked_cwd: str) -> bool:
    """True when the result looks like bwrap failing to enter *tracked_cwd*.

    bwrap prints one line, ``Can't chdir to <dir>: ...``, and exits 1
    before the command starts; the wrapper then never prints the cwd
    marker (``cwd_observed`` stays unset). A command that ran and failed
    has the marker, and a timed-out one has the timeout note appended, so
    neither is a single bwrap line. A command can forge the shape by
    printing the line and replacing its shell, so the caller must not
    treat a match as proof that nothing ran: masked_inside decides that
    before the spawn, and this is only the backstop.
    """
    if result.get("returncode", 0) == 0 or result.get("cwd_observed"):
        return False
    lines = str(result.get("output", "")).strip().splitlines()
    return len(lines) == 1 and lines[0].startswith(f"bwrap: Can't chdir to {tracked_cwd}:")


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
        # The OS user's home anchors the sensitive set even when this process
        # runs with HOME pointed at the profile home. Every mount path is
        # taken through realpath: bwrap resolves a mount destination inside
        # the sandbox root, where an absolute symlink points nowhere.
        self._home = os.path.realpath(get_real_home() or os.path.expanduser("~"))
        self._hermes_home = os.path.realpath(str(get_hermes_home()))
        # Resolved once and kept for the life of the environment: the set
        # never follows a symlink swapped in later.
        self._hidden_paths = sensitive_paths(self._home, self._hermes_home)
        # The operator binds are filtered, their sources expanded and their
        # destinations resolved once here, like the hidden set and the cwd,
        # so the mount paths are fixed for the life of the environment
        # and a dropped bind warns once; the builder's own filter
        # then drops nothing.
        self._config = replace(
            self._config,
            binds=tuple(resolve_bind_dests(expand_bind_srcs(filter_binds(self._config.binds, self._hidden_paths)))),
        )
        self._check_profile_home()
        # The mount paths are fixed here; only --chdir follows the tracked cwd.
        self._initial_cwd = os.path.realpath(_resolve_local_initial_cwd(cwd))
        self._check_initial_cwd()
        self._check_bind_sources()
        sandbox_root = os.path.realpath(get_sandbox_dir())
        self._check_sandbox_root(sandbox_root)
        # BaseEnvironment.__init__ derives the snapshot and cwd file paths
        # from get_temp_dir() and LocalEnvironment.__init__ runs the login
        # bootstrap straight away, so the state dir must exist first.
        self._state_dir = os.path.join(sandbox_root, f"bwrap-{uuid.uuid4().hex[:12]}")
        os.makedirs(self._state_dir, mode=0o700)
        # Read-only and outside the state dir: nothing in a sandbox can
        # write to what shows at the hidden file paths.
        self._empty_file = empty_file_path(self._state_dir)
        os.close(os.open(self._empty_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400))
        try:
            super().__init__(cwd=self._initial_cwd, timeout=timeout, env=env)
        except BaseException:
            self._remove_state()
            raise

    def _check_profile_home(self) -> None:
        """Refuse a profile home that holds a hidden path or lies under one.

        Under a profile home_mode the builder binds HERMES_HOME/home
        read-write on top of every overlay, and bwrap resolves the bind
        source on the host. When that path is a symlink to a tree holding
        a hidden path (HOME, HOME/.config, HERMES_HOME itself or a parent
        of it), or to a directory at or under one (HOME/.ssh), the bind is
        a second view of the same host tree and shows the secrets at
        HERMES_HOME/home, the shape filter_binds drops for operator binds.
        The plain directory under
        HERMES_HOME, a link to another directory under HERMES_HOME and a
        link to a clean directory outside every hidden path stay allowed.
        """
        if self._config.home_mode not in PROFILE_HOME_MODES:
            return
        link = os.path.join(self._hermes_home, "home")
        profile_home = os.path.realpath(link)
        for hidden in self._hidden_paths:
            if _is_within(hidden, profile_home):
                relation = f"contains {hidden}"
            elif hidden != self._hermes_home and _is_within(profile_home, hidden):
                relation = f"lies under {hidden}"
            else:
                continue
            raise ValueError(
                f"terminal.home_mode={self._config.home_mode} binds the profile home "
                f"{link} read-write inside the sandbox with the bubblewrap backend, but "
                f"it resolves to {profile_home}, which {relation}, a path the backend "
                "hides: the bind would show it again. Make HERMES_HOME/home a plain "
                "directory, or a link to a directory outside HOME's hidden dotfiles "
                "and the rest of HERMES_HOME, or set terminal.home_mode to auto or real."
            )

    def _check_initial_cwd(self) -> None:
        """Refuse a cwd of / or under a hidden path, warn about HOME: the cwd is the writable set."""
        cwd = self._initial_cwd.rstrip(os.sep) or os.sep
        if cwd == os.sep:
            raise ValueError(
                "terminal.cwd must not be / with the bubblewrap backend: the whole "
                "root would be writable inside the sandbox. Set terminal.cwd to a "
                "project or scratch directory."
            )
        # The overlays land after the cwd bind, so a cwd under a hidden path
        # is masked in every spawn and no command could run there. The profile
        # home is bound back on top of the HERMES_HOME overlay under
        # home_mode=profile, so a cwd under it is fine
        # when HERMES_HOME/home is a plain directory. The bind lands at the
        # link path when it is a symlink, so a cwd under a link target inside
        # HERMES_HOME (or another hidden path) stays masked and gets no
        # exemption; a link target outside every hidden path needs none.
        profile_home = os.path.join(self._hermes_home, "home")
        in_profile_home = (
            self._config.home_mode in PROFILE_HOME_MODES
            and not os.path.islink(profile_home)
            and _is_within(cwd, os.path.realpath(profile_home))
        )
        if not in_profile_home:
            for hidden in self._hidden_paths:
                if _is_within(cwd, hidden):
                    raise ValueError(
                        f"terminal.cwd {self._initial_cwd} lies under {hidden}, which the "
                        "bubblewrap backend hides inside every sandbox: no command could run "
                        "there. Set terminal.cwd to a project or scratch directory outside "
                        "HERMES_HOME and the hidden dotfiles (a checkout under ~/.hermes "
                        "needs to be launched from elsewhere or moved)."
                    )
        home = os.path.abspath(self._home).rstrip(os.sep) or os.sep
        if cwd == home or _is_within(home, cwd):
            logger.warning(
                "bubblewrap cwd %s covers the home directory: every dotfile outside "
                "the hidden set is writable inside the sandbox. Set terminal.cwd to "
                "a project or scratch directory for a smaller writable set.",
                self._initial_cwd,
            )

    def _check_bind_sources(self) -> None:
        """Refuse a read-write bind whose source a sandbox could swap.

        bwrap resolves a bind source on the host at every spawn. When a
        component of the source path sits inside the writable set (the cwd
        under a writable profile, another read-write bind's source, the
        profile home under a profile home_mode) and is a symlink or a
        directory a command can rename, the command replaces it and
        chooses the next spawn's mount source, gaining read-write access
        to any host directory without a hidden path below it. A mount
        point cannot be renamed (EBUSY), so
        a source bound at its own real path directly under a writable
        root is fixed: the source and its parent are mount points inside.
        One level deeper the parent is a plain directory: renamed,
        recreated and given a relative symlink at the source path, it
        steers the next spawn's mount. A read-only bind shows nothing the
        root bind does not.
        """
        writable: dict[str, str] = {}
        if resolve_profile(self._config.profile).writable_cwd:
            writable[self._initial_cwd] = "terminal.cwd"
        for bind in self._config.binds:
            if not bind.readonly:
                writable.setdefault(_real_host_path(bind.src), "the read-write bind source")
        if self._config.home_mode in PROFILE_HOME_MODES:
            writable.setdefault(os.path.realpath(os.path.join(self._hermes_home, "home")), "the profile home")
        for bind in self._config.binds:
            if bind.readonly:
                continue
            given = os.path.abspath(os.path.expanduser(bind.src))
            real = os.path.realpath(given)
            link = _swappable_link(given, writable)
            if link is not None:
                problem = f"its source path goes through the symlink {link}, which a command could replace"
            else:
                inside = [
                    root for root in writable
                    if any(path != root and _is_within(path, root) for path in (given, real))
                ]
                if not inside or (bind.dest == real and os.path.dirname(real) in writable):
                    continue
                problem = (
                    f"its source lies inside {writable[inside[0]]} {inside[0]}, which is "
                    "writable inside the sandbox"
                )
            raise ValueError(
                f"terminal.bubblewrap_binds entry {bind.src} -> {bind.dest} is read-write "
                f"and {problem} with the bubblewrap backend: a command could swap the "
                "source and choose the next spawn's mount. Bind it read-write only at "
                "its own path (dest equal to src) directly under terminal.cwd, a "
                "read-write bind source or the profile home, with no symlink on the "
                "way, where the source and its parent are mount points inside and "
                "cannot be moved; or make it read-only."
            )

    def _check_sandbox_root(self, sandbox_root: str) -> None:
        """Refuse a sandbox dir a sandbox could write to.

        The empty file bound over hidden files sits in the sandbox dir
        beside the state dir. Inside a writable bind a command could
        replace it with a symlink to a hidden file, and bwrap resolves the
        bind source on the next spawn, showing the whole secret. Under a
        hidden path (the default HERMES_HOME/sandboxes) the overlay covers
        it and nothing in a sandbox can reach it, unless a later bind lands
        on top of the overlay. The profile home under home_mode=profile
        is the one such bind, so a sandbox dir under it is refused
        first, whether the profile home is a directory under HERMES_HOME
        or a symlink to a directory outside every hidden path. An operator
        bind cannot re-expose the dir: filter_binds
        drops a source under a hidden path and a source containing one
        that maps elsewhere, and a source containing one at its own path
        sits below the overlay.
        """
        profile_home = os.path.realpath(os.path.join(self._hermes_home, "home"))
        if self._config.home_mode in PROFILE_HOME_MODES and _is_within(sandbox_root, profile_home):
            raise ValueError(
                f"terminal.sandbox_dir {sandbox_root} lies inside the profile home "
                f"{profile_home}, which terminal.home_mode={self._config.home_mode} binds "
                "read-write inside the sandbox with the bubblewrap backend: a command "
                "could replace the empty file bound over hidden files. Set "
                "terminal.sandbox_dir to a directory outside HERMES_HOME/home; the "
                "default HERMES_HOME/sandboxes is covered by the HERMES_HOME overlay."
            )
        if any(_is_within(sandbox_root, hidden) for hidden in self._hidden_paths):
            return
        writable: list[str] = [self._initial_cwd] if resolve_profile(self._config.profile).writable_cwd else []
        writable += [
            os.path.realpath(os.path.expanduser(bind.src))
            for bind in self._config.binds
            if not bind.readonly
        ]
        for root in writable:
            if _is_within(sandbox_root, root):
                raise ValueError(
                    f"terminal.sandbox_dir {sandbox_root} lies inside {root}, which is "
                    "writable inside the sandbox with the bubblewrap backend: a command "
                    "could replace the empty file bound over hidden files. Set "
                    "terminal.sandbox_dir to a directory outside terminal.cwd and "
                    "outside every read-write terminal.bubblewrap_binds source; the "
                    "default HERMES_HOME/sandboxes is covered by the HERMES_HOME overlay."
                )

    def get_temp_dir(self) -> str:
        return self._state_dir

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Run a command; report a tracked cwd that the sandbox mounts hide.

        The tracked cwd is checked against the host (LocalEnvironment's
        missing-cwd recovery), not against what the overlays mask
        inside the sandbox. After ``cd`` into a host directory under a
        hidden path, or under the fresh /tmp, ``--chdir`` would fail on
        every later spawn before the shell runs, and no cwd marker could
        move the tracked cwd again. _reset_masked_cwd decides that from the
        fixed mount layout before the wrapper and the spawn are built and
        resets the tracked cwd to the initial cwd, so the command runs
        exactly once there; the note tells the caller. The chdir_failed
        backstop below only resets the tracked cwd and never re-runs the
        command: its shape can be forged by a command that prints the
        bwrap line and replaces its shell.
        """
        note = self._reset_masked_cwd()
        result = super().execute(command, cwd, **kwargs)
        if note is not None:
            result["output"] = note + result.get("output", "")
        elif self.cwd != self._initial_cwd and chdir_failed(result, self.cwd):
            stale = self.cwd
            logger.warning(
                "bubblewrap could not enter the tracked cwd %s; resetting the working "
                "directory to %s for the next command.",
                stale, self._initial_cwd,
            )
            self.cwd = self._initial_cwd
            result["output"] = (
                result.get("output", "")
                + f"\n[bubblewrap: could not enter working directory {stale}; reset to "
                f"{self._initial_cwd}, run the command again]"
            )
        return result

    def _bwrap_prefix(self, tracked_cwd: str) -> list[str]:
        return build_bwrap_args(
            self._config,
            self._initial_cwd,
            self._state_dir,
            self._home,
            self._hermes_home,
            tracked_cwd,
            bwrap_path=self._bwrap_path,
            hidden_paths=self._hidden_paths,
        )

    def _reset_masked_cwd(self) -> str | None:
        """Reset a tracked cwd the mounts hide to the initial cwd; return the note.

        Runs before the command wrapper (which cd's to the tracked cwd) and
        the argv (whose --chdir carries it) are built. The initial cwd is
        never reset: it is bound at its own path. A tracked cwd gone from
        the host is not reset either: LocalEnvironment's recovery in
        _run_bash lands it on the nearest existing parent, as for the local
        backend, and when the mounts mask that parent the chdir_failed
        backstop in execute resets it on that spawn.
        """
        if self.cwd == self._initial_cwd or not os.path.isdir(self.cwd):
            return None
        if not masked_inside(self._bwrap_prefix(self.cwd), self.cwd):
            return None
        stale = self.cwd
        logger.warning(
            "bubblewrap: the tracked cwd %s is not visible inside the sandbox; resetting "
            "the working directory to %s.",
            stale, self._initial_cwd,
        )
        self.cwd = self._initial_cwd
        return (
            f"[bubblewrap: working directory {stale} is not visible inside the sandbox; "
            f"reset to {self._initial_cwd}]\n"
        )

    def _wrap_popen_args(self, args: list[str]) -> list[str]:
        return self._bwrap_prefix(self.cwd) + list(args)

    def _wrap_command(self, command: str, cwd: str) -> str:
        # --unsetenv strips the socket variables from the environment bwrap
        # receives, but the login bootstrap sources the shell init files
        # into the snapshot afterwards, and a 1Password or gpg-agent setup
        # exports SSH_AUTH_SOCK from ~/.bashrc. Unset them in front of the
        # command: that runs after the snapshot is sourced, and the export
        # dump that follows the command then omits them too.
        return super()._wrap_command(f"unset {' '.join(HOST_SOCKET_VARS)}; {command}", cwd)

    def _popen_preexec(self):
        # uid_thread_count scans /proc once per spawn, and only when max_procs
        # is non-zero (the one limit that needs it). The count must be fresh:
        # the kernel checks RLIMIT_NPROC against the uid's live thread count
        # when the sandbox forks, so a count taken at construction goes stale
        # as the host starts threads, and a limit that falls below the live
        # count stops bwrap from creating its namespace at all.
        uid_threads = uid_thread_count(os.getuid()) if self._config.max_procs else 0
        return make_preexec(rlimit_values(self._config, uid_threads=uid_threads))

    def _live_sandbox_pids(self) -> list[int]:
        """PIDs of this instance's bwrap wrappers still running.

        A wrapper is a direct child of this process whose argv is the bwrap
        path with this instance's state dir bound; the state dir is unique
        per instance and fixed at construction, so nothing from inside a
        sandbox can forge it. Zombies are left for the thread that spawned
        them to reap. A child that is between fork and exec still shows
        Python's cmdline and is missed; it then fails to bind the removed
        state dir and exits, exposing nothing.
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

    def _remove_state(self) -> None:
        shutil.rmtree(self._state_dir, ignore_errors=True)
        try:
            os.unlink(self._empty_file)
        except OSError:
            pass

    def cleanup(self):
        self._kill_live_sandboxes()
        super().cleanup()
        self._remove_state()
