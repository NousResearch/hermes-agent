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
5. the pins: each ancestor of a hidden path that lies strictly inside a
   writable bind is bound over itself, so a command cannot rename the
   parent of a hidden path out from under its overlay
6. the HOME layout (bubblewrap_home.home_layout_args), which makes HOME
   default-deny for dot entries: a tmpfs over HOME, the non-dot entries
   and the allowed dot entries bound back, a tmpfs with the allowed
   children for HOME/.config, HOME/.local and HOME/.local/share, an
   overlay for each hidden path that is still visible, and at the very
   end a read-only remount of each tmpfs. The cwd, the operator binds
   and the pins of steps 3 to 5 that land under HOME, and the binds of
   steps 8 to 10 that do, are emitted inside the layout, before the
   remount: a mount point under a sealed HOME could not be made
7. the overlays for hidden paths outside HOME: a tmpfs over a directory
   and an empty file over a file that exists on the host. HERMES_HOME is
   hidden as a whole, and so is the default HOME/.hermes when HERMES_HOME
   points elsewhere, so a profile's sandbox cannot read the default
   home's credentials
8. an empty read-only directory over the path of the Hermes scratch dir,
   so nothing that other Hermes processes or sessions keep in the shared
   scratch dir is in view, and inside it the scratch directory of this
   environment alone, at its own host path and writable with the cwd
   (TMPDIR of a command points at it); and the staged data roots of the
   cache registry,
   read-only, on top of that overlay; then once more the overlay of each
   hidden path that lies under one of those roots
9. under terminal.home_mode=profile, HERMES_HOME/home read-write on top of
   the overlay (it is the subprocess HOME then)
10. the per-environment state dir read-write at the same path; between
    commands it holds the shell snapshot and the cwd file, and for the
    duration of an execute_code call the hermes_exec_<id> dir with the
    script, the tools module and the rpc files
11. ``--chdir`` to the tracked cwd, then ``--`` so the caller can append
    the process limit and the shell argv

The paths in the argv are fixed at construction: the hidden set and the
HOME allowlist are resolved once, so a symlinked entry is hidden at its
target and a command cannot widen the allowlist, and the cwd, state dir
and operator bind destinations use their real paths (bwrap resolves a
mount destination inside the sandbox root, where an absolute symlink
points nowhere), so the pins are computed in the real tree of each bind.
What varies per spawn is presence only: the listing of the top of HOME,
an overlay for a hidden path that exists on the host at spawn time and
never for one that does not, a pin for an ancestor directory that exists
inside a bind that is writable anyway, and nothing at all for a path that
has become a symlink since construction. A hidden path that does not
exist cannot be created either: the top of HOME and the default-deny
directories are read-only, and a writable bind that holds such a path is
refused at construction. So host changes can only add hiding mounts and
pins, never expose anything. The pins are what keep that true below a
writable bind: a hidden entry is a mount point and cannot be renamed
from inside the sandbox, but without the pins a writable bind covering
its parent lets a command rename the parent, and the next spawn then
finds nothing to hide at the old path while the secret is readable under
the new one.

One set of mounts does grow after construction: the archive files of
oversized tool results that the host hands to this environment
(expose_spillover_file). Each is a single file, a copy kept beside the
state dir and bound read-only at the path of the archive, named by the
Hermes process and never by a command.

Resource limits are applied by prlimit(1) from util-linux, in two
places. In front of the bwrap argv above, prlimit sets RLIMIT_AS and
RLIMIT_CPU on itself and execs bwrap, so those limits are in place before
the sandbox starts. After bwrap's ``--`` separator, a second prlimit sets
RLIMIT_NPROC and execs the shell: inside the user namespace bwrap made,
the kernel counts the processes of that sandbox alone, so the value is a
ceiling for the sandbox whatever the host runs (see rlimit_values). No
Python code runs in the forked child. (Popen's pre-exec callback is
documented as unsafe in a threaded process, and the gateway is one.) A
value above the inherited hard limit is clamped to it in the parent,
since raising a hard limit needs CAP_SYS_RESOURCE.
"""

from __future__ import annotations

import json
import logging
import os
import resource  # windows-footgun: ok — bubblewrap is a Linux-only backend
import shlex
import shutil
import signal
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from hermes_constants import (
    SCRATCH_DIR_MARKER_ENV,
    SCRATCH_TMP_ENV_VARS,
    get_hermes_home,
    get_real_home,
    get_scratch_dir,
)
from tools.environments import bubblewrap_home
from tools.environments.base import EnvironmentConnectionError, get_sandbox_dir
from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd

logger = logging.getLogger(__name__)

# Credential stores under HOME, relative to it. The set and the rule that
# hides everything else under HOME by default live in bubblewrap_home.
SENSITIVE_HOME_PATHS: tuple[str, ...] = bubblewrap_home.DENIED_HOME_PATHS

# The default Hermes home under the real HOME (the Linux default of
# hermes_constants.get_hermes_home). It joins the hidden set beside the
# resolved HERMES_HOME so a relocated HERMES_HOME (a profile at
# HOME/.hermes/profiles/<name>, or any other directory) does not leave the
# default home's .env and auth.json readable inside the sandbox.
DEFAULT_HERMES_HOME_NAME = ".hermes"

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
ENV_HOME_ALLOW = "TERMINAL_BUBBLEWRAP_HOME_ALLOW"
ENV_HIDE = "TERMINAL_BUBBLEWRAP_HIDE"
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
    # terminal.bubblewrap_home_allow: more allowlist units under HOME
    # (bubblewrap_home.allow_unit). terminal.bubblewrap_hide: more paths to
    # hide, absolute or relative to HOME.
    home_allow: tuple[str, ...] = ()
    hide: tuple[str, ...] = ()


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


def _parse_string_list(name: str, raw: str) -> tuple[str, ...]:
    """A JSON list of strings from the env value *raw*; blank gives the empty tuple."""
    if not raw.strip():
        return ()
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ValueError(f"{name} must be a JSON list of strings, got {raw!r}") from None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{name} must be a JSON list of strings, got {raw!r}")
    return tuple(parsed)


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
        home_allow=_parse_string_list(ENV_HOME_ALLOW, env.get(ENV_HOME_ALLOW, "")),
        hide=_parse_string_list(ENV_HIDE, env.get(ENV_HIDE, "")),
    )


def home_replaced_by_bind(home_root: str, binds: Iterable[BindMount]) -> bool:
    """True when an operator bind puts another directory over HOME or over a parent of it."""
    return any(_is_within(home_root, bind.dest) and not _same_host_path(bind.src, bind.dest) for bind in binds)


# How many archive files of oversized tool results one environment binds.
SPILLOVER_BIND_MAX = 256

# How many symlinks the walk of one chain follows; the kernel stops at 40.
DOT_LINK_MAX_HOPS = 40


def dot_link_entries(home_root: str, name: str) -> list[str]:
    """Every directory entry the kernel passes when it resolves *name* at the top of *home_root*.

    The walk goes one path component at a time and follows each symlink
    it meets, as path resolution does, so the list holds the links in the
    middle of a chain and the directories that lead to them, not just the
    final target that realpath gives. The last item is the final target,
    or the first entry that does not exist: a command that can make it
    decides what the link resolves to. The entry *name* itself is left
    out. A chain longer than the kernel allows resolves to nothing on the
    host, so the walk stops there.
    """
    entries: list[str] = []
    current = home_root
    try:
        pending = os.readlink(os.path.join(home_root, name)).split(os.sep)
    except OSError:
        return entries
    if pending and pending[0] == "":
        current = os.sep
    hops = 1
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            current = os.path.dirname(current)
            continue
        entry = os.path.join(current, part)
        entries.append(entry)
        if os.path.islink(entry):
            hops += 1
            if hops > DOT_LINK_MAX_HOPS:
                break
            try:
                target = os.readlink(entry)
            except OSError:
                break
            if os.path.isabs(target):
                current = os.sep
            pending = target.split(os.sep) + pending
        elif os.path.lexists(entry):
            current = entry
        else:
            break
    return entries


def staged_data_roots() -> tuple[str, ...]:
    """Real host paths of the staged data directories under the active HERMES_HOME.

    The list is the cache directory registry of tools.credential_files, the
    one the container backends mount from, read through its own accessor:
    a root added there is bound here with no change to this module. A
    registry that cannot be read binds nothing and warns; the sandbox
    still starts, with those paths hidden as before.

    One root of the registry is left out: the archive of oversized tool
    results. It holds the tool output of every session of the profile,
    which can carry secrets, and nothing in it says which session a file
    belongs to. An environment shows its commands the archive files it
    was handed, one by one (BubblewrapEnvironment.expose_spillover_file).
    """
    try:
        from tools import credential_files
        from tools.tool_result_storage import get_spillover_dir

        mounts = credential_files.get_cache_directory_mounts()
        spillover = os.path.realpath(str(get_spillover_dir()))
    except Exception:
        logger.warning("bubblewrap: could not read the staged data registry; staged data paths stay hidden", exc_info=True)
        return ()
    roots: list[str] = []
    for mount in mounts:
        real = os.path.realpath(mount["host_path"])
        if real not in roots and not _is_within(real, spillover):
            roots.append(real)
    return tuple(roots)


def operator_hidden_paths(home: str, items: Iterable[str]) -> tuple[str, ...]:
    """Real host paths the operator hides through terminal.bubblewrap_hide.

    An item is absolute, starts with ``~/``, or is relative to *home*. An
    item that names HOME, a parent of it or the root would hide every
    command's working set and is ignored with a warning.
    """
    home = os.path.realpath(os.path.abspath(os.path.expanduser(home)))
    resolved: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if text == "~" or text.startswith("~/"):
            text = os.path.join(home, text[2:])
        real = os.path.realpath(text if os.path.isabs(text) else os.path.join(home, text))
        if _is_within(home, real):
            logger.warning("Ignoring terminal.bubblewrap_hide entry %r: it covers the home directory", item)
            continue
        if real not in resolved:
            resolved.append(real)
    return tuple(resolved)


def sensitive_paths(home: str, hermes_home: str, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Real host paths that must stay hidden: the HOME set, HERMES_HOME and HOME/.hermes.

    The default HOME/.hermes rides along with the resolved HERMES_HOME so a
    relocated HERMES_HOME (a profile at HOME/.hermes/profiles/<name>) does
    not leave the default home readable; when the two are the same
    directory it is listed once. A missing HOME/.hermes stays in the set
    and emits no mount (sensitive_overlay_args skips absent paths). *extra* is
    the operator's own hidden paths (operator_hidden_paths).

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
    resolved = list(bubblewrap_home.denied_home_paths(home))
    for path in (os.path.abspath(os.path.expanduser(hermes_home)), os.path.join(home, DEFAULT_HERMES_HOME_NAME), *extra):
        real = os.path.realpath(path)
        if real not in resolved:
            resolved.append(real)
    # An entry under another one is covered by it (the file safety policy
    # names files inside HERMES_HOME, which is hidden as a whole). An
    # operator entry is kept even then: a root under HERMES_HOME that is
    # bound back on top of the overlay would show it again, and the
    # builder hides it once more on top of that bind.
    kept = {os.path.realpath(path) for path in extra}
    return tuple(
        p for p in resolved
        if p in kept or not any(p != other and _is_within(p, other) for other in resolved)
    )


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


def scratch_view_path(state_dir: str) -> str:
    """Host path of the directory bound read-only over the Hermes scratch path.

    It holds one empty directory, the mount point of the environment's
    own scratch dir, and nothing else: it is what a command sees of the
    scratch path in place of the directory every Hermes process shares.
    Beside the state dir, like the empty file and for the same reason:
    at a path a command can write to it could be swapped for a symlink
    to a hidden directory, which bwrap would follow on the next spawn.
    """
    return state_dir.rstrip(os.sep) + ".scratch"


def archive_copies_path(state_dir: str) -> str:
    """Host path of the directory that holds this environment's copies of tool result archives.

    Beside the state dir, like the empty file: the copies are bind
    sources, and no command may reach or replace them.
    """
    return state_dir.rstrip(os.sep) + ".archives"


def private_scratch_path(scratch_dir: str, state_dir: str) -> str:
    """Host path of the scratch directory of one environment, inside the Hermes scratch dir.

    The same path on the host and in the sandbox, so the cwd tracking and
    the host-side file tools see what a command wrote there. The name
    carries the id of the state dir, so two environments never share one.
    It is a top-level entry of the scratch dir: the pruning of idle
    scratch entries covers one that a crashed process left behind.
    """
    return os.path.join(scratch_dir, "hermes-" + os.path.basename(state_dir.rstrip(os.sep)))


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


# Marks an argument the caller left out, where None already has a meaning.
_UNRESOLVED: object = object()


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
    home_root: str | None | object = _UNRESOLVED,
    home_allow: Sequence[str] | None = None,
    scratch_dir: str | None = None,
    scratch_view: str | None = None,
    scratch_private: str | None = None,
    staged_roots: Sequence[str] = (),
    staged_files: Sequence[tuple[str, str]] = (),
    sandbox_overlays: Sequence[str] = (),
) -> list[str]:
    """Build the bwrap argv prefix; the caller appends the shell argv after the trailing ``--``.

    All arguments are fixed at environment construction except *tracked_cwd*,
    which only sets ``--chdir``. *hidden_paths*, *home_root* (the real HOME
    the default-deny layout is built over, None for no layout) and
    *home_allow* (the allowlist units) are what BubblewrapEnvironment
    resolved at construction; when omitted they are resolved from *home*
    and *hermes_home* on this call, with no PATH and no operator items,
    which suits tests of the pure builder only. *scratch_dir* is the path of the
    Hermes scratch directory, *scratch_view* the empty directory bound
    read-only over it on top of the HERMES_HOME overlay, and
    *scratch_private* the scratch directory of this environment under
    that path, bound at its own path; without the first two nothing is
    bound. *staged_roots* are the staged data directories under
    HERMES_HOME, bound back read-only, and *staged_files* the single
    files under it that this environment was handed, as (source, path)
    pairs, bound the same way.
    *sandbox_overlays* are the paths at which a sandbox dir that no hidden
    path covers would show; each gets a tmpfs below the state dir bind. The listing of the top of
    HOME is the one input read from the host at each call, so a directory
    made on the host later shows in the next spawn.
    """
    profile = resolve_profile(config.profile)
    if hidden_paths is None:
        hidden_paths = sensitive_paths(home, hermes_home)
    if home_root is _UNRESOLVED:
        home_root = bubblewrap_home.resolve_home_root(home)
    if home_allow is None and home_root is not None:
        home_allow = bubblewrap_home.resolve_allowlist(
            home_root, "", (),
            denied_names=bubblewrap_home.denied_home_names(home_root), denied_paths=hidden_paths,
        )

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
        "--tmpfs", "/tmp",  # no-tmp: ok — bwrap mount argument, names the mountpoint it replaces
    ]
    argv += runtime_overlay_args(state_dir, os.getuid())  # windows-footgun: ok — bubblewrap is a Linux-only backend

    # The cwd is always bound at its own path so --chdir resolves even when
    # it sits under the masked /tmp; the profile decides whether it is
    # writable. The -try form skips the bind when the directory is gone
    # from the host, so LocalEnvironment's cwd recovery (a parent dir on
    # the read-only root) keeps commands running instead of every spawn
    # failing on a missing bind source.
    # Dests are emitted as given: BubblewrapEnvironment resolved them at
    # construction, and a realpath here would let a symlink planted under a
    # dest between spawns move the mount.
    binds = filter_binds(config.binds, hidden_paths)
    mounts: list[tuple[str, str, str]] = [
        ("--bind-try" if profile.writable_cwd else "--ro-bind-try", initial_cwd, initial_cwd),
    ]
    mounts += [("--ro-bind" if bind.readonly else "--bind", bind.src, bind.dest) for bind in binds]
    writable = [(initial_cwd, initial_cwd)] if profile.writable_cwd else []
    writable += [(bind.src, bind.dest) for bind in binds if not bind.readonly]
    mount_points = [initial_cwd, *(bind.dest for bind in binds)]

    # A bind that puts another directory over HOME (or over a parent of it)
    # replaces HOME on purpose. The layout would put the host entries back
    # on top of it, so it stands down and the plain overlays apply.
    if home_root is not None and home_replaced_by_bind(home_root, binds):
        home_root = None

    # The state dir holds the shell snapshot and the cwd file between
    # commands (and the execute_code sandbox dir during a call). Under
    # home_mode=profile the subprocess HOME is HERMES_HOME/home
    # (hermes_constants.get_subprocess_home). Both are bound read-write on
    # top of the overlays, so the rest of HERMES_HOME stays hidden.
    late: list[tuple[str, ...]] = []
    # Roots bound back on top of the HERMES_HOME overlay, and the ones of
    # them a command can write to.
    restored: list[str] = []
    restored_rw: list[str] = []
    # The Hermes scratch dir lies under the hidden HERMES_HOME: left
    # hidden, a write there lands in that spawn's own tmpfs and is gone by
    # the next command, with no error. The host directory is TMPDIR of
    # every Hermes process, so it also holds control sockets and message
    # payloads of other sessions, and a new one can appear while a command
    # runs. So the sandbox gets an empty read-only directory at that path,
    # and inside it the scratch dir of this environment alone, at its host
    # path. That one follows the cwd: writable with it, read-only under
    # the restricted profile. A write to the scratch path itself fails
    # where the command can see it.
    if scratch_dir and scratch_view:
        late.append(("--ro-bind", scratch_view, scratch_dir))
        if scratch_private:
            late.append(("--bind" if profile.writable_cwd else "--ro-bind", scratch_private, scratch_private))
    # Attachments, cached documents and the other staged data live under
    # HERMES_HOME, and Hermes hands the model their host paths. Read-only:
    # a command opens them, it does not produce them. The -try form lets a
    # root that is gone from the host drop out instead of failing the spawn.
    late += [("--ro-bind-try", root, root) for root in staged_roots]
    late += [("--ro-bind-try", src, path) for src, path in staged_files]
    restored += staged_roots
    if config.home_mode in PROFILE_HOME_MODES:
        profile_home = os.path.join(os.path.abspath(os.path.expanduser(hermes_home)), "home")
        if os.path.isdir(profile_home):
            late.append(("--bind", profile_home, profile_home))
            restored.append(profile_home)
            restored_rw.append(profile_home)
    # A hidden path under one of the roots just bound back is shown again
    # by that bind, so it gets its overlay once more, on top of it. Inside
    # a writable root its parents are pinned first, as under any writable
    # bind: otherwise a command renames a parent and the next spawn finds
    # the path, and the secret, under another name.
    shown = [
        path for path in hidden_paths
        if not os.path.islink(path) and any(path != root and _is_within(path, root) for root in restored)
    ]
    pins = ancestor_pin_args([(root, root) for root in restored_rw], restored, shown)
    late += [(pins[i], pins[i + 1], pins[i + 2]) for i in range(0, len(pins), 3)]
    for path in shown:
        if os.path.isdir(path):
            late.append(("--tmpfs", path))
        elif os.path.exists(path):
            late.append(("--ro-bind", empty_file_path(state_dir), path))
    # A sandbox dir outside every hidden path shows through the read-only
    # root, and with it the state dirs of the other environments: their
    # shell snapshots and the scripts of their execute_code calls. A
    # tmpfs goes over it, and the state dir of this environment on top.
    late += [("--tmpfs", path) for path in sandbox_overlays]
    late.append(("--bind", state_dir, state_dir))

    if home_root is None:
        for mount in mounts:
            argv += mount
        # Pin the parents of the hidden paths that sit inside a writable bind
        # (see ancestor_pin_args) after those binds, so the pins land on top
        # of them, and before the overlays, so the overlays land on the pins.
        argv += ancestor_pin_args(writable, mount_points, hidden_paths)
        argv += sensitive_overlay_args(hidden_paths, state_dir)
        for mount in late:
            argv += mount
    else:
        root: str = home_root  # type: ignore[assignment]

        def in_home(path: str) -> bool:
            return path != root and _is_within(path, root)

        # Listed once per spawn and shared by the pins and the layout, so
        # both see the same entries.
        try:
            listing = sorted(os.listdir(root))
        except OSError:
            listing = []
        # A writable bind that covers HOME makes the non-dot entries
        # writable and nothing else under HOME: the dot entries are
        # read-only or hidden, and the top of HOME is a read-only tmpfs.
        # So inside HOME the pins are computed against those entries, and
        # the covering bind itself only pins parents of hidden paths that
        # lie outside HOME.
        covering = [pair for pair in writable if _is_within(root, pair[1])]
        plain = [pair for pair in writable if pair not in covering]
        hidden_in = [path for path in hidden_paths if in_home(path)]
        hidden_out = [path for path in hidden_paths if not in_home(path)]
        pins = ancestor_pin_args(plain, mount_points, hidden_paths)
        pins += ancestor_pin_args(covering, mount_points, hidden_out)
        if covering:
            entries = [os.path.join(root, name) for name in listing if not name.startswith(".")]
            entries = [path for path in entries if os.path.isdir(path) and not os.path.islink(path)]
            pins += ancestor_pin_args([(path, path) for path in entries], [*mount_points, *entries], hidden_in)
        mounts += [(pins[i], pins[i + 1], pins[i + 2]) for i in range(0, len(pins), 3)]

        for mount in mounts:
            if not in_home(mount[2]):
                argv += mount
        argv += bubblewrap_home.home_layout_args(
            root,
            tuple(home_allow or ()),
            hidden_paths,
            empty_file=empty_file_path(state_dir),
            writable_roots=[dest for _src, dest in covering],
            listing=listing,
            binds=[mount for mount in mounts if in_home(mount[2])],
            late_args=[token for mount in late if in_home(mount[-1]) for token in mount],
        )
        # The overlays for hidden paths outside HOME (a HERMES_HOME kept
        # elsewhere, a credential directory that is a symlink out of HOME).
        argv += sensitive_overlay_args(hidden_out, state_dir)
        for mount in late:
            if not in_home(mount[-1]):
                argv += mount

    argv += ["--chdir", tracked_cwd, "--"]
    return argv


PROBE_ARGS: tuple[str, ...] = ("--unshare-user", "--ro-bind", "/", "/", "true")
PROBE_TIMEOUT_SECONDS = 5
INSTALL_HINT = (
    "Install the bubblewrap package (apt, dnf or pacman: bubblewrap) and "
    "util-linux (for prlimit), and make sure unprivileged user namespaces are "
    "allowed on this host, then retry; or set terminal.backend to another backend."
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
    prlimit(1), which applies the resource limits, must be on PATH too.
    """
    path = shutil.which("bwrap")
    if path is None:
        return None, "bubblewrap (bwrap) is not on PATH"
    if shutil.which("prlimit") is None:
        return path, "prlimit (util-linux) is not on PATH"
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


# The process limit the probe sets inside its sandbox: room for bwrap's
# init, the shell and the one child the shell forks, and far below what a
# uid runs on any host that has a Hermes process on it.
PROCESS_LIMIT_PROBE_VALUE = 4

# Whether RLIMIT_NPROC set inside a sandbox counts that sandbox alone on
# this kernel. Probed once per process; None until then.
_process_limit_scoped: bool | None = None


def run_process_limit_probe(bwrap_path: str, prlimit_path: str) -> bool:
    """True when a process limit set inside a sandbox is counted for that sandbox alone.

    Linux counts RLIMIT_NPROC per user namespace from 5.14 on (with
    enforcement fixes up to 5.17). Before that the count is every thread
    of the uid on the host, and a limit set inside a sandbox would stop
    every fork. The kernel exposes no flag for this, and a version check
    cannot see a distribution backport, so the probe measures it: inside
    a sandbox it sets a limit of PROCESS_LIMIT_PROBE_VALUE, well below the
    host count, and has the shell fork once. The fork passes only where
    the count is the sandbox's own. Any other outcome counts as not
    scoped, so a doubtful kernel gets no limit instead of a broken sandbox.
    """
    argv = [
        bwrap_path, "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", "--",
        prlimit_path, f"--nproc={PROCESS_LIMIT_PROBE_VALUE}", "sh", "-c", "( : ) && exit 0",
    ]
    try:
        # The exit status is the whole answer: no output is decoded.
        result = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def process_limit_is_scoped(bwrap_path: str, prlimit_path: str) -> bool:
    """run_process_limit_probe, run once per process; warns once when the limit cannot be applied."""
    global _process_limit_scoped
    if _process_limit_scoped is None:
        _process_limit_scoped = bool(run_process_limit_probe(bwrap_path, prlimit_path))
        if not _process_limit_scoped:
            logger.warning(
                "bubblewrap: this kernel does not count the process limit per sandbox (that needs "
                "Linux 5.14 or later), so terminal.bubblewrap_max_procs is not applied. The memory "
                "and CPU limits still apply."
            )
    return _process_limit_scoped


# The wrapper's own processes that exist beside the command for its whole
# run: bwrap's pid-1 init and the shell that runs the command. They count
# against the same limit, so they come on top of max_procs.
SANDBOX_BASE_PROCESSES = 2


def rlimit_values(config: BubblewrapConfig) -> dict[int, int]:
    """The rlimits a spawn gets from the three terminal.bubblewrap_* keys.

    A key at 0 leaves its limit out. The memory and CPU limits are per
    process and are set in front of bwrap. RLIMIT_NPROC is different: the
    kernel counts it per user and per user namespace, and checks a
    namespace's creator limit against the count of the namespace above. A
    value set in front of bwrap is therefore compared with every thread
    the uid runs on the host, which makes it either useless as a ceiling
    or fatal to the spawn. Set inside the sandbox, after bwrap has made
    its user namespace, it is compared with the processes of that sandbox
    alone, so max_procs is what the command may run, whatever the host
    does. A process in the sandbox cannot raise it (the hard limit is set
    too), and a nested user namespace stays inside it.
    """
    limits: dict[int, int] = {}
    if config.memory_mb:
        limits[resource.RLIMIT_AS] = config.memory_mb * 1024 * 1024
    if config.cpu_seconds:
        limits[resource.RLIMIT_CPU] = config.cpu_seconds
    if config.max_procs:
        limits[resource.RLIMIT_NPROC] = config.max_procs + SANDBOX_BASE_PROCESSES
    return limits


PRLIMIT_FLAGS: dict[int, str] = {
    resource.RLIMIT_AS: "--as", resource.RLIMIT_CPU: "--cpu", resource.RLIMIT_NPROC: "--nproc",
}


def prlimit_args(limits: Mapping[int, int], prlimit_path: str) -> list[str]:
    """The prlimit(1) argv prefix applying *limits* as soft and hard, or [] when empty.

    prlimit sets each limit on itself and execs the command that follows,
    so the limits are in place before bwrap starts and no Python code runs
    between fork and exec. It stops parsing at the first non-option
    argument, so the bwrap argv follows without a separator. A value above
    the inherited hard limit is clamped to it: raising a hard limit needs
    CAP_SYS_RESOURCE and would fail the spawn.
    """
    if not limits:
        return []
    argv = [prlimit_path]
    for res, value in limits.items():
        _, hard = resource.getrlimit(res)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        argv.append(f"{PRLIMIT_FLAGS[res]}={value}")
    return argv


_MOUNT_ARITY: dict[str, int] = {
    "--bind": 2, "--ro-bind": 2, "--bind-try": 2, "--ro-bind-try": 2,
    "--tmpfs": 1, "--dev": 1, "--proc": 1,
}
# Directives that take path operands but place no directory tree: their
# operands are stepped over so they are not read as flags.
_SKIPPED_ARITY: dict[str, int] = {"--symlink": 2, "--remount-ro": 1}


def masked_inside(argv: Sequence[str], path: str) -> bool:
    """True when host directory *path* is hidden inside the sandbox *argv* builds.

    Mounts stack in argv order, so the last directive whose destination is
    *path* or an ancestor of it decides what shows there: a mount root is
    visible (bwrap creates the mount point), a fresh --tmpfs, --dev or
    --proc hides everything below its root, and a bind shows what its
    source holds at the same relative path (the root bind of / shows the
    host itself). A mount below *path* makes *path* exist. A -try bind
    whose source is missing is skipped, as bwrap skips it. Reads only the host presence of directories, as the pins do.
    """
    visible = True
    i = 0
    while i < len(argv):
        arity = _MOUNT_ARITY.get(argv[i])
        if arity is None:
            i += 1 + _SKIPPED_ARITY.get(argv[i], 0)
            continue
        operands = argv[i + 1:i + 1 + arity]
        i += 1 + arity
        dest = operands[-1]
        if dest == path:
            visible = True
        elif _is_within(dest, path):
            # A mount below *path* makes bwrap create every directory on
            # the way to it, *path* included.
            if arity == 2 and argv[i - 1 - arity].endswith("-try") and not os.path.exists(operands[0]):
                continue
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
    kill come from LocalEnvironment. This class adds the argv prefix (the
    prlimit limits, then bwrap), a per-instance state dir for the shell
    snapshot and cwd file, the empty file bound over sensitive files, and
    their removal on cleanup.
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
        self._prlimit_path = shutil.which("prlimit") or "prlimit"
        # Probed once per process, and only when the limit is asked for.
        self._process_limit_scoped = bool(self._config.max_procs) and process_limit_is_scoped(
            self._bwrap_path, self._prlimit_path,
        )
        # The OS user's home anchors the sensitive set even when this process
        # runs with HOME pointed at the profile home. Every mount path is
        # taken through realpath: bwrap resolves a mount destination inside
        # the sandbox root, where an absolute symlink points nowhere.
        self._home = os.path.realpath(get_real_home() or os.path.expanduser("~"))
        self._hermes_home = os.path.realpath(str(get_hermes_home()))
        # Resolved once and kept for the life of the environment: the set
        # never follows a symlink swapped in later.
        self._operator_hidden = operator_hidden_paths(self._home, self._config.hide)
        self._hidden_paths = sensitive_paths(self._home, self._hermes_home, self._operator_hidden)
        # HOME is default-deny for dot entries. What a sandbox may see is
        # fixed here too: a command cannot widen it by changing PATH.
        self._home_root = bubblewrap_home.resolve_home_root(self._home)
        self._home_allow: tuple[str, ...] = ()
        if self._home_root is not None:
            self._home_allow = bubblewrap_home.resolve_allowlist(
                self._home_root, os.environ.get("PATH", ""), self._config.home_allow,
                denied_names=bubblewrap_home.denied_home_names(self._home_root),
                denied_paths=self._hidden_paths,
            )
        # The operator binds are filtered, their sources expanded and their
        # destinations resolved once here, like the hidden set and the cwd,
        # so the mount paths are fixed for the life of the environment
        # and a dropped bind warns once; the builder's own filter
        # then drops nothing.
        self._config = replace(
            self._config,
            binds=tuple(resolve_bind_dests(expand_bind_srcs(filter_binds(self._config.binds, self._hidden_paths)))),
        )
        # Resolved once, like every other mount path. The host directory,
        # which every Hermes process shares, is never bound: the sandbox
        # gets an empty view of that path and a directory of this
        # environment inside it. get_scratch_dir makes the directory;
        # pruning stays with the process that owns the home.
        self._scratch_dir: str = os.path.realpath(str(get_scratch_dir(self._hermes_home, prune=False)))
        # The spellings of that path a process environment can carry: the
        # real one, the one under HERMES_HOME as it is named (a symlinked
        # home), and whatever the temp variables of this process hold now.
        spellings = [
            self._scratch_dir,
            os.path.join(str(get_hermes_home()), "cache", "scratch"),
            *(os.environ.get(name, "") for name in (*SCRATCH_TMP_ENV_VARS, SCRATCH_DIR_MARKER_ENV)),
        ]
        self._scratch_names: tuple[str, ...] = tuple(dict.fromkeys(
            spelling for spelling in spellings if spelling and os.path.realpath(spelling) == self._scratch_dir
        ))
        self._staged_roots = staged_data_roots()
        # Archive files of oversized tool results this environment was
        # handed (expose_spillover_file), newest last: the path of each
        # and the copy that is bound there.
        self._spillover_files: dict[str, str] = {}
        # A terminal.bubblewrap_hide entry at or above one of these roots
        # wins: the root is not bound back.
        covered = [root for root in self._staged_roots if any(_is_within(root, path) for path in self._operator_hidden)]
        self._staged_roots = tuple(root for root in self._staged_roots if root not in covered)
        self._check_profile_home()
        # The mount paths are fixed here; only --chdir follows the tracked cwd.
        self._initial_cwd = os.path.realpath(_resolve_local_initial_cwd(cwd))
        self._check_initial_cwd()
        self._check_bind_sources()
        self._check_absent_denied_paths()
        self._check_writable_dot_links()
        sandbox_root = os.path.realpath(get_sandbox_dir())
        # Set by _check_sandbox_root when no hidden path covers the sandbox dir.
        self._sandbox_overlays: tuple[str, ...] = ()
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
        self._scratch_view = scratch_view_path(self._state_dir)
        self._scratch_private = private_scratch_path(self._scratch_dir, self._state_dir)
        os.mkdir(self._scratch_private, mode=0o700)
        # The view holds the mount point of the private dir and nothing else.
        os.makedirs(os.path.join(self._scratch_view, os.path.basename(self._scratch_private)), mode=0o700)
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
        Only the plain directory
        HERMES_HOME/home is exempt from the lies-under test: it sits under
        HERMES_HOME, and under the standard profile layout under the hidden
        default HOME/.hermes as well, and binding it back is the
        designed carve-out. A symlinked profile home must resolve outside
        every hidden path: a link to another directory under HERMES_HOME,
        another profile at HOME/.hermes/profiles/<name> in the default
        layout included, is a second view of the hidden tree and is
        refused. A link to a clean
        directory outside every hidden path stays allowed.
        """
        if self._config.home_mode not in PROFILE_HOME_MODES:
            return
        link = os.path.join(self._hermes_home, "home")
        profile_home = os.path.realpath(link)
        plain = not os.path.islink(link)
        for hidden in self._hidden_paths:
            if _is_within(hidden, profile_home):
                relation = f"contains {hidden}"
            elif not plain and _is_within(profile_home, hidden):
                relation = f"lies under {hidden}"
            else:
                continue
            raise ValueError(
                f"terminal.home_mode={self._config.home_mode} binds the profile home "
                f"{link} read-write inside the sandbox with the bubblewrap backend, but "
                f"it resolves to {profile_home}, which {relation}, a path the backend "
                "hides: the bind would show it again. Make HERMES_HOME/home a plain "
                "directory, or a link to a directory outside every hidden path (HOME's "
                "hidden dotfiles, the default HOME/.hermes and all of HERMES_HOME), or "
                "set terminal.home_mode to auto or real."
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
                "bubblewrap cwd %s covers the home directory: every existing non-dot "
                "entry of it is writable inside the sandbox, and no new entry can be made "
                "at the top of it (its dot entries are read-only or hidden). Set "
                "terminal.cwd to a project or scratch directory for a smaller writable set.",
                self._initial_cwd,
            )

    def _check_writable_dot_links(self) -> None:
        """Refuse a dot symlink of HOME that a command could redirect or rewrite.

        A dot entry is read-only or hidden in the sandbox, and a symlink
        at the top of HOME cannot be replaced there. What it leads to is
        another matter: ~/.bashrc linked into a dotfiles directory that the
        cwd makes writable can be rewritten through that directory, and the
        host reads it at the next login, long after the sandbox is gone.
        The same holds for any link or directory on the way: a command
        that can replace one of them decides what the host reads.

        The backend does not try to hold such a chain in place with
        mounts. It walks the chain of each dot symlink at the top of HOME
        (dot_link_entries) and refuses to start when an entry on it lies
        in the writable cwd, in a read-write operator bind or in the
        profile home that a profile home mode binds read-write. Not counted:
        an entry at or under a hidden path, which no command can reach; a
        directory on the way that holds a hidden path, which is pinned as
        a mount point and cannot be renamed or replaced; and, under a
        source that covers HOME, the dot entries of HOME, which the layout
        makes read-only or hides.
        """
        root = self._home_root
        if root is None:
            return
        sources: list[str] = [self._initial_cwd] if resolve_profile(self._config.profile).writable_cwd else []
        sources += [os.path.realpath(bind.src) for bind in self._config.binds if not bind.readonly]
        # Under a profile home mode HERMES_HOME/home is bound read-write on
        # top of the HERMES_HOME overlay, under every profile: what lies in
        # it is writable although it is under a hidden path.
        profile_home: str | None = None
        if self._config.home_mode in PROFILE_HOME_MODES:
            candidate = os.path.join(self._hermes_home, "home")
            if os.path.isdir(candidate):
                profile_home = os.path.realpath(candidate)
        if not sources and profile_home is None:
            return
        try:
            names = sorted(
                name for name in os.listdir(root)
                if name.startswith(".") and os.path.islink(os.path.join(root, name))
            )
        except OSError:
            return

        def writable(entry: str, last: bool) -> bool:
            if profile_home is not None and _is_within(entry, profile_home) and (last or entry != profile_home):
                return True
            if any(_is_within(entry, hidden) for hidden in self._hidden_paths):
                return False
            if not last and any(_is_within(hidden, entry) for hidden in self._hidden_paths):
                return False
            for source in sources:
                if not _is_within(entry, source):
                    continue
                if entry == source and not last:
                    # The top of a bind is a mount point: it cannot be replaced.
                    continue
                if _is_within(root, source) and _is_within(entry, root):
                    # A source that covers HOME makes only its non-dot entries writable.
                    if entry == root or os.path.relpath(entry, root).split(os.sep)[0].startswith("."):
                        continue
                return True
            return False

        def exposed(name: str) -> bool:
            entries = dot_link_entries(root, name)
            return any(writable(entry, index == len(entries) - 1) for index, entry in enumerate(entries))

        refused = [name for name in names if exposed(name)]
        if refused:
            raise ValueError(
                f"{', '.join(refused)} in the home directory {'is a symlink' if len(refused) == 1 else 'are symlinks'} "
                "that a command in the bubblewrap sandbox could redirect or rewrite: the link leads "
                "through, or ends in, a place that is writable inside the sandbox (terminal.cwd "
                f"{self._initial_cwd}, a read-write terminal.bubblewrap_binds source or the profile "
                "home HERMES_HOME/home). The host "
                "reads such a file after the sandbox is gone, at the next login for a shell startup "
                "file. Set terminal.cwd to a project directory that does not hold the link target, "
                "or make the bind read-only."
            )

    def _check_absent_denied_paths(self) -> None:
        """Refuse a writable bind under which a command could create a credential path.

        A denied path that exists gets an overlay. One that does not exist
        cannot be mounted over, and a placeholder made for it would land on
        the host through the same writable bind. The top of HOME and the
        default-deny directories are read-only tmpfs layers, so a name
        there cannot be created whatever the cwd is. That leaves a denied
        path inside a directory the sandbox may write to: the cwd under a
        writable profile, or a read-write operator bind. A command there
        could create ~/.config/gh or a token file and the host would use
        it later, so the environment does not start.
        """
        root = self._home_root
        if root is None:
            return
        writable: list[tuple[str, str]] = []
        if resolve_profile(self._config.profile).writable_cwd:
            writable.append(("terminal.cwd", self._initial_cwd))
        writable += [
            (f"the terminal.bubblewrap_binds entry {bind.src}", bind.dest)
            for bind in self._config.binds if not bind.readonly
        ]
        for label, bound in writable:
            for path in (*bubblewrap_home.denied_home_paths(self._home), *self._operator_hidden):
                if path == bound or not _is_within(path, bound) or os.path.lexists(path):
                    continue
                if _is_within(root, bound) and _is_within(path, root):
                    # The bind covers HOME. Under HOME only an existing
                    # non-dot entry is writable through it.
                    top = os.path.relpath(path, root).split(os.sep)
                    if len(top) == 1 or top[0].startswith(".") or not os.path.isdir(os.path.join(root, top[0])):
                        continue
                raise ValueError(
                    f"{label} makes {bound} writable inside the bubblewrap sandbox, and the "
                    f"credential path {path} does not exist on the host, so the backend cannot "
                    "hide it and a command could create it. Narrow the bind to the "
                    "subdirectory you need (for a cache, that one cache directory), or use "
                    "a project directory as terminal.cwd."
                )

    def _check_bind_sources(self) -> None:
        """Refuse a read-write bind whose source a sandbox could swap.

        bwrap resolves a bind source on the host at every spawn. When a
        component of the source path sits inside the writable set (the cwd
        under a writable profile, another read-write bind's source, the
        profile home under a profile home_mode) and is a symlink or a
        directory a command can rename, the command replaces it and
        chooses the next spawn's mount source, gaining read-write access
        to any host directory without a hidden path below it. A source
        bound at its own real path is a
        mount point inside, and the kernel refuses to rename or move a
        mount point from any path alias (EBUSY), so directly under a
        writable root the source is fixed: its parent is the root itself,
        a mount point (the cwd, a self-bound source) or a read-only
        directory entry (a source bound elsewhere, a symlinked profile
        home). One level deeper the parent is a plain writable directory:
        renamed, recreated and given a relative symlink at the source
        path, it steers the next spawn's mount. A read-only bind shows
        nothing the root bind does not.
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
                "way: bound at its own path the source is a mount point, which no "
                "command can rename or move from any path; or make it read-only."
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
        does, read-write, so a sandbox dir under it is refused first,
        whether the profile home is a directory under HERMES_HOME or a
        symlink to a directory outside every hidden path. A read-only
        directory covers the scratch path, so a sandbox dir under the
        scratch path is refused too: the state dir could not be bound at
        its own path there. The staged data roots land
        on the overlay too, but read-only. An operator
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
        if _is_within(sandbox_root, self._scratch_dir):
            raise ValueError(
                f"terminal.sandbox_dir {sandbox_root} lies inside the Hermes scratch dir "
                f"{self._scratch_dir}. The bubblewrap backend binds a directory of its own over "
                "that path, so the state dir could not be bound there. Set terminal.sandbox_dir "
                "to a directory outside the scratch dir; the default HERMES_HOME/sandboxes is "
                "covered by the HERMES_HOME overlay."
            )
        writable_profile = resolve_profile(self._config.profile).writable_cwd
        if any(_is_within(sandbox_root, hidden) for hidden in self._hidden_paths):
            return
        # No hidden path covers the sandbox dir, so it would show through
        # the read-only root with the state dirs of every other
        # environment in it. It gets an overlay of its own. That overlay
        # would also cover whatever else lies inside the sandbox dir.
        inside = [
            path for path in (self._initial_cwd, os.path.realpath(self._home), os.path.realpath(self._hermes_home),
                              *(bind.dest for bind in self._config.binds))
            if _is_within(path, sandbox_root)
        ]
        if inside:
            raise ValueError(
                f"terminal.sandbox_dir {sandbox_root} contains {', '.join(inside)}. The bubblewrap "
                "backend hides the sandbox dir from commands, since it holds the state of every "
                "sandbox, and that would hide those paths too. Set terminal.sandbox_dir to a "
                "directory of its own, or leave it at the default HERMES_HOME/sandboxes."
            )
        overlays = [sandbox_root]
        for bind in self._config.binds:
            src = os.path.realpath(bind.src)
            if src != bind.dest and _is_within(sandbox_root, src):
                # The same directory, seen through an operator bind.
                overlays.append(os.path.normpath(os.path.join(bind.dest, os.path.relpath(sandbox_root, src))))
        self._sandbox_overlays = tuple(dict.fromkeys(overlays))
        writable: list[str] = [self._initial_cwd] if writable_profile else []
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

    def expose_spillover_file(self, path: str) -> bool:
        """Show the commands of this environment one archive of an oversized tool result.

        Called on the host by tools.tool_result_storage when it hands the
        model the path of an archive. The directory of those archives is
        not bound, since it holds the tool output of every session of the
        profile. What a command sees at the path is a copy taken now, bound
        read-only in every later spawn: the name of an archive comes from
        the tool call id alone, ids repeat between sessions, and a later
        write under the same name belongs to whoever made it. Only a
        regular file directly in that directory is accepted, and a
        terminal.bubblewrap_hide entry over it wins. The newest
        SPILLOVER_BIND_MAX files are kept. Returns whether the file is
        bound.
        """
        try:
            from tools.tool_result_storage import get_spillover_dir

            spillover = os.path.realpath(str(get_spillover_dir()))
        except Exception:
            return False
        real = os.path.realpath(path)
        if os.path.dirname(real) != spillover or os.path.islink(path) or not os.path.isfile(real):
            return False
        if any(_is_within(real, hidden) for hidden in self._operator_hidden):
            return False
        copies = archive_copies_path(self._state_dir)
        copy = os.path.join(copies, os.path.basename(real))
        try:
            os.makedirs(copies, mode=0o700, exist_ok=True)
            partial = f"{copy}.{uuid.uuid4().hex[:8]}.part"
            shutil.copyfile(real, partial)
            os.replace(partial, copy)
        except OSError:
            logger.warning("bubblewrap: could not copy the tool result archive %s; it stays hidden", real, exc_info=True)
            return False
        self._spillover_files.pop(real, None)
        self._spillover_files[real] = copy
        while len(self._spillover_files) > SPILLOVER_BIND_MAX:
            dropped = self._spillover_files.pop(next(iter(self._spillover_files)))
            try:
                os.unlink(dropped)
            except OSError:
                pass
        return True

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
            home_root=self._home_root,
            home_allow=self._home_allow,
            scratch_dir=self._scratch_dir,
            scratch_view=self._scratch_view,
            scratch_private=self._scratch_private,
            staged_roots=self._staged_roots,
            staged_files=tuple((copy, path) for path, copy in self._spillover_files.items()),
            sandbox_overlays=self._sandbox_overlays,
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

    def _refresh_private_scratch(self) -> None:
        """Keep the scratch dir of this environment in place for the next spawn.

        It is an entry of the Hermes scratch dir, whose idle entries any
        Hermes process prunes. A spawn marks it as used, and makes it
        again when a prune took it while the environment sat idle: its
        bind has no source otherwise and every command would fail. A
        symlink at that path is left alone, so the bind fails rather than
        follow it.
        """
        path = self._scratch_private
        try:
            if os.path.islink(path):
                return
            os.makedirs(path, mode=0o700, exist_ok=True)
            os.utime(path)
        except OSError:
            logger.debug("bubblewrap: could not refresh the scratch dir %s", path, exc_info=True)

    def _wrap_popen_args(self, args: list[str]) -> list[str]:
        self._refresh_private_scratch()
        return self._prlimit_prefix() + self._bwrap_prefix(self.cwd) + self._process_limit_prefix() + list(args)

    def _wrap_command(self, command: str, cwd: str) -> str:
        # --unsetenv strips the socket variables from the environment bwrap
        # receives, but the login bootstrap sources the shell init files
        # into the snapshot afterwards, and a 1Password or gpg-agent setup
        # exports SSH_AUTH_SOCK from ~/.bashrc. Unset them in front of the
        # command: that runs after the snapshot is sourced, and the export
        # dump that follows the command then omits them too.
        # Hermes points TMPDIR, TMP and TEMP of its processes at the shared
        # scratch dir, which is read-only and empty here. A variable that
        # names it is pointed at the scratch dir of this environment
        # instead. One the operator set to another place is left alone, as
        # Hermes leaves it alone.
        private = shlex.quote(self._scratch_private)
        shared = "|".join(shlex.quote(path) for path in self._scratch_names)
        retarget = "".join(
            f'case "${{{name}-}}" in {shared}) export {name}={private};; esac; ' for name in SCRATCH_TMP_ENV_VARS
        )
        return super()._wrap_command(f"unset {' '.join(HOST_SOCKET_VARS)}; {retarget}{command}", cwd)

    def _prlimit_prefix(self) -> list[str]:
        """prlimit in front of bwrap: the per-process memory and CPU limits."""
        limits = rlimit_values(self._config)
        limits.pop(resource.RLIMIT_NPROC, None)
        return prlimit_args(limits, self._prlimit_path)

    def _process_limit_prefix(self) -> list[str]:
        """prlimit after bwrap's separator: the process limit, set inside the sandbox."""
        limit = rlimit_values(self._config).get(resource.RLIMIT_NPROC)
        if not limit or not self._process_limit_scoped:
            return []
        return prlimit_args({resource.RLIMIT_NPROC: limit}, self._prlimit_path)

    def _live_sandbox_pids(self) -> list[int]:
        """PIDs of this instance's bwrap wrappers still running.

        A wrapper is a direct child of this process whose argv is the bwrap
        path with this instance's state dir bound; the state dir is unique
        per instance and fixed at construction, so nothing from inside a
        sandbox can forge it. Zombies are left for the thread that spawned
        them to reap. A child that has not exec'd bwrap yet (between fork
        and exec, or still running prlimit) shows another cmdline and is
        missed; it then fails to bind the removed state dir and exits,
        exposing nothing.
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
                os.killpg(os.getpgid(pid), signal.SIGKILL)  # windows-footgun: ok — bubblewrap is a Linux-only backend
            except (ProcessLookupError, PermissionError):
                continue
        deadline = time.monotonic() + wait
        while self._live_sandbox_pids() and time.monotonic() < deadline:
            time.sleep(0.05)

    def _remove_state(self) -> None:
        shutil.rmtree(self._state_dir, ignore_errors=True)
        shutil.rmtree(scratch_view_path(self._state_dir), ignore_errors=True)
        shutil.rmtree(archive_copies_path(self._state_dir), ignore_errors=True)
        scratch_private = getattr(self, "_scratch_private", None)
        if scratch_private:
            shutil.rmtree(scratch_private, ignore_errors=True)
        try:
            os.unlink(self._empty_file)
        except OSError:
            pass

    def cleanup(self):
        self._kill_live_sandboxes()
        super().cleanup()
        self._remove_state()
