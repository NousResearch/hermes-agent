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
import shutil
import uuid
from dataclasses import dataclass
from typing import Mapping

from hermes_constants import get_hermes_home
from tools.environments.base import get_sandbox_dir
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


class BubblewrapEnvironment(LocalEnvironment):
    """LocalEnvironment whose every spawn runs inside a bwrap sandbox.

    Bash resolution, the run env, missing-cwd recovery and process-group
    kill come from LocalEnvironment. This class adds the argv prefix, a
    per-instance state dir for the shell snapshot and cwd file, the empty
    file bound over sensitive files, and their removal on cleanup.
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
        # Reject an unknown profile before anything is created on disk.
        resolve_profile(self._config.profile)
        self._home = os.path.expanduser("~")
        self._hermes_home = str(get_hermes_home())
        self._bwrap_path = shutil.which("bwrap") or "bwrap"
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

    def cleanup(self):
        super().cleanup()
        shutil.rmtree(self._state_dir, ignore_errors=True)
        try:
            os.unlink(self._empty_file)
        except OSError:
            pass
