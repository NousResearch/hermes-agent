"""Local execution environment — spawn-per-call with session snapshot."""

import contextlib
import logging
import ntpath
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from hermes_constants import get_process_hermes_home
from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _pipe_stdin
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.local_env_policy import (
    _ALWAYS_STRIP_KEYS, _HERMES_PROVIDER_ENV_BLOCKLIST, _HERMES_PROVIDER_ENV_FORCE_PREFIX,
    _is_hermes_internal_secret, _is_terminal_first_party_env,
    _matches_terminal_first_party_prefix, _plugin_terminal_env_strip_keys)
from tools.environments.local_gitbash_probe import (
    _bash_probe_details_cache, _bash_starts, _git_bash_aslr_help,
    _looks_like_msys_spawn_failure, _mandatory_aslr_enabled)
from tools.environments.local_pythonpath import (
    _build_hermes_repo_root_aliases, _strip_hermes_owned_pythonpath_and_runtime_markers)


_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)

# --- Terminal temp-cache pruning ---
# get_temp_dir() defaults to HERMES_HOME/cache/terminal (real storage, not tmpfs), so
# stale artifacts don't vanish on reboot: the gateway housekeeping loop prunes hourly
# and a once-per-process sweep covers CLI-only installs.
TERMINAL_TEMP_MAX_AGE_HOURS = 72
_terminal_temp_prune_lock = threading.Lock()
_terminal_temp_pruned_once = False
# Background artifacts come in triplets (hermes_bg_<id>.log/.pid/.exit). A live
# server's .pid never changes mtime while its .log does, so age is judged per
# GROUP (newest mtime sharing a stem) to keep pid/exit files of live sessions.
_BG_GROUP_RE = re.compile(r"^(hermes_bg_[A-Za-z0-9_-]+)\.(log|pid|exit)$")


def _default_terminal_temp_dir() -> "Path | None":
    """Return HERMES_HOME/cache/terminal, or None if unresolvable."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "cache" / "terminal"
    except Exception:
        return None


def cleanup_terminal_temp_cache(max_age_hours: int = TERMINAL_TEMP_MAX_AGE_HOURS) -> int:
    """Delete session temp artifacts older than *max_age_hours*; return count.
    Only the managed default dir is pruned — never a user-pointed ``terminal.temp_dir``."""
    root = _default_terminal_temp_dir()
    if root is None:
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0

    mtimes: dict[Path, float] = {}
    group_newest: dict[str, float] = {}
    for f in entries:
        try:
            mtimes[f] = mt = f.stat().st_mtime
        except OSError:
            continue
        if m := _BG_GROUP_RE.match(f.name):
            group_newest[m.group(1)] = max(group_newest.get(m.group(1), 0.0), mt)

    removed = 0
    for f, mt in mtimes.items():
        m = _BG_GROUP_RE.match(f.name)
        if (group_newest[m.group(1)] if m else mt) >= cutoff:
            continue
        try:
            shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _prune_terminal_temp_once() -> None:
    """Best-effort prune, at most once per process (CLI-only installs)."""
    global _terminal_temp_pruned_once
    with _terminal_temp_prune_lock:
        if _terminal_temp_pruned_once:
            return
        _terminal_temp_pruned_once = True
    try:
        cleanup_terminal_temp_cache()
    except Exception as exc:
        logger.debug("Terminal temp prune failed: %s", exc)


# --- Windows / MSYS path translation ---
def _msys_to_windows_path(cwd: str) -> str:
    """``/c/Users/x`` / ``/cygdrive/c/..`` / ``/mnt/c/..`` -> native ``C:\\Users\\x`` so
    ``isdir``/``Popen(cwd=)`` find it. No-op off Windows, for empty input and for
    multi-segment POSIX paths like ``/home/x``; idempotent on native paths."""
    m = _IS_WINDOWS and cwd and re.match(r'^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])(/.*)?$', cwd)
    if not m:
        return cwd
    tail = (m.group(2) or "").replace('/', '\\')
    return f"{m.group(1).upper()}:{tail or chr(92)}"  # chr(92) = backslash


def _resolve_local_initial_cwd(cwd: str) -> str:
    """Resolve the initial cwd to an absolute host path. A relative ``TERMINAL_CWD``
    naming the launch directory would otherwise make the wrapper ``cd`` *inside*
    the project; anchor it once so ``Popen(cwd=)`` and the in-shell ``cd`` agree."""
    expanded = os.path.expanduser(cwd) if cwd else os.getcwd()
    if _IS_WINDOWS:
        expanded = _msys_to_windows_path(expanded)
        # ntpath explicitly: with _IS_WINDOWS patched on a POSIX host,
        # os.path.isabs would reject ``C:\Users\x`` and mangle it below.
        if ntpath.isabs(expanded):
            return expanded
    if os.path.isabs(expanded):
        return expanded
    candidate = os.path.abspath(expanded)
    current = os.getcwd()
    # Relative name matching the tail of the current dir: use the current dir.
    if not os.path.isdir(candidate):
        wanted, have = Path(expanded).parts, Path(current).parts
        if wanted and len(wanted) <= len(have) and have[-len(wanted):] == wanted:
            return current
    return candidate


def _windows_to_msys_path(cwd: str) -> str:
    """Native ``C:\\Users\\x`` -> Git Bash ``/c/Users/x`` so ``builtin cd`` resolves
    it. No-op off Windows / for non-drive paths."""
    m = _IS_WINDOWS and cwd and re.match(r'^([a-zA-Z]):[\\/]*(.*)$', cwd)
    if not m:
        return cwd
    tail = (m.group(2) or "").replace('\\', '/').lstrip('/')
    return f"/{m.group(1).lower()}/{tail}"


def _bash_safe_path(path: str) -> str:
    """*path* safe to embed in a Git Bash script: ``C:\\Users\\x`` / ``C:/Users/x``
    become ``/c/Users/x`` (MSYS argument conversion mangles ``C:/`` forms) and
    leftover backslashes are normalized so bash does not eat ``\\U``. No-op off Windows."""
    return _windows_to_msys_path(path).replace("\\", "/") if _IS_WINDOWS and path else path


def _quote_bash_path(path: str) -> str:
    """Quote *path* for safe interpolation into a Git Bash script on Windows."""
    import shlex
    return shlex.quote(_bash_safe_path(path))


def _cwd_usable(path: str) -> bool:
    """True when *path* is a directory this process can actually chdir into
    (``isdir`` alone passes ``/root`` for a non-root user; ``Popen(cwd=)`` then dies)."""
    return os.path.isdir(path) and os.access(path, os.X_OK)


def _resolve_safe_cwd(cwd: str) -> str:
    """``cwd`` if enterable, else the nearest usable ancestor, else
    ``tempfile.gettempdir()``. MSYS paths are normalized first on Windows so a valid
    ``pwd -P`` result is not rejected. Lets ``_run_bash`` recover from a deleted or
    inaccessible cwd instead of ``Popen`` raising and wedging every later call.

    Used by ``_run_bash`` to recover when the configured cwd is gone — most commonly because a previous tool
    call deleted its own working directory (issue #17558) — or inaccessible to this user, e.g. ``/root``
    leaking from a root-launched CLI session into a non-root gateway's cron jobs (issue #65583). Without
    this guard, ``subprocess.Popen(..., cwd=...)`` raises ``FileNotFoundError``/``PermissionError`` before
    bash starts, wedging every subsequent terminal call until the gateway restarts.
    """
    cwd = _msys_to_windows_path(cwd)
    if cwd and _cwd_usable(cwd):
        return cwd
    if cwd and os.path.isdir(cwd):
        logger.warning(
            "Configured terminal cwd %r exists but is not accessible to "
            "this user (uid=%s) — falling back to the nearest usable "
            "directory. If this is a gateway/cron process, check for "
            "root-owned paths leaking into terminal.cwd / TERMINAL_CWD "
            "(#65583).",
            cwd, getattr(os, "getuid", lambda: "?")())
    parent = os.path.dirname(cwd) if cwd else ""
    while parent and not _cwd_usable(parent):
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            return tempfile.gettempdir()  # filesystem root itself is unusable
        parent = next_parent
    return parent or tempfile.gettempdir()


# --- Child-process environment construction ---
def _apply_profile_home(env: dict) -> None:
    """Bridge the context-local HERMES_HOME override, then the subprocess HOME contract."""
    from hermes_constants import apply_subprocess_home_env, get_hermes_home_override
    try:
        if value := get_hermes_home_override():
            env["HERMES_HOME"] = value
    except Exception:
        pass
    apply_subprocess_home_env(env)


def _inject_session_context_env(env: dict) -> None:
    """Bridge gateway session ContextVars (HERMES_SESSION_*) into a child env.
    Cross-session leak guard: the vars' last-writer-wins ``os.environ`` mirror may
    belong to another turn on a concurrent multi-session host, so once the session
    context is engaged ContextVars are authoritative — a bound value (incl. "") wins
    and an _UNSET var is STRIPPED, not inherited. An unengaged CLI keeps the mirror."""
    try:
        from gateway.session_context import _UNSET, _VAR_MAP, session_context_engaged
    except Exception:
        return
    _engaged = session_context_engaged()
    for var_name, var in _VAR_MAP.items():
        value = var.get()
        if value is not _UNSET:
            env[var_name] = "" if value is None else str(value)
        elif _engaged:
            env.pop(var_name, None)


def _filter_secret_env(
    items: Mapping[str, str], out: dict, *, unwrap_force: bool,
    plugin_strip: frozenset = frozenset()) -> None:
    """Copy *items* into *out*, dropping Hermes-managed secrets. ``_HERMES_FORCE_<NAME>``
    unwraps to ``NAME`` when ``unwrap_force`` (caller extras / terminal env), else is
    dropped. Blocklisted names survive only via env_passthrough registration or as
    context-entitled first-party ``BUZZ_*`` vars; the latter are used directly, never
    scope-resolved (UnscopedSecretError under multiplex)."""
    try:
        from tools.env_passthrough import is_env_passthrough, resolve_passthrough_value
    except Exception:
        is_env_passthrough, resolve_passthrough_value = (lambda _: False), (lambda _n, fb: fb)
    for key, value in items.items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            if not unwrap_force:
                continue
            key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if not _is_hermes_internal_secret(key):
                out[key] = value
            continue
        if _is_hermes_internal_secret(key) or key in plugin_strip:
            continue
        first_party = _is_terminal_first_party_env(key)
        passthrough = is_env_passthrough(key)
        if key in _HERMES_PROVIDER_ENV_BLOCKLIST and not (passthrough or first_party):
            continue
        if passthrough and not first_party:
            value = resolve_passthrough_value(key, value)
        if value is not None:
            out[key] = value


def _finalize_child_env(env: dict) -> dict:
    """Guards shared by every spawn surface: profile-home propagation, session-context
    bridging, Hermes-owned PYTHONPATH + venv-marker strip, MSYS defaults, delegate_task
    Kanban scrub. Returns the (possibly new) dict."""
    _apply_profile_home(env)
    _inject_session_context_env(env)
    _strip_hermes_owned_pythonpath_and_runtime_markers(env)
    _apply_windows_msys_bash_env_defaults(env)
    try:  # strip dispatcher-owned Kanban env from delegate_task child subprocesses
        from agent.delegation_context import is_delegated_child_process_context, scrub_kanban_env
        if is_delegated_child_process_context():
            return scrub_kanban_env(env)
    except Exception:
        pass
    return env


def _scrubbed_env(parts, plugin_strip: frozenset, fix_path) -> dict:
    """Filter each ``(items, unwrap_force)`` in *parts* into one env, rewrite PATH via
    *fix_path* (always prepending the hermes install dir so bare ``hermes`` resolves
    for children of a systemd/cron-launched gateway), then apply the shared guards."""
    out: dict[str, str] = {}
    for items, unwrap_force in parts:
        _filter_secret_env(items, out, unwrap_force=unwrap_force, plugin_strip=plugin_strip)
    path_key = _path_env_key(out)
    # Keep bare ``hermes`` invocations available to child jobs even when the gateway was launched by a
    # service manager or cron without the console script's directory on PATH. The terminal environment
    # already applies this invariant; Cron scripts use this sanitizer directly (#92998).
    if path_key is not None:
        out[path_key] = _prepend_hermes_bin_dir(fix_path(out.get(path_key, "")))
    return _finalize_child_env(out)


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment (background/PTY
    spawn path, search workers, computer-use driver, user-script runners)."""
    return _scrubbed_env([(base_env or {}, False), (extra_env or {}, True)],
                         _plugin_terminal_env_strip_keys(), lambda p: p)


def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
    """Sanitized env for the **non-terminal** spawn surface (browser, ACP/CLI executors,
    computer-use driver, TUI Node host). Tier 1 (``_ALWAYS_STRIP_KEYS``, plugin keys,
    force-prefixed hints, dynamic internal secrets) is always removed; Tier 2 (the
    provider/tool blocklist) unless ``inherit_credentials`` — pass that **only** for
    children that legitimately need LLM credentials (user-blessed claude/codex/gemini
    CLI, TUI Node host). Terminal/execute_code use ``_sanitize_subprocess_env``."""
    env = os.environ.copy()
    strip = _ALWAYS_STRIP_KEYS | _plugin_terminal_env_strip_keys()
    if not inherit_credentials:
        strip |= _HERMES_PROVIDER_ENV_BLOCKLIST
    for key in list(env):
        if (key in strip or key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX)
                or _is_hermes_internal_secret(key)):
            del env[key]
    env.setdefault("PYTHONUTF8", "1")  # Windows UTF-8 safety for spawned processes
    return _finalize_child_env(env)


def build_subprocess_env(
    base: "Mapping[str, str] | None" = None, *, inherit_profile_home: bool = True,
    scrub_secrets: bool = True, extra: "Mapping[str, str] | None" = None) -> dict[str, str]:
    """Single factory for child-process envs. ``base=None`` snapshots ``os.environ``.
    ``scrub_secrets=True`` -> :func:`_sanitize_subprocess_env` (profile home inherent,
    ``inherit_profile_home`` ignored). ``scrub_secrets=False`` keeps the base
    byte-for-byte (git credential flows, ``bws``/``op``); ``inherit_profile_home``
    bridges HERMES_HOME + HOME and ``extra`` is applied last so caller overrides win."""
    env: dict[str, str] = dict(base) if base is not None else os.environ.copy()
    if scrub_secrets:
        return _sanitize_subprocess_env(env, dict(extra) if extra else None)
    if inherit_profile_home:
        _apply_profile_home(env)
    if extra:
        env.update(extra)
    return env


# --- Shell discovery ---
def _windows_bash_candidates(custom: "str | None") -> list[str]:
    """Ordered bash.exe candidates on Windows: HERMES_GIT_BASH_PATH, our portable Git
    under %LOCALAPPDATA%\\hermes\\git (PortableGit ``bin`` and MinGit ``usr\\bin``),
    known Git-for-Windows dirs, then PATH last — ``shutil.which`` may return WSL's
    bash, which fails silently on Windows paths."""
    getenv = os.environ.get
    lad = getenv("LOCALAPPDATA", "")
    roots = [
        lad and os.path.join(lad, "hermes", "git", "bin"),
        lad and os.path.join(lad, "hermes", "git", "usr", "bin"),
        os.path.join(getenv("ProgramFiles", r"C:\Program Files"), "Git", "bin"),
        os.path.join(getenv("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin"),
        lad and os.path.join(lad, "Programs", "Git", "bin"),
    ]
    raw = [custom or "", *(os.path.join(r, "bash.exe") for r in roots if r)]
    candidates = list(dict.fromkeys(c for c in raw if c and os.path.isfile(c)))
    found = shutil.which("bash")
    if found and found not in candidates:
        candidates.append(found)
    return candidates


def _find_bash() -> str:
    """Find bash for command execution."""
    if not _IS_WINDOWS:
        return (shutil.which("bash")
                or next((p for p in ("/usr/bin/bash", "/bin/bash") if os.path.isfile(p)), None)
                or os.environ.get("SHELL") or "/bin/sh")
    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    candidates = _windows_bash_candidates(custom)
    # First candidate that can actually start wins: a stale HERMES_GIT_BASH_PATH
    # pointing at a broken install must not beat a healthy portable Git.
    for candidate in candidates:
        if _bash_starts(candidate):
            if candidate != custom and custom and os.path.isfile(custom):
                logger.warning(
                    "HERMES_GIT_BASH_PATH=%s fails to start; using %s instead", custom, candidate)
            return candidate
    if candidates:
        probe_details = "\n".join(
            detail for c in candidates if (detail := _bash_probe_details_cache.get(c)))
        if _mandatory_aslr_enabled() is True or _looks_like_msys_spawn_failure(probe_details):
            raise RuntimeError(_git_bash_aslr_help(candidates[0], probe_details))
        # Unknown failure class: return the first path so the caller sees the
        # real bash error instead of a less useful "not found".
        return candidates[0]
    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location.")


_git_bash_bin_dirs_cache: "list[str] | None" = None


def _git_bash_bin_dirs() -> list[str]:
    """Git Bash's coreutils dirs in ``/etc/profile`` order (mingw first so coreutils
    beat System32 lookalikes); ``[]`` off Windows. A non-login ``bash -c`` (fallback
    when ``bash -l`` is broken) never sources ``/etc/profile``, so without these
    ``cat``/``mktemp``/``mv`` are missing and commands exit 127."""
    global _git_bash_bin_dirs_cache
    if _git_bash_bin_dirs_cache is None:
        _git_bash_bin_dirs_cache = _compute_git_bash_bin_dirs() if _IS_WINDOWS else []
    return _git_bash_bin_dirs_cache


def _compute_git_bash_bin_dirs() -> list[str]:
    try:
        bash = _find_bash()
    except Exception:
        return []
    parent = os.path.dirname(os.path.dirname(bash))  # bash in <root>\bin or <root>\usr\bin (MinGit)
    root = os.path.dirname(parent) if os.path.basename(parent).lower() == "usr" else parent
    subs = ("mingw64/bin", "mingw32/bin", "usr/local/bin", "usr/bin", "bin")
    dirs = (os.path.join(root, *sub.split("/")) for sub in subs)
    return list(dict.fromkeys(d for d in dirs if os.path.isdir(d)))


def _prepend_missing_path_entries(existing_path: str, dirs: list[str]) -> str:
    """Prepend *dirs* missing from *existing_path* (``os.pathsep``); an already-listed
    dir keeps its position; unchanged input when nothing is missing."""
    entries = [e for e in existing_path.split(os.pathsep) if e]
    missing = [d for d in dirs if d not in entries]
    return os.pathsep.join([*missing, *entries]) if missing else existing_path


def _prepend_git_bash_dirs(existing_path: str) -> str:
    """Prepend Git Bash's binary dirs if missing (no-op off Windows), so the
    non-login ``bash -c`` fallback can find coreutils."""
    return _prepend_missing_path_entries(existing_path, _git_bash_bin_dirs())


# POSIX-sh-family shells that understand spawn_local's ``[shell, "-lic", "set +m; …"]``
# invocation; fish, csh/tcsh, nushell, elvish, xonsh would error, so _find_shell
# falls back to bash for them.
# (#42203)
_SPAWN_COMPATIBLE_SHELLS = frozenset({"bash", "zsh", "sh", "dash", "ksh", "mksh"})


def _find_shell() -> str:
    """User's login shell for background spawning: ``$SHELL`` on POSIX when it is an
    executable sh-family shell, else ``_find_bash``. macOS's system bash 3.2 under
    ``-l`` with stdin ``/dev/null`` sources ``~/.bash_profile``, which often
    ``exec /bin/zsh -l`` and drops ``-c`` — the command silently never runs."""
    user_shell = "" if _IS_WINDOWS else os.environ.get("SHELL")
    if (user_shell and os.path.isfile(user_shell) and os.access(user_shell, os.X_OK)
            and Path(user_shell).name in _SPAWN_COMPATIBLE_SHELLS):
        return user_shell
    return _find_bash()


# --- PATH completion for the terminal subshell ---

# Standard PATH entries for environments with minimal PATH.
_SANE_PATH = ("/opt/homebrew/bin:/opt/homebrew/sbin:"
              "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

# Cached directory containing the ``hermes`` console-script.
# ``_SENTINEL`` distinguishes "not resolved yet" from a resolved ``None``.
_SENTINEL = object()
_HERMES_BIN_DIR: "str | None | object" = _SENTINEL


def _resolve_hermes_bin_dir() -> str | None:
    """Directory holding the ``hermes`` console-script, or None (cached). A gateway
    launched by systemd/cron/a desktop launcher lacks the install dir on PATH and bare
    ``hermes`` exits 127. Order: ``which``; absolute ``sys.argv[0]`` naming a real
    hermes executable; ``sys.executable``'s dir if it holds the shim."""
    global _HERMES_BIN_DIR
    if _HERMES_BIN_DIR is not _SENTINEL:
        return _HERMES_BIN_DIR  # type: ignore[return-value]
    which = shutil.which("hermes")
    argv0 = sys.argv[0] if sys.argv else ""
    base = os.path.basename(argv0).lower()
    exe_dir = os.path.dirname(sys.executable) if sys.executable else ""
    shim = "hermes.exe" if _IS_WINDOWS else "hermes"
    if which:
        candidate = os.path.dirname(which)
    elif (os.path.isabs(argv0) and (base == "hermes" or base.startswith("hermes."))
            and os.path.isfile(argv0)):
        candidate = os.path.dirname(argv0)
    else:
        candidate = exe_dir if exe_dir and os.path.isfile(os.path.join(exe_dir, shim)) else None
    _HERMES_BIN_DIR = candidate if candidate and os.path.isdir(candidate) else None
    return _HERMES_BIN_DIR


def _prepend_hermes_bin_dir(existing_path: str) -> str:
    """Prepend the hermes install dir to ``existing_path`` if missing."""
    bin_dir = _resolve_hermes_bin_dir()
    return _prepend_missing_path_entries(existing_path, [bin_dir] if bin_dir else [])


def _managed_runtime_path_entries() -> list[str]:
    """Existing Hermes-managed runtime dirs: ``$HERMES_HOME/node`` (+``/bin``) and
    ``$HERMES_HOME/bin`` (managed ``uv``). Per call, not cached: home is
    profile-scoped and a managed tree can appear mid-process."""
    try:
        from hermes_constants import get_hermes_home, iter_hermes_node_dirs
        return [str(d) for d in (*iter_hermes_node_dirs(), get_hermes_home() / "bin") if d.is_dir()]
    except Exception:
        return []


def _append_missing_sane_path_entries(existing_path: str) -> str:
    """Normalised POSIX PATH with missing sane entries appended: empty entries
    dropped (shells read them as cwd), duplicates collapsed (first wins), then
    missing ``_SANE_PATH`` / managed-runtime dirs appended so user entries keep
    precedence. Windows is a no-op passthrough (native ``;`` PATH untouched)."""
    if _IS_WINDOWS:
        return existing_path
    # dict preserves first-occurrence order; empty entries dropped.
    ordered = dict.fromkeys(entry for entry in existing_path.split(":") if entry)
    ordered.update(dict.fromkeys([*_SANE_PATH.split(":"), *_managed_runtime_path_entries()]))
    return ":".join(ordered)


def _apply_windows_msys_bash_env_defaults(env: dict) -> None:
    """Disable MSYS argument path conversion (``/FO`` -> ``C:/.../git/FO`` breaks
    tasklist/schtasks/wmic/``cmd /c``). Git for Windows honors ``MSYS_NO_PATHCONV``;
    MSYS2/Cygwin bash honor ``MSYS2_ARG_CONV_EXCL`` — set both; users can override.

    Git Bash rewrites arguments that look like Unix paths (``/FO``, ``/TN``, ``/Create``) into
    ``C:/.../git/FO``-style paths, which breaks native Windows commands such as ``tasklist``, ``schtasks``,
    and ``wmic``. Hermes runs terminal commands through bash on Windows, so set the standard MSYS opt-out by
    default. Refs #56700.
    MSYS2-proper and Cygwin bash (which ``_find_bash`` can still return via the final ``shutil.which``
    fallback) ignore it and honor ``MSYS2_ARG_CONV_EXCL`` instead, so set both. ``*`` disables all argv
    conversion — the semantic equivalent of ``MSYS_NO_PATHCONV=1``. Also fixes ``cmd /c`` mangling (#56147).
    """
    if _IS_WINDOWS:
        env.setdefault("MSYS_NO_PATHCONV", "1")
        env.setdefault("MSYS2_ARG_CONV_EXCL", "*")


def _path_env_key(run_env: dict) -> str | None:
    """PATH env key to update without altering Windows casing (``Path`` vs ``PATH``);
    None when a Windows env has no PATH key at all."""
    return next((k for k in run_env if k.upper() == "PATH"), None) if _IS_WINDOWS else "PATH"


def _finalize_execution_env(env: dict, execution_context, *, authoritative_keys=()) -> dict:
    """Keep explicit routing exact after defaults/sanitization, without restoring secrets.

    PATH is not a credential: an explicitly chosen search path must not acquire
    host executable directories from the generic terminal convenience defaults.
    """
    if execution_context is None:
        return env
    execution_context.check()
    context = execution_context.context
    for key in context.env_unset - set(authoritative_keys):
        env.pop(key, None)
    if "PATH" in context.env_set:
        env["PATH"] = context.env_set["PATH"]
    return env


def _prepend_execution_routing(command: str, run_env: dict, execution_context) -> str:
    """Reassert sanitized routing after shell startup, or abort before user code."""
    context = execution_context.context
    routing = set(context.env_set) | set(context.env_unset)
    prelude = [f"builtin export {key}={shlex.quote(run_env[key])} || exit 126" if key in run_env
               else f"builtin unset {key} || exit 126" for key in sorted(routing)]
    return "\n".join([*prelude, command])


def _make_run_env(env: dict, execution_context=None) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    merged = dict(os.environ | env)
    if execution_context is not None:
        merged = execution_context.apply_env(merged)
    clean = _scrubbed_env([(merged, True)], frozenset(),
                          lambda p: _prepend_git_bash_dirs(_append_missing_sane_path_entries(p)))
    return _finalize_execution_env(clean, execution_context)


# --- Hermes venv / repo-root detection (module-level, computed once) ---
# Owned here; read lazily by tools.environments.local_pythonpath (tests patch here).
# The Electron app prepends the repo root to PYTHONPATH so the backend can ``import
# tools``; other subprocesses must not inherit it. Aliases: launchers may emit other
# spellings — the Windows gateway launcher renders Hermes-owned paths under the
# configured HERMES_HOME spelling (possibly a junction to another drive).
_hermes_repo_root: Path = Path(__file__).resolve().parents[2]
_hermes_repo_root_aliases: tuple[Path, ...] = _build_hermes_repo_root_aliases(
    _hermes_repo_root, Path(__file__).absolute().parents[2], get_process_hermes_home())
_in_venv: bool = (getattr(sys, "base_prefix", sys.prefix) != sys.prefix
                  or hasattr(sys, "real_prefix"))  # real_prefix: virtualenv<20
_hermes_site_packages: list[Path] | None = None  # lazily cached by local_pythonpath


# --- Login-shell init files ---
def _read_terminal_shell_init_config() -> tuple[list[str], bool]:
    """(shell_init_files, auto_source_bashrc) from config.yaml; defaults on any
    failure so terminal execution never breaks."""
    try:
        from hermes_cli.config import load_config
        terminal_cfg = (load_config() or {}).get("terminal") or {}
        files = terminal_cfg.get("shell_init_files") or []
        if not isinstance(files, list):
            files = []
        return [str(f) for f in files if f], bool(terminal_cfg.get("auto_source_bashrc", True))
    except Exception:
        return [], True


def _resolve_shell_init_files() -> list[str]:
    """Files to source before the login-shell snapshot (``~``/``${VAR}`` expanded,
    missing dropped). ``auto_source_bashrc`` applies only without an explicit list:
    ~/.profile and ~/.bash_profile first (no interactivity guard; where
    n/nvm/asdf/pyenv add PATH), ~/.bashrc last (Debian's returns early when
    non-interactive, but guard-less bashrcs keep working)."""
    explicit, auto_bashrc = _read_terminal_shell_init_config()
    candidates = explicit or (["~/.profile", "~/.bash_profile", "~/.bashrc"]
                              if auto_bashrc and not _IS_WINDOWS else [])
    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
            if path and os.path.isfile(path):
                resolved.append(path)
        except Exception:
            continue
    return resolved


def _prepend_shell_init(cmd_string: str, files: list[str]) -> str:
    """Prepend guarded, silent ``source <file>`` lines: ``set +e`` keeps going on
    errors, ``2>/dev/null`` hides noisy prompts, ``|| true`` neutralises the status."""
    if not files:
        return cmd_string
    safe = [p.replace("'", "'\\''") for p in files]
    prelude = ["set +e", *(f"[ -r '{p}' ] && . '{p}' 2>/dev/null || true" for p in safe)]
    return "\n".join(prelude) + "\n" + cmd_string


# --- Process-group teardown (POSIX) ---
def _wait_for_group_exit(proc, pgid: int, timeout: float) -> bool:
    """Wait until the process group is gone, reaping the wrapper as we go (a dead
    but unreaped group leader still makes ``killpg(pgid, 0)`` succeed).
    POSIX-only; callers are behind the _IS_WINDOWS gate."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            proc.poll()
        except Exception:
            pass
        try:
            os.killpg(pgid, 0)  # windows-footgun: ok — POSIX process-group alive probe
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # exists, even if we cannot signal it
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _sweep_escaped_descendants(descendants: list, pgid: int) -> None:
    """SIGKILL snapshotted survivors that escaped the process group via ``setsid``
    — after TERM→KILL so in-group members keep their grace; psutil's identity-aware
    Process skips recycled PIDs. POSIX-only (see _IS_WINDOWS gate in caller)."""
    for child in descendants:
        try:
            if not child.is_running():
                continue
            try:
                if os.getpgid(child.pid) == pgid:
                    continue  # group-kill already covers it
            except OSError:  # ProcessLookupError / PermissionError included
                pass
            child.kill()
        except Exception:
            continue


def _kill_process_group_posix(proc) -> None:
    """TERM the group, wait, KILL, then sweep setsid escapees. Descendants are
    snapshotted BEFORE the first signal — once the wrapper dies they reparent to
    init — and we wait on the group, not the wrapper, which can exit before
    grandchildren under load. POSIX-only (_IS_WINDOWS handled by the caller)."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        if (pgid := getattr(proc, "_hermes_pgid", None)) is None:
            raise
    try:  # psutil children snapshot; empty on any failure (must never break the kill)
        import psutil
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        descendants = []
    try:
        os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX only (see _IS_WINDOWS gate in caller)
        if not _wait_for_group_exit(proc, pgid, 1.0):
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX only (see _IS_WINDOWS gate in caller)
            _wait_for_group_exit(proc, pgid, 2.0)
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=0.2)
    except ProcessLookupError:
        pass
    _sweep_escaped_descendants(descendants, pgid)


def _kill_process_windows(proc) -> None:
    """Identity-checked terminate (start time guards against PID reuse), else kill."""
    try:
        from gateway.status import get_process_start_time, terminate_pid
        terminate_pid(proc.pid, force=True, expected_start_time=get_process_start_time(proc.pid))
    except Exception:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=2.0)


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host: every execute() spawns a fresh bash;
    the session snapshot preserves env vars across calls; CWD persists via the
    stdout marker."""

    _profile_scoped_passthrough = True
    # Commands run on the Hermes host itself — controller-side platform behavior
    # (macOS TCC pruning, etc.) legitimately applies here.
    is_local = True
    execution_context = None

    def _additional_profile_scoped_passthrough_names(self) -> tuple[str, ...]:
        """First-party ``BUZZ_*`` names present in the env, excluded from the shared
        session snapshot. env_passthrough can never list them (it refuses blocklisted
        names), so under a multiplexed gateway profile A's BUZZ_PRIVATE_KEY would land
        in the snapshot and be sourced by profile B. Prefix-only and monotonic on
        purpose: conservative even when the context-gated carve-out is inactive."""
        merged = dict(os.environ | self.env)
        return tuple(sorted(
            name for name in merged
            if isinstance(name, str) and _matches_terminal_first_party_prefix(name)))

    def _snapshot_excluded_passthrough_names(self):
        names = set(super()._snapshot_excluded_passthrough_names())
        if self.execution_context is not None:
            context = self.execution_context.context
            names.update(context.env_set)
            names.update(context.env_unset)
        return tuple(sorted(names))

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None, *, execution_context=None):
        self.execution_context = execution_context
        super().__init__(cwd=_resolve_local_initial_cwd(cwd), timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        """Shell-safe writable temp dir. Precedence: ``TERMINAL_TEMP_DIR``, TMPDIR/TMP/TEMP
        (Termux has no /tmp), ``HERMES_HOME/cache/terminal`` (real storage: tmpfs /tmp
        fills under Hermes load; pruned by ``cleanup_terminal_temp_cache``), /tmp,
        ``tempfile.gettempdir()``; backend env before process env so terminal.env
        overrides work. Windows: ``%TEMP%`` often has spaces that break unquoted bash,
        so always the HERMES_HOME cache dir with forward slashes (bash- and Python-valid)."""
        if _IS_WINDOWS:
            cache_dir = (_default_terminal_temp_dir()
                         or Path(tempfile.gettempdir()) / "hermes_terminal")
            cache_dir.mkdir(parents=True, exist_ok=True)
            _prune_terminal_temp_once()
            return str(cache_dir).replace("\\", "/")
        def _posix(p: str) -> str:
            return p.rstrip("/") or "/"
        for env_var in ("TERMINAL_TEMP_DIR", "TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/") and (
                    env_var != "TERMINAL_TEMP_DIR" or os.path.isdir(candidate)):
                return _posix(candidate)
        try:
            cache_dir = _default_terminal_temp_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            resolved = str(cache_dir)
            if resolved.startswith("/") and os.access(resolved, os.W_OK | os.X_OK):
                _prune_terminal_temp_once()
                return _posix(resolved)
        except Exception:
            pass
        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"
        fallback = tempfile.gettempdir()
        return _posix(fallback) if fallback.startswith("/") else "/tmp"

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Use native paths for Python, but Git Bash-friendly paths for cd."""
        return BaseEnvironment._quote_cwd_for_cd(_windows_to_msys_path(cwd))

    def _quote_shell_path(self, path: str) -> str:
        """Rewrite native/mixed Windows paths before quoting for Git Bash."""
        return _quote_bash_path(path)

    def _recover_cwd(self) -> None:
        """Swap ``self.cwd`` for a usable directory if it vanished or is inaccessible
        (e.g. a command ``rm -rf``'d its own cwd) — otherwise Popen raises before bash
        starts and every subsequent call fails. A benign MSYS→Windows normalization
        is not warned about."""
        # Recover when the cwd has been deleted out from under us — usually by a previous tool call that ran
        # ``rm -rf`` on its own working dir (issue #17558). On Windows, ``_resolve_safe_cwd`` also
        # normalises Git Bash-style POSIX paths (``/c/Users/...``) to native form so a perfectly valid ``pwd
        # -P`` result from bash isn't mistakenly treated as "missing" and spammed as a warning on every
        # command.
        safe_cwd = _resolve_safe_cwd(self.cwd)
        if safe_cwd == self.cwd:
            return
        if safe_cwd != _msys_to_windows_path(self.cwd):
            logger.warning(
                "LocalEnvironment cwd %r is missing on disk; "
                "falling back to %r so terminal commands keep working.",
                self.cwd, safe_cwd)
        self.cwd = safe_cwd

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        bash = _find_bash()
        run_env = _make_run_env(self.env, self.execution_context)
        if self.execution_context is not None:
            # Bash has already sourced BASH_ENV/login profiles when this runs;
            # custom init is prepended below. Pin sanitized routing before the
            # wrapper saves passthrough values or creates/restores a snapshot.
            cmd_string = _prepend_execution_routing(cmd_string, run_env, self.execution_context)
        # Login invocations (init_session's env snapshot) source the user's rc /
        # custom init files so nvm/asdf/pyenv land on PATH in the snapshot.
        if login:
            cmd_string = _prepend_shell_init(cmd_string, _resolve_shell_init_files())
        args = [bash, *(["-l"] if login else []), "-c", cmd_string]
        if self.execution_context is not None:
            args = self.execution_context.wrap_argv(args)
        self._recover_cwd()
        proc = subprocess.Popen(
            args, text=True, env=run_env, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True, cwd=self.cwd,
            **({"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}))
        if not _IS_WINDOWS:
            with contextlib.suppress(ProcessLookupError):
                proc._hermes_pgid = os.getpgid(proc.pid)
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    def _kill_process(self, proc):
        """Kill the entire process group (all children)."""
        try:
            (_kill_process_windows if _IS_WINDOWS else _kill_process_group_posix)(proc)
        except OSError:  # ProcessLookupError / PermissionError included
            with contextlib.suppress(Exception):
                proc.kill()

    def _extract_cwd_from_output(self, result: dict):
        """Base semantics plus: Git Bash ``pwd -P`` emits MSYS form on Windows —
        normalize to native and require the dir to exist, else ``_run_bash`` would
        warn every command. A stale path rolls back to the previous cwd, which this
        command did not observe, so ``cwd_observed`` is dropped."""
        prev_cwd = self.cwd
        super()._extract_cwd_from_output(result)
        if self.cwd != prev_cwd:
            normalized = _msys_to_windows_path(self.cwd)
            if normalized and os.path.isdir(normalized):
                self.cwd = normalized
                result["cwd"] = normalized
            else:
                self.cwd = prev_cwd
                result.pop("cwd_observed", None)
                result.pop("cwd", None)

    def cleanup(self):
        """Clean up temp files, including orphaned atomic-write snapshots
        (``snap.tmp.<bashpid>``) a failed/interrupted mv could leave behind."""
        # See #38249.
        import glob
        try:
            stale = glob.glob(f"{self._snapshot_path}.tmp.*")
        except Exception:
            stale = []
        for f in (self._snapshot_path, self._cwd_file, *stale):
            with contextlib.suppress(OSError):
                os.unlink(f)
