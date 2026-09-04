"""Child-process environment for execute_code: env scrubbing, interpreter and cwd resolution.

Both the per-call remote path and the local session kernel build their child
env through ``_build_child_env`` so the security rules (secret scrubbing,
PYTHONPATH hygiene, UTF-8 forcing, TZ) cannot drift between them.
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Dict

# Logger name kept as the origin module's so existing log expectations hold.
logger = logging.getLogger("tools.code_execution_tool")

_IS_WINDOWS = platform.system() == "Windows"

# Scrub order: secret-substring block first; whatever is left must match a safe
# prefix, the exact-name HERMES_ allowlist, or (Windows) an OS-essential name.
# The broad "HERMES_" prefix is deliberately NOT safe — it leaked config vars
# without a secret substring (HERMES_BASE_URL, HERMES_KANBAN_DB, *_WEBHOOK).
# HERMES_RPC_SOCKET / HERMES_RPC_DIR / TZ / HOME are injected after scrubbing.
_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM", "TMPDIR", "TMP", "TEMP", "SHELL",
                      "LOGNAME", "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
# "PASS" is intentionally absent: it false-positives on BYPASS_CACHE /
# COMPASS_DIR / PASSENGER_HOST while PASSWORD/PASSWD already cover credentials.
_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD", "AUTH", "DSN",
                      "WEBHOOK", "CREDS", "BEARER", "APIKEY")

# Non-secret runtime-location flags that repo-root modules a sandbox script
# imports may read at import time. HERMES_DELEGATED_CHILD_CONTEXT must ride
# along or a child that imports Hermes code loses the Kanban mutation guard
# while still inheriting HERMES_HOME.
_HERMES_CHILD_ALLOWED = frozenset({
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV", "HERMES_DELEGATED_CHILD_CONTEXT",
})

# Windows-only: without these the CRT itself fails — socket.socket() raises
# WinError 10106 (Winsock can't find mswsock.dll) and subprocess can't resolve
# cmd.exe. Well-known OS paths, not secrets; the substring block still runs.
_WINDOWS_ESSENTIAL_ENV_VARS = frozenset({
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "OS",
    "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "PUBLIC", "ALLUSERSPROFILE",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
    "APPDATA", "LOCALAPPDATA", "USERPROFILE", "USERDOMAIN", "USERNAME",
    "HOMEDRIVE", "HOMEPATH", "COMPUTERNAME",
})


def _scrub_child_env(source_env, is_passthrough=None, is_windows=None):
    """Produce the scrubbed child-process env for execute_code.

    Rules, in order: (1) passthrough vars (skill/config-declared) resolve
    through the active profile secret scope — an absent scoped value is
    omitted; (2) secret-substring names are blocked; (3) safe prefixes pass;
    (4) operational HERMES_* pass by exact name; (5) on Windows the
    OS-essential allowlist passes by exact name.
    """
    try:
        from tools.env_passthrough import is_env_passthrough, resolve_passthrough_value
    except Exception:
        is_env_passthrough = lambda _: False  # noqa: E731
        resolve_passthrough_value = lambda _name, _fallback: None  # noqa: E731
    if is_passthrough is None:
        is_passthrough = is_env_passthrough
    if is_windows is None:
        is_windows = _IS_WINDOWS
    scrubbed = {}
    # Non-secret HERMES_* vars no allowlist admits are dropped on purpose; a script importing a
    # repo module that reads one would see it silently unset — log the drop, point at the opt-in.
    _dropped_hermes = []
    for k, v in source_env.items():
        if is_passthrough(k):
            resolved = resolve_passthrough_value(k, v)
            if resolved is not None:
                scrubbed[k] = resolved
            continue
        if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
            continue
        if (any(k.startswith(p) for p in _SAFE_ENV_PREFIXES)
                or k in _HERMES_CHILD_ALLOWED
                or (is_windows and k.upper() in _WINDOWS_ESSENTIAL_ENV_VARS)):
            scrubbed[k] = v
        elif k.startswith("HERMES_"):
            _dropped_hermes.append(k)
    if _dropped_hermes:
        logger.debug(
            "execute_code: dropped %d non-allowlisted HERMES_* var(s) from the "
            "sandbox child env (%s). This is intentional hardening (#27303); if "
            "a sandbox script legitimately needs one, declare it via "
            "env_passthrough in the skill/config so it passes by explicit opt-in.",
            len(_dropped_hermes), ", ".join(sorted(_dropped_hermes)),
        )
    # delegate_task children are marked by a ContextVar, not os.environ, and the sandbox crosses
    # a process boundary: strip dispatcher-owned Kanban vars AFTER the scrub so an explicit
    # passthrough cannot re-grant a delegated child the parent's board mutation capability.
    try:
        from agent.delegation_context import is_delegated_child_process_context, scrub_kanban_env
        if is_delegated_child_process_context():
            scrubbed = scrub_kanban_env(scrubbed)
    except Exception:
        pass
    return scrubbed


def _build_child_env(*, rpc_endpoint: str, rpc_token: str, tmpdir: str,
                     child_python: str) -> Dict[str, str]:
    """Build the scrubbed child environment both execution paths share."""
    from hermes_constants import apply_subprocess_home_env
    child_env = _scrub_child_env(os.environ)
    child_env["HERMES_RPC_SOCKET"] = rpc_endpoint
    child_env["HERMES_RPC_TOKEN"] = rpc_token
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Force UTF-8 stdio and default file encoding: on Windows sys.stdout is bound to the console
    # code page (cp1252) and print("→") raises; harmless under a C/POSIX locale (containers).
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    # Only TZ reaches the child; HERMES_TIMEZONE is an internal setting.
    _tz_name = os.getenv("HERMES_TIMEZONE", "").strip()
    if _tz_name:
        child_env["TZ"] = _tz_name
    child_env.pop("HERMES_TIMEZONE", None)
    apply_subprocess_home_env(child_env)
    # PYTHONPATH: the staging dir (hermes_tools.py) must always be importable even when project
    # mode changes CWD. Hermes's root is added ONLY when the child runs in Hermes's Python env —
    # exposing Hermes's site-packages to an external interpreter can mix incompatible compiled
    # extensions (3.12 NumPy under a 3.9 venv). Inherited Hermes-owned entries are stripped first.
    # Before re-injecting PYTHONPATH, strip Hermes-owned entries that leaked through _scrub_child_env
    # (PYTHONPATH is in _SAFE_ENV_PREFIXES so it passes the scrub). They are redundant for same-Hermes-
    # environment children and may be incompatible with external interpreters (project mode can select a
    # different venv), so they must not shadow or poison the child's sys.path (#74817).
    from tools.environments.local_pythonpath import _strip_hermes_owned_pythonpath
    _strip_hermes_owned_pythonpath(child_env)
    _existing_pp = child_env.get("PYTHONPATH", "")
    _pp_parts = [tmpdir]
    if _uses_hermes_python_environment(child_python):
        _pp_parts.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elif child_python not in _external_env_logged:
        # Surface once per interpreter so "import hermes_constants fails" is diagnosable.
        _external_env_logged.add(child_python)
        logger.info("execute_code: child interpreter %s is outside the Hermes "
                    "environment; hermes root omitted from PYTHONPATH", child_python)
    if _existing_pp:
        _pp_parts.append(_existing_pp)
    child_env["PYTHONPATH"] = os.pathsep.join(_pp_parts)
    return child_env


# Interpreter-probe caches: success-only dicts (FIFO-evicted at the cap) rather than lru_cache —
# a transient probe failure (fork pressure, 5s timeout) must not stick for the process lifetime.
_PROBE_CACHE_MAX = 32
_usable_python_cache: dict = {}
_python_prefix_cache: dict = {}

# Interpreter paths already reported as outside the Hermes environment.
_external_env_logged: set = set()


def _cache_probe_result(cache: dict, key: str, value):
    """Insert into a bounded probe cache, FIFO-evicting at the cap."""
    if len(cache) >= _PROBE_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _probe_python(python_path: str, code: str, *, text: bool = False):
    """Run ``python_path -c code``; None if missing, unspawnable, or past the 5s timeout."""
    try:
        from agent.delegation_context import delegated_child_subprocess_env
        return subprocess.run(
            [python_path, "-c", code], timeout=5, capture_output=True, text=text,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
            stdin=subprocess.DEVNULL, env=delegated_child_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None


def _is_usable_python(python_path: str) -> bool:
    """Whether the interpreter is Python 3.8+ (what the RPC stubs need); success cached, failure retried."""
    cached = _usable_python_cache.get(python_path)
    if cached is not None:
        return cached
    result = _probe_python(python_path, "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)")
    if result is None:
        return False
    usable = result.returncode == 0
    _cache_probe_result(_usable_python_cache, python_path, usable)
    return usable


def _python_environment_prefix(python_path: str) -> str:
    """Resolved ``sys.prefix`` reported by *python_path* ("" on failure; failures are not cached)."""
    cached = _python_prefix_cache.get(python_path)
    if cached is not None:
        return cached
    result = _probe_python(python_path, "import sys; print(sys.prefix)", text=True)
    if result is not None and result.returncode == 0 and result.stdout.strip():
        prefix = os.path.realpath(result.stdout.strip())
        _cache_probe_result(_python_prefix_cache, python_path, prefix)
        return prefix
    return ""


def _uses_hermes_python_environment(python_path: str) -> bool:
    """Whether *python_path* belongs to Hermes's active Python environment. Short-circuits when
    it IS the running interpreter (by path or realpath — covers ``uv run`` venvs) so no probe
    runs on the default strict path and a flaky probe can never drop the hermes root."""
    if python_path == sys.executable or os.path.realpath(python_path) == os.path.realpath(sys.executable):
        return True
    return _python_environment_prefix(python_path) == os.path.realpath(sys.prefix)


def _resolve_child_python(mode: str) -> str:
    """Child interpreter: ``sys.executable`` in strict mode; in project mode the active
    VIRTUAL_ENV/CONDA_PREFIX python if it exists and passes the 3.8+ probe, else ``sys.executable``."""
    if mode != "project":
        return sys.executable
    subdir, exe_names = ("Scripts", ("python.exe", "python3.exe")) if _IS_WINDOWS else ("bin", ("python", "python3"))
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(var, "").strip()
        for exe in exe_names if root else ():
            candidate = os.path.join(root, subdir, exe)
            if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
                continue
            if _is_usable_python(candidate):
                return candidate
            logger.info("execute_code: skipping %s=%s (Python version < 3.8 or broken). "
                        "Using sys.executable instead.", var, candidate)
            return sys.executable
    return sys.executable


def _resolve_child_cwd(mode: str, staging_dir: str, task_id: str = "") -> str:
    """Child cwd. Strict: the staging dir. Project mirrors the terminal/file-tool ladder so every
    file-writing path agrees: session cwd record (`cd` state) → registered ``session.cwd.set``
    override → TERMINAL_CWD → os.getcwd() → staging dir (never Popen on a missing cwd).

    (#56047)
    """
    if mode != "project":
        return staging_dir
    if task_id:
        try:
            from tools.terminal_tool import get_session_cwd
            recorded = get_session_cwd(task_id)
        except Exception:
            recorded = None
        if recorded and os.path.isdir(recorded):
            return recorded
        try:
            from tools.file_tools_paths import _registered_task_cwd_override
            session_cwd = _registered_task_cwd_override(task_id)
        except Exception:
            session_cwd = None
        if session_cwd and os.path.isdir(session_cwd):
            return session_cwd
    from agent.runtime_cwd import scope_terminal_cwd
    raw = scope_terminal_cwd().strip()
    for candidate in (os.path.expanduser(raw) if raw else "", os.getcwd()):
        if candidate and os.path.isdir(candidate):
            return candidate
    return staging_dir
