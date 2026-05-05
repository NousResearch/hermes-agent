"""Local execution environment — spawn-per-call with session snapshot."""

import base64
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from tools.environments.base import BaseEnvironment, _pipe_stdin

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)


def _resolve_safe_cwd(cwd: str) -> str:
    """Return ``cwd`` if it exists as a directory, else the nearest existing
    ancestor.  Falls back to ``tempfile.gettempdir()`` only if walking up the
    path can't find any existing directory (effectively never on a healthy
    filesystem, but cheap belt-and-braces).

    Used by ``_run_bash`` to recover when the configured cwd is gone — most
    commonly because a previous tool call deleted its own working directory
    (issue #17558).  Without this guard, ``subprocess.Popen(..., cwd=...)``
    raises ``FileNotFoundError`` before bash starts, wedging every subsequent
    terminal call until the gateway restarts.
    """
    if cwd and os.path.isdir(cwd):
        return cwd
    parent = os.path.dirname(cwd) if cwd else ""
    while parent:
        if os.path.isdir(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            # Reached the filesystem root and it doesn't exist either —
            # genuinely nothing to fall back to except the temp dir.
            break
        parent = next_parent
    return tempfile.gettempdir()


# Hermes-internal env vars that should NOT leak into terminal subprocesses.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config."""
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                blocked.add(name)
            elif category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "COHERE_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "HELICONE_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_ALLOWED_USERS",
        "WHATSAPP_ENABLED",
        "WHATSAPP_MODE",
        "WHATSAPP_ALLOWED_USERS",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_IGNORE_STORIES",
        "HASS_TOKEN",
        "HASS_URL",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "GATEWAY_ALLOWED_USERS",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "VERCEL_OIDC_TOKEN",
        "VERCEL_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
    })
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()


_POWERSHELL_LEADING_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"\$|"
    r"\[[A-Za-z_][\w.]*\](?:::|\s|\(|$)|"
    r"(?:Add|Clear|Compare|Compress|ConvertFrom|ConvertTo|Copy|Export|ForEach|"
    r"Format|Get|Import|Invoke|Join|Measure|Move|New|Out|Pop|Push|Read|Remove|"
    r"Rename|Resolve|Select|Set|Sort|Split|Start|Stop|Tee|Test|Wait|Where|Write)-"
    r")",
    re.IGNORECASE,
)

_POWERSHELL_PIPELINE_RE = re.compile(
    r"(?:^|[|;]\s*)"
    r"(?:"
    r"(?:ForEach|Format|Get|Measure|Select|Sort|Where)-Object\b|"
    r"(?:Add|Clear|Compare|Compress|ConvertFrom|ConvertTo|Copy|Export|ForEach|"
    r"Format|Get|Import|Invoke|Join|Measure|Move|New|Out|Pop|Push|Read|Remove|"
    r"Rename|Resolve|Select|Set|Sort|Split|Start|Stop|Tee|Test|Wait|Where|Write)-"
    r")",
    re.IGNORECASE,
)

_POWERSHELL_NONINTERACTIVE_PREAMBLE = (
    "$ProgressPreference='SilentlyContinue'; "
    "if (Get-Variable PSStyle -ErrorAction SilentlyContinue) { "
    "$PSStyle.OutputRendering='PlainText' }; "
)

_EXPLICIT_WINDOWS_SHELLS = {
    "bash",
    "bash.exe",
    "sh",
    "sh.exe",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
}

_CMD_EXCLUSIVE_COMMANDS = {
    "assoc",
    "break",
    "call",
    "chdir",
    "chcp",
    "choice",
    "clip",
    "cls",
    "color",
    "copy",
    "date",
    "del",
    "endlocal",
    "erase",
    "findstr",
    "ftype",
    "md",
    "mklink",
    "move",
    "path",
    "pause",
    "prompt",
    "rd",
    "rem",
    "ren",
    "rename",
    "setlocal",
    "start",
    "time",
    "title",
    "type",
    "ver",
    "verify",
    "vol",
    "where",
}

_CMD_SHARED_COMMANDS = {
    "cd",
    "dir",
    "echo",
    "if",
    "for",
    "mkdir",
    "rmdir",
    "set",
}

_CMD_PERCENT_VAR_RE = re.compile(r"%(?:[A-Za-z_][A-Za-z0-9_]*|[0-9*]|ERRORLEVEL)%", re.IGNORECASE)
_CMD_SLASH_SWITCH_RE = re.compile(r"(?:^|\s)/(?:[A-Za-z?]|-)")
_CMD_OPERATOR_RE = re.compile(r"\s(?:&&|\|\||[|&])\s")
_CMD_PIPELINE_COMMAND_RE = re.compile(
    r"(?:&&|\|\||[|&])\s*"
    r"(?:"
    + "|".join(re.escape(cmd) for cmd in sorted(_CMD_EXCLUSIVE_COMMANDS))
    + r")\b",
    re.IGNORECASE,
)
_EXPLICIT_CMD_RE = re.compile(r"^\s*@?cmd(?:\.exe)?\s+(?P<args>.*)$", re.IGNORECASE)
_EXPLICIT_CMD_C_RE = re.compile(r"(?<!\S)(?://|/)c(?:\s+|$)", re.IGNORECASE)


def _strip_cmd_grouping_quotes(command: str) -> str:
    """Remove one outer quote pair used only to group a cmd /c body."""
    if len(command) >= 2 and command[0] == command[-1] and command[0] in {"'", '"'}:
        return command[1:-1]
    return command


def _powershell_version_key(dirname: str) -> tuple:
    """Sort installed PowerShell directory names with stable releases first."""
    numbers = tuple(int(part) for part in re.findall(r"\d+", dirname))
    stable = 0 if "preview" in dirname.lower() else 1
    return numbers, stable, dirname.lower()


def _iter_programfiles_pwsh_candidates() -> list[str]:
    """Return installed PowerShell 7+ pwsh.exe candidates, newest first."""
    roots = []
    for env_name in ("ProgramFiles", "ProgramW6432", "LOCALAPPDATA"):
        root = os.environ.get(env_name)
        if root:
            roots.append(root)

    candidates: list[tuple[tuple, str]] = []
    seen_roots: set[str] = set()
    for root in roots:
        ps_root = (
            os.path.join(root, "PowerShell")
            if os.path.basename(root).lower() != "local"
            else os.path.join(root, "Programs", "PowerShell")
        )
        normalized_root = os.path.normcase(os.path.abspath(ps_root))
        if normalized_root in seen_roots:
            continue
        seen_roots.add(normalized_root)
        if not os.path.isdir(ps_root):
            continue

        try:
            version_dirs = os.listdir(ps_root)
        except OSError:
            continue
        for dirname in version_dirs:
            candidate = os.path.join(ps_root, dirname, "pwsh.exe")
            if os.path.isfile(candidate):
                candidates.append((_powershell_version_key(dirname), candidate))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates]


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if not path:
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _find_powershell() -> str:
    """Find the newest native PowerShell executable for Git Bash to launch."""
    custom = os.environ.get("HERMES_POWERSHELL_PATH")
    if custom and os.path.isfile(custom):
        return _normalize_windows_shell_path(custom)

    candidates = _dedupe_paths(
        _iter_programfiles_pwsh_candidates()
        + [
            shutil.which("pwsh.exe") or "",
            shutil.which("pwsh") or "",
            os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "PowerShell",
                "7",
                "pwsh.exe",
            ),
            shutil.which("powershell.exe") or "",
            shutil.which("powershell") or "",
        ]
        + [
            os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"),
                "System32",
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe",
            )
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return _normalize_windows_shell_path(candidate)

    raise RuntimeError(
        "PowerShell not found. Native Windows PowerShell commands require "
        "pwsh.exe or powershell.exe on PATH, or HERMES_POWERSHELL_PATH set."
    )


def _looks_like_powershell(command: str) -> bool:
    """Return True when a command is PowerShell syntax, not Bash syntax."""
    stripped = command.lstrip()
    if not stripped:
        return False

    first_word = re.split(r"\s+", stripped, maxsplit=1)[0].lower()
    if first_word in _EXPLICIT_WINDOWS_SHELLS:
        return False

    return bool(
        _POWERSHELL_LEADING_COMMAND_RE.search(stripped)
        or _POWERSHELL_PIPELINE_RE.search(stripped)
    )


def _find_cmd() -> str:
    """Find native cmd.exe for Git Bash to launch on Windows."""
    custom = os.environ.get("HERMES_CMD_PATH")
    if custom and os.path.isfile(custom):
        return _normalize_windows_shell_path(custom)

    candidates = _dedupe_paths(
        [
            shutil.which("cmd.exe") or "",
            shutil.which("cmd") or "",
            os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"),
                "System32",
                "cmd.exe",
            ),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return _normalize_windows_shell_path(candidate)

    raise RuntimeError(
        "cmd.exe not found. Native Windows CMD commands require cmd.exe "
        "on PATH, or HERMES_CMD_PATH set."
    )


def _looks_like_cmd(command: str) -> bool:
    """Return True when a command is CMD syntax, not Bash or PowerShell syntax."""
    stripped = command.lstrip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return False

    raw_first_word = re.split(r"\s+", stripped, maxsplit=1)[0].lower()
    first_word = raw_first_word.lstrip("@")
    if first_word in _EXPLICIT_WINDOWS_SHELLS:
        return False
    if first_word.endswith(".exe") or first_word.endswith(".cmd") or first_word.endswith(".bat"):
        return False

    if _CMD_PIPELINE_COMMAND_RE.search(stripped):
        return True
    if first_word in _CMD_EXCLUSIVE_COMMANDS:
        return True
    if _CMD_PERCENT_VAR_RE.search(stripped):
        return True

    if first_word not in _CMD_SHARED_COMMANDS:
        return False
    if raw_first_word.startswith("@"):
        return True

    remainder = stripped[len(raw_first_word):].lstrip()
    if first_word in {"cd", "mkdir", "rmdir"}:
        return bool(_CMD_SLASH_SWITCH_RE.search(remainder) or re.search(r"(^|\s)[A-Za-z]:[\\/]", remainder))
    if first_word == "dir":
        return not remainder.startswith("-")
    if first_word == "set":
        return not remainder.startswith("-")
    if first_word == "echo":
        return bool(
            _CMD_PERCENT_VAR_RE.search(remainder)
            or remainder.lower() in {"on", "off"}
            or _CMD_OPERATOR_RE.search(stripped)
        )
    if first_word in {"if", "for"}:
        return bool(_CMD_PERCENT_VAR_RE.search(stripped) or _CMD_SLASH_SWITCH_RE.search(stripped))

    return False


def _wrap_windows_powershell_command(command: str) -> str:
    """Wrap native PowerShell syntax so Git Bash can execute it on Windows."""
    if not _IS_WINDOWS or not _looks_like_powershell(command):
        return command

    script = _POWERSHELL_NONINTERACTIVE_PREAMBLE + command
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    powershell = shlex.quote(_find_powershell())
    return (
        f"{powershell} -NoProfile -NonInteractive "
        f"-ExecutionPolicy Bypass -EncodedCommand {encoded}"
    )


def _wrap_windows_cmd_command(command: str) -> str:
    """Wrap native CMD syntax so Git Bash can execute it on Windows."""
    if not _IS_WINDOWS:
        return command

    # Git Bash/MSYS rewrites /c-style arguments for native Windows programs
    # unless they use the double-slash escape form.
    cmd = shlex.quote(_find_cmd())

    explicit = _EXPLICIT_CMD_RE.match(command)
    if explicit:
        args = explicit.group("args")
        c_switch = _EXPLICIT_CMD_C_RE.search(args)
        if not c_switch:
            return command
        body = _strip_cmd_grouping_quotes(args[c_switch.end():].strip())
        return f"{cmd} //d //s //c {shlex.quote(body)}" if body else f"{cmd} //d //s //c"

    if not _looks_like_cmd(command):
        return command

    return f"{cmd} //d //s //c {shlex.quote(command)}"


def _wrap_windows_native_command(command: str) -> str:
    """Wrap native Windows shell syntax for the Git Bash local backend."""
    wrapped = _wrap_windows_powershell_command(command)
    if wrapped != command:
        return wrapped
    return _wrap_windows_cmd_command(command)


def _normalize_windows_shell_path(path: str) -> str:
    """Convert native Windows or MSYS paths into a Git-Bash-friendly form."""
    if not path:
        return path

    expanded = os.path.expanduser(path)

    # Drive-letter paths: C:\foo or C:/foo -> C:/foo
    if re.match(r"^[A-Za-z]:[\\/]", expanded):
        return expanded.replace("\\", "/")

    # UNC paths: \\server\share -> //server/share
    if expanded.startswith("\\\\"):
        return "//" + expanded.lstrip("\\").replace("\\", "/")

    # Git Bash / MSYS drive mounts: /c/Users/foo -> C:/Users/foo
    msys_match = re.match(r"^/([A-Za-z])/(.*)$", expanded)
    if msys_match:
        drive = msys_match.group(1).upper()
        rest = msys_match.group(2)
        return f"{drive}:/{rest}"

    # Relative Windows paths: .\foo\bar -> ./foo/bar
    if "\\" in expanded:
        return expanded.replace("\\", "/")

    return expanded


def _is_wsl_bash_launcher(path: str | None) -> bool:
    """Return True when *path* is Windows' WSL launcher, not Git Bash."""
    if not path:
        return False

    normalized = os.path.normcase(os.path.abspath(path))
    return normalized.endswith(os.path.normcase(r"\windows\system32\bash.exe")) or normalized.endswith(
        os.path.normcase(r"\windows\sysnative\bash.exe")
    )


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    sanitized: dict[str, str] = {}

    for key, value in (base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    for key, value in (extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            sanitized[real_key] = value
        elif key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    # Per-profile HOME isolation for background processes (same as _make_run_env).
    from hermes_constants import get_subprocess_home
    _profile_home = get_subprocess_home()
    if _profile_home:
        sanitized["HOME"] = _profile_home

    return sanitized


def _find_bash() -> str:
    """Find bash for command execution."""
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")
            or "/bin/sh"
        )

    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    found = shutil.which("bash")
    if found and not _is_wsl_bash_launcher(found):
        return found

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


# Backward compat — process_registry.py imports this name
_find_shell = _find_bash


# Standard PATH entries for environments with minimal PATH.
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    merged = dict(os.environ | env)
    run_env = {}
    for k, v in merged.items():
        if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            run_env[real_key] = v
        elif k not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(k):
            run_env[k] = v
    existing_path = run_env.get("PATH", "")
    if not _IS_WINDOWS and "/usr/bin" not in existing_path.split(":"):
        run_env["PATH"] = f"{existing_path}:{_SANE_PATH}" if existing_path else _SANE_PATH

    # Per-profile HOME isolation: redirect system tool configs (git, ssh, gh,
    # npm …) into {HERMES_HOME}/home/ when that directory exists.  Only the
    # subprocess sees the override — the Python process keeps the real HOME.
    from hermes_constants import get_subprocess_home
    _profile_home = get_subprocess_home()
    if _profile_home:
        run_env["HOME"] = _profile_home

    return run_env


def _read_terminal_shell_init_config() -> tuple[list[str], bool]:
    """Return (shell_init_files, auto_source_bashrc) from config.yaml.

    Best-effort — returns sensible defaults on any failure so terminal
    execution never breaks because the config file is unreadable.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        terminal_cfg = cfg.get("terminal") or {}
        files = terminal_cfg.get("shell_init_files") or []
        if not isinstance(files, list):
            files = []
        auto_bashrc = bool(terminal_cfg.get("auto_source_bashrc", True))
        return [str(f) for f in files if f], auto_bashrc
    except Exception:
        return [], True


def _resolve_shell_init_files() -> list[str]:
    """Resolve the list of files to source before the login-shell snapshot.

    Expands ``~`` and ``${VAR}`` references and drops anything that doesn't
    exist on disk, so a missing ``~/.bashrc`` never breaks the snapshot.
    The ``auto_source_bashrc`` path runs only when the user hasn't supplied
    an explicit list — once they have, Hermes trusts them.
    """
    explicit, auto_bashrc = _read_terminal_shell_init_config()

    candidates: list[str] = []
    if explicit:
        candidates.extend(explicit)
    elif auto_bashrc and not _IS_WINDOWS:
        # Build a login-shell-ish source list so tools like n / nvm / asdf /
        # pyenv that self-install into the user's shell rc land on PATH in
        # the captured snapshot.
        #
        # ~/.profile and ~/.bash_profile run first because they have no
        # interactivity guard — installers like ``n`` and ``nvm`` append
        # their PATH export there on most distros, and a non-interactive
        # ``. ~/.profile`` picks that up.
        #
        # ~/.bashrc runs last. On Debian/Ubuntu the default bashrc starts
        # with ``case $- in *i*) ;; *) return;; esac`` and exits early
        # when sourced non-interactively, which is why sourcing bashrc
        # alone misses nvm/n PATH additions placed below that guard. We
        # still include it so users who put PATH logic in bashrc (and
        # stripped the guard, or never had one) keep working.
        candidates.extend(["~/.profile", "~/.bash_profile", "~/.bashrc"])

    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            continue
        if path and os.path.isfile(path):
            resolved.append(path)
    return resolved


def _prepend_shell_init(cmd_string: str, files: list[str]) -> str:
    """Prepend ``source <file>`` lines (guarded + silent) to a bash script.

    Each file is wrapped so a failing rc file doesn't abort the whole
    bootstrap: ``set +e`` keeps going on errors, ``2>/dev/null`` hides
    noisy prompts, and ``|| true`` neutralises the exit status.
    """
    if not files:
        return cmd_string

    prelude_parts = ["set +e"]
    for path in files:
        # shlex.quote isn't available here without an import; the files list
        # comes from os.path.expanduser output so it's a concrete absolute
        # path.  Escape single quotes defensively anyway.
        safe = path.replace("'", "'\\''")
        prelude_parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
    prelude = "\n".join(prelude_parts) + "\n"
    return prelude + cmd_string


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Spawn-per-call: every execute() spawns a fresh bash process.
    Session snapshot preserves env vars across calls.
    CWD persists via file-based read after each command.
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        if cwd:
            cwd = os.path.expanduser(cwd)
        self.shell_path_style = "git-bash" if _IS_WINDOWS else "posix"
        initial_cwd = self._to_shell_path(cwd or os.getcwd())
        super().__init__(cwd=initial_cwd, timeout=timeout, env=env)
        self.init_session()

    @staticmethod
    def _to_shell_path(path: str) -> str:
        """Normalize local host paths into the shell path format used by this backend."""
        if not _IS_WINDOWS:
            return path
        return _normalize_windows_shell_path(path)

    def get_temp_dir(self) -> str:
        """Return a shell-safe writable temp dir for local execution.

        Termux does not provide /tmp by default, but exposes a POSIX TMPDIR.
        Prefer POSIX-style env vars when available, keep using /tmp on regular
        Unix systems, and only fall back to tempfile.gettempdir() when it also
        resolves to a POSIX path.

        Check the environment configured for this backend first so callers can
        override the temp root explicitly (for example via terminal.env or a
        custom TMPDIR), then fall back to the host process environment.
        """
        if _IS_WINDOWS:
            for env_var in ("TMPDIR", "TMP", "TEMP"):
                candidate = self.env.get(env_var) or os.environ.get(env_var)
                normalized = self._to_shell_path(candidate) if candidate else ""
                if normalized and re.match(r"^[A-Za-z]:/", normalized):
                    return normalized.rstrip("/") or normalized

            return self._to_shell_path(tempfile.gettempdir()).rstrip("/")

        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"

        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"

        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"

        return "/tmp"

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        bash = _find_bash()
        # For login-shell invocations (used by init_session to build the
        # environment snapshot), prepend sources for the user's bashrc /
        # custom init files so tools registered outside bash_profile
        # (nvm, asdf, pyenv, …) end up on PATH in the captured snapshot.
        # Non-login invocations are already sourcing the snapshot and
        # don't need this.
        if login:
            init_files = _resolve_shell_init_files()
            if init_files:
                cmd_string = _prepend_shell_init(cmd_string, init_files)
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]
        run_env = _make_run_env(self.env)

        # Recover when the cwd has been deleted out from under us — usually by
        # a previous tool call that ran ``rm -rf`` on its own working dir
        # (issue #17558).  Popen would otherwise raise FileNotFoundError on
        # the cwd before bash starts, wedging every subsequent call until the
        # gateway restarts.
        cwd_for_popen = self._to_shell_path(self.cwd) if _IS_WINDOWS else self.cwd
        safe_cwd = _resolve_safe_cwd(cwd_for_popen)
        if safe_cwd != cwd_for_popen:
            logger.warning(
                "LocalEnvironment cwd %r is missing on disk; "
                "falling back to %r so terminal commands keep working.",
                self.cwd,
                safe_cwd,
            )
            self.cwd = self._to_shell_path(safe_cwd) if _IS_WINDOWS else safe_cwd
            cwd_for_popen = safe_cwd

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
            cwd=cwd_for_popen,
        )
        if not _IS_WINDOWS:
            try:
                proc._hermes_pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        command = _wrap_windows_native_command(command)
        shell_cwd = self._to_shell_path(cwd) if cwd else ""
        return super().execute(
            command,
            cwd=shell_cwd,
            timeout=timeout,
            stdin_data=stdin_data,
        )

    def _kill_process(self, proc):
        """Kill the entire process group (all children)."""

        def _group_alive(pgid: int) -> bool:
            try:
                # POSIX-only: _IS_WINDOWS is handled before this helper is used.
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # The group exists, even if this process cannot signal it.
                return True

        def _wait_for_group_exit(pgid: int, timeout: float) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                # Reap the wrapper promptly. A dead but unreaped group leader
                # still makes killpg(pgid, 0) report the group as alive.
                try:
                    proc.poll()
                except Exception:
                    pass
                if not _group_alive(pgid):
                    return True
                time.sleep(0.05)
            try:
                proc.poll()
            except Exception:
                pass
            return not _group_alive(pgid)

        try:
            if _IS_WINDOWS:
                proc.terminate()
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = getattr(proc, "_hermes_pgid", None)
                    if pgid is None:
                        raise

                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    return

                # Wait on the process group, not just the shell wrapper. Under
                # load the wrapper can exit before grandchildren do; returning
                # at that point leaves orphaned process-group members behind.
                if _wait_for_group_exit(pgid, 1.0):
                    return

                try:
                    # POSIX-only: _IS_WINDOWS is handled by the outer branch.
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                _wait_for_group_exit(pgid, 2.0)
                try:
                    proc.wait(timeout=0.2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def _update_cwd(self, result: dict):
        """Read CWD from temp file (local-only, no round-trip needed).

        Skip the assignment when the path no longer exists as a directory —
        ``pwd -P`` on a deleted cwd can leave a stale value in the marker
        file, and propagating it would re-wedge the next ``Popen``.  The
        ``_run_bash`` recovery path will resolve a safe fallback if needed.
        """
        try:
            with open(self._cwd_file) as f:
                cwd_path = f.read().strip()
            if cwd_path and os.path.isdir(cwd_path):
                self.cwd = cwd_path
        except (OSError, FileNotFoundError):
            pass

        # Still strip the marker from output so it's not visible
        self._extract_cwd_from_output(result)

    def cleanup(self):
        """Clean up temp files."""
        for f in (self._snapshot_path, self._cwd_file):
            try:
                os.unlink(f)
            except OSError:
                pass
