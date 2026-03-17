"""Local execution environment with interrupt support and non-blocking I/O."""

import glob
import os
import platform
import shutil
import signal
import subprocess
import threading
import time

_IS_WINDOWS = platform.system() == "Windows"

from tools.environments.base import BaseEnvironment
from tools.environments.persistent_shell import PersistentShellMixin
from tools.interrupt import is_interrupted

# Unique marker to isolate real command output from shell init/exit noise.
# printf (no trailing newline) keeps the boundaries clean for splitting.
_OUTPUT_FENCE = "__HERMES_FENCE_a9f7b3__"

# Hermes-internal env vars that should NOT leak into terminal subprocesses.
# These are loaded from ~/.hermes/.env for Hermes' own LLM/provider calls
# but can break external CLIs (e.g. codex) that also honor them.
# See: https://github.com/NousResearch/hermes-agent/issues/1002
#
# Built dynamically from the provider registry so new providers are
# automatically covered without manual blocklist maintenance.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config.

    Automatically picks up api_key_env_vars and base_url_env_var from
    every registered provider, plus tool/messaging env vars from the
    optional config registry, so new Hermes-managed secrets are blocked
    in subprocesses without having to maintain multiple static lists.
    """
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

    # Vars not covered above but still Hermes-internal / conflict-prone.
    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",         # legacy alias
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_TOKEN",         # OAuth token (not in registry as env var)
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_MODEL",
        # Expanded isolation for other major providers (Issue #1002)
        "GOOGLE_API_KEY",          # Gemini / Google AI Studio
        "DEEPSEEK_API_KEY",        # DeepSeek
        "MISTRAL_API_KEY",         # Mistral AI
        "GROQ_API_KEY",            # Groq
        "TOGETHER_API_KEY",        # Together AI
        "PERPLEXITY_API_KEY",      # Perplexity
        "COHERE_API_KEY",          # Cohere
        "FIREWORKS_API_KEY",       # Fireworks AI
        "XAI_API_KEY",             # xAI (Grok)
        "HELICONE_API_KEY",        # LLM Observability proxy
        # Gateway/runtime config not represented in OPTIONAL_ENV_VARS.
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
        # Skills Hub / GitHub app auth paths and aliases.
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        # Remote sandbox backend credentials.
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
    })
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment.

    `_HERMES_FORCE_<VAR>` entries in ``extra_env`` opt a blocked variable back in
    intentionally for callers that truly need it.
    """
    sanitized: dict[str, str] = {}

    for key, value in (base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST:
            sanitized[key] = value

    for key, value in (extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            sanitized[real_key] = value
        elif key not in _HERMES_PROVIDER_ENV_BLOCKLIST:
            sanitized[key] = value

    return sanitized


def _find_bash() -> str:
    """Find bash for command execution.

    The fence wrapper uses bash syntax (semicolons, $?, printf), so we
    must use bash — not the user's $SHELL which could be fish/zsh/etc.
    On Windows: uses Git Bash (bundled with Git for Windows).
    """
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")  # last resort: whatever they have
            or "/bin/sh"
        )

    # Windows: look for Git Bash (installed with Git for Windows).
    # Allow override via env var (same pattern as Claude Code).
    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    # shutil.which finds bash.exe if Git\bin is on PATH
    found = shutil.which("bash")
    if found:
        return found

    # Check common Git for Windows install locations
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


# Backward compat — process_registry.py imports this name
_find_shell = _find_bash


# Noise lines emitted by interactive shells when stdin is not a terminal.
# Used as a fallback when output fence markers are missing.
_SHELL_NOISE_SUBSTRINGS = (
    # bash
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
    # zsh / oh-my-zsh / macOS terminal session
    "Restored session:",
    "Saving session...",
    "Last login:",
    "command not found:",
    "Oh My Zsh",
    "compinit:",
)


def _clean_shell_noise(output: str) -> str:
    """Strip shell startup/exit warnings that leak when using -i without a TTY.

    Removes lines matching known noise patterns from both the beginning
    and end of the output.  Lines in the middle are left untouched.
    """

    def _is_noise(line: str) -> bool:
        return any(noise in line for noise in _SHELL_NOISE_SUBSTRINGS)

    lines = output.split("\n")

    # Strip leading noise
    while lines and _is_noise(lines[0]):
        lines.pop(0)

    # Strip trailing noise (walk backwards, skip empty lines from split)
    end = len(lines) - 1
    while end >= 0 and (not lines[end] or _is_noise(lines[end])):
        end -= 1

    if end < 0:
        return ""

    cleaned = lines[: end + 1]
    result = "\n".join(cleaned)

    # Preserve trailing newline if original had one
    if output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


_SANE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    merged = dict(os.environ | env)
    run_env = {}
    for k, v in merged.items():
        if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            run_env[real_key] = v
        elif k not in _HERMES_PROVIDER_ENV_BLOCKLIST:
            run_env[k] = v
    existing_path = run_env.get("PATH", "")
    if "/usr/bin" not in existing_path.split(":"):
        run_env["PATH"] = f"{existing_path}:{_SANE_PATH}" if existing_path else _SANE_PATH
    return run_env


def _extract_fenced_output(raw: str) -> str:
    """Extract real command output from between fence markers.

    The execute() method wraps each command with printf(FENCE) markers.
    This function finds the first and last fence and returns only the
    content between them, which is the actual command output free of
    any shell init/exit noise.

    Falls back to pattern-based _clean_shell_noise if fences are missing.
    """
    first = raw.find(_OUTPUT_FENCE)
    if first == -1:
        return _clean_shell_noise(raw)

    start = first + len(_OUTPUT_FENCE)
    last = raw.rfind(_OUTPUT_FENCE)

    if last <= first:
        # Only start fence found (e.g. user command called `exit`)
        return _clean_shell_noise(raw[start:])

    return raw[start:last]


class LocalEnvironment(PersistentShellMixin, BaseEnvironment):
    """Run commands directly on the host machine.

    Features:
    - Popen + polling for interrupt support (user can cancel mid-command)
    - Background stdout drain thread to prevent pipe buffer deadlocks
    - stdin_data support for piping content (bypasses ARG_MAX limits)
    - sudo -S transform via SUDO_PASSWORD env var
    - Uses interactive login shell so full user env is available
    - Optional persistent shell mode (cwd/env vars survive across calls)
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None,
                 persistent: bool = False):
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)
        self.persistent = persistent
        if self.persistent:
            self._init_persistent_shell()

    @property
    def _temp_prefix(self) -> str:
        return f"/tmp/hermes-local-{self._session_id}"

    def _spawn_shell_process(self) -> subprocess.Popen:
        user_shell = _find_bash()
        run_env = _make_run_env(self.env)
        return subprocess.Popen(
            [user_shell, "-l"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=run_env,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
        )

    def _read_temp_files(self, *paths: str) -> list[str]:
        results = []
        for path in paths:
            if os.path.exists(path):
                with open(path) as f:
                    results.append(f.read())
            else:
                results.append("")
        return results

    def _kill_shell_children(self):
        if self._shell_pid is None:
            return
        try:
            subprocess.run(
                ["pkill", "-P", str(self._shell_pid)],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _cleanup_temp_files(self):
        for f in glob.glob(f"{self._temp_prefix}-*"):
            if os.path.exists(f):
                os.remove(f)

    def _execute_oneshot(self, command: str, cwd: str = "", *,
                         timeout: int | None = None,
                         stdin_data: str | None = None) -> dict:
        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        exec_command, sudo_stdin = self._prepare_command(command)

        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        user_shell = _find_bash()
        fenced_cmd = (
            f"printf '{_OUTPUT_FENCE}';"
            f" {exec_command};"
            f" __hermes_rc=$?;"
            f" printf '{_OUTPUT_FENCE}';"
            f" exit $__hermes_rc"
        )
        run_env = _make_run_env(self.env)

        proc = subprocess.Popen(
            [user_shell, "-lic", fenced_cmd],
            text=True,
            cwd=work_dir,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if effective_stdin is not None else subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
        )

        if effective_stdin is not None:
            def _write_stdin():
                try:
                    proc.stdin.write(effective_stdin)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_write_stdin, daemon=True).start()

        _output_chunks: list[str] = []

        def _drain_stdout():
            try:
                for line in proc.stdout:
                    _output_chunks.append(line)
            except ValueError:
                pass
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()
        deadline = time.monotonic() + effective_timeout

        while proc.poll() is None:
            if is_interrupted():
                try:
                    if _IS_WINDOWS:
                        proc.terminate()
                    else:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                reader.join(timeout=2)
                return {
                    "output": "".join(_output_chunks) + "\n[Command interrupted — user sent a new message]",
                    "returncode": 130,
                }
            if time.monotonic() > deadline:
                try:
                    if _IS_WINDOWS:
                        proc.terminate()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                reader.join(timeout=2)
                return self._timeout_result(effective_timeout)
            time.sleep(0.2)

        reader.join(timeout=5)
        output = _extract_fenced_output("".join(_output_chunks))
        return {"output": output, "returncode": proc.returncode}
