"""Base class for all Hermes execution environment backends."""

from abc import ABC, abstractmethod
import os
import subprocess
from pathlib import Path


_TERMINAL_BLOCKED_PROVIDER_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "KIMI_API_KEY",
    "KIMI_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_BASE_URL",
    "MINIMAX_CN_API_KEY",
    "MINIMAX_CN_BASE_URL",
    "NOUS_API_KEY",
    "NOUS_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
})


def get_sandbox_dir() -> Path:
    """Return the host-side root for all sandbox storage (Docker workspaces,
    Singularity overlays/SIF cache, etc.).

    Configurable via TERMINAL_SANDBOX_DIR. Defaults to ~/.hermes/sandboxes/.
    """
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    if custom:
        p = Path(custom)
    else:
        p = Path.home() / ".hermes" / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_terminal_subprocess_env(extra_env: dict | None = None) -> dict:
    """Build a subprocess env without leaking Hermes runtime provider state."""
    env = dict(os.environ)
    for key in _TERMINAL_BLOCKED_PROVIDER_ENV_KEYS:
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    return env


class BaseEnvironment(ABC):
    """Common interface for all Hermes execution backends.

    Subclasses implement execute() and cleanup(). Shared helpers eliminate
    duplicated subprocess boilerplate across backends.
    """

    def __init__(self, cwd: str, timeout: int, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}

    @abstractmethod
    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command, return {"output": str, "returncode": int}."""
        ...

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared helpers (eliminate duplication across backends)
    # ------------------------------------------------------------------

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available.

        Returns:
            (transformed_command, sudo_stdin) — see _transform_sudo_command
            for the full contract.  Callers that drive a subprocess directly
            should prepend sudo_stdin (when not None) to any stdin_data they
            pass to Popen.  Callers that embed stdin via heredoc (modal,
            daytona) handle sudo_stdin in their own execute() method.
        """
        from tools.terminal_tool import _transform_sudo_command
        return _transform_sudo_command(command)

    def _build_run_kwargs(self, timeout: int | None,
                          stdin_data: str | None = None) -> dict:
        """Build common subprocess.run kwargs for non-interactive execution."""
        kw = {
            "text": True,
            "timeout": timeout or self.timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if stdin_data is not None:
            kw["input"] = stdin_data
        else:
            kw["stdin"] = subprocess.DEVNULL
        return kw

    def _timeout_result(self, timeout: int | None) -> dict:
        """Standard return dict when a command times out."""
        return {
            "output": f"Command timed out after {timeout or self.timeout}s",
            "returncode": 124,
        }
