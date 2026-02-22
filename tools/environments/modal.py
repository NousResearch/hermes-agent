"""Modal cloud execution environment wrapping mini-swe-agent's SwerexModalEnvironment."""

import uuid

from tools.environments.base import BaseEnvironment


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via mini-swe-agent.

    Wraps SwerexModalEnvironment and adds sudo -S support.
    Async-safety patches are applied once before first use so Modal
    works inside any event loop (Atropos, gateway, etc.).
    """

    _patches_applied = False

    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60):
        super().__init__(cwd=cwd, timeout=timeout)

        if not ModalEnvironment._patches_applied:
            try:
                from environments.patches import apply_patches
                apply_patches()
            except ImportError:
                pass
            ModalEnvironment._patches_applied = True

        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
        self._inner = SwerexModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            startup_timeout=180.0, runtime_timeout=3600.0,
        )

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command = self._prepare_command(command)
        return self._inner.execute(exec_command, cwd=cwd, timeout=timeout)

    def cleanup(self):
        if hasattr(self._inner, 'stop'):
            self._inner.stop()
