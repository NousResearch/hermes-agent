"""tools/claude_session/tmux_interface.py — Low-level tmux operations."""

import logging
import os
import signal
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TmuxInterface:
    """Encapsulates all tmux CLI interactions for a single session."""

    def __init__(self, session_name: str):
        self.session_name = session_name

    def _run(self, args: list, timeout: int = 10) -> subprocess.CompletedProcess:
        """Run a tmux command."""
        cmd = ["tmux"] + args
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("tmux command timed out: %s", " ".join(args))
            raise
        except FileNotFoundError:
            raise RuntimeError("tmux is not installed or not in PATH")

    def session_exists(self) -> bool:
        """Check if the tmux session exists."""
        r = self._run(["has-session", "-t", self.session_name])
        return r.returncode == 0

    def create_session(self, workdir: str, env: Optional[dict] = None) -> str:
        """Create a new detached tmux session. Returns session name."""
        cmd = [
            "new-session",
            "-d",
            "-s", self.session_name,
            "-c", workdir,
        ]
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
        r = self._run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to create tmux session: {r.stderr}")
        return self.session_name

    def capture_pane(self, lines: int = 200) -> str:
        """Capture visible pane output. Returns raw text with ANSI codes."""
        r = self._run([
            "capture-pane",
            "-t", self.session_name,
            "-p",
            "-S", f"-{lines}",
        ])
        return r.stdout if r.returncode == 0 else ""

    def send_keys(self, text: str, enter: bool = False) -> None:
        """Send text to the tmux session, optionally pressing Enter.

        Uses send-keys -l for literal text (no special key name interpretation).
        This avoids the need for manual escaping since subprocess.run passes
        arguments directly without shell interpretation.

        IMPORTANT: This method performs NO delays. The caller is responsible
        for timing — multi-line text needs a delay before Enter to allow
        bracketed paste to complete, but that delay must NOT be inside a lock.
        """
        cmd = ["send-keys", "-t", self.session_name, "-l", text]
        self._run(cmd)
        if enter:
            self.send_special_key("Enter")

    def send_special_key(self, key: str) -> None:
        """Send a special key sequence (e.g., C-c, C-d, Enter)."""
        self._run(["send-keys", "-t", self.session_name, key])

    def kill_session(self) -> None:
        """Kill the tmux session.

        tmux kill-session closes the pty and sends SIGHUP to child processes.
        However, Claude Code may setsid and escape the SIGHUP — fall back to
        SIGTERM the pane PID if the session leader is gone.
        """
        try:
            self._run(["kill-session", "-t", self.session_name], timeout=5)
            logger.info("Tmux session %s killed", self.session_name)
        except Exception as e:
            logger.warning("tmux kill-session failed for %s: %s", self.session_name, e)
            # Fallback: find the pane PID and SIGTERM it
            try:
                result = self._run(
                    ["list-panes", "-t", self.session_name, "-F", "#{pane_pid}"], timeout=5
                )
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.lstrip("-").isdigit():
                        pid = int(line)
                        try:
                            os.kill(pid, signal.SIGTERM)
                            logger.info("Sent SIGTERM to pane PID %d", pid)
                        except ProcessLookupError:
                            pass
            except Exception as sig_e:
                logger.warning("SIGTERM fallback also failed for %s: %s", self.session_name, sig_e)
