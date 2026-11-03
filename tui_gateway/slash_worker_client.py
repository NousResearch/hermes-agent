"""The gateway side of the persistent slash worker: ``_SlashWorker`` starts ``tui_gateway.slash_worker`` as
an owned child (``spawn_owned``), talks JSON lines to it, and stops its whole process group on close.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module), so they reference
server.py globals bare: ``_hermes_home`` and ``_SLASH_WORKER_TIMEOUT_S``. Tests patch the names on
``tui_gateway.server`` (``spawn_owned``, ``_stop_worker_group``, ``os``), where these bodies read them.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

from hermes_cli.process_identity import release_process_group_once_empty, spawn_owned
from hermes_constants import get_hermes_home

from .method_ctx import bind_module


def _prepend_tool_paths(env: dict[str, str]) -> dict[str, str]:
    """Prepend managed bin (managed-first policy for the Browser Use CLI), venv bin and
    ~/.local/bin to PATH so slash_worker children resolve Hermes-managed CLIs under the Desktop's minimal PATH.
    The PM store dirs go in front of all of them: a user's node/uv in ~/.local/bin never wins."""
    managed_bin = ""
    with contextlib.suppress(Exception):
        managed_bin = str(Path(get_hermes_home()) / "bin")
    venv_bin = str(Path(sys.executable).parent)  # <venv>/bin (POSIX) or <venv>/Scripts (Windows)
    parts = [p for p in (managed_bin, venv_bin, str(Path.home() / ".local" / "bin"), env.get("PATH") or "") if p]
    env["PATH"] = os.pathsep.join(parts)
    with contextlib.suppress(Exception):
        import pm

        env["PATH"] = pm.store_first_path(env["PATH"])
    return env


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str, profile_home: str | None = None,
                 provider: str | None = None):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()
        # ``--provider`` pins the child to the parent agent's virtual provider: without it the
        # worker re-resolves provider from config, so a MoA session (provider=moa, model=<preset>)
        # dispatched its preset NAME to the configured real provider and 402/503'd (#57283).
        argv = [sys.executable, "-P", "-m", "tui_gateway.slash_worker", "--session-key", session_key] \
            + (["--model", model] if model else []) \
            + (["--provider", provider] if provider else [])
        self._closed = False
        from hermes_cli._subprocess_compat import windows_hide_flags
        # slash_worker runs the Hermes agent → needs provider credentials. Tier-1 secrets
        # (gateway/GitHub/infra) are still stripped (#29157). Global-remote / multi-profile sessions: the
        # worker must resolve config/skills/state against the session's profile home, not the gateway's
        # launch HERMES_HOME (#40677).
        from tools.environments.local import served_profile_child_env
        from agent.secret_scope import is_multiplex_active

        # The worker runs the agent → needs provider credentials; tier-1 secrets (gateway/GitHub/
        # infra) are still stripped. A served profile's worker gets THAT profile's home + secrets and
        # none of the launch profile's .env / TERMINAL_* residue, exactly what a standalone
        # `hermes -p X` would load itself. The launch profile is a profile too: once the process hosts
        # a second home (multiplex flipped), its worker must name its own home or the fail-closed
        # no-target/no-scope path raises UnscopedSecretError (#115427).
        env = _prepend_tool_paths(served_profile_child_env(
            target_home=profile_home or (_hermes_home if is_multiplex_active() else None),
            inherit_credentials=True))
        # Internal slash workers must import the same checkout as their parent; -P keeps their cwd off sys.path.
        module_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (module_root, env.get("PYTHONPATH", "")) if part
        )
        # spawn_owned: the worker leads its own process group (otherwise it inherits the gateway's
        # pgid and mcp_tool's orphan sweep, racing the spawn, killpg()s the TUI parent itself), and
        # that group dies with this process however it dies: SIGKILL/OOM of the gateway used to leave
        # a quick command's shell, and anything else the worker started in its group, running forever.
        # errors="replace": bytes invalid in the system locale (GBK Windows) must not raise
        # UnicodeDecodeError in the drain threads.
        # Prepend the Hermes venv bin dir and the user-local bin dir to PATH so slash_worker child processes
        # can resolve Hermes-managed CLIs (browser-use, uvx) even when the parent gateway was launched with
        # a minimal PATH (e.g. by the Desktop/Dashboard app). See #83845.
        self.proc = spawn_owned(
            argv, purpose="slash-worker", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=os.getcwd(), env=env, creationflags=windows_hide_flags())
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            with contextlib.suppress(json.JSONDecodeError):
                self.stdout_queue.put(json.loads(line))
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in self.proc.stderr or []:
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        """Run one command; return its output text.

        A command like /prompt may also have parked a next-turn prompt (a "seed")
        on the worker CLI; it rides back on the reply's ``seed`` field and is
        retrieved separately via ``pop_seed()``.
        """
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")
        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()
            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                self._last_seed = str(msg.get("seed", "") or "")
                return str(msg.get("output", "")).rstrip()
            raise RuntimeError(
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}")

    def pop_seed(self) -> str:
        """Return and clear the seed from the last ``run()`` (empty when none)."""
        seed, self._last_seed = getattr(self, "_last_seed", ""), ""
        return seed

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self.proc
        try:
            _stop_worker_group(proc)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                with contextlib.suppress(Exception):
                    stream.close()


def _stop_worker_group(proc) -> None:
    """TERM the slash worker, give it 1 s, then KILL; never raises.

    POSIX signals the worker's whole process group (``spawn_owned`` made it the leader): signalling
    the leader alone left a running quick command's shell, or anything else the worker started in
    its group, alive after every session close. The group is released from the owner-death
    supervisor once it is empty; a group that is not stays covered.
    """
    if os.name != "posix":
        for stop in (proc.terminate, proc.kill):
            if proc.poll() is not None:
                return
            with contextlib.suppress(Exception):
                stop()
                proc.wait(timeout=1)  # also reaps the zombie a kill leaves behind
        return
    pgid = proc.pid
    # Only a leader still alive on entry proves the number is still this worker's group. A worker
    # that already died leaves its group to the owner-death supervisor rather than risk signalling
    # a recycled pgid.
    if proc.poll() is None:
        for sig in (signal.SIGTERM, signal.SIGKILL):  # windows-footgun: ok — POSIX-only, Windows returned above
            # Members that outlive the leader keep the group (and its number) alive.
            with contextlib.suppress(OSError):
                os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only, Windows returned above
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
    # Killed members are reaped by init asynchronously; one still alive after that stays covered.
    release_process_group_once_empty(pgid, proc=proc)


def register(server) -> None:
    """Publish the worker class and its helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server)
