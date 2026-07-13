"""Sprites execution environment.

Uses the sprites-py SDK (https://github.com/superfly/sprites-py) to run
commands in Sprites — stateful cloud sandboxes on Fly.io, with
checkpoint & restore. Persistent by default: each Sprite outlives the session
and is reused via a deterministic, profile-scoped ``hermes-{profile}-{task_id}``
name (``hermes-{task_id}`` on the default profile). Cleanup leaves the Sprite
running when ``persistent_filesystem`` is True; the Sprite is deleted otherwise.
"""

import logging
import os
import re
import shlex
import threading
from pathlib import Path

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
)

logger = logging.getLogger(__name__)


def _slugify_name_component(value: str) -> str:
    """Reduce an arbitrary string to a Sprite/Fly-safe name component.

    Sprite names are DNS-ish: lowercase ``[a-z0-9-]`` with no leading/trailing
    or doubled hyphens. Anything else (``/``, ``.``, uppercase, unicode from a
    profile directory or subagent id) is collapsed to a single hyphen.
    """
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def _resolve_sprite_name(task_id: str) -> str:
    """Deterministic, profile-scoped Sprite name.

    A Sprite is persistent and resumed *by name*, so its name is the durable
    identity of a session's live sandbox (processes, sockets, PID space — not
    just a filesystem snapshot). We scope the name by the active Hermes profile
    so two independent profiles never resume into one another's live Sprite,
    while the same ``(profile, task_id)`` always resumes the same Sprite.

    The default profile keeps the historical ``hermes-{task_id}`` name so
    already-created Sprites keep resolving after this change.
    """
    try:
        from agent.file_safety import _resolve_active_profile_name
        profile = _resolve_active_profile_name()
    except Exception:
        profile = "default"
    task_slug = _slugify_name_component(task_id) or "default"
    profile_slug = _slugify_name_component(profile) if profile else ""
    if profile_slug and profile_slug != "default":
        return f"hermes-{profile_slug}-{task_slug}"
    return f"hermes-{task_slug}"


class SpritesEnvironment(BaseEnvironment):
    """Sprites backend: stateful cloud sandboxes on Fly.io.

    Spawn-per-call via ``_ThreadedProcessHandle`` wrapping blocking
    ``sprite.command(...).combined_output()`` calls. The SDK timeout is
    used (rather than wrapping the shell), since the SDK already cancels
    the underlying WebSocket exec on deadline.
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        cwd: str = "/root",
        timeout: int = 60,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.sprites", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))

        from sprites import SpritesClient
        from sprites.exceptions import NotFoundError, SpriteError

        self._NotFoundError = NotFoundError
        self._SpriteError = SpriteError

        token = os.getenv("SPRITES_TOKEN") or os.getenv("SPRITE_TOKEN")
        if not token:
            raise ValueError(
                "Sprites backend requires SPRITES_TOKEN. "
                "Run `hermes setup terminal` or set SPRITES_TOKEN in .env."
            )
        self._client = SpritesClient(
            token=token,
            timeout=max(30.0, float(timeout)),
        )
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._lock = threading.Lock()
        self._sprite = None

        # Sprites does not yet honor SpriteConfig sizing knobs (cpu / ram /
        # storage / region) — sandboxes get default sizing. We omit SpriteConfig
        # entirely so the wire format stays minimal until the platform exposes
        # these knobs.
        sprite_name = _resolve_sprite_name(task_id)
        self._sprite_name = sprite_name
        try:
            self._sprite = self._client.get_sprite(sprite_name)
            logger.info(
                "Sprites: resumed existing sprite %s for task %s",
                self._sprite.name, task_id,
            )
        except NotFoundError:
            self._sprite = self._client.create_sprite(sprite_name)
            logger.info(
                "Sprites: created sprite %s for task %s",
                self._sprite.name, task_id,
            )

        # Detect remote home dir for .hermes sync target.
        self._remote_home = "/root"
        try:
            from sprites.exceptions import ExitError
            cmd = self._sprite.command("bash", "-c", "echo $HOME", timeout=15)
            home = cmd.combined_output().decode().strip()
            if home:
                self._remote_home = home
                if requested_cwd in {"~", "/root"}:
                    self.cwd = home
        except Exception:
            pass

        self._fs = self._sprite.filesystem("/")
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._sprite_upload,
            delete_fn=self._sprite_delete,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    # ------------------------------------------------------------------
    # File sync callbacks
    # ------------------------------------------------------------------

    def _sprite_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via the SpriteFilesystem API."""
        data = Path(host_path).read_bytes()
        remote = self._fs / remote_path
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_bytes(data)

    def _sprite_delete(self, remote_paths: list[str]) -> None:
        """Delete remote files; missing entries are tolerated."""
        for rp in remote_paths:
            try:
                (self._fs / rp).unlink(missing_ok=True)
            except Exception as e:
                logger.debug("Sprites: delete %s failed: %s", rp, e)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking SDK call."""
        sprite = self._sprite
        from sprites.exceptions import ExitError, TimeoutError as SpritesTimeout

        if login:
            shell_cmd = ["bash", "-l", "-c", cmd_string]
        else:
            shell_cmd = ["bash", "-c", cmd_string]

        # The SDK timeout cancels the WebSocket cleanly, so prefer it over
        # the shell-level ``timeout`` wrapper used by other backends.
        cmd_timeout = float(timeout) if timeout and timeout > 0 else None

        def exec_fn() -> tuple[str, int]:
            cmd = sprite.command(*shell_cmd, timeout=cmd_timeout)
            try:
                output = cmd.combined_output()
                return (output.decode("utf-8", errors="replace"), 0)
            except ExitError as e:
                # ``e.stdout`` carries the combined output when raised from
                # combined_output(); ``e.stderr`` is empty in that path.
                buf = (e.stdout or b"") + (e.stderr or b"")
                return (buf.decode("utf-8", errors="replace"),
                        e.exit_code() if callable(getattr(e, "exit_code", None)) else 1)
            except SpritesTimeout:
                return (f"command timed out after {cmd_timeout}s\n", 124)

        # No external cancel: the SDK does not expose a kill hook on a
        # running Cmd. The deadline above is the cancellation path.
        return _ThreadedProcessHandle(exec_fn, cancel_fn=None)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        with self._lock:
            if self._sprite is None:
                return

            # No sync_back: the Sprite's persistent ext4 filesystem IS the
            # authoritative store. Files the agent touched stay in the Sprite
            # and are visible again on the next session that resumes by the
            # same task_id. (For ephemeral runs with persistent=False, the
            # Sprite is intentionally deleted with its filesystem.)

            try:
                if self._persistent:
                    logger.info(
                        "Sprites: leaving sprite %s running (persistent)",
                        self._sprite.name,
                    )
                else:
                    self._sprite.delete()
                    logger.info("Sprites: deleted sprite %s", self._sprite.name)
            except Exception as e:
                logger.warning("Sprites: cleanup failed: %s", e)
            finally:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._sprite = None
