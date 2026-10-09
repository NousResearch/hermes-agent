"""Where this profile's desktop lives: the gateway host or inside the configured terminal backend.

``bot_desktop.placement``:
  ``auto``      (default) follow the terminal backend when it is a sandbox that can host a stream
                (docker / ssh / singularity); the gateway host when ``terminal.backend`` is local. A
                sandbox backend that CANNOT host one (modal, daytona, vercel) resolves to ``refused``:
                the user chose a sandbox for the agent's actions, so quietly running the screen, cua-driver
                and the browser on the host beside it would hand the agent a desktop outside that sandbox.
  ``terminal``  always the terminal backend; error when it cannot host one.
  ``gateway``   always the gateway host (the pre-#108914 behaviour for sandboxed users, now an explicit
                opt-in because it is the boundary-crossing shape).

``resolve()`` is pure config + backend-class reasoning: it never starts a sandbox. ``terminal_environment()``
does acquire the profile's terminal environment (creating the container if needed) because a screen cannot
exist before the sandbox does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

GATEWAY = "gateway"
TERMINAL = "terminal"
REFUSED = "refused"
_STREAM_BACKENDS = ("docker", "ssh", "singularity")


@dataclass(frozen=True)
class Placement:
    where: str            # GATEWAY | TERMINAL | REFUSED
    backend: str          # terminal.backend as configured
    reason: str = ""      # human sentence when REFUSED (or why TERMINAL was not possible)


def _setting() -> str:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    value = str(cfg.get("placement") or "auto").strip().lower()
    return value if value in ("auto", TERMINAL, GATEWAY) else "auto"


def _terminal_backend() -> str:
    from tools.terminal_tool import _get_env_config
    return str(_get_env_config().get("env_type") or "local")


def resolve() -> Placement:
    # Backend first (env only): a local terminal IS the gateway host whatever the setting says, and that is
    # the common case — it must not cost a config load (this runs on every browser / cua-driver spawn).
    backend = _terminal_backend()
    if backend == "local":
        return Placement(GATEWAY, backend, "terminal.backend is local, so the terminal IS the gateway host")
    setting = _setting()
    if setting == GATEWAY:
        return Placement(GATEWAY, backend)
    if backend in _STREAM_BACKENDS:
        return Placement(TERMINAL, backend)
    reason = (f"terminal.backend is {backend}, which cannot host a screen yet, and running the desktop on the "
              f"gateway host would put the agent's screen, browser and computer_use outside the sandbox you chose "
              f"for it. Set bot_desktop.placement: gateway to allow that explicitly.")
    return Placement(REFUSED, backend, reason)


def terminal_environment(*, create: bool = True) -> Optional[Any]:
    """This profile's terminal environment object (the one ``terminal`` runs commands in). ``create=False``
    only returns an already-running one.

    On a cache-miss create, a Bot Desktop marker naming a still-alive container rides along as a
    one-shot hint (:func:`_marker_container_hint`) so the planner tries THAT exact container before
    its own label search: a persistent container's reuse key can legitimately drift from what this
    process would compute right now (a profile migrating from a standalone to a multiplexed
    gateway changes persistent Docker's task-id key; any other config/session-context change can
    too), and a restart must not orphan a container the marker proves is still running (#t_2ca05370).
    """
    from tools import terminal_tool as tt
    task_id = tt._resolve_container_task_id(None)
    with tt._env_lock:
        env = tt._lookup_active_env(task_id, None)
    if env is not None or not create:
        return env
    # Acquire through the same planner the terminal tool uses so the container carries the same mounts,
    # limits and identity label; a trivial command warms it.
    tt.terminal_tool("true", task_id=None, _docker_expected_container_id=_marker_container_hint())
    with tt._env_lock:
        return tt._lookup_active_env(task_id, None)


def _marker_container_hint() -> Optional[str]:
    """The Bot Desktop sandbox marker's recorded container id, when it names a Docker-hosted
    screen that is still alive; None otherwise (the planner falls back to its normal behaviour,
    unchanged)."""
    from tools.bot_desktop import sandbox_host
    marker = sandbox_host._read_marker()
    if not marker or marker.get("backend") != "DockerEnvironment":
        return None
    container = marker.get("container")
    if container and sandbox_host.marker_sandbox_alive(marker):
        return container
    return None
