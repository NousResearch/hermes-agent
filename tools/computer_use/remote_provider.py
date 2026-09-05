"""The remote computer-use provider: a desktop on another machine, over MCP.

The agent dials out to a host bridge (see ``host_bridge.py``) running on the
machine whose keyboard and mouse should be driven. The bridge owns cua-driver,
its display, and its permissions; this side only needs the transport. That
inverts the deployment story of the built-in provider: a headless gateway
with no display and no cua-driver binary still gets the tool, because the
machine that matters is elsewhere.

Configuration — ``config.yaml``::

    computer_use:
      provider: remote
      remote:
        enabled: true          # optional; ``provider: remote`` implies intent
        url: https://host:8765/mcp   # bare host normalizes to /mcp

and a bearer token of at least 32 bytes in ``.env`` as
``HERMES_CUA_REMOTE_TOKEN``. Everything fails closed: unreadable config or a
missing/short token means no backend, never a quiet fall back to the local
desktop (``tools.computer_use.remote`` documents each rule).

``is_available`` is config-only by contract: it must be cheap and must not
touch the network, and it deliberately does not install or probe the local
cua-driver — the remote host owns the driver, so the local binary is
irrelevant to whether this provider can service calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import register_provider

if TYPE_CHECKING:
    from tools.computer_use.backend import ComputerUseBackend

logger = logging.getLogger(__name__)


class RemoteCuaProvider(ComputerUseProvider):
    """cua-driver on a remote machine, reached through an authenticated MCP host bridge."""

    @property
    def name(self) -> str:
        return "remote"

    @property
    def display_name(self) -> str:
        return "Remote desktop (MCP host bridge)"

    def _resolve_config(self) -> Optional[Any]:
        """Active remote config, or None when the transport is not configured.

        Broken config (bad URL, token missing/short, control characters) raises rather
        than resolving to None: an operator who wrote ``provider: remote`` must not
        have the tool silently fall back to the local desktop.
        """
        from tools.computer_use.remote import resolve_remote_cua_config

        cfg = self._computer_use_cfg()
        if not cfg:
            return None
        return resolve_remote_cua_config(cfg, permission_mode="standard")

    @staticmethod
    def _computer_use_cfg() -> Dict[str, Any]:
        """The ``computer_use`` config block, or ``{}`` when config is unreadable."""
        import contextlib

        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config

            return (load_config() or {}).get("computer_use") or {}
        return {}

    def is_available(self) -> bool:
        """True iff remote config resolves. Config-only — no network, no local binary probe.

        A resolve that raises is an unavailable provider (the seam's contract: a throwing
        provider is an absent one); the dispatcher's error surfaces the cause.
        """
        try:
            return self._resolve_config() is not None
        except Exception:  # noqa: BLE001 — config errors make the runtime absent
            logger.debug("remote CUA provider availability check failed", exc_info=True)
            return False

    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        """Build the cua-driver backend with the remote transport attached.

        ``create_backend`` raising here is the provider contract for "the runtime is
        known to be gone" — a misconfigured remote surfaces at dispatch with its cause,
        not minutes later as a start() timeout.
        """
        remote_cfg = self._resolve_config()
        if remote_cfg is None:
            raise RuntimeError(
                "computer_use.provider is 'remote' but no remote transport is configured — "
                "set computer_use.remote.url and HERMES_CUA_REMOTE_TOKEN (see "
                "`hermes computer-use doctor`)"
            )

        from tools.computer_use.cua_backend import CuaDriverBackend

        return CuaDriverBackend(permission_mode=permission_mode, remote_config=remote_cfg)


register_provider(RemoteCuaProvider())