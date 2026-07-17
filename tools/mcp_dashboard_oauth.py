"""Dashboard-mediated callback bridge for MCP OAuth.

The MCP SDK remains responsible for discovery, DCR, PKCE, state validation and
token exchange. This module only moves the two human/browser callbacks from a
loopback listener into the already-authenticated dashboard session.
"""

from __future__ import annotations

import asyncio
import contextvars
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import parse_qs, urlparse


@dataclass
class DashboardOAuthFlow:
    flow_id: str
    server_name: str
    profile: str | None
    redirect_uri: str
    created_at: float = field(default_factory=time.time)
    status: str = "starting"
    authorization_url: str | None = None
    error: str | None = None
    tools: list[dict] = field(default_factory=list)
    expected_state: str | None = field(default=None, init=False)
    _callback: tuple[str, str | None] | None = field(default=None, init=False, repr=False)
    _callback_error: str | None = field(default=None, init=False, repr=False)
    _authorization_ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _callback_ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    async def publish_authorization_url(self, url: str) -> None:
        if self.status in {"approved", "error"}:
            raise RuntimeError("OAuth flow already ended")
        state = parse_qs(urlparse(url).query).get("state", [None])[0]
        if not state:
            raise ValueError("OAuth authorization URL did not include state")
        self.expected_state = state
        self.authorization_url = url
        self.status = "authorization_required"
        self._authorization_ready.set()

    async def wait_for_authorization_url(self, timeout: float = 30.0) -> str:
        ready = await asyncio.to_thread(self._authorization_ready.wait, timeout)
        if not ready:
            raise TimeoutError("Timed out waiting for MCP authorization URL")
        if not self.authorization_url:
            raise RuntimeError(self.error or "MCP OAuth flow ended before authorization")
        return self.authorization_url

    def deliver_callback(
        self,
        *,
        code: str | None,
        state: str | None,
        error: str | None,
    ) -> None:
        if self._callback_ready.is_set():
            raise ValueError("OAuth callback already received")
        if (
            self.expected_state is None
            or state is None
            or not secrets.compare_digest(self.expected_state, state)
        ):
            raise ValueError("OAuth callback state mismatch")
        if error:
            self._callback_error = error
        elif code:
            self._callback = (code, state)
        else:
            self._callback_error = "OAuth callback did not include code or error"
        self._callback_ready.set()

    async def wait_for_callback(self, timeout: float = 300.0) -> tuple[str, str | None]:
        ready = await asyncio.to_thread(self._callback_ready.wait, timeout)
        if not ready:
            raise TimeoutError("Timed out waiting for MCP OAuth callback")
        if self._callback_error:
            raise RuntimeError(f"OAuth authorization failed: {self._callback_error}")
        if self._callback is None:
            raise RuntimeError("OAuth callback did not include an authorization code")
        return self._callback

    def mark_approved(self) -> None:
        self.status = "approved"
        self.error = None

    def mark_error(self, error: str) -> None:
        self.status = "error"
        self.error = error
        self._authorization_ready.set()
        self._callback_ready.set()

    def snapshot(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "server_name": self.server_name,
            "status": self.status,
            "authorization_url": self.authorization_url,
            "error": self.error,
        }


_current_dashboard_flow: contextvars.ContextVar[DashboardOAuthFlow | None] = (
    contextvars.ContextVar("mcp_dashboard_oauth_flow", default=None)
)


@contextmanager
def dashboard_oauth_flow(flow: DashboardOAuthFlow) -> Iterator[None]:
    token = _current_dashboard_flow.set(flow)
    try:
        yield
    finally:
        _current_dashboard_flow.reset(token)


def get_dashboard_oauth_flow() -> DashboardOAuthFlow | None:
    return _current_dashboard_flow.get()
