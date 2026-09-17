"""Unified BrowserSession and Human Takeover / Browser Lease infrastructure.

Unifies browser capabilities (Workstation Chromium, Desktop Preview, Lightpanda)
behind a single contract and provides strict human-agent control leases to prevent
accidental destruction of human logins, CAPTCHAs, or MFA sessions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from uuid import uuid4

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrowserControlMode(str, Enum):
    AGENT = "agent"
    HUMAN = "human"
    SHARED = "shared"


class HumanTakeoverActiveError(RuntimeError):
    """Raised when an automated action attempts a destructive browser operation while human has control."""


DESTRUCTIVE_ACTIONS = frozenset({
    "close_preview",
    "close_tab",
    "browser_close",
    "browser_navigate",
    "navigate",
    "browser_reset",
    "clear_session",
    "logout",
    "refresh",
    "close_terminal",
    "browser_click",
    "click",
    "browser_type",
    "type",
    "browser_press",
    "press",
    "submit",
    "browser_fill",
    "fill",
})


@dataclass(slots=True)
class BrowserControlLease:
    task_id: str
    mode: BrowserControlMode = BrowserControlMode.AGENT
    reason: Optional[str] = None
    acquired_at: str = field(default_factory=_utc_now)
    acquired_by: str = "agent"
    lock_destructive: bool = True
    generation: int = 1
    fence_token: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data


class BrowserControlLeaseManager:
    """Thread-safe manager for browser control ownership and human leases."""

    _instance: Optional["BrowserControlLeaseManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._leases: Dict[str, BrowserControlLease] = {}
        self._mutex = threading.Lock()

    @classmethod
    def get_instance(cls) -> "BrowserControlLeaseManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_lease(self, task_id: str) -> BrowserControlLease:
        with self._mutex:
            if task_id not in self._leases:
                self._leases[task_id] = BrowserControlLease(task_id=task_id)
            return self._leases[task_id]

    def request_human_control(
        self,
        task_id: str,
        reason: str,
        *,
        acquired_by: str = "human",
    ) -> BrowserControlLease:
        """Lock the browser for human intervention (login, MFA, CAPTCHA)."""
        with self._mutex:
            prev = self._leases.get(task_id)
            prev_gen = prev.generation if prev else 0
            lease = BrowserControlLease(
                task_id=task_id,
                mode=BrowserControlMode.HUMAN,
                reason=reason,
                acquired_at=_utc_now(),
                acquired_by=acquired_by,
                lock_destructive=True,
                generation=prev_gen + 1,
                fence_token=uuid4().hex,
            )
            self._leases[task_id] = lease
            logger.info("Human control lease GRANTED for task %s (gen %s): %s", task_id, lease.generation, reason)
            return lease

    def resume_agent_control(
        self,
        task_id: str,
        *,
        acquired_by: str = "agent",
    ) -> BrowserControlLease:
        """Restore agent control after human user finishes intervention."""
        with self._mutex:
            prev = self._leases.get(task_id)
            prev_gen = prev.generation if prev else 0
            lease = BrowserControlLease(
                task_id=task_id,
                mode=BrowserControlMode.AGENT,
                reason=None,
                acquired_at=_utc_now(),
                acquired_by=acquired_by,
                lock_destructive=False,
                generation=prev_gen + 1,
                fence_token=uuid4().hex,
            )
            self._leases[task_id] = lease
            logger.info("Agent control lease RESUMED for task %s (gen %s)", task_id, lease.generation)
            return lease

    def assert_action_allowed(
        self,
        task_id: str,
        action: str,
        *,
        fence_token: Optional[str] = None,
    ) -> None:
        """Enforce protection: raise HumanTakeoverActiveError if action is destructive during human lease or fence token is stale."""
        lease = self.get_lease(task_id)
        if fence_token is not None and fence_token != lease.fence_token:
            raise HumanTakeoverActiveError(
                f"Action '{action}' is BLOCKED: Stale fence token for task '{task_id}'. "
                f"Execution authority was revoked or superseded by generation {lease.generation}."
            )
        if lease.mode == BrowserControlMode.HUMAN and lease.lock_destructive:
            normalized_action = action.strip().lower()
            if normalized_action in DESTRUCTIVE_ACTIONS:
                raise HumanTakeoverActiveError(
                    f"Action '{action}' is BLOCKED: Human Takeover is active for task '{task_id}'. "
                    f"Reason: {lease.reason or 'User manual intervention in progress'}. "
                    f"Please wait for user to complete interaction or call resume_agent_control()."
                )


class BrowserProvider(ABC):
    """Abstract provider for concrete browser implementations."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def navigate(self, url: str, task_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def read(self, task_id: str, *, start: Optional[int] = None, count: Optional[int] = None) -> str: ...

    @abstractmethod
    def execute(self, script: str, task_id: str) -> Any: ...


class WorkstationBrowserProvider(BrowserProvider):
    """Provider backed by the primary Electron Chromium Workstation Controller."""

    def name(self) -> str:
        return "workstation_browser"

    def is_available(self) -> bool:
        from tools.browser_workstation import workstation_controller_available
        return workstation_controller_available()

    def navigate(self, url: str, task_id: str) -> Dict[str, Any]:
        from tools.browser_workstation import workstation_routed_browser_handler
        res = workstation_routed_browser_handler(
            "browser_navigate",
            {"url": url},
            fallback=lambda: {"error": "workstation controller unavailable"},
            task_id=task_id,
        )
        return json.loads(res) if isinstance(res, str) else res

    def read(self, task_id: str, *, start: Optional[int] = None, count: Optional[int] = None) -> str:
        from tools.browser_workstation import workstation_routed_browser_handler
        args: Dict[str, Any] = {}
        res = workstation_routed_browser_handler(
            "browser_snapshot",
            args,
            fallback=lambda: "Error: Workstation browser unavailable",
            task_id=task_id,
        )
        return str(res)

    def execute(self, script: str, task_id: str) -> Any:
        from tools.browser_workstation import workstation_routed_browser_handler
        res = workstation_routed_browser_handler(
            "browser_console",
            {"script": script},
            fallback=lambda: None,
            task_id=task_id,
        )
        return res


class PreviewBrowserProvider(BrowserProvider):
    """Provider backed by the Desktop Preview bridge."""

    def name(self) -> str:
        return "preview_browser"

    def is_available(self) -> bool:
        from tools.desktop_ui import is_desktop_session
        return is_desktop_session()

    def navigate(self, url: str, task_id: str) -> Dict[str, Any]:
        from tools.open_preview_tool import open_preview_tool
        open_preview_tool(url=url)
        return {"success": True, "provider": "preview_browser", "url": url}

    def read(self, task_id: str, *, start: Optional[int] = None, count: Optional[int] = None) -> str:
        from tools.read_preview_tool import read_preview_tool
        return read_preview_tool(start=start, count=count)

    def execute(self, script: str, task_id: str) -> Any:
        from tools.drive_preview_tool import drive_preview_tool
        return drive_preview_tool(action="eval", script=script)


class UnifiedBrowserSession:
    """Unified session facade exposing seamless browser operations to agents and tasks."""

    def __init__(
        self,
        task_id: str,
        *,
        lease_manager: Optional[BrowserControlLeaseManager] = None,
        primary_provider: Optional[BrowserProvider] = None,
        fallback_provider: Optional[BrowserProvider] = None,
    ) -> None:
        self.task_id = task_id
        self.lease_mgr = lease_manager or BrowserControlLeaseManager.get_instance()
        self.primary = primary_provider or WorkstationBrowserProvider()
        self.fallback = fallback_provider or PreviewBrowserProvider()

    def active_provider(self) -> BrowserProvider:
        if self.primary.is_available():
            return self.primary
        if self.fallback.is_available():
            return self.fallback
        return self.primary

    def request_human_control(self, reason: str) -> BrowserControlLease:
        return self.lease_mgr.request_human_control(self.task_id, reason)

    def resume_agent_control(self) -> BrowserControlLease:
        return self.lease_mgr.resume_agent_control(self.task_id)

    def navigate(self, url: str) -> Dict[str, Any]:
        self.lease_mgr.assert_action_allowed(self.task_id, "browser_navigate")
        provider = self.active_provider()
        return provider.navigate(url, self.task_id)

    def read(self, *, start: Optional[int] = None, count: Optional[int] = None) -> str:
        self.lease_mgr.assert_action_allowed(self.task_id, "read")
        provider = self.active_provider()
        return provider.read(self.task_id, start=start, count=count)

    def execute(self, script: str) -> Any:
        self.lease_mgr.assert_action_allowed(self.task_id, "execute")
        provider = self.active_provider()
        return provider.execute(script, self.task_id)

    def close(self) -> None:
        self.lease_mgr.assert_action_allowed(self.task_id, "close_tab")
        # Allowed if lease permits
