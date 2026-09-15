"""Browser Runtime Supervisor for Hermes Workstation.

Supervises the live connection to the Electron Chromium controller, actively
detects stale control descriptors, handles connection heartbeats, classifies
network/process errors, and manages exponential backoff and circuit-breaker retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControllerErrorKind(str, Enum):
    NONE = "none"
    DESCRIPTOR_NOT_FOUND = "descriptor_not_found"
    DESCRIPTOR_STALE = "descriptor_stale"
    PROCESS_DEAD = "process_dead"
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "timeout"
    AUTH_MISMATCH = "auth_mismatch"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    ACTION_FAILED = "action_failed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SupervisorHealth:
    healthy: bool
    state: str
    error_kind: ControllerErrorKind = ControllerErrorKind.NONE
    pid: Optional[int] = None
    url: Optional[str] = None
    checked_at: str = field(default_factory=_utc_now)
    detail: str = ""


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BrowserRuntimeSupervisor:
    """Active supervisor for the Workstation Browser controller."""

    def __init__(
        self,
        *,
        control_path: Optional[Path] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        circuit_cooldown_seconds: float = 10.0,
    ) -> None:
        self.control_path = control_path
        self.max_retries = max(1, max_retries)
        self.backoff_factor = max(0.1, backoff_factor)
        self.circuit_cooldown = circuit_cooldown_seconds
        self._consecutive_failures = 0
        self._circuit_state = CircuitBreakerState.CLOSED
        self._circuit_tripped_at: float = 0.0

    def resolve_control_path(self) -> Path:
        if self.control_path is not None:
            return self.control_path
        from tools.browser_workstation import workstation_control_path
        return workstation_control_path()

    def is_pid_alive(self, pid: int) -> bool:
        """Check if process with given PID exists and is running."""
        if pid <= 0:
            return False
        if psutil is not None:
            try:
                p = psutil.Process(pid)
                return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
            except Exception:
                pass
        if os.name == "nt":
            import ctypes
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def inspect_descriptor(self) -> Tuple[Optional[Dict[str, Any]], ControllerErrorKind, str]:
        """Read and validate the descriptor, checking process liveness."""
        path = self.resolve_control_path()
        if not path.exists():
            return None, ControllerErrorKind.DESCRIPTOR_NOT_FOUND, f"Descriptor {path} does not exist"

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, ControllerErrorKind.DESCRIPTOR_STALE, f"Descriptor unparseable: {exc}"

        if not isinstance(data, dict):
            return None, ControllerErrorKind.DESCRIPTOR_STALE, "Descriptor is not a valid JSON object"

        pid = data.get("pid")
        if isinstance(pid, int) and pid > 0:
            if not self.is_pid_alive(pid):
                # Process died abruptly! Clean up or mark stale descriptor
                self._clean_stale_descriptor(path)
                return (
                    data,
                    ControllerErrorKind.PROCESS_DEAD,
                    f"Process PID {pid} is dead, stale descriptor was removed",
                )

        url = data.get("url")
        token = data.get("token")
        if not url or not str(url).startswith("http://127.0.0.1:") or not token:
            return None, ControllerErrorKind.PROTOCOL_MISMATCH, "Descriptor failed loopback/auth validation"

        return data, ControllerErrorKind.NONE, "Descriptor format and PID valid"

    def _clean_stale_descriptor(self, path: Path) -> None:
        try:
            logger.warning("Cleaning up stale browser descriptor from dead process: %s", path)
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Failed removing stale descriptor: %s", exc)

    def classify_error(self, exc: Exception) -> ControllerErrorKind:
        """Classify a connection or runtime error."""
        text = str(exc)
        lowered = text.lower()

        if "10061" in text or "connection refused" in lowered or "actively refused" in lowered:
            return ControllerErrorKind.CONNECTION_REFUSED
        if "timeout" in lowered or "timed out" in lowered:
            return ControllerErrorKind.TIMEOUT
        if "401" in text or "unauthorized" in lowered or "auth" in lowered:
            return ControllerErrorKind.AUTH_MISMATCH
        if "stale" in lowered or "dead" in lowered:
            return ControllerErrorKind.PROCESS_DEAD
        if "protocol mismatch" in lowered:
            return ControllerErrorKind.PROTOCOL_MISMATCH
        return ControllerErrorKind.UNKNOWN

    def check_health(self, *, force: bool = False) -> SupervisorHealth:
        """Probe overall controller and process health."""
        descriptor, err_kind, detail = self.inspect_descriptor()
        if err_kind != ControllerErrorKind.NONE:
            return SupervisorHealth(
                healthy=False,
                state="unavailable",
                error_kind=err_kind,
                detail=detail,
            )

        assert descriptor is not None
        url = descriptor.get("url")
        token = descriptor.get("token")
        pid = descriptor.get("pid")

        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        req = Request(
            f"{url}/health",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(req, timeout=0.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("success"):
                    self._on_success()
                    return SupervisorHealth(
                        healthy=True,
                        state="running",
                        pid=pid,
                        url=url,
                        detail="Controller responding normally",
                    )
        except HTTPError as exc:
            self._on_failure()
            return SupervisorHealth(
                healthy=False,
                state="error",
                error_kind=ControllerErrorKind.AUTH_MISMATCH if exc.code == 401 else ControllerErrorKind.ACTION_FAILED,
                pid=pid,
                url=url,
                detail=f"HTTP error {exc.code}",
            )
        except (URLError, TimeoutError, OSError) as exc:
            self._on_failure()
            kind = self.classify_error(exc)
            return SupervisorHealth(
                healthy=False,
                state="unreachable",
                error_kind=kind,
                pid=pid,
                url=url,
                detail=str(exc),
            )

        self._on_failure()
        return SupervisorHealth(healthy=False, state="unknown", error_kind=ControllerErrorKind.UNKNOWN)

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_state = CircuitBreakerState.CLOSED

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_retries and self._circuit_state == CircuitBreakerState.CLOSED:
            self._circuit_state = CircuitBreakerState.OPEN
            self._circuit_tripped_at = time.monotonic()
            logger.warning("Browser supervisor circuit breaker TRIPPED (open)")

    def can_attempt(self) -> bool:
        if self._circuit_state == CircuitBreakerState.CLOSED:
            return True
        if self._circuit_state == CircuitBreakerState.OPEN:
            now = time.monotonic()
            if now - self._circuit_tripped_at >= self.circuit_cooldown:
                self._circuit_state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN allows 1 probe

    def execute_with_supervision(
        self,
        action_fn: Callable[[], Any],
        *,
        action_name: str = "browser_action",
    ) -> Any:
        """Execute a browser operation with active supervision, error classification, and backoff."""
        if not self.can_attempt():
            raise RuntimeError(
                f"Browser controller circuit breaker is OPEN due to repeated failures. Cooldown: {self.circuit_cooldown}s"
            )

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = action_fn()
                self._on_success()
                return result
            except Exception as exc:
                last_exc = exc
                kind = self.classify_error(exc)
                self._on_failure()
                logger.warning(
                    "Browser action '%s' failed attempt %d/%d [classified as %s]: %s",
                    action_name,
                    attempt,
                    self.max_retries,
                    kind.value,
                    exc,
                )

                if kind in {ControllerErrorKind.PROCESS_DEAD, ControllerErrorKind.DESCRIPTOR_STALE}:
                    # Fast fail if process is confirmed dead; clean up descriptor
                    path = self.resolve_control_path()
                    self._clean_stale_descriptor(path)
                    raise

                if attempt < self.max_retries:
                    backoff = self.backoff_factor * (2 ** (attempt - 1))
                    time.sleep(backoff)

        assert last_exc is not None
        raise last_exc
