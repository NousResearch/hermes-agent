"""errors.py — Unified error taxonomy for claude_session.

All session-layer errors are wrapped in SessionError subclasses so callers
(gateway, tools) can branch on error type instead of parsing strings.

Severity levels guide the gateway's response:
  - TRANSIENT: auto-retry is reasonable (network blip, tmux hiccup)
  - RECOVERABLE: user action or restart can fix (permission denied, startup fail)
  - FATAL: cannot be recovered in-session (tmux not installed, auth revoked)
"""

from enum import Enum


class ErrorSeverity(str, Enum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class SessionError(Exception):
    """Base for all session errors."""

    severity: ErrorSeverity = ErrorSeverity.RECOVERABLE
    retryable: bool = False

    def __init__(self, message: str, *, detail: str = ""):
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error": str(self),
            "error_type": type(self).__name__,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "detail": self.detail,
        }


# --- Infrastructure errors (tmux / subprocess) ---

class TmuxError(SessionError):
    """Tmux command failed or session was lost."""
    severity = ErrorSeverity.TRANSIENT
    retryable = True


class TmuxNotFoundError(SessionError):
    """tmux binary is not installed."""
    severity = ErrorSeverity.FATAL
    retryable = False


class TmuxTimeoutError(SessionError):
    """Tmux command timed out."""
    severity = ErrorSeverity.TRANSIENT
    retryable = True


class SessionDisconnectedError(SessionError):
    """Tmux session disappeared (crashed / killed externally)."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


# --- Session lifecycle errors ---

class SessionNotActiveError(SessionError):
    """Operation attempted on a session that isn't running."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


class StartupFailedError(SessionError):
    """Claude Code failed to become IDLE within the startup timeout."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = True


class SessionExitedError(SessionError):
    """Claude Code process has exited."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


# --- Permission errors ---

class PermissionError(SessionError):
    """Permission prompt handling failed."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


class InvalidPermissionResponseError(SessionError):
    """Caller passed an invalid permission response value."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


# --- Wait / timeout errors ---

class WaitTimeoutError(SessionError):
    """wait_for_idle exceeded its deadline."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = True


class StallDetectedError(SessionError):
    """No output growth for the stall threshold duration."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = True


# --- Validation errors ---

class ValidationError(SessionError):
    """Invalid input parameter."""
    severity = ErrorSeverity.RECOVERABLE
    retryable = False


# --- Detection / parsing errors ---

class StateDetectionError(SessionError):
    """Failed to parse Claude Code output state."""
    severity = ErrorSeverity.TRANSIENT
    retryable = True


def wrap_tmux_error(exc: Exception) -> SessionError:
    """Convert a raw tmux/subprocess exception into the appropriate SessionError."""
    import subprocess

    if isinstance(exc, FileNotFoundError):
        return TmuxNotFoundError("tmux is not installed or not in PATH")
    if isinstance(exc, subprocess.TimeoutExpired):
        return TmuxTimeoutError(f"tmux command timed out: {exc}", detail=str(exc))
    msg = str(exc).lower()
    if "session not found" in msg or "can't find session" in msg or "no session" in msg:
        return SessionDisconnectedError(f"tmux session lost: {exc}", detail=str(exc))
    return TmuxError(f"tmux error: {exc}", detail=str(exc))
