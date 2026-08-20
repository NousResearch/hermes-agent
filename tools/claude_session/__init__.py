"""Claude Session — context pipeline for Claude Code."""

from tools.claude_session.session import ClaudeSession
from tools.claude_session.errors import (
    SessionError, ErrorSeverity,
    TmuxError, TmuxNotFoundError, TmuxTimeoutError,
    SessionDisconnectedError, SessionNotActiveError,
    StartupFailedError, SessionExitedError,
    PermissionError, InvalidPermissionResponseError,
    WaitTimeoutError, StallDetectedError, ValidationError,
    StateDetectionError, wrap_tmux_error,
)
from tools.claude_session.stream_output import SessionOutputStreamer

# Backward compatibility alias
ClaudeSessionManager = ClaudeSession

__all__ = [
    "ClaudeSession", "ClaudeSessionManager",
    "SessionError", "ErrorSeverity",
    "TmuxError", "TmuxNotFoundError", "TmuxTimeoutError",
    "SessionDisconnectedError", "SessionNotActiveError",
    "StartupFailedError", "SessionExitedError",
    "PermissionError", "InvalidPermissionResponseError",
    "WaitTimeoutError", "StallDetectedError", "ValidationError",
    "StateDetectionError", "wrap_tmux_error",
    "SessionOutputStreamer",
]
