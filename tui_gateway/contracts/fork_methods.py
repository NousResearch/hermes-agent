"""Contract registrations for fork-specific passthrough methods.

These methods are registered as lambda handlers in tui_gateway/server.py
but don't have upstream contract entries. Minimal contracts keep the
contract-registry completeness test happy.
"""

from __future__ import annotations

from pydantic import Field

from .base import Params, Result
from .common import ProfileParams
from .registry import method


class RespondParams(Params):
    """Passthrough respond methods carry a free-form payload."""
    pass


class RespondResult(Result):
    """Passthrough respond methods return nothing structured."""
    pass


class WorktreeCleanupParams(Params):
    session_id: str = ""


class WorktreeCleanupResult(Result):
    cleaned: bool = False


# Passthrough respond methods (lambda handlers in server.py)
method("mcp.setup.respond", params=RespondParams, result=RespondResult, doc="MCP setup respond passthrough")
method("preview.act.respond", params=RespondParams, result=RespondResult, doc="Preview act respond passthrough")
method("preview.read.respond", params=RespondParams, result=RespondResult, doc="Preview read respond passthrough")
method("secret.respond", params=RespondParams, result=RespondResult, doc="Secret respond passthrough")
method("sudo.respond", params=RespondParams, result=RespondResult, doc="Sudo respond passthrough")
method("terminal.read.respond", params=RespondParams, result=RespondResult, doc="Terminal read respond passthrough")
method("tour.respond", params=RespondParams, result=RespondResult, doc="Tour respond passthrough")
method("window.read.respond", params=RespondParams, result=RespondResult, doc="Window read respond passthrough")

# Worktree cleanup
method("session.worktree_cleanup", params=WorktreeCleanupParams, result=WorktreeCleanupResult,
       doc="Clean up a conversation worktree")
