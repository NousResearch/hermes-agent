"""Slash workers must not own the parent TUI conversation worktree."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch


def test_slash_worker_disables_conversation_worktree_management(monkeypatch):
    """Internal slash workers reuse the parent TUI session's lifecycle."""
    from tui_gateway import slash_worker

    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "draft-key"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    with (
        patch.object(slash_worker, "_start_parent_death_watchdog"),
        patch.object(slash_worker, "_prepare_slash_worker_runtime"),
        patch.object(slash_worker, "HermesCLI", return_value=MagicMock()) as cli_type,
    ):
        slash_worker.main()

    assert cli_type.call_args.kwargs["resume"] == "draft-key"
    assert cli_type.call_args.kwargs["manage_conversation_worktree"] is False
