"""Regression coverage for browser_back() when a backend omits data."""
from __future__ import annotations

import json

from tools import browser_tool as bt


def test_browser_back_handles_success_with_none_data(monkeypatch):
    """A successful backend response may legally omit the data payload."""
    monkeypatch.setattr(
        bt._session,
        "_run_browser_command",
        lambda *args, **kwargs: {"success": True, "data": None},
    )
    monkeypatch.setattr(bt, "_blocked_private_page", lambda *args, **kwargs: None)

    result = json.loads(bt.browser_back("qa-back-none"))

    assert result["success"] is True
    assert result["url"] == ""
