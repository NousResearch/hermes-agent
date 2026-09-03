"""Regression tests for the rollback.diff truncation flags (#101744).

``rollback.diff`` caps the returned diff at 4000 chars. Before the fix the
payload was indistinguishable from a complete diff; it now carries
``truncated`` and ``total_length`` so clients can surface that the preview
was cut short mid-line.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    yield mod


class _FakeMgr:
    """Stands in for CheckpointManager — only ``diff`` is reachable."""

    def __init__(self, diff: str, stat: str = "1 file changed"):
        self._diff = diff
        self._stat = stat

    def diff(self, cwd: str, commit_hash: str) -> dict:
        return {"success": True, "stat": self._stat, "diff": self._diff}


def _call(server, monkeypatch, diff_text: str) -> dict:
    monkeypatch.setattr(server, "_sess", lambda params, rid: ({"cols": 80}, None))
    monkeypatch.setattr(
        server,
        "_with_checkpoints",
        lambda session, fn: fn(_FakeMgr(diff_text), "/tmp/proj"),
    )
    monkeypatch.setattr(server, "_resolve_checkpoint_hash", lambda mgr, cwd, ref: ref)
    monkeypatch.setattr(server, "render_diff", lambda raw, cols: "")
    resp = server._methods["rollback.diff"]("r1", {"session_id": "sid", "hash": "abc123"})
    assert "error" not in resp
    return resp["result"]


def test_long_diff_is_flagged_truncated(server, monkeypatch):
    full = "x" * 4500
    result = _call(server, monkeypatch, full)
    assert result["diff"] == full[:4000]
    assert result["truncated"] is True
    assert result["total_length"] == 4500


def test_short_diff_is_not_flagged(server, monkeypatch):
    result = _call(server, monkeypatch, "short diff")
    assert result["diff"] == "short diff"
    assert result["truncated"] is False
    assert result["total_length"] == len("short diff")


def test_exactly_4000_chars_is_not_truncated(server, monkeypatch):
    full = "y" * 4000
    result = _call(server, monkeypatch, full)
    assert result["diff"] == full
    assert result["truncated"] is False
    assert result["total_length"] == 4000


def test_empty_diff_keeps_flags(server, monkeypatch):
    result = _call(server, monkeypatch, "")
    assert result["diff"] == ""
    assert result["truncated"] is False
    assert result["total_length"] == 0
