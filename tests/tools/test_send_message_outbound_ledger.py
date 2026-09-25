"""Outbound-first sends are booked in the delivery ledger.

``hermes send``, cron in-channel sends, the kanban notifier and the MCP server all
funnel through ``send_message_tool._handle_send``; until now none of them left a
trace in ``delivery_obligations``, so delivery accounting saw only the
gateway-response half of outbound traffic (production receipt: a
``hermes send -t weixin`` approval push invisible in the ledger, gateway.log AND
the session mirror on the same day). These tests drive the real
``send_message_tool`` entrypoint against a temp-HERMES_HOME ledger.
"""

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gateway.delivery_ledger as dl
from gateway.config import Platform
from tools.send_message_tool import send_message_tool

CHAT_ID = "c1-home"


@pytest.fixture
def ledger_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    db = home / "state.db"
    monkeypatch.setattr(dl, "_db_path", lambda: db)
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    return db


def _send(message, *, ok=True):
    """Invoke the real tool entrypoint against a patched standalone send path."""
    config = SimpleNamespace(
        platforms={Platform.SIGNAL: SimpleNamespace(enabled=True, token=None, extra={})},
        get_home_channel=lambda _p: SimpleNamespace(chat_id=CHAT_ID),
    )

    async def _record(platform, pconfig, chat_id, message, **kwargs):
        if ok:
            return {"success": True, "message_id": "m1"}
        return {"error": "connection refused"}

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "tools.interrupt.is_interrupted", return_value=False
    ), patch("model_tools._run_async", side_effect=lambda c: asyncio.run(c)), patch(
        "tools.send_message_tool._send_to_platform", side_effect=_record
    ), patch("gateway.mirror.mirror_to_session", return_value=False):
        return json.loads(
            send_message_tool({"action": "send", "target": "signal", "message": message})
        )


def _rows(db):
    with sqlite3.connect(db) as conn:
        # A send that books nothing must leave the ledger empty — including the
        # never-created-table case the real opener would have initialized lazily.
        dl._initialize_schema(conn)
        return conn.execute(
            "SELECT platform, chat_id, thread_id, state, attempts, content "
            "FROM delivery_obligations"
        ).fetchall()


def test_successful_outbound_send_books_delivered_ledger_row(ledger_db):
    result = _send("deploy finished")
    assert result["success"] is True
    assert _rows(ledger_db) == [("signal", CHAT_ID, None, "delivered", 0, "deploy finished")]


def test_failed_outbound_send_books_no_row(ledger_db):
    result = _send("deploy finished", ok=False)
    assert "error" in result
    assert _rows(ledger_db) == []


def test_ledger_disabled_books_no_row(ledger_db, monkeypatch):
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)  # override the fixture's on
    result = _send("deploy finished")
    assert result["success"] is True
    assert _rows(ledger_db) == []
