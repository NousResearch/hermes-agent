"""Desktop prompt-turn delivery gates for the optional original Slack thread mirror."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hermes_cli.slack_desktop_sync import mirror_row


def accepted_user(db, session_id: str, row_id: Any, *, desktop: bool, visible: bool,
                  send: Callable | None = None, auth_test: Callable | None = None) -> bool:
    if not desktop or not visible:
        return False
    return mirror_row(db, session_id, row_id, 'user', send=send, auth_test=auth_test)


def completed_final(db, session_id: str, receipt: dict | None, *, status: str, text: Any,
                    desktop: bool, successful: bool = True,
                    send: Callable | None = None, auth_test: Callable | None = None) -> bool:
    from gateway.response_filters import is_intentional_silence_response
    if (not desktop or not successful or status != 'complete' or not isinstance(text, str) or not text.strip()
            or is_intentional_silence_response(text) or not isinstance(receipt, dict)
            or receipt.get('complete') is not True):
        return False
    return mirror_row(db, session_id, receipt.get('final_assistant_row_id'), 'assistant',
                      send=send, auth_test=auth_test)
