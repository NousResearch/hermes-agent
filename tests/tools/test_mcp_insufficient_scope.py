"""A grant that authenticated and still can't run the tool.

The failure looks like every other auth error and needs the opposite
treatment: refreshing the token hands back the same scopes, so the retry
fails identically and the user is told to re-authenticate — which, done
unchanged, fails again. These tests pin the two things that break that loop:
recognizing the challenge, and answering it with re-consent instead of a
refresh.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


pytest.importorskip("mcp.types")

from tools.mcp_tool_handlers import (  # noqa: E402  -- after importorskip
    _handle_auth_error_and_retry,
    _insufficient_scope,
)


def _refused(challenge: str) -> Exception:
    """An exception carrying a server's WWW-Authenticate challenge."""
    error = RuntimeError("403 Forbidden")
    error.response = SimpleNamespace(  # type: ignore[attr-defined]
        status_code=403,
        headers={"www-authenticate": challenge},
    )
    return error


class TestRecognizingTheChallenge:
    def test_scopes_are_read_off_the_challenge(self):
        assert (
            _insufficient_scope(
                _refused('Bearer error="insufficient_scope", scope="docs.readonly drive.file"')
            )
            == "docs.readonly drive.file"
        )

    def test_a_challenge_without_a_scope_list_is_still_a_scope_failure(self):
        # Empty string, not None: the server said the grant is too narrow but
        # not what would be wide enough.
        assert _insufficient_scope(_refused('Bearer error="insufficient_scope"')) == ""

    def test_an_expired_token_is_not_a_scope_failure(self):
        assert _insufficient_scope(_refused('Bearer error="invalid_token"')) is None

    def test_an_exception_with_no_response_is_not_a_scope_failure(self):
        assert _insufficient_scope(RuntimeError("connection reset")) is None


class TestAnsweringIt:
    def test_the_grant_is_widened_instead_of_refreshed(self):
        """No refresh, no retry, and the model is pointed at the card."""
        retried = []

        with patch("tools.mcp_tool._bump_server_error"):
            result = _handle_auth_error_and_retry(
                "gdocs",
                _refused('Bearer error="insufficient_scope", scope="drive.file"'),
                lambda: retried.append(True),
                "call tool 'create_doc'",
            )

        assert retried == [], "retrying re-sends the same too-narrow grant"

        payload = json.loads(result)
        assert payload["needs_reauth"] is True
        assert payload["server"] == "gdocs"
        assert "drive.file" in payload["error"]
        assert "setup_mcp" in payload["error"]

    def test_an_ordinary_auth_error_still_takes_the_refresh_path(self):
        """The scope branch must not swallow the case it sits in front of."""
        with (
            patch("tools.mcp_tool_handlers._is_auth_error", return_value=True),
            patch("tools.mcp_tool_loop._run_on_mcp_loop", return_value=False) as recover,
            patch("tools.mcp_tool._bump_server_error"),
        ):
            result = _handle_auth_error_and_retry(
                "gdocs", _refused('Bearer error="invalid_token"'), lambda: None, "list tools"
            )

        assert recover.called, "an expired token is exactly what refresh is for"
        assert json.loads(result)["needs_reauth"] is True
