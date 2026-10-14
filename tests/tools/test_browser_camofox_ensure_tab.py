"""Regression test: _ensure_tab must send ``listItemId`` (not ``sessionKey``).

The Camoufox REST API server requires ``listItemId`` in the ``POST /tabs``
body.  A previous version sent ``sessionKey`` which caused a 400 Bad Request
on every ``browser_navigate`` call.  See issue #37960.
"""

from unittest.mock import patch, MagicMock


def test_ensure_tab_sends_list_item_id():
    """POST /tabs body must contain ``listItemId``, not ``sessionKey``."""
    # Import the module under test
    from tools import browser_camofox as mod

    fake_session = {
        "user_id": "hermes_test123",
        "tab_id": None,
        "session_key": "task_my-session",
        "managed": False,
        "adopt_existing_tab": False,
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {"tabId": "tab-42"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mod, "_get_session", return_value=fake_session), \
         patch.object(mod, "get_camofox_url", return_value="http://localhost:9377"), \
         patch("tools.browser_camofox.requests.post", return_value=mock_response) as mock_post:
        result = mod._ensure_tab("test-task", url="https://example.com")

    # Verify the POST was called
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # Core assertion: listItemId present, sessionKey absent
    assert "listItemId" in body, f"Expected 'listItemId' in POST body, got: {body}"
    assert "sessionKey" not in body, f"'sessionKey' should not be in POST body: {body}"
    assert body["listItemId"] == "task_my-session"
    assert body["userId"] == "hermes_test123"
    assert body["url"] == "https://example.com"

    # Verify tab_id was set from response
    assert result["tab_id"] == "tab-42"


def test_ensure_tab_skips_creation_when_tab_exists():
    """If session already has a tab_id, no POST should be made."""
    from tools import browser_camofox as mod

    fake_session = {
        "user_id": "hermes_test123",
        "tab_id": "existing-tab",
        "session_key": "task_my-session",
        "managed": False,
    }

    with patch.object(mod, "_get_session", return_value=fake_session), \
         patch("tools.browser_camofox.requests.post") as mock_post:
        result = mod._ensure_tab("test-task")

    # No POST should be made — tab already exists
    mock_post.assert_not_called()
    assert result["tab_id"] == "existing-tab"
