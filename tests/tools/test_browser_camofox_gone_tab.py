"""Regression test: ``_navigate_tab`` must recreate the tab on 410 Gone, not only 404.

Camofox >= 1.17 answers ``410 Gone`` for a tab id it no longer knows (for example
after the server container restarted). Hermes caches the tab id per task, so if
only 404 is treated as "tab is gone" every later ``browser_navigate`` in the
session fails with ``Navigation failed: 410 Client Error: Gone`` until Hermes is
restarted.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests


def _http_error(status: int) -> requests.HTTPError:
    resp = MagicMock()
    resp.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=resp)


@pytest.mark.parametrize("status", [404, 410])
def test_navigate_tab_recreates_tab_when_server_says_gone(status):
    from tools import browser_camofox as mod

    session = {"user_id": "hermes_test", "tab_id": "dead-tab", "session_key": "task_x",
               "managed": False, "adopt_existing_tab": False}
    calls = []

    def fake_post(path, body, timeout=None):
        calls.append(path)
        if path.endswith("/navigate"):
            raise _http_error(status)
        return {"tabId": "fresh-tab"}

    with patch.object(mod, "_get_session", return_value=session), \
         patch.object(mod, "_post", side_effect=fake_post):
        result_session, data = mod._navigate_tab("task", "https://example.com")

    assert result_session["tab_id"] == "fresh-tab"
    assert data == {"ok": True, "url": "https://example.com"}
    assert calls == ["/tabs/dead-tab/navigate", "/tabs"]


def test_navigate_tab_reraises_other_http_errors():
    from tools import browser_camofox as mod

    session = {"user_id": "hermes_test", "tab_id": "tab", "session_key": "task_x",
               "managed": False, "adopt_existing_tab": False}

    with patch.object(mod, "_get_session", return_value=session), \
         patch.object(mod, "_post", side_effect=_http_error(500)):
        with pytest.raises(requests.HTTPError):
            mod._navigate_tab("task", "https://example.com")
    assert session["tab_id"] == "tab"
