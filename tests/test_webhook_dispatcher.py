"""Tests for the outbound webhook dispatcher (agent/webhook_dispatcher.py)."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def tmp_hermes_home(tmp_path):
    """Provide a temporary hermes_home directory for webhook storage."""
    return tmp_path


@pytest.fixture()
def dispatcher(tmp_hermes_home):
    """Return a fresh WebhookDispatcher using the temp storage."""
    from agent.webhook_dispatcher import WebhookDispatcher
    return WebhookDispatcher(hermes_home=tmp_hermes_home)


class TestWebhookRegistration:
    """Test webhook register/unregister/list operations."""

    def test_register_returns_id(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        assert wh_id is not None
        assert isinstance(wh_id, str)
        assert len(wh_id) > 0

    def test_register_and_list(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook1",
            events=["task.completed"],
        )
        dispatcher.register(
            url="https://example.com/hook2",
            events=["cron.completed", "cron.failed"],
        )
        webhooks = dispatcher.list_webhooks()
        assert len(webhooks) == 2
        urls = {wh["url"] for wh in webhooks}
        assert "https://example.com/hook1" in urls
        assert "https://example.com/hook2" in urls

    def test_unregister(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        assert len(dispatcher.list_webhooks()) == 1
        result = dispatcher.unregister(wh_id)
        assert result is True
        assert len(dispatcher.list_webhooks()) == 0

    def test_unregister_nonexistent(self, dispatcher):
        result = dispatcher.unregister("nonexistent-id")
        assert result is False

    def test_secret_redacted_in_list(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
            secret="my-secret-key",
        )
        webhooks = dispatcher.list_webhooks()
        assert len(webhooks) == 1
        assert webhooks[0]["secret"] == "***"

    def test_secret_not_redacted_when_none(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        webhooks = dispatcher.list_webhooks()
        assert len(webhooks) == 1
        assert webhooks[0]["secret"] is None

    def test_get_webhook_by_id(self, dispatcher):
        wh_id = dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
            secret="test-secret",
        )
        cfg = dispatcher.get_webhook(wh_id)
        assert cfg is not None
        assert cfg.url == "https://example.com/hook"
        assert cfg.secret == "test-secret"
        assert cfg.events == ["task.completed"]

    def test_get_webhook_nonexistent(self, dispatcher):
        cfg = dispatcher.get_webhook("nonexistent-id")
        assert cfg is None

    def test_persistence(self, tmp_hermes_home):
        """Webhooks survive a new dispatcher instance."""
        from agent.webhook_dispatcher import WebhookDispatcher

        d1 = WebhookDispatcher(hermes_home=tmp_hermes_home)
        wh_id = d1.register(
            url="https://example.com/persist",
            events=["task.completed"],
        )
        # New instance should load from file
        d2 = WebhookDispatcher(hermes_home=tmp_hermes_home)
        assert len(d2.list_webhooks()) == 1
        cfg = d2.get_webhook(wh_id)
        assert cfg is not None
        assert cfg.url == "https://example.com/persist"


class TestEventMatching:
    """Test the event matching logic used in dispatch()."""

    def test_exact_match(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        results = dispatcher.dispatch("task.completed", {"task_id": "abc"})
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_no_match(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        results = dispatcher.dispatch("cron.completed", {"job_id": "xyz"})
        assert len(results) == 0

    def test_star_matches_all(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["*"],
        )
        results = dispatcher.dispatch("task.completed", {"task_id": "abc"})
        assert len(results) == 1

    def test_multiple_event_patterns(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed", "cron.completed"],
        )
        results1 = dispatcher.dispatch("task.completed", {"task_id": "abc"})
        results2 = dispatcher.dispatch("cron.completed", {"job_id": "xyz"})
        results3 = dispatcher.dispatch("task.failed", {"task_id": "abc"})
        assert len(results1) == 1
        assert len(results2) == 1
        assert len(results3) == 0


class TestHMACSigning:
    """Test HMAC-SHA256 payload signing."""

    def test_sign_format(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig = WebhookDispatcher._sign_payload(b"hello", "secret")
        assert sig.startswith("sha256=")
        assert len(sig) == len("sha256=") + 64  # hex digest

    def test_sign_deterministic(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret")
        sig2 = WebhookDispatcher._sign_payload(b"hello", "secret")
        assert sig1 == sig2

    def test_sign_different_secret(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret1")
        sig2 = WebhookDispatcher._sign_payload(b"hello", "secret2")
        assert sig1 != sig2

    def test_sign_different_payload(self):
        from agent.webhook_dispatcher import WebhookDispatcher
        sig1 = WebhookDispatcher._sign_payload(b"hello", "secret")
        sig2 = WebhookDispatcher._sign_payload(b"world", "secret")
        assert sig1 != sig2


class TestDispatch:
    """Test dispatching events to webhooks."""

    def test_dispatch_calls_matching_webhooks(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            results = dispatcher.dispatch("task.completed", {"task_id": "abc"})

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["status_code"] == 200

        # Verify the request was made
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["event"] == "task.completed"
        assert body["data"]["task_id"] == "abc"

    def test_dispatch_skips_non_matching(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        results = dispatcher.dispatch("cron.completed", {"job_id": "xyz"})
        assert len(results) == 0

    def test_dispatch_no_webhooks(self, dispatcher):
        """Dispatching when no webhooks are registered returns empty results."""
        results = dispatcher.dispatch("task.completed", {"task_id": "abc"})
        assert results == []

    def test_dispatch_with_hmac_signature(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
            secret="my-test-secret",
        )
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            dispatcher.dispatch("task.completed", {"task_id": "abc"})

        req = mock_urlopen.call_args[0][0]
        assert "X-Hermes-Signature" in req.headers
        assert req.headers["X-Hermes-Signature"].startswith("sha256=")

    def test_dispatch_retries_on_failure(self, dispatcher):
        dispatcher.register(
            url="https://example.com/hook",
            events=["task.completed"],
        )
        import urllib.error
        with patch("urllib.request.urlopen") as mock_urlopen, \
             patch("agent.webhook_dispatcher.time.sleep") as mock_sleep:
            # First two attempts fail, third succeeds
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.side_effect = [
                urllib.error.URLError("connection refused"),
                urllib.error.URLError("timeout"),
                mock_resp,
            ]

            results = dispatcher.dispatch("task.completed", {"task_id": "abc"})

        assert len(results) == 1
        assert results[0]["success"] is True
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2  # slept between attempts
