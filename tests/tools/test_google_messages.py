import json
import sys
import types
from pathlib import Path

from tools import google_messages


def test_default_profile_path_uses_dedicated_google_messages_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert google_messages._default_profile_path() == (
        Path(tmp_path) / "browser-profiles" / "google-messages"
    )


def test_parse_conversation_item_extracts_sender_timestamp_snippet_and_unread():
    item = google_messages._parse_conversation_item(
        {
            "text": "Alice\nYesterday\nCan you grab tomatoes?",
            "aria_label": "Unread conversation with Alice",
            "class_name": "conversation unread",
        }
    )

    assert item == {
        "sender": "Alice",
        "timestamp": "Yesterday",
        "snippet": "Can you grab tomatoes?",
        "unread": True,
    }


def test_parse_conversation_items_deduplicates_and_limits():
    raw = [
        {"text": "Alice\n10:34 AM\nFirst"},
        {"text": "Alice\n10:34 AM\nFirst"},
        {"text": "Bob\nTue\nSecond"},
    ]

    conversations = google_messages._parse_conversation_items(raw, limit=1)

    assert conversations == [
        {"sender": "Alice", "timestamp": "10:34 AM", "snippet": "First", "unread": False}
    ]


def test_classify_status_pairing_required_when_qr_hint_and_no_conversations():
    status = google_messages._classify_status(
        "https://messages.google.com/web",
        "Messages for web Pair with QR code Scan the code on your phone",
        [],
    )

    assert status["state"] == "pairing_required"
    assert status["pairing_required"] is True
    assert status["ready"] is False


def test_classify_status_pairing_required_wins_over_setup_list_items():
    status = google_messages._classify_status(
        "https://messages.google.com/web",
        "Messages for web Pair with QR code Scan the code on your phone",
        [
            {"text": "Set up Messages\nChoose an account\nContinue"},
            {"text": "Keep your phone connected\nLearn more"},
        ],
    )

    assert status["state"] == "pairing_required"
    assert status["pairing_required"] is True
    assert status["ready"] is False


def test_classify_status_login_required_for_google_sign_in_url():
    status = google_messages._classify_status(
        "https://accounts.google.com/signin/v2/identifier",
        "Sign in with your Google Account to continue to Messages",
        [],
    )

    assert status["state"] == "login_required"
    assert status["login_required"] is True
    assert status["ready"] is False


def test_classify_status_login_required_for_sign_in_page_text():
    status = google_messages._classify_status(
        "https://messages.google.com/web",
        "Choose an account Sign in Google Account",
        [],
    )

    assert status["state"] == "login_required"
    assert status["login_required"] is True
    assert status["ready"] is False


def test_classify_status_login_required_wins_over_generic_pairing_hint():
    status = google_messages._classify_status(
        "https://accounts.google.com/signin/v2/identifier",
        "Choose an account Sign in with your Google Account to use Messages on the web",
        [],
    )

    assert status["state"] == "login_required"
    assert status["login_required"] is True
    assert status["pairing_required"] is True
    assert status["ready"] is False


def test_classify_status_ready_when_conversation_candidates_parse():
    status = google_messages._classify_status(
        "https://messages.google.com/web/conversations",
        "Messages",
        [{"text": "Alice\nToday\nHello"}],
    )

    assert status["state"] == "ready"
    assert status["pairing_required"] is False
    assert status["ready"] is True
    assert status["conversation_candidates"] == 1


def test_parse_conversation_item_does_not_treat_month_name_contact_as_timestamp():
    item = google_messages._parse_conversation_item(
        {"text": "May Flowers\nToday\nGarden party?"}
    )

    assert item == {
        "sender": "May Flowers",
        "timestamp": "Today",
        "snippet": "Garden party?",
        "unread": False,
    }


def test_tool_handlers_return_setup_hint_when_playwright_launch_fails(monkeypatch, tmp_path):
    def boom(profile_path, headless, timeout_ms):
        raise RuntimeError("no browser goblin installed")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(google_messages, "_with_page", boom)

    payload = json.loads(google_messages.google_messages_status(headless=True))

    assert payload["ok"] is False
    assert "no browser goblin" in payload["error"]
    assert payload["profile_path"] == str(tmp_path / "browser-profiles" / "google-messages")
    assert "python -m playwright install chromium" in payload["setup_hint"]


def test_with_page_cleans_up_context_and_playwright_when_goto_fails(monkeypatch, tmp_path):
    calls = []

    class FakePage:
        def set_default_timeout(self, timeout_ms):
            calls.append(("timeout", timeout_ms))

        def goto(self, *args, **kwargs):
            calls.append(("goto", args, kwargs))
            raise RuntimeError("navigation exploded")

    class FakeContext:
        pages = [FakePage()]

        def close(self):
            calls.append(("context.close",))

    class FakeChromium:
        def launch_persistent_context(self, *args, **kwargs):
            calls.append(("launch", args, kwargs))
            return FakeContext()

    class FakePlaywright:
        chromium = FakeChromium()

        def stop(self):
            calls.append(("pw.stop",))

    class FakeSyncPlaywright:
        def start(self):
            calls.append(("start",))
            return FakePlaywright()

    fake_playwright_pkg = types.ModuleType("playwright")
    fake_sync_api = types.ModuleType("playwright.sync_api")
    fake_sync_api.sync_playwright = lambda: FakeSyncPlaywright()
    monkeypatch.setitem(sys.modules, "playwright", fake_playwright_pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api)

    try:
        google_messages._with_page(tmp_path / "profile", headless=True, timeout_ms=1234)
    except RuntimeError as exc:
        assert "navigation exploded" in str(exc)
    else:
        raise AssertionError("_with_page should re-raise navigation failures")

    assert ("context.close",) in calls
    assert ("pw.stop",) in calls
