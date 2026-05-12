import json
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
