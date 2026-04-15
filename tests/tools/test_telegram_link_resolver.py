import json
import importlib
import sys
import types
import asyncio

import pytest

from hermes_state import SessionDB
from tools.telegram_link_resolver import (
    parse_private_telegram_link,
    resolve_private_telegram_link,
)


def test_parse_private_telegram_link_with_thread():
    parsed = parse_private_telegram_link("https://t.me/c/3712897238/712/1031")

    assert parsed == {
        "chat_id": "-1003712897238",
        "thread_id": "712",
        "message_id": "1031",
        "internal_chat_id": "3712897238",
    }


def test_parse_private_telegram_link_without_thread():
    parsed = parse_private_telegram_link("https://t.me/c/3712897238/1031")

    assert parsed == {
        "chat_id": "-1003712897238",
        "thread_id": None,
        "message_id": "1031",
        "internal_chat_id": "3712897238",
    }


def test_resolve_private_telegram_link_from_session_db(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="s1", source="telegram")
    db.append_message(
        "s1",
        role="user",
        content="historic telegram note",
        message_metadata={
            "platform": "telegram",
            "chat_id": "-1003712897238",
            "thread_id": "712",
            "message_id": "1031",
        },
    )
    db.close()

    resolved = resolve_private_telegram_link(
        "https://t.me/c/3712897238/712/1031",
        db_path=db_path,
    )

    assert resolved is not None
    assert resolved["title"] == "Telegram message 1031"
    assert resolved["content"] == "historic telegram note"
    assert resolved["metadata"]["session_id"] == "s1"


def test_resolve_private_telegram_link_returns_clear_error_when_unknown(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="s1", source="telegram")
    db.close()

    resolved = resolve_private_telegram_link(
        "https://t.me/c/3712897238/712/1031",
        db_path=db_path,
    )

    assert resolved is not None
    assert resolved["content"] == ""
    assert "could not resolve" in resolved["error"].lower()


def test_web_extract_short_circuits_private_telegram_links(monkeypatch):
    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")

    sys.modules.setdefault(
        "openai",
        types.SimpleNamespace(OpenAI=_DummyClient, AsyncOpenAI=_DummyClient),
    )
    sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=_DummyClient))
    web_tools = importlib.import_module("tools.web_tools")
    web_extract_tool = web_tools.web_extract_tool

    monkeypatch.setattr(
        "tools.web_tools.resolve_private_telegram_link",
        lambda url: {
            "url": url,
            "title": "Telegram message 1031",
            "content": "native telegram content",
        },
    )
    monkeypatch.setattr(
        "tools.web_tools._get_backend",
        lambda: (_ for _ in ()).throw(AssertionError("backend should not run")),
    )

    result = json.loads(asyncio.run(
        web_extract_tool(
            ["https://t.me/c/3712897238/712/1031"],
            use_llm_processing=False,
        )
    ))

    assert result["results"][0]["title"] == "Telegram message 1031"
    assert result["results"][0]["content"] == "native telegram content"
    assert result["results"][0]["error"] is None


def test_browser_navigate_rejects_private_telegram_links():
    sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
    browser_tool = importlib.import_module("tools.browser_tool")
    browser_navigate = browser_tool.browser_navigate

    result = json.loads(browser_navigate("https://t.me/c/3712897238/712/1031"))

    assert result["success"] is False
    assert result["suggested_tool"] == "web_extract"
    assert "not browser-accessible" in result["error"]
