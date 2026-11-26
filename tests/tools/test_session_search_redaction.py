"""Discovery output is derived recall text: strict redaction on every model-visible field (#113165)."""

import json

import pytest

from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tools.session_search_tool import session_search

SECRET_BODY = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
SECRET = "ghp_" + SECRET_BODY
SK_TAIL = "abcdefghij1234567890abcdefghij"
SK_KEY = "sk-" + "proj-" + SK_TAIL


def _search(db, **kwargs):
    """The tool as the agent calls it: ``db`` and ``persistence`` both the store handle."""
    return json.loads(session_search(db=db, persistence=db, **kwargs))


@pytest.fixture
def db(tmp_path):
    db = storage_handle(tmp_path)
    seed_session(db, "s_secret", "cli")
    seed_messages(db, "s_secret", {"role": "user", "content": "please deploy the service"})
    seed_messages(db, "s_secret", {"role": "assistant", "content": "sure", "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": f"export GITHUB_TOKEN={SECRET}"})}}]})
    seed_messages(db, "s_secret", {"role": "user", "content": f"deployneedle token is {SECRET}"})
    seed_messages(db, "s_secret", {"role": "assistant", "content": "deployed"})
    return db


def _dump(entry) -> str:
    return json.dumps(entry, ensure_ascii=False)


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_discovery_never_returns_secret_in_message_bodies(db, monkeypatch, detail):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)  # user opted out; derived text stays strict
    out = _search(db, query="deployneedle", detail=detail)
    assert out["results"], out
    hit = out["results"][0]
    assert any(m.get("anchor") for m in hit["messages"])
    assert SECRET not in _dump(hit)


def test_title_match_discovery_redacts_window(db, monkeypatch):
    db.sessions.set_title("s_secret", "release runbook")
    out = _search(db, query="release runbook")
    hit = out["results"][0]
    assert hit["matched_role"] == "session_title"
    assert SECRET not in _dump(hit)


def test_explicit_session_read_stays_verbatim(db):
    read = _search(db, session_id="s_secret")
    assert SECRET in _dump(read["messages"])


@pytest.fixture
def snippet_db(tmp_path):
    db = storage_handle(tmp_path)
    seed_session(db, "s_snip", "cli")
    seed_messages(db, "s_snip", {"role": "user", "content": f"github token {SECRET} and openai key {SK_KEY} saved"})
    seed_messages(db, "s_snip", {"role": "assistant", "content": "ok"})
    seed_session(db, "s_crop", "cli")
    filler = " ".join(f"w{i}" for i in range(38))
    seed_messages(db, "s_crop", {"role": "user", "content": f"{SECRET} {filler} cropneedle"})
    seed_messages(db, "s_crop", {"role": "assistant", "content": "ok"})
    return db


# FTS snippet() puts highlight markers inside the token and crops its prefix; both broke
# post-hoc redaction of the snippet field.
@pytest.mark.parametrize("detail", ["compact", "full"])
@pytest.mark.parametrize("query", ["ghp", SECRET_BODY, "proj", "cropneedle"])
def test_snippet_never_leaks_secret(snippet_db, monkeypatch, detail, query):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    out = _search(snippet_db, query=query, detail=detail)
    assert out["results"], out
    blob = _dump(out["results"])
    assert SECRET_BODY not in blob and SK_TAIL not in blob
    if query == "cropneedle":
        assert ">>>cropneedle<<<" in out["results"][0]["snippet"]


IMAGE_URL = "data:image/png;base64," + "iVBORw0KGgo" + "A" * 64


@pytest.fixture
def multipart_db(tmp_path):
    db = storage_handle(tmp_path)
    seed_session(db, "s_mm", "cli")
    seed_messages(db, "s_mm", {"role": "user", "content": [
        {"type": "text", "text": f"multimodal {SECRET} mmneedle"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}}]})
    seed_messages(db, "s_mm", {"role": "assistant", "content": "ok"})
    db.sessions.set_title("s_mm", "vision runbook")
    return db


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_multipart_hit_is_redacted_field_wise(multipart_db, monkeypatch, detail):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    hit = _search(multipart_db, query="mmneedle", detail=detail)["results"][0]
    assert SECRET_BODY not in _dump(hit) and IMAGE_URL not in _dump(hit)
    anchor = next(m for m in hit["messages"] if m.get("anchor"))
    assert [p["type"] for p in anchor["content"]] == ["text", "image_url"]
    assert "mmneedle" in anchor["content"][0]["text"]
    assert hit["snippet"].startswith("multimodal ") and ">>>mmneedle<<<" in hit["snippet"]


def test_multipart_title_match_bookends_are_redacted(multipart_db, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    hit = _search(multipart_db, query="vision runbook")["results"][0]
    assert hit["matched_role"] == "session_title"
    assert hit["bookend_start"] and hit["bookend_end"]
    assert SECRET_BODY not in _dump(hit) and IMAGE_URL not in _dump(hit)


# Only base64 data URLs are inline data; text that merely starts with "data:" (SSE, prose) is not.
@pytest.mark.parametrize("content", [
    "data: the quarterlyneedle table has three columns",
    'data: {"delta": "quarterlyneedle"}\n\ndata: [DONE]',
])
def test_text_starting_with_data_is_not_an_inline_data_url(tmp_path, content):
    db = storage_handle(tmp_path)
    seed_session(db, "s_sse", "cli")
    seed_messages(db, "s_sse", {"role": "user", "content": content})
    seed_messages(db, "s_sse", {"role": "assistant", "content": "ok"})
    hit = _search(db, query="quarterlyneedle")["results"][0]
    assert next(m for m in hit["messages"] if m.get("anchor"))["content"] == content
    assert ">>>quarterlyneedle<<<" in hit["snippet"]


AIZA_KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
AKIA_KEY = "AKIA" + "IOSFODNN7EXAMPLQ"


# The preview crop (63 chars in SQL, 60 after shaping) can cut a fixed-length key below its pattern's length.
@pytest.mark.parametrize("lead,pad,key", [("", 26, AIZA_KEY), ("", 40, AKIA_KEY), ("\n\n\t  ", 24, AIZA_KEY)])
def test_browse_preview_drops_key_cut_by_the_crop(tmp_path, lead, pad, key):
    db = storage_handle(tmp_path)
    seed_session(db, "s_cut", "cli")
    seed_messages(db, "s_cut", {"role": "user", "content": f"{lead}{'x' * pad} {key} is the key"})
    seed_session(db, "s_dots", "cli")
    seed_messages(db, "s_dots", {"role": "user", "content": "wait..."})
    rows = {r["session_id"]: r for r in _search(db)["results"]}
    assert key[:12] not in rows["s_cut"]["preview"]
    assert rows["s_dots"]["preview"] == "wait..."


# Adaptive discovery fully hydrates only the top hit; lower-ranked hits are "compact" entries
# (anchor message + snippet). Seed several matching sessions so compact entries are exercised.
def test_lower_ranked_compact_results_are_redacted(tmp_path, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    db = storage_handle(tmp_path)
    for i, content in enumerate([f"rankneedle token {SECRET}", f"rankneedle key {SK_KEY}",
                                 [{"type": "text", "text": f"rankneedle multimodal {SECRET}"},
                                  {"type": "image_url", "image_url": {"url": IMAGE_URL}}]]):
        seed_session(db, f"s_rank{i}", "cli")
        seed_messages(db, f"s_rank{i}", {"role": "user", "content": content})
        seed_messages(db, f"s_rank{i}", {"role": "assistant", "content": "ok"})
    results = _search(db, query="rankneedle")["results"]
    compact = [r for r in results if r["detail"] == "compact"]
    assert len(results) == 3 and len(compact) == 2, results
    for hit in compact:
        assert [m.get("anchor") for m in hit["messages"]] == [True]
        assert ">>>rankneedle<<<" in hit["snippet"]
    blob = _dump(results)
    assert SECRET_BODY not in blob and SK_TAIL not in blob and IMAGE_URL not in blob
