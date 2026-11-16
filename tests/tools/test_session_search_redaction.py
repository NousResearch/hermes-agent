"""Discovery output is derived recall text: strict redaction on every model-visible field (#113165)."""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search

SECRET_BODY = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
SECRET = "ghp_" + SECRET_BODY
SK_TAIL = "abcdefghij1234567890abcdefghij"
SK_KEY = "sk-" + "proj-" + SK_TAIL


@pytest.fixture
def db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_secret", source="cli")
    db.append_message("s_secret", role="user", content="please deploy the service")
    db.append_message("s_secret", role="assistant", content="sure", tool_calls=[{
        "id": "c1", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": f"export GITHUB_TOKEN={SECRET}"})}}])
    db.append_message("s_secret", role="user", content=f"deployneedle token is {SECRET}")
    db.append_message("s_secret", role="assistant", content="deployed")
    return db


def _dump(entry) -> str:
    return json.dumps(entry, ensure_ascii=False)


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_discovery_never_returns_secret_in_message_bodies(db, monkeypatch, detail):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)  # user opted out; derived text stays strict
    out = json.loads(session_search(query="deployneedle", detail=detail, db=db))
    assert out["results"], out
    hit = out["results"][0]
    assert any(m.get("anchor") for m in hit["messages"])
    assert SECRET not in _dump(hit)


def test_title_match_discovery_redacts_window(db, monkeypatch):
    db.set_session_title("s_secret", "release runbook")
    out = json.loads(session_search(query="release runbook", db=db))
    hit = out["results"][0]
    assert hit["matched_role"] == "session_title"
    assert SECRET not in _dump(hit)


def test_explicit_session_read_stays_verbatim(db):
    read = json.loads(session_search(session_id="s_secret", db=db))
    assert SECRET in _dump(read["messages"])


@pytest.fixture
def snippet_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_snip", source="cli")
    db.append_message("s_snip", role="user", content=f"github token {SECRET} and openai key {SK_KEY} saved")
    db.append_message("s_snip", role="assistant", content="ok")
    db.create_session("s_crop", source="cli")
    filler = " ".join(f"w{i}" for i in range(38))
    db.append_message("s_crop", role="user", content=f"{SECRET} {filler} cropneedle")
    db.append_message("s_crop", role="assistant", content="ok")
    return db


# FTS snippet() puts highlight markers inside the token and crops its prefix; both broke
# post-hoc redaction of the snippet field.
@pytest.mark.parametrize("detail", ["compact", "full"])
@pytest.mark.parametrize("query", ["ghp", SECRET_BODY, "proj", "cropneedle"])
def test_snippet_never_leaks_secret(snippet_db, monkeypatch, detail, query):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    out = json.loads(session_search(query=query, detail=detail, db=snippet_db))
    assert out["results"], out
    blob = _dump(out["results"])
    assert SECRET_BODY not in blob and SK_TAIL not in blob
    if query == "cropneedle":
        assert ">>>cropneedle<<<" in out["results"][0]["snippet"]


IMAGE_URL = "data:image/png;base64," + "iVBORw0KGgo" + "A" * 64


@pytest.fixture
def multipart_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_mm", source="cli")
    db.append_message("s_mm", role="user", content=[
        {"type": "text", "text": f"multimodal {SECRET} mmneedle"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}}])
    db.append_message("s_mm", role="assistant", content="ok")
    db.set_session_title("s_mm", "vision runbook")
    return db


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_multipart_hit_is_redacted_field_wise(multipart_db, monkeypatch, detail):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    hit = json.loads(session_search(query="mmneedle", detail=detail, db=multipart_db))["results"][0]
    assert SECRET_BODY not in _dump(hit) and IMAGE_URL not in _dump(hit)
    anchor = next(m for m in hit["messages"] if m.get("anchor"))
    assert [p["type"] for p in anchor["content"]] == ["text", "image_url"]
    assert "mmneedle" in anchor["content"][0]["text"]
    assert hit["snippet"].startswith("multimodal ") and ">>>mmneedle<<<" in hit["snippet"]


def test_multipart_title_match_bookends_are_redacted(multipart_db, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    hit = json.loads(session_search(query="vision runbook", db=multipart_db))["results"][0]
    assert hit["matched_role"] == "session_title"
    assert hit["bookend_start"] and hit["bookend_end"]
    assert SECRET_BODY not in _dump(hit) and IMAGE_URL not in _dump(hit)


# Only base64 data URLs are inline data; text that merely starts with "data:" (SSE, prose) is not.
@pytest.mark.parametrize("content", [
    "data: the quarterlyneedle table has three columns",
    'data: {"delta": "quarterlyneedle"}\n\ndata: [DONE]',
])
def test_text_starting_with_data_is_not_an_inline_data_url(tmp_path, content):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_sse", source="cli")
    db.append_message("s_sse", role="user", content=content)
    db.append_message("s_sse", role="assistant", content="ok")
    hit = json.loads(session_search(query="quarterlyneedle", db=db))["results"][0]
    assert next(m for m in hit["messages"] if m.get("anchor"))["content"] == content
    assert ">>>quarterlyneedle<<<" in hit["snippet"]


AIZA_KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
AKIA_KEY = "AKIA" + "IOSFODNN7EXAMPLQ"


# The preview crop (63 chars in SQL, 60 after shaping) can cut a fixed-length key below its pattern's length.
@pytest.mark.parametrize("lead,pad,key", [("", 26, AIZA_KEY), ("", 40, AKIA_KEY), ("\n\n\t  ", 24, AIZA_KEY)])
def test_browse_preview_drops_key_cut_by_the_crop(tmp_path, lead, pad, key):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_cut", source="cli")
    db.append_message("s_cut", role="user", content=f"{lead}{'x' * pad} {key} is the key")
    db.create_session("s_dots", source="cli")
    db.append_message("s_dots", role="user", content="wait...")
    rows = {r["session_id"]: r for r in json.loads(session_search(db=db))["results"]}
    assert key[:12] not in rows["s_cut"]["preview"]
    assert rows["s_dots"]["preview"] == "wait..."


# Adaptive discovery fully hydrates only the top hit; lower-ranked hits are "compact" entries
# (anchor message + snippet). Seed several matching sessions so compact entries are exercised.
def test_lower_ranked_compact_results_are_redacted(tmp_path, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    db = SessionDB(tmp_path / "state.db")
    for i, content in enumerate([f"rankneedle token {SECRET}", f"rankneedle key {SK_KEY}",
                                 [{"type": "text", "text": f"rankneedle multimodal {SECRET}"},
                                  {"type": "image_url", "image_url": {"url": IMAGE_URL}}]]):
        db.create_session(f"s_rank{i}", source="cli")
        db.append_message(f"s_rank{i}", role="user", content=content)
        db.append_message(f"s_rank{i}", role="assistant", content="ok")
    results = json.loads(session_search(query="rankneedle", db=db))["results"]
    compact = [r for r in results if r["detail"] == "compact"]
    assert len(results) == 3 and len(compact) == 2, results
    for hit in compact:
        assert [m.get("anchor") for m in hit["messages"]] == [True]
        assert ">>>rankneedle<<<" in hit["snippet"]
    blob = _dump(results)
    assert SECRET_BODY not in blob and SK_TAIL not in blob and IMAGE_URL not in blob
