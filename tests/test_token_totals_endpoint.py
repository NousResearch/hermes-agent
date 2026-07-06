"""GET /api/sessions/{id}/token-totals — decoded aggregate endpoint.

Drives the real handler against an in-memory SessionDB with stubbed auth /
404 helpers, so the wiring (scope routing, decode, validation) is covered
without standing up the full gateway.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

# The handler builds aiohttp web.Response objects; skip where aiohttp is not
# installed (e.g. minimal test runtimes that don't carry the gateway deps).
pytest.importorskip("aiohttp")

from hermes_state import SessionDB
from hermes_token_codec import pack_assistant_tokens, pack_input_tokens


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "totals_ep.db")
    yield d
    d.close()


def _adapter(db):
    from gateway.platforms.api_server import APIServerAdapter
    a = APIServerAdapter.__new__(APIServerAdapter)
    a._check_auth = lambda request: None
    a._get_existing_session_or_404 = lambda sid: ({"id": sid}, None)
    a._ensure_session_db = lambda: db
    db.resolve_resume_session_id = lambda sid: sid  # identity for the test
    return a


def _req(session_id, scope=None):
    return SimpleNamespace(
        match_info={"session_id": session_id},
        query=({"scope": scope} if scope is not None else {}),
    )


def _seed(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(session_id="s1", role="user", content="q",
                      token_count=pack_input_tokens(1000, 400))
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(300, 50))


def _body(resp):
    return json.loads(resp.text)


def test_session_scope_default(db):
    _seed(db)
    resp = asyncio.run(_adapter(db)._handle_session_token_totals(_req("s1")))
    body = _body(resp)
    assert body["scope"] == "session"
    assert body["session_id"] == "s1"
    assert body["tokens"]["input"] == 1000
    assert body["tokens"]["cache_read"] == 400
    assert body["tokens"]["output"] == 300
    assert body["tokens"]["reasoning"] == 50
    assert body["tokens"]["messages"] == 2


def test_conversation_scope(db):
    _seed(db)
    resp = asyncio.run(_adapter(db)._handle_session_token_totals(_req("s1", scope="conversation")))
    body = _body(resp)
    assert body["scope"] == "conversation"
    assert body["tokens"]["input"] == 1000
    assert body["tokens"]["output"] == 300


def test_invalid_scope_400(db):
    _seed(db)
    resp = asyncio.run(_adapter(db)._handle_session_token_totals(_req("s1", scope="bogus")))
    assert resp.status == 400
