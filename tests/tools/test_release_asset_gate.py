"""release_asset_gate — the human signals autonomy levels read: the Notion
Release Assets row (sign-off / Hold) and ❌ reactions on a Slack announcement.
HTTP is mocked at _http_json; credentials at get_secret."""
import pytest

import tools.release_asset_gate as g
from model_tools import _run_async

DS = "11111111-2222-3333-4444-555555555555"


def _page(status="Approved", asset_type="Help article",
          destination="everafterai.zendesk.com", parent=None):
    return {
        "object": "page",
        "id": "abc",
        "url": "https://www.notion.so/abc",
        "parent": parent or {"type": "data_source_id", "data_source_id": DS,
                             "database_id": "99999999-0000-0000-0000-000000000000"},
        "properties": {
            "Status": {"type": "select", "select": {"name": status} if status else None},
            "Asset type": {"type": "select", "select": {"name": asset_type}},
            "Destination": {"type": "select", "select": {"name": destination}},
        },
    }


def test_row_state_reads_select_props():
    st = g.row_state(_page(), DS)
    assert st == g.RowState(status="Approved", asset_type="Help article",
                            destination="everafterai.zendesk.com",
                            url="https://www.notion.so/abc")
    # The expected id may be given without dashes; the database id also counts.
    assert g.row_state(_page(), DS.replace("-", "")).status == "Approved"
    assert g.row_state(_page(), "99999999000000000000000000000000").status == "Approved"
    # Status as a `status`-type property, and an empty select.
    p = _page()
    p["properties"]["Status"] = {"type": "status", "status": {"name": "Hold"}}
    assert g.row_state(p, DS).status == "Hold"
    assert g.row_state(_page(status=None), DS).status == ""


def test_row_state_refuses_foreign_parent():
    for parent in ({"type": "data_source_id", "data_source_id": "deadbeef" * 4},
                   {"type": "page_id", "page_id": DS},
                   {"type": "workspace", "workspace": True}):
        with pytest.raises(g.GateError):
            g.row_state(_page(parent=parent), DS)
    with pytest.raises(g.GateError):
        g.row_state(_page(), "")  # unconfigured expected parent fails closed


def test_fetch_requires_notion_token(monkeypatch):
    import agent.secret_scope as ss
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: None)
    with pytest.raises(g.GateError, match="Notion"):
        _run_async(g.fetch_notion_page("abc"))


def test_fetch_rejects_bad_page_id(monkeypatch):
    import agent.secret_scope as ss
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: "tok" if n == "NOTION_API_KEY" else None)
    for bad in ("../users", "abc/def", "", "x" * 80):
        with pytest.raises(g.GateError):
            _run_async(g.fetch_notion_page(bad))


def test_fetch_page_calls_notion(monkeypatch):
    import agent.secret_scope as ss
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: "tok" if n == "NOTION_API_TOKEN" else None)
    seen = {}

    async def fake(method, url, headers, **kw):
        seen.update(method=method, url=url, headers=headers)
        return 200, _page()

    monkeypatch.setattr(g, "_http_json", fake)
    page = _run_async(g.fetch_notion_page(DS))
    assert page["id"] == "abc"
    assert seen["method"] == "GET"
    assert seen["url"] == f"https://api.notion.com/v1/pages/{DS}"
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert seen["headers"]["Notion-Version"]


@pytest.fixture
def slack_tok(monkeypatch):
    import agent.secret_scope as ss
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: "xoxb" if n == "SLACK_BOT_TOKEN" else None)


def test_veto_reactions_ignores_bot_and_other_emoji(monkeypatch, slack_tok):
    async def fake(method, url, headers, **kw):
        if url.endswith("auth.test"):
            return 200, {"ok": True, "user_id": "UBOT"}
        assert url.endswith("reactions.get")
        assert kw["params"] == {"channel": "C1", "timestamp": "1.2", "full": "true"}
        return 200, {"ok": True, "message": {"reactions": [
            {"name": "x", "users": ["UBOT", "U1"]},
            {"name": "thumbsup", "users": ["U2"]},
            {"name": "no_entry_sign", "users": ["U3"]},
        ]}}

    monkeypatch.setattr(g, "_http_json", fake)
    assert sorted(_run_async(g.veto_reactions("C1", "1.2"))) == ["U1", "U3"]


def test_veto_reactions_none(monkeypatch, slack_tok):
    async def fake(method, url, headers, **kw):
        if url.endswith("auth.test"):
            return 200, {"ok": True, "user_id": "UBOT"}
        return 200, {"ok": True, "message": {}}

    monkeypatch.setattr(g, "_http_json", fake)
    assert _run_async(g.veto_reactions("C1", "1.2")) == []


def test_veto_reactions_error_raises(monkeypatch, slack_tok):
    async def fake(method, url, headers, **kw):
        if url.endswith("auth.test"):
            return 200, {"ok": True, "user_id": "UBOT"}
        return 200, {"ok": False, "error": "missing_scope"}

    monkeypatch.setattr(g, "_http_json", fake)
    with pytest.raises(g.GateError, match="missing_scope"):
        _run_async(g.veto_reactions("C1", "1.2"))


def test_post_announcement(monkeypatch, slack_tok):
    seen = {}

    async def fake(method, url, headers, **kw):
        seen.update(url=url, body=kw.get("json"))
        return 200, {"ok": True, "ts": "9.9"}

    monkeypatch.setattr(g, "_http_json", fake)
    assert _run_async(g.post_announcement("C1", "hello", thread_ts="1.0")) == "9.9"
    assert seen["url"].endswith("chat.postMessage")
    assert seen["body"] == {"channel": "C1", "text": "hello", "thread_ts": "1.0"}


def test_slack_unscoped_is_gate_error(monkeypatch):
    import agent.secret_scope as ss

    def unscoped(n, d=None):
        raise ss.UnscopedSecretError(n)

    monkeypatch.setattr(ss, "get_secret", unscoped)
    with pytest.raises(g.GateError):
        _run_async(g.post_announcement("C1", "x"))
