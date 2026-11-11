"""session_search chat-origin scope: a gateway chat recalls its own conversation by default.

Every result carries provenance (``origin`` / ``same_origin``); ``scope="all"`` is the explicit
cross-conversation opt-in; an explicit ``scope="chat"`` without an origin fails closed.
"""

import json

import pytest

from hermes_state import SessionDB
from tools.registry import registry
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _session(db, sid, *, source="signal", chat_id=None, chat_type="group", name=None, key=None, title=None,
             started_at=None, content=None):
    db.create_session(sid, source=source)
    db._conn.execute(
        "UPDATE sessions SET chat_id=?, chat_type=?, display_name=?, session_key=?, title=?, "
        "started_at=COALESCE(?, started_at) WHERE id=?",
        (chat_id, chat_type if chat_id else None, name, key, title, started_at, sid))
    if content:
        db.append_message(sid, role="user", content=content)
    db._conn.commit()


def _search(db, **kwargs):
    return json.loads(session_search(db=db, current_session_id="current", **kwargs))


def _ids(result):
    return [r["session_id"] for r in result["results"]]


def test_schema_exposes_scope():
    scope = SESSION_SEARCH_SCHEMA["parameters"]["properties"]["scope"]
    assert scope["enum"] == ["chat", "all"]


def test_group_search_defaults_to_the_current_session_key(db):
    _session(db, "current", chat_id="chat-a", name="Group A", key="agent:main:signal:group:chat-a:user-a")
    _session(db, "same_user", chat_id="chat-a", name="Group A", key="agent:main:signal:group:chat-a:user-a",
             title="A links", content="ambiguous local links")
    _session(db, "other_user", chat_id="chat-a", name="Group A", key="agent:main:signal:group:chat-a:user-b",
             title="B links", content="ambiguous other user links ambiguous ambiguous")

    result = _search(db, query="ambiguous")

    assert result["scope"] == "chat"
    assert _ids(result) == ["same_user"]
    assert result["results"][0]["same_origin"] is True
    assert result["results"][0]["origin"] == {"source": "signal", "chat_type": "group", "display_name": "Group A"}


def test_forum_topic_search_stays_in_the_current_topic(db):
    """Two topics of one forum chat share the chat id; the topic is part of the session key."""
    _session(db, "current", source="telegram", chat_id="forum-a", key="agent:main:telegram:group:forum-a:topic-1")
    _session(db, "same_topic", source="telegram", chat_id="forum-a", key="agent:main:telegram:group:forum-a:topic-1",
             content="release checklist topic one")
    _session(db, "other_topic", source="telegram", chat_id="forum-a", key="agent:main:telegram:group:forum-a:topic-2",
             content="release checklist topic two")

    result = _search(db, query="release checklist")

    assert _ids(result) == ["same_topic"]
    labelled = _search(db, query="release checklist", scope="all")["results"]
    assert {r["session_id"]: r["same_origin"] for r in labelled} == {"same_topic": True, "other_topic": False}


def test_dm_search_defaults_to_the_current_chat(db):
    _session(db, "current", source="telegram", chat_id="dm-a", chat_type="dm", key="agent:main:telegram:dm:dm-a")
    _session(db, "same_dm", source="telegram", chat_id="dm-a", chat_type="dm", key="agent:main:telegram:dm:dm-a",
             content="private recall marker")
    _session(db, "other_dm", source="telegram", chat_id="dm-b", chat_type="dm", key="agent:main:telegram:dm:dm-b",
             content="private recall marker foreign")

    assert _ids(_search(db, query="private recall")) == ["same_dm"]


def test_chat_id_scope_when_there_is_no_session_key_and_source_must_match(db):
    _session(db, "current", chat_id="shared-id", name="Signal Group")
    _session(db, "same", chat_id="shared-id", name="Signal Group", content="collision marker here")
    _session(db, "other_platform", source="telegram", chat_id="shared-id", name="Telegram Group",
             content="collision marker there")

    scoped = _search(db, query="collision")
    everything = _search(db, query="collision", scope="all")

    assert _ids(scoped) == ["same"]
    assert {r["session_id"]: r["same_origin"] for r in everything["results"]} == {"same": True, "other_platform": False}


def test_scope_all_ranks_same_origin_first_and_labels_foreign_results(db):
    _session(db, "current", chat_id="chat-a", name="Group A")
    _session(db, "foreign", chat_id="chat-b", name="Group B", content="ambiguous foreign links ambiguous ambiguous")
    _session(db, "same", chat_id="chat-a", name="Group A", content="ambiguous same links")

    result = _search(db, query="ambiguous", scope="all")

    assert result["scope"] == "all"
    assert _ids(result)[:2] == ["same", "foreign"]
    assert result["results"][1]["same_origin"] is False
    assert result["results"][1]["origin"]["display_name"] == "Group B"
    assert "different conversation" in result["notice"]


def test_chat_scoped_miss_reports_matches_outside_scope(db):
    _session(db, "current", chat_id="chat-a", name="Group A")
    _session(db, "foreign", chat_id="chat-b", name="Group B", content="outside-scope needle")
    _session(db, "legacy", content="outside-scope needle from a legacy row")  # no chat_id / session_key

    scoped = _search(db, query="outside-scope")

    assert scoped["results"] == []
    assert scoped["recall_scope"]["scope_level"] == "chat"
    assert scoped["recall_scope"]["matches_outside_scope"] == 2
    assert "scope='all'" in scoped["message"]
    assert {r["session_id"] for r in _search(db, query="outside-scope", scope="all")["results"]} == {"foreign", "legacy"}


def test_explicit_chat_scope_without_an_origin_fails_closed(db):
    _session(db, "current")  # CLI-like: no chat origin
    _session(db, "other", chat_id="chat-b", content="needle text")

    assert _search(db, query="needle")["scope"] == "all"  # default without an origin
    closed = _search(db, query="needle", scope="chat")
    assert closed["results"] == [] and "No active chat origin" in closed["message"]


def test_stale_fts_like_fallback_honours_chat_scope(db):
    _session(db, "current", chat_id="chat-a", key="k-a")
    _session(db, "same", chat_id="chat-a", key="k-a", content="stale fallback scoped needle")
    _session(db, "foreign", chat_id="chat-b", key="k-b", content="stale fallback scoped needle foreign")
    db._fts_stale = True
    db._fts_enabled = False

    assert _ids(_search(db, query="stale fallback")) == ["same"]


def test_cjk_route_honours_chat_scope(db):
    _session(db, "current", chat_id="chat-a", key="k-a")
    _session(db, "same", chat_id="chat-a", key="k-a", content="关于大别山项目的本地记录")
    _session(db, "foreign", chat_id="chat-b", key="k-b", content="关于大别山项目的外部记录")

    assert _ids(_search(db, query="大别山项目")) == ["same"]


def test_title_match_prefers_this_chat_over_a_newer_foreign_title(db):
    _session(db, "current", chat_id="chat-a", key="k-a")
    _session(db, "same_title", chat_id="chat-a", key="k-a", title="Shared Title #2", started_at=1,
             content="same chat title content")
    _session(db, "foreign_title", chat_id="chat-b", key="k-b", title="Shared Title", started_at=2,
             content="foreign title content")

    assert _search(db, query="Shared Title")["results"][0]["session_id"] == "same_title"


def test_browse_defaults_to_the_current_chat_beyond_the_recent_window(db):
    _session(db, "current", chat_id="chat-a", name="Group A")
    for idx in range(12):
        _session(db, f"foreign_{idx}", chat_id="chat-b", title=f"Foreign {idx}", started_at=1000 + idx,
                 content="hello")
    _session(db, "same_old", chat_id="chat-a", title="Same old", started_at=1, content="hello")

    result = _search(db, limit=3)

    assert result["scope"] == "chat"
    assert _ids(result) == ["same_old"]
    assert result["results"][0]["same_origin"] is True


def test_read_and_scroll_by_id_stay_allowed_but_warn_on_a_foreign_session(db):
    _session(db, "current", chat_id="chat-a")
    _session(db, "foreign", chat_id="chat-b")
    msg_id = db.append_message("foreign", role="user", content="foreign scroll content")
    db._conn.commit()

    read = _search(db, session_id="foreign")
    scroll = _search(db, session_id="foreign", around_message_id=msg_id)

    for result in (read, scroll):
        assert result["success"] is True
        assert result["same_origin"] is False
        assert "different conversation" in result["cross_context_warning"]


def test_origin_falls_back_to_the_turn_session_context(db, monkeypatch):
    """A turn whose session row is not written yet still scopes by the gateway session context."""
    _session(db, "same", chat_id="chat-a", key="k-a", content="context fallback needle")
    _session(db, "foreign", chat_id="chat-b", key="k-b", content="context fallback needle foreign")
    env = {"HERMES_SESSION_KEY": "k-a", "HERMES_SESSION_CHAT_ID": "chat-a", "HERMES_SESSION_PLATFORM": "signal"}
    monkeypatch.setattr("gateway.session_context.get_session_env", lambda name, default="": env.get(name, default))

    result = json.loads(session_search(query="context fallback", db=db, current_session_id="not-yet-written"))

    assert result["scope"] == "chat"
    assert _ids(result) == ["same"]


def test_registered_handler_passes_scope(db):
    _session(db, "current", chat_id="chat-a")
    _session(db, "foreign", chat_id="chat-b", title="Group B result", content="registry scope marker")

    entry = registry.get_entry("session_search")
    result = json.loads(entry.handler({"query": "registry", "scope": "all"}, db=db, current_session_id="current"))

    assert result["scope"] == "all"
    assert result["results"][0]["session_id"] == "foreign"
    assert result["results"][0]["same_origin"] is False


@pytest.mark.parametrize("source", ["tui", "desktop", "cli"])
def test_non_messaging_turn_with_a_session_key_defaults_to_all(db, source):
    """TUI/desktop/CLI bind a per-conversation session key, but they are not a chat: recall stays global."""
    from gateway.session_context import clear_session_vars, set_session_vars
    _session(db, "other_conv", source=source, key=f"{source}-conv-2", title="Other conversation",
             content="desk recall needle")
    tokens = set_session_vars(session_key=f"{source}-conv-1", source=source)
    try:
        found = json.loads(session_search(query="desk recall", db=db, current_session_id="not-yet-written"))
        browse = json.loads(session_search(db=db, current_session_id="not-yet-written"))
    finally:
        clear_session_vars(tokens)

    assert found["scope"] == "all" and _ids(found) == ["other_conv"]
    assert browse["scope"] == "all" and "other_conv" in _ids(browse)


def test_api_server_turn_defaults_to_all(db):
    _session(db, "current", source="api_server", key="api-conv-1")
    _session(db, "other_conv", source="api_server", key="api-conv-2", title="Other API run",
             content="api recall needle")

    found = _search(db, query="api recall")
    browse = _search(db)

    assert found["scope"] == "all" and _ids(found) == ["other_conv"]
    assert browse["scope"] == "all" and _ids(browse) == ["other_conv"]


def test_title_lookup_applies_the_origin_platform_before_picking_the_winner(db):
    """Same chat id on Signal and Telegram: the newer Telegram "#3" must not shadow this chat's title."""
    _session(db, "current", chat_id="shared-id")
    _session(db, "local_title", chat_id="shared-id", title="Roadmap", started_at=1, content="first draft notes")
    _session(db, "foreign_title", source="telegram", chat_id="shared-id", title="Roadmap #3", started_at=2,
             content="other platform notes")

    assert db.resolve_session_by_title("Roadmap", chat_id="shared-id", source="signal") == "local_title"
    result = _search(db, query="Roadmap")
    assert result["scope"] == "chat"
    assert _ids(result) == ["local_title"]


def test_browse_applies_the_origin_platform_before_the_candidate_limit(db):
    """Foreign-platform rows with the same chat id must not fill the bounded candidate window."""
    _session(db, "current", chat_id="shared-id")
    _session(db, "local_old", chat_id="shared-id", title="Local old", started_at=1, content="hello")
    for idx in range(160):
        _session(db, f"tg_{idx}", source="telegram", chat_id="shared-id", started_at=1000 + idx, content="hello")

    result = _search(db, limit=3)

    assert result["scope"] == "chat"
    assert _ids(result) == ["local_old"]


def test_delegated_child_in_a_chat_recalls_that_chat(db):
    """A subagent row has no chat fields; the inherited chat context supplies key and platform."""
    from gateway.session_context import clear_session_vars, set_session_vars
    key = "agent:main:telegram:dm:dm-a"
    _session(db, "current", source="subagent")
    _session(db, "parent_chat", source="telegram", chat_id="dm-a", chat_type="dm", key=key,
             content="delegated recall needle")
    _session(db, "other_chat", source="telegram", chat_id="dm-b", chat_type="dm", key="agent:main:telegram:dm:dm-b",
             content="delegated recall needle elsewhere")
    tokens = set_session_vars(platform="telegram", chat_id="dm-a", session_key=key)
    try:
        result = _search(db, query="delegated recall")
    finally:
        clear_session_vars(tokens)

    assert result["scope"] == "chat"
    assert _ids(result) == ["parent_chat"]


def test_delegated_child_in_a_tui_turn_keeps_global_recall(db):
    from gateway.session_context import clear_session_vars, set_session_vars
    _session(db, "current", source="subagent")
    _session(db, "other_conv", source="tui", key="tui-conv-2", title="Other conversation",
             content="delegated desk needle")
    tokens = set_session_vars(session_key="tui-conv-1", source="tui")
    try:
        found = _search(db, query="delegated desk")
        browse = _search(db)
    finally:
        clear_session_vars(tokens)

    assert found["scope"] == "all" and _ids(found) == ["other_conv"]
    assert browse["scope"] == "all" and "other_conv" in _ids(browse)
