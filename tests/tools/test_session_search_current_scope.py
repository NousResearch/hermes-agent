"""Gateway current-chat scoping contracts for session_search."""

from contextlib import contextmanager
from contextvars import copy_context
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import (
    SessionContext,
    SessionSource,
    SessionStore,
    build_session_key,
)
from gateway.session_context import (
    clear_session_vars,
    gateway_context_active,
    get_session_env,
    reset_session_vars,
    set_session_vars,
)
from gateway.recall_scope import canonical_recall_identity
from hermes_state import SessionDB
from tools.session_search_tool import session_search


_DEFAULT_ORIGIN = object()


@pytest.fixture(autouse=True)
def _clean_session_context():
    reset_session_vars()
    yield
    reset_session_vars()


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _source(
    *,
    platform: Platform = Platform.DISCORD,
    chat_id: str,
    user_id: str,
    chat_type: str = "group",
    thread_id: str | None = None,
    parent_chat_id: str | None = None,
    prospective_thread_id: str | None = None,
    scope_id: str | None = "guild-1",
    profile: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
        prospective_thread_id=prospective_thread_id,
        scope_id=scope_id,
        profile=profile,
    )


def _seed(
    db: SessionDB,
    session_id: str,
    source: SessionSource,
    content: str,
    *,
    origin_json: object = _DEFAULT_ORIGIN,
) -> tuple[str, int]:
    session_key = build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=True,
        profile=source.profile,
    )
    durable_origin = (
        json.dumps(source.to_dict()) if origin_json is _DEFAULT_ORIGIN else origin_json
    )
    db.create_session(
        session_id,
        source=source.platform.value,
        user_id=source.user_id,
        session_key=session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
        thread_id=source.thread_id,
        profile_name=source.profile,
        origin_json=durable_origin,
    )
    message_id = db.append_message(session_id, role="user", content=content)
    return session_key, message_id


@contextmanager
def _bound(source: SessionSource, session_key: str, session_id: str):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key=session_key,
        session_id=session_id,
    )
    tokens = runner._set_session_env(context)
    try:
        yield
    finally:
        runner._clear_session_env(tokens)


def _result_ids(payload: dict) -> set[str]:
    return {item["session_id"] for item in payload.get("results", [])}


def test_gateway_default_includes_chat_peers_excludes_other_chats_and_all_opts_out(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    other_profile_source = _source(
        chat_id="chat-a", user_id="user-3", profile="work"
    )
    other_chat_type_source = _source(
        chat_id="chat-a", user_id="user-4", chat_type="channel"
    )

    current_key, _ = _seed(db, "current", current_source, "active turn")
    peer_key, _ = _seed(db, "same-chat-peer", peer_source, "scope needle")
    _, other_message_id = _seed(
        db,
        "other-chat",
        other_source,
        "scope needle",
    )
    _seed(db, "other-profile", other_profile_source, "scope needle")
    _seed(db, "other-chat-type", other_chat_type_source, "scope needle")

    # Live agent sessions remain per-user even though recall is shared by chat.
    assert current_key != peer_key

    with _bound(current_source, current_key, "current"):
        current = json.loads(
            session_search(
                query="scope needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )
        global_result = json.loads(
            session_search(
                query="scope needle",
                db=db,
                current_session_id="current",
                scope="all",
                limit=10,
            )
        )
        browse = json.loads(
            session_search(
                db=db,
                current_session_id="current",
                limit=10,
            )
        )
        wrong_read = json.loads(
            session_search(
                session_id="other-chat",
                db=db,
                current_session_id="current",
            )
        )
        wrong_scroll = json.loads(
            session_search(
                session_id="other-chat",
                around_message_id=other_message_id,
                db=db,
                current_session_id="current",
            )
        )

    assert current["scope"] == "current"
    assert _result_ids(current) == {"same-chat-peer"}
    assert global_result["scope"] == "all"
    assert _result_ids(global_result) == {
        "same-chat-peer",
        "other-chat",
        "other-profile",
        "other-chat-type",
    }
    assert _result_ids(browse) == {"same-chat-peer"}
    assert wrong_read["success"] is False
    assert "outside the current chat scope" in wrong_read["error"]
    assert wrong_scroll["success"] is False
    assert "outside the current chat scope" in wrong_scroll["error"]


def test_gateway_thread_scope_includes_users_but_excludes_parent_and_sibling_threads(
    db,
):
    current_source = _source(
        platform=Platform.TELEGRAM,
        chat_id="group-1",
        user_id="user-1",
        chat_type="forum",
        thread_id="topic-1",
        scope_id=None,
    )
    peer_source = _source(
        platform=Platform.TELEGRAM,
        chat_id="group-1",
        user_id="user-2",
        chat_type="forum",
        thread_id="topic-1",
        scope_id=None,
    )
    sibling_source = _source(
        platform=Platform.TELEGRAM,
        chat_id="group-1",
        user_id="user-1",
        chat_type="forum",
        thread_id="topic-2",
        scope_id=None,
    )
    parent_source = _source(
        platform=Platform.TELEGRAM,
        chat_id="group-1",
        user_id="user-1",
        scope_id=None,
    )

    current_key, _ = _seed(db, "topic-current", current_source, "active turn")
    peer_key, _ = _seed(db, "topic-peer", peer_source, "thread needle")
    _seed(db, "topic-sibling", sibling_source, "thread needle")
    _seed(db, "topic-parent", parent_source, "thread needle")
    assert current_key != peer_key

    with _bound(current_source, current_key, "topic-current"):
        result = json.loads(
            session_search(
                query="thread needle",
                db=db,
                current_session_id="topic-current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"topic-peer"}


@pytest.mark.parametrize(
    ("platform", "scope_id"),
    [
        (Platform.TELEGRAM, None),
        (Platform.SLACK, "workspace-1"),
    ],
)
def test_dm_topics_and_slack_dm_threads_retain_thread_identity(
    db, platform, scope_id
):
    current_source = _source(
        platform=platform,
        chat_id="dm-1",
        user_id="user-1",
        chat_type="dm",
        thread_id="topic-1",
        scope_id=scope_id,
    )
    peer_source = _source(
        platform=platform,
        chat_id="dm-1",
        user_id="user-2",
        chat_type="dm",
        thread_id="topic-1",
        scope_id=scope_id,
    )
    sibling_source = _source(
        platform=platform,
        chat_id="dm-1",
        user_id="user-1",
        chat_type="dm",
        thread_id="topic-2",
        scope_id=scope_id,
    )
    unthreaded_source = _source(
        platform=platform,
        chat_id="dm-1",
        user_id="user-1",
        chat_type="dm",
        scope_id=scope_id,
    )
    other_workspace_source = _source(
        platform=platform,
        chat_id="dm-1",
        user_id="user-3",
        chat_type="dm",
        thread_id="topic-1",
        scope_id="workspace-2" if platform == Platform.SLACK else scope_id,
    )

    current_key, _ = _seed(db, "dm-current", current_source, "active turn")
    _seed(db, "dm-peer", peer_source, "dm thread needle")
    _seed(db, "dm-sibling", sibling_source, "dm thread needle")
    _seed(db, "dm-parent", unthreaded_source, "dm thread needle")
    if platform == Platform.SLACK:
        _seed(db, "other-workspace", other_workspace_source, "dm thread needle")

    with _bound(current_source, current_key, "dm-current"):
        result = json.loads(
            session_search(
                query="dm thread needle",
                db=db,
                current_session_id="dm-current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"dm-peer"}


def test_nonstandard_supported_chat_type_uses_exact_chat_identity(db):
    platform = Platform.MATRIX
    chat_type = "room"
    current_source = _source(
        platform=platform,
        chat_id="room-1",
        user_id="user-1",
        chat_type=chat_type,
        scope_id=None,
    )
    peer_source = _source(
        platform=platform,
        chat_id="room-1",
        user_id="user-2",
        chat_type=chat_type,
        scope_id=None,
    )
    other_source = _source(
        platform=platform,
        chat_id="room-2",
        user_id="user-1",
        chat_type=chat_type,
        scope_id=None,
    )
    current_key, _ = _seed(db, "plugin-current", current_source, "active turn")
    _seed(db, "plugin-peer", peer_source, "plugin shape needle")
    _seed(db, "plugin-other", other_source, "plugin shape needle")

    with _bound(current_source, current_key, "plugin-current"):
        result = json.loads(
            session_search(
                query="plugin shape needle",
                db=db,
                current_session_id="plugin-current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"plugin-peer"}


def test_authenticated_relay_plugin_shape_uses_underlying_platform_identity(db):
    from gateway.relay.ws_transport import _event_from_wire

    current_source = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "irc",
                "chat_id": "#room-1",
                "chat_type": "group",
                "user_id": "user-1",
            },
        }
    ).source
    peer_source = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "irc",
                "chat_id": "#room-1",
                "chat_type": "group",
                "user_id": "user-2",
            },
        }
    ).source
    other_source = _event_from_wire(
        {
            "text": "hello",
            "source": {
                "platform": "irc",
                "chat_id": "#room-2",
                "chat_type": "group",
                "user_id": "user-1",
            },
        }
    ).source
    assert current_source.delivered_via_upstream_relay is True
    assert current_source.platform == Platform("irc")

    current_key, _ = _seed(db, "relay-current", current_source, "active turn")
    _seed(db, "relay-peer", peer_source, "relay plugin needle")
    _seed(db, "relay-other", other_source, "relay plugin needle")

    with _bound(current_source, current_key, "relay-current"):
        result = json.loads(
            session_search(
                query="relay plugin needle",
                db=db,
                current_session_id="relay-current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"relay-peer"}


def test_whatsapp_jid_lid_aliases_match_only_with_durable_alias_proof(
    db, tmp_path, monkeypatch
):
    hermes_home = tmp_path / "whatsapp-home"
    mapping_dir = hermes_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "lid-mapping-999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"), encoding="utf-8"
    )
    (mapping_dir / "lid-mapping-15551234567_reverse.json").write_text(
        json.dumps("999@lid"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    durable_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="999@lid",
        user_id="999@lid",
        chat_type="dm",
        scope_id=None,
    )
    live_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="15551234567@s.whatsapp.net",
        user_id="15551234567@s.whatsapp.net",
        chat_type="dm",
        scope_id=None,
    )
    peer_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="15551234567@s.whatsapp.net",
        user_id="15551234567@s.whatsapp.net",
        chat_type="dm",
        scope_id=None,
    )
    other_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="15550000000@s.whatsapp.net",
        user_id="15550000000@s.whatsapp.net",
        chat_type="dm",
        scope_id=None,
    )

    current_key, _ = _seed(db, "wa-current", durable_source, "active turn")
    _seed(db, "wa-peer", peer_source, "whatsapp alias needle")
    _seed(db, "wa-other", other_source, "whatsapp alias needle")

    with _bound(live_source, current_key, "wa-current"):
        result = json.loads(
            session_search(
                query="whatsapp alias needle",
                db=db,
                current_session_id="wa-current",
                limit=10,
            )
        )

    assert result["success"] is True
    assert _result_ids(result) == {"wa-peer"}


def test_whatsapp_unproven_alias_change_fails_closed(db, tmp_path, monkeypatch):
    hermes_home = tmp_path / "whatsapp-empty-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    durable_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="999@lid",
        user_id="999@lid",
        chat_type="dm",
        scope_id=None,
    )
    live_source = _source(
        platform=Platform.WHATSAPP,
        chat_id="15551234567@s.whatsapp.net",
        user_id="15551234567@s.whatsapp.net",
        chat_type="dm",
        scope_id=None,
    )
    current_key, _ = _seed(db, "wa-unproven", durable_source, "active turn")

    with _bound(live_source, current_key, "wa-unproven"):
        result = json.loads(
            session_search(
                db=db,
                current_session_id="wa-unproven",
            )
        )

    assert result["success"] is False
    assert "does not match" in result["error"]


def test_discord_auto_thread_initial_origin_matches_later_thread_context(db):
    initial_source = _source(
        chat_id="parent-channel",
        user_id="user-1",
        chat_type="channel",
        prospective_thread_id="thread-123",
    )
    live_source = _source(
        chat_id="thread-123",
        user_id="user-1",
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="parent-channel",
    )
    peer_source = _source(
        chat_id="thread-123",
        user_id="user-2",
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="parent-channel",
    )
    other_parent_source = _source(
        chat_id="thread-123",
        user_id="user-3",
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="other-parent",
    )

    current_key, _ = _seed(db, "auto-current", initial_source, "active turn")
    _seed(db, "auto-peer", peer_source, "auto thread needle")
    _seed(db, "other-parent", other_parent_source, "auto thread needle")

    with _bound(live_source, current_key, "auto-current"):
        result = json.loads(
            session_search(
                query="auto thread needle",
                db=db,
                current_session_id="auto-current",
                limit=10,
            )
        )

    wrong_parent_live_source = _source(
        chat_id="thread-123",
        user_id="user-1",
        chat_type="thread",
        thread_id="thread-123",
        parent_chat_id="other-parent",
    )
    with _bound(wrong_parent_live_source, current_key, "auto-current"):
        mismatch = json.loads(
            session_search(
                query="auto thread needle",
                db=db,
                current_session_id="auto-current",
                limit=10,
            )
        )

    assert result["success"] is True
    assert _result_ids(result) == {"auto-peer"}
    assert mismatch["success"] is False
    assert "does not match the active session" in mismatch["error"]


def test_short_cjk_fallback_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "桂林路线")
    _seed(db, "other", other_source, "桂林路线")

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="桂林",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"peer"}


def test_stale_fts_like_fallback_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "stale index needle")
    _seed(db, "other", other_source, "stale index needle")
    db._fts_stale = True
    db._fts_enabled = False

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="stale index needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"peer"}


def test_latin_fts_scope_uses_sessions_first_query_plan(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "commonterm route")
    _seed(db, "other", other_source, "commonterm route")

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="commonterm",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"peer"}

    from gateway.recall_scope import canonical_recall_identity

    identity = canonical_recall_identity(current_source.to_dict())
    from_sql, join_where = db._fts_search_from_sql("messages_fts", identity)
    scope_sql, scope_params = db._recall_scope_sql("s", identity)
    sql = (
        f"EXPLAIN QUERY PLAN SELECT m.id {from_sql} WHERE "
        + " AND ".join([*join_where, "messages_fts MATCH ?", scope_sql])
        + " LIMIT ?"
    )
    plan = db._conn.execute(sql, ["commonterm", *scope_params, 10]).fetchall()
    details = [str(row[3]) for row in plan]
    session_step = next(i for i, detail in enumerate(details) if "SCAN s" in detail)
    message_step = next(
        i
        for i, detail in enumerate(details)
        if "idx_messages_session_id" in detail
    )
    fts_step = next(
        i
        for i, detail in enumerate(details)
        if "messages_fts VIRTUAL TABLE" in detail
    )
    assert session_step < message_step < fts_step, details


def test_current_scope_query_deadline_fails_closed_without_affecting_all(
    db, monkeypatch
):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "deadline needle")
    _seed(db, "other", other_source, "deadline needle")
    monkeypatch.setattr(
        "hermes_state_search._CURRENT_SCOPE_SEARCH_DEADLINE_SECONDS", 0.0
    )

    with _bound(current_source, current_key, "current"):
        timed_out = json.loads(
            session_search(
                query="deadline needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )
        explicit_all = json.loads(
            session_search(
                query="deadline needle",
                scope="all",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert timed_out["success"] is False
    assert "bounded query deadline" in timed_out["error"]
    assert _result_ids(explicit_all) == {"peer", "other"}


def test_cjk_bigram_backend_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "桂林路线规划")
    _seed(db, "other", other_source, "桂林路线规划")

    # CI does not require the optional cjk_unicode61 extension. Build the
    # same external-content table name with the bundled trigram tokenizer so
    # this test forces the CJK-index branch rather than silently testing LIKE.
    db._conn.execute(
        "CREATE VIRTUAL TABLE messages_fts_cjk USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id', "
        "tokenize='trigram')"
    )
    db._conn.execute(
        "INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls) "
        "SELECT id, content, tool_name, tool_calls FROM messages WHERE role <> 'tool'"
    )
    db._conn.commit()
    db._fts_cjk_available = True

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="桂林路线",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert db._describe_search_path("桂林路线") == "fts_cjk"
    assert _result_ids(result) == {"peer"}


def test_cjk_trigram_backend_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "大别山项目计划")
    _seed(db, "other", other_source, "大别山项目计划")
    db._fts_cjk_available = False
    assert db._trigram_available is True

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="大别山项目",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert db._describe_search_path("大别山项目") == "trigram"
    assert _result_ids(result) == {"peer"}


def test_forced_short_cjk_like_backend_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "桂林路线")
    _seed(db, "other", other_source, "桂林路线")
    db._fts_cjk_available = False
    db._trigram_available = False

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="桂林",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert db._describe_search_path("桂林") == "like_scan"
    assert _result_ids(result) == {"peer"}


def test_latin_adjacent_cjk_fallback_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "修改youer服务端")
    _seed(db, "other", other_source, "修改youer服务端")
    db._fts_cjk_available = False
    assert db._trigram_available is True

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="youer",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"peer"}


def test_deferred_rebuild_gap_backend_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _, peer_message_id = _seed(db, "peer", peer_source, "deferredgap needle")
    _, other_message_id = _seed(db, "other", other_source, "deferredgap needle")
    db._conn.execute(
        "DELETE FROM messages_fts WHERE rowid IN (?, ?)",
        (peer_message_id, other_message_id),
    )
    db._conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES "
        "('fts_rebuild_progress', '0'), ('fts_rebuild_high_water', ?)",
        (str(max(peer_message_id, other_message_id)),),
    )
    db._conn.commit()

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="deferredgap",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert _result_ids(result) == {"peer"}


def test_title_route_obeys_current_chat_scope(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    other_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "unrelated peer content")
    _seed(db, "other", other_source, "unrelated foreign content")
    db.set_session_title("peer", "Current Scope Exact Title")
    db.set_session_title("other", "Foreign Scope Exact Title")

    with _bound(current_source, current_key, "current"):
        peer_result = json.loads(
            session_search(
                query="Current Scope Exact Title",
                db=db,
                current_session_id="current",
            )
        )
        foreign_result = json.loads(
            session_search(
                query="Foreign Scope Exact Title",
                db=db,
                current_session_id="current",
            )
        )

    assert _result_ids(peer_result) == {"peer"}
    assert _result_ids(foreign_result) == set()


@pytest.mark.parametrize(
    "origin_json",
    [None, "{not-json", json.dumps({"platform": "discord"})],
)
def test_gateway_missing_or_malformed_durable_scope_fails_closed(db, origin_json):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    current_key, _ = _seed(
        db,
        "current",
        current_source,
        "active turn",
        origin_json=origin_json,
    )
    _seed(db, "peer", peer_source, "fail closed needle")

    with _bound(current_source, current_key, "current"):
        failed = json.loads(
            session_search(
                query="fail closed needle",
                db=db,
                current_session_id="current",
            )
        )
        explicit_all = json.loads(
            session_search(
                query="fail closed needle",
                db=db,
                current_session_id="current",
                scope="all",
            )
        )

    assert failed["success"] is False
    assert "scope" in failed["error"]
    assert _result_ids(explicit_all) == {"peer"}


def test_unknown_candidate_origin_is_hidden_until_explicit_all(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "known-peer", peer_source, "unknown origin needle")
    _seed(
        db,
        "unknown-peer",
        peer_source,
        "unknown origin needle",
        origin_json=None,
    )
    _seed(
        db,
        "malformed-peer",
        peer_source,
        "unknown origin needle",
        origin_json=json.dumps(
            {
                "platform": "discord",
                "chat_id": "chat-a",
                "chat_type": "💣",
                "scope_id": "guild-1",
            }
        ),
    )

    with _bound(current_source, current_key, "current"):
        current = json.loads(
            session_search(
                query="unknown origin needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )
        explicit_all = json.loads(
            session_search(
                query="unknown origin needle",
                db=db,
                current_session_id="current",
                scope="all",
                limit=10,
            )
        )

    assert _result_ids(current) == {"known-peer"}
    assert _result_ids(explicit_all) == {
        "known-peer",
        "unknown-peer",
        "malformed-peer",
    }


def test_cross_scope_lineage_root_cannot_leak_metadata_or_results(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    foreign_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "foreign-root", foreign_source, "foreign root content")
    db.set_session_title("foreign-root", "Foreign Root Private Title")
    db.create_session(
        "current-child",
        source="discord",
        parent_session_id="foreign-root",
        origin_json=json.dumps(current_source.to_dict()),
    )
    db.append_message(
        "current-child", role="user", content="cross lineage needle"
    )

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                query="cross lineage needle",
                db=db,
                current_session_id="current",
            )
        )

    assert _result_ids(result) == set()
    assert "Foreign Root Private Title" not in json.dumps(result)


def test_current_scoped_exact_miss_never_scans_other_profiles(db, monkeypatch):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")

    def forbidden_scan(_session_id):
        raise AssertionError("current-scoped miss broadened to all profiles")

    monkeypatch.setattr(
        "tools.session_search_tool._locate_session_db", forbidden_scan
    )
    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(
                session_id="missing-exact-session",
                db=db,
                current_session_id="current",
            )
        )

    assert result["success"] is False
    assert "not found" in result["error"]


def test_ordinary_unbound_is_global_but_bound_gateway_mismatch_fails_closed(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")

    unbound = json.loads(
        session_search(
            db=db,
            current_session_id="current",
        )
    )
    mismatches = []
    for wrong_source in (
        _source(chat_id="chat-b", user_id="user-1"),
        _source(chat_id="chat-a", user_id="user-1", scope_id="guild-2"),
    ):
        with _bound(wrong_source, current_key, "current"):
            mismatches.append(
                json.loads(
                    session_search(
                        db=db,
                        current_session_id="current",
                    )
                )
            )

    assert unbound["success"] is True
    assert unbound["scope"] == "all"
    assert all(result["success"] is False for result in mismatches)
    assert all("does not match" in result["error"] for result in mismatches)


def test_non_gateway_surfaces_keep_profile_global_default(db):
    _seed(db, "one", _source(chat_id="chat-a", user_id="u1"), "global needle")
    _seed(db, "two", _source(chat_id="chat-b", user_id="u2"), "global needle")

    result = json.loads(
        session_search(
            query="global needle",
            db=db,
            limit=10,
        )
    )

    assert result["scope"] == "all"
    assert _result_ids(result) == {"one", "two"}


@pytest.mark.parametrize(
    "surface",
    ["cli", "tui", "desktop", "api_server", "acp", "cron", "subagent"],
)
def test_non_gateway_explicit_current_is_unavailable_but_omitted_stays_global(
    db, surface
):
    _seed(db, "one", _source(chat_id="chat-a", user_id="u1"), "surface needle")
    _seed(db, "two", _source(chat_id="chat-b", user_id="u2"), "surface needle")
    tokens = set_session_vars(
        platform=surface,
        source=surface,
        session_id=f"{surface}-session",
        gateway_context=False,
    )
    try:
        explicit = json.loads(
            session_search(
                query="surface needle",
                scope="current",
                db=db,
                current_session_id=f"{surface}-session",
                limit=10,
            )
        )
        omitted = json.loads(
            session_search(query="surface needle", db=db, limit=10)
        )
    finally:
        clear_session_vars(tokens)

    assert explicit["success"] is False
    assert "unavailable outside a live messaging gateway" in explicit["error"]
    assert omitted["scope"] == "all"
    assert _result_ids(omitted) == {"one", "two"}


def test_profile_requires_all_for_query_browse_but_exact_read_scroll_survive(
    db, tmp_path, monkeypatch
):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")

    profile_home = tmp_path / "work-profile"
    profile_home.mkdir()
    profile_db = SessionDB(profile_home / "state.db")
    profile_db.create_session("work-session", source="cli")
    message_id = profile_db.append_message(
        "work-session", role="user", content="profile scope needle"
    )
    profile_db.close()

    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda name: name)
    monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda _name: None)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda _name: True)
    current_home = db.db_path.parent
    monkeypatch.setattr(
        profiles_mod,
        "get_profile_dir",
        lambda name: current_home if name == "default" else profile_home,
    )

    with _bound(current_source, current_key, "current"):
        current_query = json.loads(
            session_search(
                query="profile scope needle",
                profile="work",
                scope="current",
                db=db,
                current_session_id="current",
            )
        )
        implicit_query = json.loads(
            session_search(
                query="profile scope needle",
                profile="work",
                db=db,
                current_session_id="current",
            )
        )
        implicit_browse = json.loads(
            session_search(
                profile="work",
                db=db,
                current_session_id="current",
            )
        )
        all_query = json.loads(
            session_search(
                query="profile scope needle",
                profile="work",
                scope="all",
                db=db,
                current_session_id="current",
            )
        )
        all_browse = json.loads(
            session_search(
                profile="work",
                scope="all",
                db=db,
                current_session_id="current",
            )
        )
        exact_read = json.loads(
            session_search(
                session_id="work-session",
                profile="work",
                db=db,
                current_session_id="current",
            )
        )
        exact_scroll = json.loads(
            session_search(
                session_id="work-session",
                around_message_id=message_id,
                profile="work",
                db=db,
                current_session_id="current",
            )
        )
        exact_current = json.loads(
            session_search(
                session_id="work-session",
                profile="work",
                scope="current",
                db=db,
                current_session_id="current",
            )
        )

    assert current_query["success"] is False
    assert "cannot be combined" in current_query["error"]
    assert implicit_query["success"] is False
    assert implicit_browse["success"] is False
    assert _result_ids(all_query) == {"work-session"}
    assert _result_ids(all_browse) == {"work-session"}
    assert exact_read["success"] is True
    assert exact_scroll["success"] is True
    assert exact_current["success"] is False


def test_same_active_profile_exact_read_and_scroll_remain_current_scoped(
    db, monkeypatch
):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    foreign_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _peer_key, peer_message_id = _seed(db, "peer", peer_source, "peer exact")
    _foreign_key, foreign_message_id = _seed(
        db, "foreign", foreign_source, "foreign exact"
    )

    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda name: name)
    monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda _name: None)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda _name: db.db_path.parent)
    monkeypatch.setattr(
        "tools.session_search_tool._resolve_profile_db",
        lambda _profile: (_ for _ in ()).throw(
            AssertionError("same active profile opened a cross-profile DB")
        ),
    )

    with _bound(current_source, current_key, "current"):
        peer_read = json.loads(
            session_search(
                session_id="peer",
                profile="default",
                db=db,
                current_session_id="current",
            )
        )
        peer_scroll = json.loads(
            session_search(
                session_id="peer",
                around_message_id=peer_message_id,
                profile="default",
                scope="current",
                db=db,
                current_session_id="current",
            )
        )
        foreign_read = json.loads(
            session_search(
                session_id="foreign",
                profile="default",
                db=db,
                current_session_id="current",
            )
        )
        foreign_scroll = json.loads(
            session_search(
                session_id="foreign",
                around_message_id=foreign_message_id,
                profile="default",
                scope="current",
                db=db,
                current_session_id="current",
            )
        )
        foreign_all = json.loads(
            session_search(
                session_id="foreign",
                profile="default",
                scope="all",
                db=db,
                current_session_id="current",
            )
        )

    assert peer_read["success"] is True and peer_read["scope"] == "current"
    assert peer_scroll["success"] is True and peer_scroll["scope"] == "current"
    assert foreign_read["success"] is False
    assert foreign_scroll["success"] is False
    assert foreign_all["success"] is True and foreign_all["scope"] == "all"


def test_delegated_child_clears_copied_gateway_capability_and_defaults_global(db):
    from agent.delegation_context import delegated_child_context

    current_source = _source(chat_id="chat-a", user_id="user-1")
    peer_source = _source(chat_id="chat-a", user_id="user-2")
    foreign_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "peer", peer_source, "delegation scope needle")
    _seed(db, "foreign", foreign_source, "delegation scope needle")
    db.create_session("child", source="subagent", parent_session_id="current")

    with _bound(current_source, current_key, "current"):
        parent_before = json.loads(
            session_search(
                query="delegation scope needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )
        copied = copy_context()

        def run_child():
            with delegated_child_context("child"):
                assert gateway_context_active() is False
                assert get_session_env("HERMES_SESSION_ID") == "child"
                payload = json.loads(
                    session_search(
                        query="delegation scope needle",
                        db=db,
                        current_session_id="child",
                        limit=10,
                    )
                )
            return payload, gateway_context_active(), get_session_env(
                "HERMES_SESSION_ID"
            )

        child_result, copied_parent_marker, copied_parent_id = copied.run(run_child)
        parent_after = json.loads(
            session_search(
                query="delegation scope needle",
                db=db,
                current_session_id="current",
                limit=10,
            )
        )

    assert parent_before["scope"] == parent_after["scope"] == "current"
    assert _result_ids(parent_before) == _result_ids(parent_after) == {"peer"}
    assert child_result["scope"] == "all"
    assert _result_ids(child_result) == {"peer", "foreign"}
    assert copied_parent_marker is True
    assert copied_parent_id == "current"


def test_gateway_reset_preserves_scope_and_recalls_predecessor(db, tmp_path):
    source = _source(chat_id="chat-a", user_id="user-1")
    store = SessionStore(tmp_path / "gateway-sessions", GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = db

    original = store.get_or_create_session(source)
    db.append_message(
        original.session_id,
        role="user",
        content="reset persistence needle",
    )
    current = store.reset_session(original.session_key)
    assert current is not None
    assert (
        json.loads(db.get_session(current.session_id)["origin_json"])["chat_id"]
        == "chat-a"
    )

    with _bound(source, current.session_key, current.session_id):
        result = json.loads(
            session_search(
                query="reset persistence needle",
                db=db,
                current_session_id=current.session_id,
            )
        )

    assert original.session_id in _result_ids(result)


def test_compression_child_inherits_scope_and_recalls_parent(db):
    source = _source(chat_id="chat-a", user_id="user-1")
    parent_key, _ = _seed(
        db,
        "compression-parent",
        source,
        "compression persistence needle",
    )
    db.end_session("compression-parent", "compression")
    db.create_session(
        "compression-child",
        source="discord",
        parent_session_id="compression-parent",
    )

    child = db.get_session("compression-child")
    assert child["origin_json"] == db.get_session("compression-parent")["origin_json"]

    with _bound(source, parent_key, "compression-child"):
        result = json.loads(
            session_search(
                query="compression persistence needle",
                db=db,
                current_session_id="compression-child",
            )
        )

    assert "compression-parent" in _result_ids(result)


def test_scoped_browse_stops_before_foreign_compression_child_projection(db):
    current_source = _source(chat_id="chat-a", user_id="user-1")
    foreign_source = _source(chat_id="chat-b", user_id="user-1")
    current_key, _ = _seed(db, "current", current_source, "active turn")
    _seed(db, "root", current_source, "scoped root preview")
    db.set_session_title("root", "Scoped Root Title")
    db.end_session("root", "compression")
    db.create_session(
        "foreign-child",
        source="discord",
        user_id=foreign_source.user_id,
        session_key=_gateway_session_key_for_test(foreign_source),
        chat_id=foreign_source.chat_id,
        chat_type=foreign_source.chat_type,
        parent_session_id="root",
        origin_json=json.dumps(foreign_source.to_dict()),
    )
    db.append_message(
        "foreign-child", role="user", content="FOREIGN PRIVATE PREVIEW"
    )
    db.set_session_title("foreign-child", "FOREIGN PRIVATE TITLE")
    current_identity = canonical_recall_identity(current_source.to_dict())
    assert current_identity is not None
    assert db.get_compression_tip("root", recall_scope=current_identity) == "root"
    assert db._get_session_rich_rows_batch(
        ["foreign-child"], recall_scope=current_identity
    ) == {}

    with _bound(current_source, current_key, "current"):
        result = json.loads(
            session_search(db=db, current_session_id="current", limit=10)
        )

    serialized = json.dumps(result)
    assert result["success"] is True
    assert "root" in _result_ids(result)
    assert "foreign-child" not in _result_ids(result)
    assert "FOREIGN PRIVATE TITLE" not in serialized
    assert "FOREIGN PRIVATE PREVIEW" not in serialized


def _gateway_session_key_for_test(source: SessionSource) -> str:
    return build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=True,
        profile=source.profile,
    )


def test_python_and_sql_recall_normalization_are_equal_or_sql_narrower(db):
    base = {
        "platform": "discord",
        "chat_id": "chat-a",
        "chat_type": "group",
        "scope_id": "guild-1",
        "profile": "default",
        "user_id": "user-a",
    }
    identity = canonical_recall_identity(base)
    assert identity is not None
    cases = [
        ("canonical", base, True),
        (
            "whitespace strings",
            {
                **base,
                "platform": " discord ",
                "chat_id": " chat-a ",
                "chat_type": " group ",
                "scope_id": " guild-1 ",
                "profile": " default ",
            },
            True,
        ),
        ("optional nulls", {**base, "thread_id": None, "guild_id": None}, True),
        ("chat bool", {**base, "chat_id": True}, False),
        ("chat numeric", {**base, "chat_id": 1}, False),
        ("chat object", {**base, "chat_id": {"id": "chat-a"}}, False),
        ("chat array", {**base, "chat_id": ["chat-a"]}, False),
        ("chat null", {**base, "chat_id": None}, False),
        ("platform numeric", {**base, "platform": 1}, False),
        ("chat type bool", {**base, "chat_type": False}, False),
        ("thread numeric", {**base, "thread_id": 123}, False),
        ("prospective array", {**base, "prospective_thread_id": []}, False),
        ("parent object", {**base, "parent_chat_id": {}}, False),
        ("user bool", {**base, "user_id": True}, False),
        ("alternate user numeric", {**base, "user_id_alt": 7}, False),
        ("scope numeric", {**base, "scope_id": 1}, False),
        ("guild bool", {**base, "guild_id": True}, False),
        ("profile numeric", {**base, "profile": 1}, False),
        ("profile object", {**base, "profile": {}}, False),
        ("profile traversal string", {**base, "profile": "../default"}, False),
        ("malformed json", "{not-json", False),
    ]

    for index, (label, candidate, expected) in enumerate(cases):
        session_id = f"parity-{index}"
        origin_json = (
            candidate if isinstance(candidate, str) else json.dumps(candidate)
        )
        db.create_session(session_id, source="discord", origin_json=origin_json)
        python_candidate = None
        if not isinstance(candidate, str):
            python_candidate = canonical_recall_identity(candidate)
        python_matches = python_candidate == identity
        sql_matches = db.session_matches_recall_scope(session_id, identity)
        assert python_matches is expected, label
        assert sql_matches is expected, label
        assert not sql_matches or python_matches, label
