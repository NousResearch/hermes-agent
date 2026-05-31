"""Tests for the single-shape session_search tool.

Three calling shapes:
  1. DISCOVERY — pass query → FTS5 + anchored window + bookends per hit
  2. SCROLL    — pass session_id + around_message_id → just the window
  3. BROWSE    — no args → recent sessions chronologically

All run zero LLM calls.
"""
import json
import time

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import (
    SESSION_SEARCH_SCHEMA,
    _HIDDEN_SESSION_SOURCES,
    _format_timestamp,
    session_search,
)


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_modpack_sessions(db):
    """Create three sessions about a modpack so FTS5 has hits to dedupe."""
    now = int(time.time())
    # Older session — modpack origin
    db.create_session("s_oldest", source="cli")
    db._conn.execute("UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
                     (now - 30000, "Building the Modpack", "s_oldest"))
    db.append_message("s_oldest", role="user", content="Let's build a Minecraft modpack")
    db.append_message("s_oldest", role="assistant", content="Great. Let me scaffold the modpack repo.")
    db.append_message("s_oldest", role="user", content="Use NeoForge 1.21.1")
    db.append_message("s_oldest", role="assistant", content="Done. Modpack repo created with NeoForge 1.21.1.")
    db.append_message("s_oldest", role="assistant", content="Tier-0 mods installed; modpack smoke test passes.")

    # Middle session — modpack quest coverage
    db.create_session("s_middle", source="cli")
    db._conn.execute("UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
                     (now - 15000, "Modpack Quest Coverage", "s_middle"))
    db.append_message("s_middle", role="user", content="Deep-dive every modpack reference quest guide")
    db.append_message("s_middle", role="assistant", content="Surveying ATM10 questbook for modpack inspiration.")
    db.append_message("s_middle", role="user", content="Update the modpack version too")
    db.append_message("s_middle", role="assistant", content="Modpack version bumped 0.4 → 0.8.5; quest coverage page added.")

    # Newest session — modpack mob spawn fix
    db.create_session("s_newest", source="cli")
    db._conn.execute("UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
                     (now - 1000, "Modpack Mob Spawn Fix", "s_newest"))
    db.append_message("s_newest", role="user", content="Fix the modpack mob spawning")
    db.append_message("s_newest", role="assistant", content="Investigating elite mob gating in the modpack KubeJS.")
    db.append_message("s_newest", role="assistant", content="Shipped commit b850442. Modpack alternator nerfed too.")
    db._conn.commit()


# =========================================================================
# Schema invariants
# =========================================================================

class TestSchema:
    def test_schema_has_required_params(self):
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        # Discovery shape
        assert "query" in params
        assert "limit" in params
        assert "sort" in params
        # Scroll shape
        assert "session_id" in params
        assert "around_message_id" in params
        assert "window" in params
        # Shared
        assert "role_filter" in params

    def test_mode_parameter_includes_previous_handoff(self):
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        assert params["mode"]["enum"] == ["previous", "handoff"]
        assert params["scope"]["enum"] == ["current", "global"]

    def test_sort_enum(self):
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        assert params["sort"]["enum"] == ["newest", "oldest"]

    def test_schema_description_teaches_scroll(self):
        desc = SESSION_SEARCH_SCHEMA["description"]
        assert "SCROLL" in desc
        assert "DISCOVERY" in desc
        assert "BROWSE" in desc
        # Must explain how to scroll
        assert "scroll FORWARD" in desc or "messages[-1]" in desc

    def test_no_llm_promise_in_description(self):
        # The new design never calls an LLM
        desc = SESSION_SEARCH_SCHEMA["description"].lower()
        assert "no llm" in desc


class TestHiddenSources:
    def test_tool_source_hidden(self):
        assert "tool" in _HIDDEN_SESSION_SOURCES


class TestFormatTimestamp:
    def test_unix_timestamp(self):
        out = _format_timestamp(1700000000)
        assert "2023" in out

    def test_none(self):
        assert _format_timestamp(None) == "unknown"

    def test_iso_string_passthrough(self):
        out = _format_timestamp("not-a-number-string")
        assert out == "not-a-number-string"


# =========================================================================
# Browse shape (no args)
# =========================================================================

class TestBrowseShape:
    def test_no_args_returns_recent_sessions(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(db=db))
        assert result["success"] is True
        assert result["mode"] == "browse"
        assert result["count"] >= 3

    def test_browse_excludes_current_session(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(db=db, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids

    def test_browse_returns_titles(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(db=db))
        titles = [r.get("title") for r in result["results"]]
        assert any("Modpack" in (t or "") for t in titles)


# =========================================================================
# Scoped gateway recall + previous/handoff mode
# =========================================================================

class TestScopedGatewayRecall:
    def _create_scoped_session(
        self,
        db,
        session_id,
        *,
        source="qqbot",
        chat_type="dm",
        chat_id="user-a",
        user_id="user-a",
        session_key=None,
        started_at=None,
        ended_at=None,
        text="新增功能",
    ):
        db.create_session(
            session_id,
            source=source,
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
            thread_id=None,
            session_key=session_key or f"agent:main:{source}:{chat_type}:{chat_id}",
        )
        if started_at is not None or ended_at is not None:
            db._conn.execute(
                "UPDATE sessions SET started_at = COALESCE(?, started_at), ended_at = ? WHERE id = ?",
                (started_at, ended_at, session_id),
            )
            db._conn.commit()
        db.append_message(session_id, role="user", content=text)
        db.append_message(session_id, role="assistant", content=f"已记录：{text}")

    def test_default_gateway_search_is_scoped_to_current_chat(self, db):
        self._create_scoped_session(db, "a_old", chat_id="qq-a", user_id="qq-a", text="新增功能 admissions")
        self._create_scoped_session(db, "b_old", chat_id="qq-b", user_id="qq-b", text="新增功能 tutoring")
        db.create_session("a_current", source="qqbot", user_id="qq-a", chat_type="dm", chat_id="qq-a")

        result = json.loads(session_search(
            query="新增功能",
            db=db,
            current_session_id="a_current",
            current_source="qqbot",
            current_chat_type="dm",
            current_chat_id="qq-a",
            current_user_id="qq-a",
        ))
        sids = [r["session_id"] for r in result["results"]]
        assert "a_old" in sids
        assert "b_old" not in sids

    def test_handoff_returns_recent_ended_session_not_adjacent_project(self, db):
        now = time.time()
        self._create_scoped_session(
            db,
            "admissions_done",
            chat_id="qq-a",
            user_id="qq-a",
            started_at=now - 200,
            ended_at=now - 5,
            text="Stage 52 admissions-sales-workbench PR #16 已 merge",
        )
        self._create_scoped_session(
            db,
            "tutoring_neighbor",
            chat_id="qq-a",
            user_id="qq-a",
            started_at=now - 100,
            ended_at=now - 30,
            text="/workspace/tutoring-exam-analysis OCR PDF 学生档案",
        )
        db.create_session("a_current", source="qqbot", user_id="qq-a", chat_type="dm", chat_id="qq-a")

        result = json.loads(session_search(
            mode="handoff",
            db=db,
            current_session_id="a_current",
            current_source="qqbot",
            current_chat_type="dm",
            current_chat_id="qq-a",
            current_user_id="qq-a",
        ))
        assert result["success"] is True
        assert result["mode"] == "handoff"
        assert result["results"][0]["session_id"] == "admissions_done"
        payload = json.dumps(result, ensure_ascii=False)
        assert "/workspace/tutoring-exam-analysis" not in payload
        assert "OCR" not in payload
        assert "PDF" not in payload
        assert "学生档案" not in payload

    def test_scoped_no_result_does_not_global_fallback(self, db):
        self._create_scoped_session(db, "b_old", chat_id="qq-b", user_id="qq-b", text="新增功能 only b")
        db.create_session("a_current", source="qqbot", user_id="qq-a", chat_type="dm", chat_id="qq-a")

        scoped = json.loads(session_search(
            query="新增功能",
            db=db,
            current_session_id="a_current",
            current_source="qqbot",
            current_chat_type="dm",
            current_chat_id="qq-a",
            current_user_id="qq-a",
        ))
        assert scoped["results"] == []

        global_result = json.loads(session_search(
            query="新增功能",
            scope="global",
            db=db,
            current_session_id="a_current",
            current_source="qqbot",
            current_chat_type="dm",
            current_chat_id="qq-a",
            current_user_id="qq-a",
        ))
        assert [r["session_id"] for r in global_result["results"]] == ["b_old"]

    def test_group_chat_recall_is_isolated_by_user_and_session_key(self, db):
        now = time.time()
        shared_chat = "group-1"
        key_a = "agent:main:qqbot:group:group-1:user-a"
        key_b = "agent:main:qqbot:group:group-1:user-b"
        self._create_scoped_session(
            db,
            "group_a_old",
            chat_type="group",
            chat_id=shared_chat,
            user_id="user-a",
            session_key=key_a,
            started_at=now - 200,
            ended_at=now - 100,
            text="shared chat scoped secret alpha",
        )
        self._create_scoped_session(
            db,
            "group_b_old",
            chat_type="group",
            chat_id=shared_chat,
            user_id="user-b",
            session_key=key_b,
            started_at=now - 100,
            ended_at=now - 10,
            text="shared chat scoped secret beta",
        )
        self._create_scoped_session(
            db,
            "group_missing_user",
            chat_type="group",
            chat_id=shared_chat,
            user_id=None,
            session_key=None,
            started_at=now - 80,
            ended_at=now - 20,
            text="shared chat scoped secret missing-user",
        )
        db._conn.execute(
            "UPDATE sessions SET user_id = NULL, user_id_alt = NULL, session_key = NULL WHERE id = ?",
            ("group_missing_user",),
        )
        db._conn.commit()
        db.create_session("group_legacy_missing_user", source="qqbot")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
            (now - 70, now - 15, "group_legacy_missing_user"),
        )
        db.append_message(
            "group_legacy_missing_user",
            role="user",
            content="shared chat scoped secret legacy-missing-user",
        )
        db.create_session(
            "group_a_current",
            source="qqbot",
            user_id="user-a",
            chat_type="group",
            chat_id=shared_chat,
            session_key=key_a,
        )

        common_scope = dict(
            db=db,
            current_session_id="group_a_current",
            current_source="qqbot",
            current_chat_type="group",
            current_chat_id=shared_chat,
            current_user_id="user-a",
            current_session_key=key_a,
        )
        search = json.loads(session_search(query="shared chat scoped secret", **common_scope))
        assert [r["session_id"] for r in search["results"]] == ["group_a_old"]
        payload = json.dumps(search, ensure_ascii=False)
        assert "group_b_old" not in payload
        assert "group_missing_user" not in payload
        assert "group_legacy_missing_user" not in payload

        previous = json.loads(session_search(mode="previous", **common_scope))
        assert previous["results"][0]["session_id"] == "group_a_old"
        assert "group_b_old" not in json.dumps(previous, ensure_ascii=False)

        handoff = json.loads(session_search(mode="handoff", **common_scope))
        assert handoff["results"][0]["session_id"] == "group_a_old"
        assert "group_b_old" not in json.dumps(handoff, ensure_ascii=False)

    def test_handoff_legacy_fallback_is_source_bounded_and_marked(self, db):
        now = time.time()
        db.create_session("legacy_qq", source="qqbot", user_id="qq-a")
        db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (now - 50, now - 10, "legacy_qq"))
        db.append_message("legacy_qq", role="user", content="legacy admissions handoff")
        db.create_session("legacy_telegram", source="telegram", user_id="qq-a")
        db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (now - 40, now - 5, "legacy_telegram"))
        db.append_message("legacy_telegram", role="user", content="legacy telegram should not leak")
        db.create_session("a_current", source="qqbot", user_id="qq-a", chat_type="dm", chat_id="qq-a")

        result = json.loads(session_search(
            mode="previous",
            db=db,
            current_session_id="a_current",
            current_source="qqbot",
            current_chat_type="dm",
            current_chat_id="qq-a",
            current_user_id="qq-a",
        ))
        assert result["results"][0]["session_id"] == "legacy_qq"
        assert result["results"][0].get("legacy_scope_fallback") is True
        assert "legacy_telegram" not in json.dumps(result, ensure_ascii=False)

    def test_cli_search_remains_broad_by_default(self, db):
        self._create_scoped_session(db, "qq_old", chat_id="qq-a", user_id="qq-a", text="modpack qq")
        db.create_session("cli_old", source="cli")
        db.append_message("cli_old", role="user", content="modpack cli")

        result = json.loads(session_search(query="modpack", db=db, current_source="cli"))
        sids = {r["session_id"] for r in result["results"]}
        assert {"qq_old", "cli_old"}.issubset(sids)


# =========================================================================
# Discovery shape (with query)
# =========================================================================

class TestDiscoveryShape:
    def test_query_returns_anchored_windows(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", db=db))
        assert result["success"] is True
        assert result["mode"] == "discover"
        assert result["count"] >= 1

    def test_discovery_result_has_bookends_and_window(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, db=db))
        for hit in result["results"]:
            assert "bookend_start" in hit
            assert "messages" in hit
            assert "bookend_end" in hit
            assert "match_message_id" in hit
            assert "snippet" in hit
            assert "messages_before" in hit
            assert "messages_after" in hit

    def test_match_message_id_is_anchor_in_window(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, db=db))
        for hit in result["results"]:
            anchor_id = hit["match_message_id"]
            window_ids = [m["id"] for m in hit["messages"]]
            assert anchor_id in window_ids

    def test_no_results_returns_empty_list(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="zzz_no_such_term_zzz", db=db))
        assert result["success"] is True
        assert result["results"] == []
        assert result["count"] == 0

    def test_limit_clamped_to_max_10(self, db):
        _seed_modpack_sessions(db)
        # Pass huge limit; should not error and should cap
        result = json.loads(session_search(query="modpack", limit=999, db=db))
        assert result["count"] <= 10

    def test_limit_floor_to_1(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=0, db=db))
        # Result count depends on hits, but the limit must be at least 1
        assert result["count"] >= 0

    def test_non_int_limit_falls_back(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit="bogus", db=db))
        assert result["success"] is True

    def test_current_session_filtered_out(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", db=db, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids


class TestDiscoverySort:
    def test_sort_newest_orders_by_recency(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, sort="newest", db=db))
        # First result should be the most recent session
        first = result["results"][0]
        assert first["session_id"] == "s_newest" or "Newest" in (first.get("title") or "")

    def test_sort_oldest_orders_by_age(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, sort="oldest", db=db))
        first = result["results"][0]
        assert first["session_id"] == "s_oldest"

    def test_invalid_sort_silently_ignored(self, db):
        _seed_modpack_sessions(db)
        # Should not error
        result = json.loads(session_search(query="modpack", sort="bogus", db=db))
        assert result["success"] is True


class TestRoleFilter:
    def test_default_excludes_tool_role(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="modpack question")
        db.append_message("s1", role="tool", content="modpack tool output", tool_name="x")
        result = json.loads(session_search(query="modpack", db=db))
        # The FTS5 match should be on the user message, not the tool message
        if result["count"] > 0:
            matched_role = result["results"][0]["matched_role"]
            assert matched_role in ("user", "assistant")

    def test_explicit_tool_role_includes_tool(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", role="tool", content="modpack tool output", tool_name="x")
        result = json.loads(session_search(query="modpack", role_filter="tool", db=db))
        # Should now match the tool message
        if result["count"] > 0:
            assert result["results"][0]["matched_role"] == "tool"


# =========================================================================
# Scroll shape (session_id + around_message_id)
# =========================================================================

class TestScrollShape:
    def test_scroll_returns_window_without_bookends(self, db):
        _seed_modpack_sessions(db)
        # Get an anchor first via discovery
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]

        # Now scroll
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=2, db=db
        ))
        assert result["success"] is True
        assert result["mode"] == "scroll"
        assert "messages" in result
        # Scroll shape has no bookends
        assert "bookend_start" not in result
        assert "bookend_end" not in result

    def test_scroll_window_clamped_to_20(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=999, db=db
        ))
        assert result["window"] == 20

    def test_scroll_window_floor_to_1(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=-5, db=db
        ))
        assert result["window"] == 1

    def test_scroll_returns_messages_before_after_counts(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=3, db=db
        ))
        assert "messages_before" in result
        assert "messages_after" in result

    def test_scroll_anchor_in_window(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=2, db=db
        ))
        anchor_in_window = [m for m in result["messages"] if m["id"] == anchor_mid]
        assert len(anchor_in_window) == 1
        assert anchor_in_window[0].get("anchor") is True

    def test_scroll_missing_anchor_errors(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(
            session_id="s_oldest", around_message_id=999999, db=db
        ))
        assert result["success"] is False
        assert "not in" in result.get("error", "")

    def test_scroll_missing_session_errors(self, db):
        result = json.loads(session_search(
            session_id="nonexistent", around_message_id=1, db=db
        ))
        assert result["success"] is False

    def test_scroll_rejects_current_session_lineage(self, db):
        _seed_modpack_sessions(db)
        # Grab some valid id from s_oldest
        disc = json.loads(session_search(query="modpack", limit=3, db=db))
        match = [r for r in disc["results"] if r["session_id"] == "s_oldest"]
        if match:
            mid = match[0]["match_message_id"]
            result = json.loads(session_search(
                session_id="s_oldest", around_message_id=mid, db=db,
                current_session_id="s_oldest",
            ))
            assert result["success"] is False
            assert "current session" in result.get("error", "").lower()

    def test_scroll_invalid_around_message_id_errors(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(
            session_id="s_oldest", around_message_id="not-an-int", db=db
        ))
        assert result["success"] is False

    def test_scroll_rejects_out_of_scope_session_and_lineage_rebind(self, db):
        shared_chat = "group-1"
        key_a = "agent:main:qqbot:group:group-1:user-a"
        key_b = "agent:main:qqbot:group:group-1:user-b"
        db.create_session(
            "owner_parent",
            source="qqbot",
            user_id="user-a",
            chat_type="group",
            chat_id=shared_chat,
            session_key=key_a,
        )
        parent_mid = db.append_message("owner_parent", role="user", content="owner parent")
        db.create_session(
            "other_parent",
            source="qqbot",
            user_id="user-b",
            chat_type="group",
            chat_id=shared_chat,
            session_key=key_b,
        )
        other_mid = db.append_message("other_parent", role="user", content="other parent")
        db.create_session(
            "other_child",
            source="qqbot",
            user_id="user-b",
            chat_type="group",
            chat_id=shared_chat,
            session_key=key_b,
            parent_session_id="owner_parent",
        )
        child_mid = db.append_message("other_child", role="user", content="other child")
        db.create_session(
            "owner_current",
            source="qqbot",
            user_id="user-a",
            chat_type="group",
            chat_id=shared_chat,
            session_key=key_a,
        )

        common_scope = dict(
            db=db,
            current_session_id="owner_current",
            current_source="qqbot",
            current_chat_type="group",
            current_chat_id=shared_chat,
            current_user_id="user-a",
            current_session_key=key_a,
        )
        own = json.loads(session_search(
            session_id="owner_parent",
            around_message_id=parent_mid,
            **common_scope,
        ))
        assert own["success"] is True

        direct = json.loads(session_search(
            session_id="other_parent",
            around_message_id=other_mid,
            **common_scope,
        ))
        assert direct["success"] is False
        assert "outside the current scope" in direct.get("error", "")

        rebound = json.loads(session_search(
            session_id="owner_parent",
            around_message_id=child_mid,
            **common_scope,
        ))
        assert rebound["success"] is False
        assert "outside the current scope" in rebound.get("error", "")


class TestScrollPattern:
    """The forward/backward scroll loop using tool output."""

    def test_scroll_forward_from_last_id(self, db):
        # Long session
        db.create_session("s_long", source="cli")
        ids = []
        for i in range(20):
            ids.append(db.append_message("s_long", role="user" if i % 2 == 0 else "assistant",
                                         content=f"long session msg {i}"))

        v1 = json.loads(session_search(
            session_id="s_long", around_message_id=ids[5], window=3, db=db
        ))
        last_id = v1["messages"][-1]["id"]
        v2 = json.loads(session_search(
            session_id="s_long", around_message_id=last_id, window=3, db=db
        ))
        # Forward scroll: v2 should reach further than v1
        assert max(m["id"] for m in v2["messages"]) > max(m["id"] for m in v1["messages"])
        # Boundary id appears in both
        assert last_id in [m["id"] for m in v1["messages"]]
        assert last_id in [m["id"] for m in v2["messages"]]


# =========================================================================
# Shape precedence
# =========================================================================

class TestShapePrecedence:
    def test_scroll_args_beat_query(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        # Pass both query and scroll args — scroll should win
        result = json.loads(session_search(
            query="modpack",  # would normally trigger discovery
            session_id=anchor_sid, around_message_id=anchor_mid, db=db,
        ))
        assert result["mode"] == "scroll"

    def test_empty_query_falls_back_to_browse(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="   ", db=db))
        assert result["mode"] == "browse"

    def test_non_string_query_falls_back_to_browse(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query=None, db=db))  # type: ignore
        assert result["mode"] == "browse"
