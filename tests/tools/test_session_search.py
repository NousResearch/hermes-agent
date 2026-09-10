"""Tests for the single-shape session_search tool.

Four calling shapes:
  1. DISCOVERY — pass query → FTS5 + adaptive/full hydration
  2. SCROLL    — pass session_id + around_message_id → just the window
  3. READ      — pass session_id → whole or head/tail-truncated session
  4. BROWSE    — no args → recent sessions chronologically

All run zero LLM calls.
"""
import inspect
import json
import time
from datetime import datetime, timezone

import pytest

from hermes_state import SessionDB
from tools.registry import registry
from tools.session_search_tool import (
    SESSION_SEARCH_SCHEMA,
    _branch_copy_edge,
    _format_timestamp,
    _is_compacted_message,
    _resolve_to_parent,
    _session_link,
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
    def test_schema_params_cover_every_shape(self):
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        # Discovery shape
        assert "query" in params
        assert "limit" in params
        assert params["sort"]["enum"] == ["newest", "oldest"]
        assert params["detail"]["enum"] == ["adaptive", "full"]
        assert params["detail"]["default"] == "adaptive"
        # Scroll shape
        assert "session_id" in params
        assert "around_message_id" in params
        assert "window" in params
        # Shared
        assert "role_filter" in params
        # Mode is inferred from which args are set — no explicit mode param
        assert "mode" not in params

    def test_detail_parameter_is_appended_for_positional_compatibility(self):
        parameters = list(inspect.signature(session_search).parameters)
        historical_prefix = [
            "query",
            "role_filter",
            "limit",
            "db",
            "current_session_id",
            "session_id",
            "around_message_id",
            "window",
            "sort",
            "profile",
        ]
        assert parameters == [*historical_prefix, "detail"]


class TestFormatTimestamp:
    def test_formats_unix_and_passes_through_the_rest(self):
        assert "2023" in _format_timestamp(1700000000)
        assert _format_timestamp(None) == "unknown"
        assert _format_timestamp("not-a-number-string") == "not-a-number-string"


# =========================================================================
# Browse shape (no args)
# =========================================================================

class TestBrowseShape:
    def test_browse_uses_bounded_recent_path(self):
        class _DB:
            rich_called = False
            bounded_kwargs = None

            def list_recent_sessions_bounded(self, **kwargs):
                self.bounded_kwargs = kwargs
                return []

            def list_sessions_rich(self, **_kwargs):
                self.rich_called = True
                raise AssertionError("unbounded rich listing must not be used")

        db = _DB()
        result = json.loads(session_search(db=db))

        assert result["success"] is True
        assert db.rich_called is False
        assert db.bounded_kwargs["timeout_seconds"] == 3.0

    def test_browse_fails_closed_without_bounded_database_capability(self):
        class _LegacyDB:
            def list_sessions_rich(self, **_kwargs):
                raise AssertionError("known-unbounded fallback must not be called")

        result = json.loads(session_search(db=_LegacyDB()))

        assert result["success"] is False
        assert "does not support bounded recent-session browse" in result["error"]

    def test_lazy_database_is_released_after_search(self, monkeypatch):
        class _DB:
            released = 0

            def list_recent_sessions_bounded(self, **_kwargs):
                return []

        db = _DB()
        monkeypatch.setattr("hermes_state_registry.acquire", lambda: db)
        monkeypatch.setattr(
            "hermes_state_registry.release_or_close",
            lambda _: setattr(db, "released", db.released + 1),
        )

        result = json.loads(session_search())

        assert result["success"] is True
        assert db.released == 1

    def test_cross_profile_database_is_closed_but_shared_database_is_not(
        self, monkeypatch
    ):
        class _DB:
            def __init__(self):
                self.closed = 0

            def list_recent_sessions_bounded(self, **_kwargs):
                return []

            def close(self):
                self.closed += 1

        shared_db = _DB()
        profile_db = _DB()
        monkeypatch.setattr(
            "tools.session_search_tool._resolve_profile_db",
            lambda _profile: profile_db,
        )

        result = json.loads(session_search(db=shared_db, profile="work"))

        assert result["success"] is True
        assert profile_db.closed == 1
        assert shared_db.closed == 0

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


# =========================================================================
# Discovery shape (with query)
# =========================================================================

class TestDiscoveryShape:
    def test_discovery_field_plan_preserves_full_default_result(self, db, monkeypatch):
        _seed_modpack_sessions(db)
        original = db.search_messages
        requested_fields = None

        def search_spy(*args, **kwargs):
            nonlocal requested_fields
            requested_fields = kwargs.get("fields")
            return original(*args, **kwargs)

        monkeypatch.setattr(db, "search_messages", search_spy)

        result = json.loads(session_search(query="modpack", limit=1, db=db))

        assert result["success"] is True
        assert requested_fields is not None
        assert "context" not in requested_fields
        assert len(result["results"]) == 1
        hit = result["results"][0]
        assert hit["detail"] == "full"
        assert "bookend_start" in hit
        assert hit["messages"]
        assert "bookend_end" in hit

    def test_full_detail_returns_bookends_and_window_for_every_hit(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(
            query="modpack", limit=3, detail="full", db=db
        ))
        assert result["success"] is True
        assert result["mode"] == "discover"
        assert result["detail"] == "full"
        assert result["count"] >= 1
        for hit in result["results"]:
            assert hit["detail"] == "full"
            assert "bookend_start" in hit
            assert "messages" in hit
            assert "bookend_end" in hit
            assert "match_message_id" in hit
            assert "snippet" in hit
            assert "messages_before" in hit
            assert "messages_after" in hit

    def test_default_discovery_keeps_top_full_and_compacts_lower_hits(self, db):
        _seed_modpack_sessions(db)

        result = json.loads(session_search(query="modpack", limit=3, db=db))

        assert result["success"] is True
        assert result["detail"] == "adaptive"
        assert len(result["results"]) == 3

        top, *lower = result["results"]
        assert top["detail"] == "full"
        assert "bookend_start" in top
        assert len(top["messages"]) > 1
        assert "bookend_end" in top

        for hit in lower:
            assert hit["detail"] == "compact"
            assert hit["bookend_start"] == []
            assert len(hit["messages"]) == 1
            assert hit["messages"][0]["id"] == hit["match_message_id"]
            assert hit["messages"][0]["anchor"] is True
            assert hit["bookend_end"] == []

    def test_adaptive_detail_preserves_ranking_and_reduces_payload(self, db):
        now = int(time.time())
        for session_index in range(3):
            session_id = f"payload_{session_index}"
            db.create_session(session_id, source="cli")
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (now - session_index, session_id),
            )
            for message_index in range(8):
                db.append_message(
                    session_id,
                    role="user" if message_index % 2 == 0 else "assistant",
                    content=f"opening {session_index}-{message_index} " + "o" * 2500,
                )
            db.append_message(
                session_id,
                role="user",
                content=f"payloadneedle anchor {session_index} " + "a" * 3500,
            )
            for message_index in range(8):
                db.append_message(
                    session_id,
                    role="assistant" if message_index % 2 == 0 else "user",
                    content=f"closing {session_index}-{message_index} " + "c" * 2500,
                )
        db._conn.commit()

        adaptive_json = session_search(query="payloadneedle", limit=3, db=db)
        full_json = session_search(
            query="payloadneedle", limit=3, detail="full", db=db
        )
        adaptive = json.loads(adaptive_json)
        full = json.loads(full_json)

        assert [r["session_id"] for r in adaptive["results"]] == [
            r["session_id"] for r in full["results"]
        ]
        assert [r["match_message_id"] for r in adaptive["results"]] == [
            r["match_message_id"] for r in full["results"]
        ]
        assert len(adaptive_json.encode("utf-8")) < len(full_json.encode("utf-8")) * 0.6


    def test_current_session_filtered_out(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", db=db, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids


class TestGatewayRestartLineageDiscovery:
    @pytest.fixture
    def restarted_lineage(self, db, request):
        """Gateway restart chain in production shape: each successor row carries the
        ``_reset_from`` marker stamped atomically at CREATE time
        (gateway/session_recovery.py ``_session_create_kwargs``); the predecessor's
        end write is a separate best-effort and may be missing ("the old row remains
        open"). ``request.param=False`` seeds the pre-marker bare-link shape."""
        markers = getattr(request, "param", True)
        root, owner, current = (
            "20260812_gateway_root", "20260903_gateway_owner", "20260904_gateway_current",
        )
        parent = None
        for sid, day, title in (
            (root, "2026-08-12", "MCP серверы Hermes"),
            (owner, "2026-09-03", "September gateway recovery"),
            (current, "2026-09-04", "Current gateway conversation"),
        ):
            db.create_session(
                sid, source="telegram", parent_session_id=parent,
                session_key="tg:restart:1",
                model_config={"_reset_from": parent} if markers and parent else None,
            )
            started_at = datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
                (started_at, title, sid),
            )
            parent = sid
        # A restart can create a successor without ever ending its predecessor.
        for index in range(6):
            db.append_message(
                owner, role="user" if index % 2 == 0 else "assistant",
                content=f"restartneedle recovery note {index}",
                timestamp=datetime(2026, 9, 3, 12, index, tzinfo=timezone.utc).timestamp(),
            )
        db.append_message(current, role="user", content="currentneedle live context")
        db._conn.commit()
        return owner, current

    # markers=True: production rows carry _reset_from; markers=False: pre-marker bare
    # links. Both must behave identically for every end_reason — the marker decides
    # the boundary, not the (optional, overwritable) end write.
    @pytest.mark.parametrize("restarted_lineage", [True, False], indirect=True)
    @pytest.mark.parametrize("end_reason", [None, "compression", "session_reset"])
    def test_previous_owner_is_searchable_but_current_context_is_not(
        self, db, restarted_lineage, end_reason,
    ):
        owner, current = restarted_lineage
        if end_reason is not None:
            db.end_session(owner, end_reason)
        result = json.loads(registry.dispatch(
            "session_search", {"query": "restartneedle"},
            db=db, current_session_id=current,
        ))
        assert result["success"] is True
        assert [hit["session_id"] for hit in result["results"]] == [owner]
        assert result["sessions_searched"] == result["count"] == 1
        live = json.loads(registry.dispatch(
            "session_search", {"query": "currentneedle"},
            db=db, current_session_id=current,
        ))
        assert live["success"] is True
        assert live["results"] == []

    def test_metadata_describes_the_message_owner_not_the_lineage_root(
        self, db, restarted_lineage,
    ):
        owner, _ = restarted_lineage
        # Omit current_session_id so the visibility bug cannot mask the metadata bug.
        result = json.loads(registry.dispatch(
            "session_search", {"query": "restartneedle"}, db=db,
        ))
        assert result["success"] is True
        hit, = result["results"]
        assert hit["session_id"] == owner
        # The tool formats timestamps in the host's local zone — compute the expected
        # string the same way instead of hardcoding a zone-dependent literal.
        expected_when = datetime.fromtimestamp(
            datetime.fromisoformat("2026-09-03").replace(tzinfo=timezone.utc).timestamp()
        ).strftime("%B %d, %Y at %I:%M %p")
        assert (hit["when"], hit["title"]) == (
            expected_when, "September gateway recovery",
        )

    def test_unended_restart_predecessor_hit_is_scrollable(self, db, restarted_lineage):
        """PR-head gap: discovery surfaced the unended restart predecessor but scroll
        rejected its anchor — "scroll never rejects a discovery result"."""
        owner, current = restarted_lineage
        disc = json.loads(session_search(
            query="restartneedle", db=db, current_session_id=current, limit=1,
        ))
        assert disc["count"] == 1
        hit = disc["results"][0]
        scrolled = json.loads(session_search(
            session_id=hit["session_id"],
            around_message_id=hit["match_message_id"],
            db=db,
            current_session_id=current,
        ))
        assert scrolled["success"] is True
        assert scrolled["mode"] == "scroll"
        contents = " ".join(m.get("content") or "" for m in scrolled["messages"])
        assert "restartneedle" in contents


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


# =========================================================================
# Scroll shape (session_id + around_message_id)
# =========================================================================

class TestScrollShape:
    def test_scroll_returns_anchored_window_without_bookends(self, db):
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
        # Scroll shape has no bookends
        assert "bookend_start" not in result
        assert "bookend_end" not in result
        # The anchor is in the window and flagged
        anchor_in_window = [m for m in result["messages"] if m["id"] == anchor_mid]
        assert len(anchor_in_window) == 1
        assert anchor_in_window[0].get("anchor") is True

    def test_scroll_window_clamped_to_20(self, db):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=999, db=db
        ))
        assert result["window"] == 20


    def test_scroll_rejects_active_delegation_child_in_current_lineage(self, db):
        """Production shape (run_agent.py source fallback): a delegate run can land
        with source='cli', so the ``_delegate_from`` marker — not the source — is what
        keeps live delegation children out of recall. A bare child row without a
        marker is indistinguishable from a restart successor and must stay
        scrollable (see TestGatewayRestartLineageDiscovery)."""
        db.create_session("s_current", source="cli")
        db.create_session(
            "s_delegate", source="cli", parent_session_id="s_current",
            model_config={"_delegate_from": "s_current"},
        )
        mid = db.append_message(
            "s_delegate", role="assistant", content="live delegated result"
        )

        result = json.loads(session_search(
            session_id="s_delegate", around_message_id=mid, db=db,
            current_session_id="s_current",
        ))

        assert result["success"] is False
        assert "current session" in result.get("error", "").lower()


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


    def test_session_id_without_anchor_reads(self, db):
        _seed_modpack_sessions(db)
        # session_id alone (no anchor, no query) → read shape, not browse.
        result = json.loads(session_search(session_id="s_oldest", db=db))
        assert result["mode"] == "read"


# =========================================================================
# Read shape — dump a whole session by id (serves @session links)
# =========================================================================

class TestReadShape:
    def test_read_returns_full_session(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(session_id="s_oldest", db=db))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["session_id"] == "s_oldest"
        assert result["message_count"] == 5
        assert result["truncated"] is False
        assert len(result["messages"]) == 5
        assert result["session_meta"]["title"] == "Building the Modpack"

    def test_read_strips_ansi_sequences_from_messages(self, db):
        db.create_session("s_ansi", source="cli")
        db.append_message("s_ansi", role="user", content="plain")
        db.append_message(
            "s_ansi", role="assistant", content="\u001b[31mred text\u001b[0m and more"
        )
        db._conn.commit()
        result = json.loads(session_search(session_id="s_ansi", db=db))
        assert result["success"] is True
        rendered = [m["content"] for m in result["messages"] if m.get("content")]
        assert any(text == "red text and more" for text in rendered)
        assert all("\u001b" not in text for text in rendered)

    def test_read_truncates_large_session(self, db):
        db.create_session("s_big", source="cli")
        for i in range(50):
            db.append_message("s_big", role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        db._conn.commit()
        result = json.loads(session_search(session_id="s_big", db=db))
        assert result["mode"] == "read"
        assert result["message_count"] == 50
        assert result["truncated"] is True
        assert len(result["messages"]) == 30  # head 20 + tail 10


# =========================================================================
# Session links — the value the agent writes to point the user at a session
# =========================================================================

def _linked_session_id(link: str) -> str:
    """Recover the session id from an `@session:[<profile>/]<id>` value."""
    assert link.startswith("@session:"), link
    value = link[len("@session:"):]

    return value.rsplit("/", 1)[-1]


class TestSessionLink:
    def test_link_carries_the_named_profile(self):
        assert _session_link("s_oldest", "work") == "@session:work/s_oldest"


    def test_every_discovery_result_links_to_its_own_session(self, db):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=5, db=db))

        assert result["results"]
        for entry in result["results"]:
            assert _linked_session_id(entry["link"]) == entry["session_id"]


# =========================================================================
# Cross-profile read — `profile` swaps in another profile's DB (read-only)
# =========================================================================

class TestCrossProfileRead:
    def _patch_profiles(self, monkeypatch, home, exists=True):
        from hermes_cli import profiles as profiles_mod
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: exists)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: home)

    def test_bare_id_locates_across_profiles(self, db, tmp_path, monkeypatch):
        # The real-world failure: model dropped the owning profile and passed a
        # bare id. The tool must scan profiles and find it anyway.
        other_home = tmp_path / "asdf_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        other.create_session("s_far", source="cli")
        other.append_message("s_far", role="user", content="hi")
        other._conn.commit()

        from collections import namedtuple
        from hermes_cli import profiles as profiles_mod
        Info = namedtuple("Info", "name path")
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: tmp_path / "default_home")
        monkeypatch.setattr(profiles_mod, "list_profiles", lambda: [Info("asdf", other_home)])

        # `db` (current profile) lacks s_far; no profile passed → scan finds it.
        result = json.loads(session_search(session_id="s_far", db=db))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["profile"] == "asdf"


    def test_combined_value_autosplits(self, db, tmp_path, monkeypatch):
        # Agent passed the raw "@session:<profile>/<id>" value as session_id with
        # no separate profile — the tool should recover both.
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        other.create_session("s_other", source="cli")
        other.append_message("s_other", role="user", content="hi")
        other._conn.commit()

        self._patch_profiles(monkeypatch, other_home)

        # Every permutation the model might send must resolve to (asdf, s_other).
        for kwargs in (
            {"session_id": "asdf/s_other"},                    # full value, no profile
            {"session_id": "asdf/s_other", "profile": "asdf"},  # full value AND profile
            {"session_id": "s_other", "profile": "asdf"},       # bare id + profile
        ):
            result = json.loads(session_search(db=db, **kwargs))
            assert result["success"] is True, kwargs
            assert result["mode"] == "read"
            assert result["session_id"] == "s_other"


# =========================================================================
# Cron demotion in discover ranking (#19434)
# =========================================================================

class TestCronDemotion:
    def _seed_cron_and_interactive(self, db):
        """One interactive (telegram) session and several cron sessions, all
        matching the same query. Cron rows accumulate repetitive vocabulary
        and out-number the user's single interactive session — the live-data
        symptom in #19434.
        """
        now = int(time.time())
        # Interactive user session — older, so it loses on bare recency too.
        db.create_session("s_user", source="telegram")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                         (now - 90000, "s_user"))
        db.append_message("s_user", role="user", content="how is the venom project going")
        db.append_message("s_user", role="assistant", content="The venom project shipped its first milestone.")
        # Several cron sessions, all newer and all stuffed with the same terms.
        for i in range(8):
            sid = f"cron_{i}"
            db.create_session(sid, source="cron")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                             (now - 1000 - i, sid))
            db.append_message(sid, role="user", content="venom project daily status")
            db.append_message(sid, role="assistant", content="venom project venom project venom summary")
        db._conn.commit()

    def test_interactive_session_surfaces_above_cron(self, db):
        self._seed_cron_and_interactive(db)
        result = json.loads(session_search(query="venom project", limit=1, db=db))
        assert result["success"] is True
        assert result["count"] == 1
        # With cron drowning FTS, bare BM25/recency would return a cron_* hit.
        # Demotion must put the user's interactive session first.
        assert result["results"][0]["source"] == "telegram"
        assert result["results"][0]["session_id"] == "s_user"

    def test_cron_still_reachable_when_only_match(self, db):
        """Demotion must not exclude cron — when only cron matches, it still
        comes back."""
        now = int(time.time())
        db.create_session("cron_only", source="cron")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                         (now - 500, "cron_only"))
        db.append_message("cron_only", role="user", content="quarterly archive sweep")
        db.append_message("cron_only", role="assistant", content="Archive sweep complete.")
        db._conn.commit()
        result = json.loads(session_search(query="archive sweep", db=db))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["source"] == "cron"


# =========================================================================
# Compaction summary filtering (#43175)
# =========================================================================

class TestCompactionSummaryFiltering:
    """session_search discovery must exclude compaction handoffs from bookends."""

    def test_is_compaction_summary_detects_prefix(self):
        from tools.session_search_tool import _is_compaction_summary
        assert _is_compaction_summary("[CONTEXT COMPACTION — REFERENCE ONLY] foo")
        assert _is_compaction_summary("[CONTEXT SUMMARY]: old summary")
        assert not _is_compaction_summary("Hello, how can I help?")
        assert not _is_compaction_summary("")
        assert not _is_compaction_summary(None)

    def test_compaction_summary_excluded_from_bookend_start(self, db):
        """Compaction handoff in bookend_start position must be filtered out."""
        db.create_session("s_compact", source="cli")
        # First message: a compaction handoff (should be filtered)
        db.append_message("s_compact", role="user",
                          content="[CONTEXT COMPACTION — REFERENCE ONLY] "
                                  "Earlier turns were compacted into the summary below. " + "x" * 50000)
        # Second message: normal user message
        db.append_message("s_compact", role="user", content="Fix the zorgblat rendering bug")
        # Padding messages to push window away from session start (so bookend has room)
        for i in range(10):
            db.append_message("s_compact", role="user", content=f"setup step {i}")
            db.append_message("s_compact", role="assistant", content=f"setup done {i}")
        # Match target: uses a unique term so FTS5 anchors here, not at the start
        db.append_message("s_compact", role="user", content="investigate the frobnitz mob spawning in KubeJS")
        db.append_message("s_compact", role="assistant", content="I'll look into the frobnitz mob spawning issue.")
        # Tail messages
        for i in range(5):
            db.append_message("s_compact", role="user", content=f"tail {i}")
            db.append_message("s_compact", role="assistant", content=f"done tail {i}")
        db._conn.commit()

        result = json.loads(session_search(query="frobnitz mob spawning", db=db, limit=1))
        assert result["success"] is True
        assert len(result["results"]) >= 1
        entry = result["results"][0]
        # bookend_start must NOT contain the compaction handoff
        for msg in entry.get("bookend_start", []):
            assert "[CONTEXT COMPACTION" not in (msg.get("content") or "")
        # The normal message should still be present in bookend_start
        bookend_contents = [m.get("content", "") for m in entry.get("bookend_start", [])]
        assert any("zorgblat" in c for c in bookend_contents)


# =========================================================================
# Compression-aware discovery (#6256)
#
# After compression (in-place compaction or legacy rotation), pre-compaction
# content is no longer in the live context but MUST stay discoverable via
# session_search. The old code skipped any FTS hit on the current session or
# lineage, creating a "memory black hole". Live context is scoped to the
# current session; hidden subagent sources remain excluded separately.
# =========================================================================

class TestResolveToParent:
    """Unit tests for _resolve_to_parent's compression-aware tuple return."""

    def test_legacy_rotation_detects_compression(self, db):
        """Parent ended with end_reason='compression', child has parent_session_id."""
        db.create_session("s_parent", source="cli")
        db.end_session("s_parent", "compression")
        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        root, has_compression = _resolve_to_parent(db, "s_child")
        assert root == "s_parent"
        assert has_compression is True


    def test_chain_with_mixed_edges(self, db):
        """Compression grandparent → parent → child (no end_reason on parent)."""
        db.create_session("s_gp", source="cli")
        db.end_session("s_gp", "compression")
        db.create_session("s_p", source="cli", parent_session_id="s_gp")
        # s_p does NOT end with compression — but ancestor s_gp does
        db.create_session("s_c", source="cli", parent_session_id="s_p")
        root, has_compression = _resolve_to_parent(db, "s_c")
        assert root == "s_gp"
        assert has_compression is True


class TestIsCompactedMessage:
    """Unit tests for the _is_compacted_message helper."""

    def test_active_message_returns_false(self, db):
        db.create_session("s1", source="cli")
        mid = db.append_message("s1", role="user", content="hello")
        assert _is_compacted_message(db, mid) is False

    def test_compacted_message_returns_true(self, db):
        db.create_session("s1", source="cli")
        mid = db.append_message("s1", role="user", content="archived content")
        db.archive_and_compact("s1", [
            {"role": "assistant", "content": "compacted summary"},
        ])
        # mid is now active=0, compacted=1
        assert _is_compacted_message(db, mid) is True


class TestInPlaceCompactionDiscovery:
    """In-place compaction: archived turns on the SAME session_id must be
    discoverable from the current session."""

    def test_archived_content_discoverable_after_compaction(self, db):
        """The core regression: pre-compaction content on the current session
        must surface in discovery even though raw_sid == current_session_id."""
        db.create_session("s_compact", source="cli")
        db.append_message("s_compact", role="user",
                          content="The spectral phoenix only spawns during full moons")
        db.append_message("s_compact", role="assistant",
                          content="Spectral phoenix requires moonstone bait")
        db.archive_and_compact("s_compact", [
            {"role": "user", "content": "Summary: spectral phoenix discussed"},
            {"role": "assistant", "content": "Acknowledged spectral phoenix info"},
        ])

        result = json.loads(session_search(
            query="spectral phoenix", db=db, current_session_id="s_compact",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        # The hit should be from the same session (archived rows)
        hit = result["results"][0]
        assert hit["session_id"] == "s_compact"

    def test_live_content_still_filtered_on_current_session(self, db):
        """Non-compacted (active) content on the current session stays filtered."""
        db.create_session("s_live", source="cli")
        db.append_message("s_live", role="user", content="crystal golem farming route")
        result = json.loads(session_search(
            query="crystal golem", db=db, current_session_id="s_live",
        ))
        assert result["count"] == 0


class TestLegacyRotationDiscovery:
    """Legacy rotation: parent session ended with end_reason='compression',
    child session created. Parent's pre-compaction content must be discoverable
    from the child."""

    def test_compression_parent_discoverable_from_child(self, db):
        db.create_session("s_parent", source="cli")
        db.append_message("s_parent", role="user",
                          content="The void crystal mining requires diamond pickaxe")
        db.append_message("s_parent", role="assistant",
                          content="Void crystal found in the deep caverns")
        db.end_session("s_parent", "compression")

        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        db.append_message("s_child", role="user", content="Continue void crystal work")

        result = json.loads(session_search(
            query="void crystal", db=db, current_session_id="s_child",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_parent" in sids


class TestSameLineageDiscovery:
    """A bare parent link proves nothing either way: with no marker and no branch end
    write (pre-marker legacy rows) the predecessor is treated as NOT projected into
    the caller's context — recall errs toward surfacing (#20856/#85756) — while the
    current session's own rows stay excluded."""

    def test_unended_parent_surfaces_but_current_child_is_excluded(self, db):
        db.create_session("s_parent", source="cli")
        db.append_message("s_parent", role="user",
                          content="nebula deployment infrastructure setup")
        db.append_message("s_parent", role="assistant",
                          content="Nebula deployment configured successfully")

        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        db.append_message("s_child", role="user",
                          content="delegated nebula deployment subtask")

        result = json.loads(session_search(
            query="nebula deployment", db=db, current_session_id="s_child",
        ))
        assert [hit["session_id"] for hit in result["results"]] == ["s_parent"]


# =========================================================================
# Both layers together: discovery scope (#63144) × bookend bounding (#69334)
#
# Compaction touches two independent layers of session_search:
#   1. Discovery scope — compaction-archived rows on the current session must
#      surface in discovery (this PR).
#   2. Content bounding — bookends must exclude generated compaction handoff
#      summaries and cap message content length (#43175 / #69334).
# A compacted session exercises both at once: its archived content is the FTS
# hit, while the compaction summary row it produced sits at the session tail,
# exactly where bookend_end is sampled.
# =========================================================================

class TestCompactionDiscoveryBothLayers:
    """Compacted-session content is discoverable AND its bookends still
    exclude compaction summaries / cap content length."""

    def _seed_compacted_session(self, db):
        db.create_session("s_both", source="cli")
        # Long normal opening — exercises the 1200-char bookend cap.
        db.append_message("s_both", role="user",
                          content="Kick off the obsidian gateway migration. " + "o" * 5000)
        db.append_message("s_both", role="assistant",
                          content="Starting the obsidian gateway migration plan.")
        # Padding so the anchored window doesn't swallow the bookends.
        for i in range(10):
            db.append_message("s_both", role="user", content=f"migration step {i}")
            db.append_message("s_both", role="assistant", content=f"migration step {i} done")
        # The FTS match target — will be archived by compaction below.
        db.append_message("s_both", role="user",
                          content="the obsidian gateway needs a quartz keystone to activate")
        db.append_message("s_both", role="assistant",
                          content="Noted: quartz keystone required for the obsidian gateway.")
        for i in range(5):
            db.append_message("s_both", role="user", content=f"wrap-up {i}")
            db.append_message("s_both", role="assistant", content=f"wrapped {i}")
        # Compact in place: everything above becomes active=0/compacted=1 and
        # the handoff summary is inserted as the new live tail.
        db.archive_and_compact("s_both", [
            {"role": "user",
             "content": "[CONTEXT COMPACTION — REFERENCE ONLY] "
                        "Earlier turns were compacted into this summary. " + "s" * 50000},
            {"role": "assistant", "content": "Continuing after compaction."},
        ])
        db._conn.commit()

    def test_archived_hit_surfaces_with_bounded_summary_free_bookends(self, db):
        self._seed_compacted_session(db)

        result = json.loads(session_search(
            query="quartz keystone", db=db, current_session_id="s_both",
        ))

        # Layer 1 — discovery scope: the archived (active=0, compacted=1)
        # content on the CURRENT session must surface.
        assert result["success"] is True
        assert result["count"] >= 1
        entry = result["results"][0]
        assert entry["session_id"] == "s_both"

        # Layer 2a — summary exclusion: the compaction handoff row sits at the
        # session tail (freshly inserted by archive_and_compact), exactly where
        # bookend_end samples — it must be filtered out.
        for msg in entry.get("bookend_start", []) + entry.get("bookend_end", []):
            assert "[CONTEXT COMPACTION" not in (msg.get("content") or "")

        # Layer 2b — content caps: bookends ≤1200 chars, window ≤4000 chars.
        for msg in entry.get("bookend_start", []) + entry.get("bookend_end", []):
            assert len(msg.get("content") or "") <= 1210
        for msg in entry.get("messages", []):
            assert len(msg.get("content") or "") <= 4010

        # The long-but-legitimate opening survives (capped, not dropped).
        bookend_contents = [m.get("content") or "" for m in entry.get("bookend_start", [])]
        assert any("obsidian gateway migration" in c for c in bookend_contents)


# =========================================================================
# Teknium review round 2: rewind exclusion + delegation-under-compression
# =========================================================================

class TestRewindExclusion:
    """Rewind/undo rows (active=0, compacted=0) must STAY hidden — only
    compaction archives (active=0, compacted=1) should surface."""

    def test_compacted_messages_still_surface_alongside_rewind(self, db):
        """On the same session: compacted rows surface, rewind rows don't."""
        db.create_session("s_mixed", source="cli")
        # Message that will be compacted
        db.append_message("s_mixed", role="user",
                          content="compaction archived content beta")
        db.archive_and_compact("s_mixed", [
            {"role": "assistant", "content": "Summary of beta"},
        ])
        # Now add a post-compaction message and rewind it
        mid2 = db.append_message("s_mixed", role="user",
                                 content="rewound content gamma")
        db._conn.execute(
            "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
            (mid2,),
        )
        db._conn.commit()

        # Compacted content should be discoverable
        result_compact = json.loads(session_search(
            query="compaction archived content beta", db=db,
            current_session_id="s_mixed",
        ))
        assert result_compact["count"] >= 1

        # Rewound content should NOT be discoverable
        result_rewind = json.loads(session_search(
            query="rewound content gamma", db=db,
            current_session_id="s_mixed",
        ))
        assert result_rewind["count"] == 0


class TestLegacyContinuationPlusDelegation:
    """Regression: a delegation child created under a compression continuation
    must stay excluded because subagent runs are not the user's history.
    Only the compression-ended ancestor's content should surface."""

    def test_compression_parent_surfaces_but_delegate_child_excluded(self, db):
        """Setup: grandparent (compression) → parent (compression) → child
        (active, current session). A delegation grandchild is created under
        the parent. Searching from the child should find grandparent/parent
        content but NOT the delegation grandchild's content."""
        # Grandparent: compression-ended, has searchable content
        db.create_session("s_gp", source="cli")
        db.append_message("s_gp", role="user",
                          content="grandparent cosmic anomaly research data")
        db.end_session("s_gp", "compression")

        # Parent: compression-ended continuation
        db.create_session("s_p", source="cli", parent_session_id="s_gp")
        db.append_message("s_p", role="user",
                          content="parent cosmic anomaly follow-up notes")
        db.end_session("s_p", "compression")

        # Current session: active child
        db.create_session("s_current", source="cli", parent_session_id="s_p")

        # Delegation child under s_p (not compression-ended)
        db.create_session("s_delegate", source="subagent", parent_session_id="s_p")
        db.append_message("s_delegate", role="assistant",
                          content="delegated cosmic anomaly subtask results")

        result = json.loads(session_search(
            query="cosmic anomaly", db=db,
            current_session_id="s_current",
        ))

        # Compression-ended ancestors should be discoverable
        sids = [r["session_id"] for r in result["results"]]
        assert "s_gp" in sids or "s_p" in sids

        # Delegation child must NOT appear
        assert "s_delegate" not in sids


# =========================================================================
# /new-reset lineage must stay discoverable (#85756)
#
# Gateway /new creates a child with parent_session_id and ends the parent
# with end_reason='session_reset'. That child carries no transcript, so the
# current-lineage exclusion (which assumes same-root content is already in
# context) goes blind: FTS hits in last-night's session are dropped, and
# browse hides every recent interactive row because they all have a parent.
# Hidden subagent sources must stay excluded independently of parent links.
# =========================================================================

def _seed_gateway_new_reset_chain(db, *, needle="ibuprofen night-dose protocol"):
    """A → B → C gateway /new chain. C is the empty current session."""
    db.create_session(
        "s_aug12", source="telegram", session_key="tg:user:1",
    )
    db.append_message("s_aug12", role="user", content="older unrelated chat")
    db.end_session("s_aug12", "session_reset")

    db.create_session(
        "s_night", source="telegram",
        parent_session_id="s_aug12",
        session_key="tg:user:1",
        model_config={"_reset_from": "s_aug12"},
    )
    db._conn.execute(
        "UPDATE sessions SET title = ? WHERE id = ?",
        ("Night ibuprofen plan", "s_night"),
    )
    db.append_message("s_night", role="user", content=f"Remember the {needle}")
    db.append_message(
        "s_night", role="assistant", content=f"Noted {needle} at 21:00",
    )
    db.end_session("s_night", "session_reset")

    db.create_session(
        "s_today", source="telegram",
        parent_session_id="s_night",
        session_key="tg:user:1",
        model_config={"_reset_from": "s_night"},
    )
    db._conn.commit()
    return needle


class TestNewResetLineageDiscovery:
    """After /new, yesterday's session must be searchable from the empty child."""

    def test_session_reset_parent_discoverable_from_child(self, db):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(
            query="ibuprofen", db=db, current_session_id="s_today",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_night" in sids
        blob = json.dumps(result["results"], ensure_ascii=False).lower()
        assert "ibuprofen" in blob

    def test_cli_new_session_parent_discoverable_from_child(self, db):
        db.create_session("s_cli_old", source="cli")
        db.append_message(
            "s_cli_old", role="user",
            content="quartz lantern wiring diagram from yesterday",
        )
        db.end_session("s_cli_old", "new_session")
        db.create_session(
            "s_cli_new", source="cli", parent_session_id="s_cli_old",
        )
        result = json.loads(session_search(
            query="quartz lantern", db=db, current_session_id="s_cli_new",
        ))
        assert result["count"] >= 1
        assert "s_cli_old" in [r["session_id"] for r in result["results"]]

    def test_hidden_subagent_child_still_excluded(self, db):
        """Source filtering must still hide subagent runs after lineage filtering changes."""
        db.create_session("s_parent", source="cli")
        db.create_session(
            "s_child", source="subagent", parent_session_id="s_parent",
        )
        db.append_message(
            "s_child", role="user",
            content="nebula deployment infrastructure setup",
        )
        result = json.loads(session_search(
            query="nebula deployment", db=db, current_session_id="s_parent",
        ))
        assert result["count"] == 0

    def test_title_match_reset_parent_not_dropped(self, db):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(
            query="Night ibuprofen plan", db=db, current_session_id="s_today",
        ))
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_night" in sids

    def test_scroll_into_reset_parent_is_allowed(self, db):
        _seed_gateway_new_reset_chain(db)
        disc = json.loads(session_search(
            query="ibuprofen", db=db, current_session_id="s_today", limit=1,
        ))
        assert disc["count"] >= 1
        hit = disc["results"][0]
        scrolled = json.loads(session_search(
            session_id=hit["session_id"],
            around_message_id=hit["match_message_id"],
            db=db,
            current_session_id="s_today",
        ))
        assert scrolled["success"] is True
        assert scrolled["mode"] == "scroll"
        contents = " ".join(m.get("content") or "" for m in scrolled["messages"])
        assert "ibuprofen" in contents.lower()


class TestNewResetLineageBrowse:
    """Browse must list /new-reset conversations, not only cron/root rows."""

    def test_reset_parent_appears_in_browse(self, db):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(db=db, current_session_id="s_today"))
        assert result["mode"] == "browse"
        sids = [r["session_id"] for r in result["results"]]
        assert "s_today" not in sids
        assert "s_night" in sids

    def test_browse_still_hides_live_delegation_child(self, db):
        db.create_session("s_main", source="cli")
        db.append_message("s_main", role="user", content="parent work")
        db.create_session(
            "s_delegate", source="cli", parent_session_id="s_main",
        )
        db.append_message("s_delegate", role="assistant", content="subagent work")
        result = json.loads(session_search(db=db, current_session_id="s_other"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_delegate" not in sids
        assert "s_main" in sids

    def test_browse_lists_legacy_premarker_reset_child(self, db):
        """A pre-marker reset child (no _reset_from, admitted by the SQL
        same-key heuristic because its parent ended at a reset boundary on
        the same session_key) must not be re-hidden by a Python re-check.
        Regression guard for the follow-up to #85756."""
        db.create_session("s_old", source="telegram", session_key="tg:legacy:1")
        db.append_message("s_old", role="user", content="legacy era chat")
        db.end_session("s_old", "session_reset")
        # Legacy child: parent link + same session_key, NO _reset_from marker,
        # still live (end_reason=None).
        db.create_session(
            "s_legacy_child", source="telegram",
            parent_session_id="s_old", session_key="tg:legacy:1",
        )
        db.append_message("s_legacy_child", role="user", content="current era chat")
        result = json.loads(session_search(db=db, current_session_id="s_other"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_legacy_child" in sids


# =========================================================================
# Branch-copy boundary (review rework of #105147 / upstream #103184)
#
# /branch is the only writer that carries a transcript verbatim into a child
# session row, and it stamps ``_branched_from`` atomically in the child's
# INSERT. A predecessor whose transcript is already projected into the
# caller's live context must STAY excluded from recall (the review's core
# ask); reset/restart predecessors (empty child) and compression ancestors
# (summary, not transcript) must surface. The marker — not the parent link,
# not the end write — is the boundary in both directions.
# =========================================================================

def _seed_branch_copy(db, parent_id, child_id, needle, *, end_parent=None):
    """Seed a /branch in production shape: the child row's INSERT stamps the
    ``_branched_from`` marker atomically (gateway/slash_commands_session.py,
    tui_gateway/methods_session.py, hermes_cli/cli_commands_mixin.py) and the
    transcript is then copied verbatim; the gateway /branch path does NOT end
    the parent row. Returns the parent's needle message id."""
    db.create_session(parent_id, source="cli")
    parent_mid = db.append_message(parent_id, role="user", content=needle)
    if end_parent:
        db.end_session(parent_id, end_parent)
    db.create_session(child_id, source="cli", parent_session_id=parent_id,
                      model_config={"_branched_from": parent_id})
    db.append_message(child_id, role="user", content=needle)  # verbatim copy
    db._conn.commit()
    return parent_mid


class TestBranchCopyExclusion:
    """Negative controls: predecessors whose transcripts were verbatim-copied into
    the current session stay excluded — restoring the semantic distinction the
    pre-PR tests pinned (this flip-back is the review's purpose, not a regression
    of the restart-chain fix below)."""

    def test_branched_parent_still_excluded(self, db):
        _seed_branch_copy(db, "s_p", "s_q", "zephyr crystal cache design")
        result = json.loads(session_search(
            query="zephyr crystal", db=db, current_session_id="s_q",
        ))
        sids = [r["session_id"] for r in result.get("results", [])]
        assert "s_p" not in sids
        assert result["count"] == 0

    def test_branched_parent_still_excluded_after_re_end(self, db):
        """A later end write on the parent (tui_shutdown and friends overwrite
        end_reason — #20856) must not turn the copied transcript back into
        recall; the marker is the boundary, not the end_reason."""
        _seed_branch_copy(db, "s_p", "s_q", "zephyr crystal cache design")
        db.end_session("s_p", "session_reset")
        result = json.loads(session_search(
            query="zephyr crystal", db=db, current_session_id="s_q",
        ))
        assert result["count"] == 0

    def test_legacy_api_fork_parent_still_excluded(self, db):
        """api_server fork: no marker, parent ended 'branched' before the child row
        (the legacy arm). Seed the child's started_at from the parent's actual
        ended_at + 1 — end_session stamps wall-clock now, so a fake small timestamp
        would make the ordering arm false-negative."""
        db.create_session("s_p", source="api_server")
        db.append_message("s_p", role="user", content="zephyr crystal cache design")
        db.end_session("s_p", "branched")
        ended_at = db.get_session("s_p")["ended_at"]
        db.create_session("s_q", source="api_server", parent_session_id="s_p")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?", (ended_at + 1, "s_q"),
        )
        db.append_message("s_q", role="user", content="zephyr crystal cache design")
        db._conn.commit()
        result = json.loads(session_search(
            query="zephyr crystal", db=db, current_session_id="s_q",
        ))
        assert result["count"] == 0

    def test_current_sessions_branch_descendant_excluded(self, db):
        """Downward closure: the branch child of the CURRENT session holds a copy of
        the caller's own transcript (ancestor-prefix copy), so it is a duplicate
        source, not new recall."""
        db.create_session("s_cur", source="cli")
        db.append_message("s_cur", role="user", content="zephyr crystal cache design")
        db.create_session("s_bc", source="cli", parent_session_id="s_cur",
                          model_config={"_branched_from": "s_cur"})
        db.append_message("s_bc", role="user", content="zephyr crystal cache design")
        db._conn.commit()
        result = json.loads(session_search(
            query="zephyr crystal", db=db, current_session_id="s_cur",
        ))
        assert result["count"] == 0

    def test_branch_of_branch_excludes_both_ancestors(self, db):
        db.create_session("s_a", source="cli")
        db.append_message("s_a", role="user", content="aurora spindle calibration")
        db.create_session("s_b", source="cli", parent_session_id="s_a",
                          model_config={"_branched_from": "s_a"})
        db.append_message("s_b", role="user", content="aurora spindle calibration")
        db.append_message("s_b", role="user", content="aurora beacon alignment")
        db.create_session("s_c", source="cli", parent_session_id="s_b",
                          model_config={"_branched_from": "s_b"})
        db.append_message("s_c", role="user", content="aurora spindle calibration")
        db._conn.commit()
        result = json.loads(session_search(
            query="aurora", db=db, current_session_id="s_c",
        ))
        assert [r["session_id"] for r in result.get("results", [])] == []

    def test_scroll_into_branched_parent_anchor_is_rejected(self, db):
        """Scroll duality: the branch parent's transcript is already in the child's
        active context, so scrolling its anchor must be rejected — while reset
        predecessors stay scrollable (test_unended_restart_predecessor_hit_is_scrollable)."""
        parent_mid = _seed_branch_copy(db, "s_p", "s_q", "zephyr crystal cache design")
        result = json.loads(session_search(
            session_id="s_p", around_message_id=parent_mid, db=db,
            current_session_id="s_q",
        ))
        assert result["success"] is False
        assert "current session" in result.get("error", "").lower()

    def test_branch_parent_title_suppressed_from_child_side(self, db):
        db.create_session("s_p", source="cli")
        db.append_message("s_p", role="user", content="plain work note")
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?",
                         ("Zephyr Branch Origin Story", "s_p"))
        db.create_session("s_q", source="cli", parent_session_id="s_p",
                          model_config={"_branched_from": "s_p"})
        db._conn.commit()
        from_child = json.loads(session_search(
            query="Zephyr Branch Origin Story", db=db, current_session_id="s_q",
        ))
        assert from_child["count"] == 0
        # From outside the lineage the same title must still resolve.
        from_outside = json.loads(session_search(
            query="Zephyr Branch Origin Story", db=db,
        ))
        assert [r["session_id"] for r in from_outside["results"]] == ["s_p"]


class TestContinuationProjectionDiscovery:
    """Positive controls: only branch-copy edges keep a predecessor inside the
    caller's live context; every continuation that does NOT carry the transcript
    (reset, restart, compression) surfaces."""

    def test_compression_continuation_with_inherited_marker_surfaces_parent(self, db):
        """s_a --branch--> s_b --compression--> s_c. The compression continuation
        inherits ``_branched_from=s_a`` verbatim while its parent is s_b: binding
        the marker BY VALUE must not misread s_c as a branch child of s_b."""
        db.create_session("s_a", source="cli")
        db.append_message("s_a", role="user", content="velvet compass assembly")
        db.create_session("s_b", source="cli", parent_session_id="s_a",
                          model_config={"_branched_from": "s_a"})
        db.append_message("s_b", role="user", content="velvet compass assembly")
        db.append_message("s_b", role="user", content="harbor winch calibration")
        db.end_session("s_b", "compression")
        db.create_session("s_c", source="cli", parent_session_id="s_b",
                          model_config={"_branched_from": "s_a"})
        db.append_message("s_c", role="user", content="continue after summary")
        db._conn.commit()
        # s_b's own post-branch message crossed the edge only as a summary — it must
        # be reachable from the continuation (existence-only marker matching would
        # misread s_c as s_b's branch child and hide it).
        from_sc = json.loads(session_search(
            query="harbor winch", db=db, current_session_id="s_c",
        ))
        assert [r["session_id"] for r in from_sc["results"]] == ["s_b"]
        # From s_b itself the branch edge to s_a still binds: s_a and s_b's own copy
        # of it stay hidden.
        from_sb = json.loads(session_search(
            query="velvet compass", db=db, current_session_id="s_b",
        ))
        assert from_sb["count"] == 0

    def test_reset_edge_cuts_the_projection(self, db):
        """s_a --branch--> s_b, then a fresh reset child s_r of s_b: the branch
        chain is reachable from s_r (the reset child carries no transcript) while
        staying hidden from s_b (the branch child carries a copy)."""
        db.create_session("s_a", source="cli")
        db.append_message("s_a", role="user", content="solstice ledger audit")
        db.create_session("s_b", source="cli", parent_session_id="s_a",
                          model_config={"_branched_from": "s_a"})
        db.append_message("s_b", role="user", content="solstice ledger audit")
        db.append_message("s_b", role="user", content="harbor winch calibration")
        db.create_session("s_r", source="cli", parent_session_id="s_b",
                          model_config={"_reset_from": "s_b"})
        db._conn.commit()
        from_sr = json.loads(session_search(
            query="solstice ledger", db=db, current_session_id="s_r",
        ))
        assert from_sr["count"] == 1
        assert from_sr["results"][0]["session_id"] in {"s_a", "s_b"}
        from_sb = json.loads(session_search(
            query="solstice ledger", db=db, current_session_id="s_b",
        ))
        assert from_sb["count"] == 0

    def test_branch_sibling_under_non_projected_ancestor_surfaces(self, db):
        """The projection stops at the reset edge below s_p, so s_p AND its branch
        child (an alternate future of s_p, not of the caller) are both reachable
        from the reset child — and stay hidden from the branch sibling itself."""
        db.create_session("s_p", source="cli")
        db.append_message("s_p", role="user", content="solstice ledger audit")
        db.create_session("s_sib", source="cli", parent_session_id="s_p",
                          model_config={"_branched_from": "s_p"})
        db.append_message("s_sib", role="user", content="solstice ledger audit")
        db.create_session("s_r", source="cli", parent_session_id="s_p",
                          model_config={"_reset_from": "s_p"})
        db._conn.commit()
        from_sr = json.loads(session_search(
            query="solstice ledger", db=db, current_session_id="s_r",
        ))
        assert from_sr["count"] == 1
        assert from_sr["results"][0]["session_id"] in {"s_p", "s_sib"}
        from_sib = json.loads(session_search(
            query="solstice ledger", db=db, current_session_id="s_sib",
        ))
        assert from_sib["count"] == 0

    def test_delegate_marker_hides_cli_sourced_delegate_child_from_discovery(self, db):
        """run_agent.py can stamp delegate runs source='cli' (fallback chain), so the
        ``_delegate_from`` marker — not the source filter — hides the live
        delegation run from discovery."""
        db.create_session("s_current", source="cli")
        db.create_session(
            "s_delegate", source="cli", parent_session_id="s_current",
            model_config={"_delegate_from": "s_current"},
        )
        db.append_message(
            "s_delegate", role="assistant", content="quasar delegation trace results",
        )
        db._conn.commit()
        result = json.loads(session_search(
            query="quasar delegation", db=db, current_session_id="s_current",
        ))
        assert result["count"] == 0


class TestBranchCopyEdge:
    """Unit matrix for the pure edge classifier behind the live-context projection."""

    def test_marker_bound_by_value_to_the_parent(self):
        assert _branch_copy_edge(
            {"model_config": {"_branched_from": "p1"}}, {"id": "p1"}) is True

    def test_marker_survives_json_text_storage(self):
        # sessions.model_config is stored as JSON text; get_session returns it unparsed.
        assert _branch_copy_edge(
            {"model_config": '{"_branched_from": "p1"}'}, {"id": "p1"}) is True

    def test_inherited_marker_names_the_grandparent_not_this_edge(self):
        # compression copies model_config onto the continuation row: the inherited
        # value names s_a while the parent here is s_b — not a branch copy.
        assert _branch_copy_edge(
            {"model_config": {"_branched_from": "s_a"}, "started_at": 200.0},
            {"id": "s_b"}) is False

    def test_marker_pointing_at_an_unrelated_id(self):
        assert _branch_copy_edge(
            {"model_config": {"_branched_from": "someone-else"}}, {"id": "p1"}) is False

    def test_legacy_branched_end_with_later_child_start(self):
        # api_server fork shape: parent ended 'branched', child started afterwards.
        assert _branch_copy_edge(
            {"started_at": 101.0},
            {"id": "p1", "end_reason": "branched", "ended_at": 100.0}) is True

    def test_legacy_arm_requires_an_ended_at(self):
        assert _branch_copy_edge(
            {"started_at": 101.0},
            {"id": "p1", "end_reason": "branched", "ended_at": None}) is False

    def test_legacy_arm_requires_child_started_after_parent_ended(self):
        assert _branch_copy_edge(
            {"started_at": 99.0},
            {"id": "p1", "end_reason": "branched", "ended_at": 100.0}) is False

    def test_reset_marker_is_not_a_copy_edge(self):
        assert _branch_copy_edge(
            {"model_config": {"_reset_from": "p1"}, "started_at": 101.0},
            {"id": "p1", "end_reason": "session_reset", "ended_at": 100.0}) is False

    def test_delegate_marker_is_not_a_copy_edge(self):
        assert _branch_copy_edge(
            {"model_config": {"_delegate_from": "p1"}, "started_at": 101.0},
            {"id": "p1"}) is False

    def test_bare_parent_link_is_not_a_copy_edge(self):
        assert _branch_copy_edge({"started_at": 101.0}, {"id": "p1"}) is False

    def test_compression_ended_parent_is_not_a_copy_edge(self):
        assert _branch_copy_edge(
            {"started_at": 101.0},
            {"id": "p1", "end_reason": "compression", "ended_at": 100.0}) is False

