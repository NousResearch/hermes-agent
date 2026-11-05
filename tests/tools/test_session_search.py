"""Tests for the single-shape session_search tool.

Four calling shapes:
  1. DISCOVERY — pass query → FTS5 + adaptive/full hydration
  2. SCROLL    — pass session_id + around_message_id → just the window
  3. READ      — pass session_id → whole or head/tail-truncated session
  4. BROWSE    — no args → recent sessions chronologically

All run zero LLM calls.
"""
import json
import time
from datetime import datetime, timezone, UTC
from types import SimpleNamespace

import pytest

from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tools.session_search_tool import (
    _format_timestamp,
    _is_compacted_message,
    _resolve_to_parent,
    _session_link,
    session_search,
)


@pytest.fixture
def db(tmp_path):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as handle:
        yield handle


@pytest.fixture
def persistence(tmp_path):
    """A read-only storage interface handle on the ``db`` fixture's store."""
    from persistence_assembly import open_persistence_reader

    with open_persistence_reader(tmp_path) as handle:
        yield handle


def _seed_modpack_sessions(db):
    """Create three sessions about a modpack so FTS5 has hits to dedupe."""
    now = int(time.time())
    # Older session — modpack origin
    seed_session(db, "s_oldest", "cli")
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 30000, "Building the Modpack", "s_oldest"),
    ))
    seed_messages(db, "s_oldest", {"role": "user", "content": "Let's build a Minecraft modpack"})
    seed_messages(db, "s_oldest", {"role": "assistant", "content": "Great. Let me scaffold the modpack repo."})
    seed_messages(db, "s_oldest", {"role": "user", "content": "Use NeoForge 1.21.1"})
    seed_messages(db, "s_oldest", {"role": "assistant", "content": "Done. Modpack repo created with NeoForge 1.21.1."})
    seed_messages(db, "s_oldest", {"role": "assistant", "content": "Tier-0 mods installed; modpack smoke test passes."})

    # Middle session — modpack quest coverage
    seed_session(db, "s_middle", "cli")
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 15000, "Modpack Quest Coverage", "s_middle"),
    ))
    seed_messages(db, "s_middle", {"role": "user", "content": "Deep-dive every modpack reference quest guide"})
    seed_messages(db, "s_middle", {
        "role": "assistant",
        "content": "Surveying ATM10 questbook for modpack inspiration.",
    })
    seed_messages(db, "s_middle", {"role": "user", "content": "Update the modpack version too"})
    seed_messages(db, "s_middle", {
        "role": "assistant",
        "content": "Modpack version bumped 0.4 → 0.8.5; quest coverage page added.",
    })

    # Newest session — modpack mob spawn fix
    seed_session(db, "s_newest", "cli")
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 1000, "Modpack Mob Spawn Fix", "s_newest"),
    ))
    seed_messages(db, "s_newest", {"role": "user", "content": "Fix the modpack mob spawning"})
    seed_messages(db, "s_newest", {
        "role": "assistant",
        "content": "Investigating elite mob gating in the modpack KubeJS.",
    })
    seed_messages(db, "s_newest", {
        "role": "assistant",
        "content": "Shipped commit b850442. Modpack alternator nerfed too.",
    })


# =========================================================================
# Schema invariants
# =========================================================================


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
        class _Sessions:
            rich_called = False
            bounded_query = None

            def list_recent_sessions(self, query):
                self.bounded_query = query
                return ()

            def list_sessions(self, *_args, **_kwargs):
                self.rich_called = True
                raise AssertionError("unbounded rich listing must not be used")

        sessions = _Sessions()
        result = json.loads(session_search(persistence=SimpleNamespace(sessions=sessions)))

        assert result["success"] is True
        assert sessions.rich_called is False
        assert sessions.bounded_query is not None

    def test_browse_fails_closed_without_bounded_database_capability(self):
        class _LegacySessions:
            def list_sessions(self, *_args, **_kwargs):
                raise AssertionError("known-unbounded fallback must not be called")

        result = json.loads(session_search(persistence=SimpleNamespace(sessions=_LegacySessions())))

        assert result["success"] is False

    def test_lazy_database_is_released_after_search(self, monkeypatch):
        from contextlib import contextmanager

        class _Reader:
            released = 0
            sessions = SimpleNamespace(list_recent_sessions=lambda _query: ())

            def ready(self):
                pass

        reader = _Reader()

        @contextmanager
        def _open_reader(_target):
            try:
                yield reader
            finally:
                reader.released += 1

        monkeypatch.setattr("persistence_assembly.open_persistence_reader", _open_reader)

        result = json.loads(session_search())

        assert result["success"] is True
        assert reader.released == 1

    def test_cross_profile_database_is_closed_but_shared_database_is_not(
        self, monkeypatch
    ):
        class _Handle:
            def __init__(self):
                self.closed = 0
                self.sessions = SimpleNamespace(list_recent_sessions=lambda _query: ())

            def close(self):
                self.closed += 1

        shared = _Handle()
        profile_handle = _Handle()

        def _resolve(_profile, handles):
            handles.callback(profile_handle.close)
            return profile_handle

        monkeypatch.setattr("tools.session_search_tool._resolve_profile_db", _resolve)

        result = json.loads(session_search(persistence=shared, profile="work"))

        assert result["success"] is True
        assert profile_handle.closed == 1
        assert shared.closed == 0

    def test_no_args_returns_recent_sessions(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(db=db, persistence=persistence))
        assert result["success"] is True
        assert result["mode"] == "browse"
        assert result["count"] >= 3

    def test_browse_excludes_current_session(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(db=db, persistence=persistence, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids


# =========================================================================
# Discovery shape (with query)
# =========================================================================

class TestDiscoveryShape:

    def test_full_detail_returns_bookends_and_window_for_every_hit(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(
            query="modpack", limit=3, detail="full", db=db, persistence=persistence
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

    def test_default_discovery_keeps_top_full_and_compacts_lower_hits(self, db, persistence):
        _seed_modpack_sessions(db)

        result = json.loads(session_search(query="modpack", limit=3, db=db, persistence=persistence))

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

    def test_adaptive_detail_preserves_ranking_and_reduces_payload(self, db, persistence):
        now = int(time.time())
        for session_index in range(3):
            session_id = f"payload_{session_index}"
            seed_session(db, session_id, "cli")
            raw_holder(db)._execute_write(lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (now - session_index, session_id),
            ))
            for message_index in range(8):
                seed_messages(db, session_id, {
                    "role": "user" if message_index % 2 == 0 else "assistant",
                    "content": f"opening {session_index}-{message_index} " + "o" * 2500,
                })
            seed_messages(db, session_id, {
                "role": "user",
                "content": f"payloadneedle anchor {session_index} " + "a" * 3500,
            })
            for message_index in range(8):
                seed_messages(db, session_id, {
                    "role": "assistant" if message_index % 2 == 0 else "user",
                    "content": f"closing {session_index}-{message_index} " + "c" * 2500,
                })

        adaptive_json = session_search(query="payloadneedle", limit=3, db=db, persistence=persistence)
        full_json = session_search(
            query="payloadneedle", limit=3, detail="full", db=db, persistence=persistence
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


    def test_current_session_filtered_out(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", db=db, persistence=persistence, current_session_id="s_newest"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_newest" not in sids


class TestDiscoverySort:
    def test_sort_newest_orders_by_recency(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, sort="newest", db=db, persistence=persistence))
        # First result should be the most recent session
        first = result["results"][0]
        assert first["session_id"] == "s_newest" or "Newest" in (first.get("title") or "")

    def test_sort_oldest_orders_by_age(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=3, sort="oldest", db=db, persistence=persistence))
        first = result["results"][0]
        assert first["session_id"] == "s_oldest"


# =========================================================================
# Scroll shape (session_id + around_message_id)
# =========================================================================

class TestScrollShape:
    def test_scroll_returns_anchored_window_without_bookends(self, db, persistence):
        _seed_modpack_sessions(db)
        # Get an anchor first via discovery
        disc = json.loads(session_search(query="modpack", limit=1, db=db, persistence=persistence))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]

        # Now scroll
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=2, db=db, persistence=persistence
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

    def test_scroll_window_clamped_to_20(self, db, persistence):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db, persistence=persistence))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        result = json.loads(session_search(
            session_id=anchor_sid, around_message_id=anchor_mid, window=999, db=db, persistence=persistence
        ))
        assert result["window"] == 20


    def test_scroll_rejects_active_delegation_child_in_current_lineage(self, db, persistence):
        seed_session(db, "s_current", "cli")
        seed_session(
            db,
            "s_delegate",
            "delegate",
            parent_session_id="s_current",
        )
        mid = seed_messages(db, "s_delegate", {
            "role": "assistant",
            "content": "live delegated result",
        })[0]

        result = json.loads(session_search(
            session_id="s_delegate", around_message_id=mid, db=db, persistence=persistence,
            current_session_id="s_current",
        ))

        assert result["success"] is False
        assert "current session" in result.get("error", "").lower()


class TestScrollPattern:
    """The forward/backward scroll loop using tool output."""

    def test_scroll_forward_from_last_id(self, db, persistence):
        # Long session
        seed_session(db, "s_long", "cli")
        ids = []
        for i in range(20):
            ids.append(seed_messages(db, "s_long", {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"long session msg {i}",
            })[0])

        v1 = json.loads(session_search(
            session_id="s_long", around_message_id=ids[5], window=3, db=db, persistence=persistence
        ))
        last_id = v1["messages"][-1]["id"]
        v2 = json.loads(session_search(
            session_id="s_long", around_message_id=last_id, window=3, db=db, persistence=persistence
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
    def test_scroll_args_beat_query(self, db, persistence):
        _seed_modpack_sessions(db)
        disc = json.loads(session_search(query="modpack", limit=1, db=db, persistence=persistence))
        anchor_sid = disc["results"][0]["session_id"]
        anchor_mid = disc["results"][0]["match_message_id"]
        # Pass both query and scroll args — scroll should win
        result = json.loads(session_search(
            query="modpack",  # would normally trigger discovery
            session_id=anchor_sid, around_message_id=anchor_mid, db=db, persistence=persistence,
        ))
        assert result["mode"] == "scroll"


    def test_session_id_without_anchor_reads(self, db, persistence):
        _seed_modpack_sessions(db)
        # session_id alone (no anchor, no query) → read shape, not browse.
        result = json.loads(session_search(session_id="s_oldest", db=db, persistence=persistence))
        assert result["mode"] == "read"


# =========================================================================
# Read shape — dump a whole session by id (serves @session links)
# =========================================================================

class TestReadShape:
    def test_read_returns_full_session(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(session_id="s_oldest", db=db, persistence=persistence))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["session_id"] == "s_oldest"
        assert result["message_count"] == 5
        assert result["truncated"] is False
        assert len(result["messages"]) == 5
        assert result["session_meta"]["title"] == "Building the Modpack"

    def test_read_strips_ansi_sequences_from_messages(self, db, persistence):
        seed_session(db, "s_ansi", "cli")
        seed_messages(db, "s_ansi", {"role": "user", "content": "plain"})
        seed_messages(db, "s_ansi", {
            "role": "assistant",
            "content": "\u001b[31mred text\u001b[0m and more",
        })
        result = json.loads(session_search(session_id="s_ansi", db=db, persistence=persistence))
        assert result["success"] is True
        rendered = [m["content"] for m in result["messages"] if m.get("content")]
        assert any(text == "red text and more" for text in rendered)
        assert all("\u001b" not in text for text in rendered)

    def test_read_truncates_large_session(self, db, persistence):
        seed_session(db, "s_big", "cli")
        for i in range(50):
            seed_messages(db, "s_big", {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
        result = json.loads(session_search(session_id="s_big", db=db, persistence=persistence))
        assert result["mode"] == "read"
        assert result["message_count"] == 50
        assert result["truncated"] is True
        assert len(result["messages"]) < 50

    def test_read_caps_oversized_message_content(self, db, persistence):
        # #114344: a huge archived tool result stored as a message must not come
        # back whole on the read shape - discovery/scroll already cap (#69334).
        seed_session(db, "s_huge", "cli")
        seed_messages(db, "s_huge", {"role": "user", "content": "run it"})
        seed_messages(db, "s_huge", {"role": "assistant", "content": "x" * 80_000})
        seed_messages(db, "s_huge", {"role": "user", "content": "thanks"})
        result = json.loads(session_search(session_id="s_huge", db=db, persistence=persistence))
        assert result["mode"] == "read"
        assert result["truncated"] is False  # 3 messages, count-wise it all fits
        big = next(m for m in result["messages"] if m.get("content_truncated"))
        assert len(big["content"]) <= 2001  # read cap (2000) plus ellipsis
        assert big["original_content_chars"] == 80_000
        assert sum(len(m.get("content") or "") for m in result["messages"]) < 5_000

    def test_title_match_entry_caps_content_like_fts_hits(self, db, persistence):
        """A session-title match is a discovery entry: bookends 1200, window 4000, same as FTS hits."""
        seed_session(db, "s_titled", "cli")
        db.sessions.set_title("s_titled", "quasar ledger reconciliation")
        seed_messages(db, "s_titled", {"role": "user", "content": "reconcile the quasar ledger"})
        seed_messages(db, "s_titled", {"role": "tool", "content": "y" * 80_000})
        db.sessions.end("s_titled", "cli_exit")
        result = json.loads(session_search(query="quasar ledger reconciliation", db=db, persistence=persistence, detail="full"))
        entry = next(r for r in result["results"] if r["matched_role"] == "session_title")
        shaped = entry["bookend_start"] + entry["messages"] + entry["bookend_end"]
        big = [m for m in shaped if m.get("original_content_chars") == 80_000]
        assert big and all(m["content_truncated"] and len(m["content"]) <= 4001 for m in big)


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


    def test_every_discovery_result_links_to_its_own_session(self, db, persistence):
        _seed_modpack_sessions(db)
        result = json.loads(session_search(query="modpack", limit=5, db=db, persistence=persistence))

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

    def test_bare_id_never_reads_another_profiles_store(self, db, persistence, tmp_path, monkeypatch):
        # #106761: profiles are isolated islands. A bare id that misses the caller's
        # store must NOT be located by scanning every other profile's state.db.
        other_home = tmp_path / "asdf_home"
        other_home.mkdir()
        other = storage_handle(other_home)
        seed_session(other, "s_far", "cli")
        seed_messages(other, "s_far", {"role": "user", "content": "secret"})

        from collections import namedtuple
        from hermes_cli import profiles as profiles_mod
        Info = namedtuple("Info", "name path")
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: tmp_path / "default_home")
        monkeypatch.setattr(profiles_mod, "list_profiles", lambda: [Info("asdf", other_home)])

        result = json.loads(session_search(session_id="s_far", db=db, persistence=persistence))
        assert result["success"] is False
        assert "messages" not in result and "secret" not in json.dumps(result)

        # Naming the owning profile is still the sanctioned cross-profile read.
        self._patch_profiles(monkeypatch, other_home)
        named = json.loads(session_search(session_id="s_far", profile="asdf", db=db, persistence=persistence))
        assert named["success"] is True and named["message_count"] == 1

    def test_combined_value_autosplits(self, db, persistence, tmp_path, monkeypatch):
        # Agent passed the raw "@session:<profile>/<id>" value as session_id with
        # no separate profile — the tool should recover both.
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other = storage_handle(other_home)
        seed_session(other, "s_other", "cli")
        seed_messages(other, "s_other", {"role": "user", "content": "hi"})

        self._patch_profiles(monkeypatch, other_home)

        # Every permutation the model might send must resolve to (asdf, s_other).
        for kwargs in (
            {"session_id": "asdf/s_other"},                    # full value, no profile
            {"session_id": "asdf/s_other", "profile": "asdf"},  # full value AND profile
            {"session_id": "s_other", "profile": "asdf"},       # bare id + profile
        ):
            result = json.loads(session_search(db=db, persistence=persistence, **kwargs))
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
        seed_session(db, "s_user", "telegram")
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (now - 90000, "s_user"),
        ))
        seed_messages(db, "s_user", {"role": "user", "content": "how is the venom project going"})
        seed_messages(db, "s_user", {"role": "assistant", "content": "The venom project shipped its first milestone."})
        # Several cron sessions, all newer and all stuffed with the same terms.
        for i in range(8):
            sid = f"cron_{i}"
            seed_session(db, sid, "cron")
            raw_holder(db)._execute_write(lambda conn: conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (now - 1000 - i, sid),
            ))
            seed_messages(db, sid, {"role": "user", "content": "venom project daily status"})
            seed_messages(db, sid, {"role": "assistant", "content": "venom project venom project venom summary"})

    def test_interactive_session_surfaces_above_cron(self, db, persistence):
        self._seed_cron_and_interactive(db)
        result = json.loads(session_search(query="venom project", limit=1, db=db, persistence=persistence))
        assert result["success"] is True
        assert result["count"] == 1
        # With cron drowning FTS, bare BM25/recency would return a cron_* hit.
        # Demotion must put the user's interactive session first.
        assert result["results"][0]["source"] == "telegram"
        assert result["results"][0]["session_id"] == "s_user"

    def test_cron_still_reachable_when_only_match(self, db, persistence):
        """Demotion must not exclude cron — when only cron matches, it still
        comes back."""
        now = int(time.time())
        seed_session(db, "cron_only", "cron")
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (now - 500, "cron_only"),
        ))
        seed_messages(db, "cron_only", {"role": "user", "content": "quarterly archive sweep"})
        seed_messages(db, "cron_only", {"role": "assistant", "content": "Archive sweep complete."})
        result = json.loads(session_search(query="archive sweep", db=db, persistence=persistence))
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

    def test_compaction_summary_excluded_from_bookend_start(self, db, persistence):
        """Compaction handoff in bookend_start position must be filtered out."""
        seed_session(db, "s_compact", "cli")
        # First message: a compaction handoff (should be filtered)
        seed_messages(db, "s_compact", {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY] "
            "Earlier turns were compacted into the summary below. " + "x" * 50000,
        })
        # Second message: normal user message
        seed_messages(db, "s_compact", {"role": "user", "content": "Fix the zorgblat rendering bug"})
        # Padding messages to push window away from session start (so bookend has room)
        for i in range(10):
            seed_messages(db, "s_compact", {"role": "user", "content": f"setup step {i}"})
            seed_messages(db, "s_compact", {"role": "assistant", "content": f"setup done {i}"})
        # Match target: uses a unique term so FTS5 anchors here, not at the start
        seed_messages(db, "s_compact", {"role": "user", "content": "investigate the frobnitz mob spawning in KubeJS"})
        seed_messages(db, "s_compact", {
            "role": "assistant",
            "content": "I'll look into the frobnitz mob spawning issue.",
        })
        # Tail messages
        for i in range(5):
            seed_messages(db, "s_compact", {"role": "user", "content": f"tail {i}"})
            seed_messages(db, "s_compact", {"role": "assistant", "content": f"done tail {i}"})

        result = json.loads(session_search(query="frobnitz mob spawning", db=db, persistence=persistence, limit=1))
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
# lineage, creating a "memory black hole". Delegation children must STAY
# excluded — their content is still visible to the parent agent.
# =========================================================================

class TestResolveToParent:
    """Unit tests for _resolve_to_parent's compression-aware tuple return."""

    def test_legacy_rotation_detects_compression(self, db, persistence):
        """Parent ended with end_reason='compression', child has parent_session_id."""
        seed_session(db, "s_parent", "cli")
        db.sessions.end("s_parent", "compression")
        seed_session(db, "s_child", "cli", parent_session_id="s_parent")
        root, has_compression = _resolve_to_parent(persistence, "s_child")
        assert root == "s_parent"
        assert has_compression is True


    def test_chain_with_mixed_edges(self, db, persistence):
        """Compression grandparent → parent → child (no end_reason on parent)."""
        seed_session(db, "s_gp", "cli")
        db.sessions.end("s_gp", "compression")
        seed_session(db, "s_p", "cli", parent_session_id="s_gp")
        # s_p does NOT end with compression — but ancestor s_gp does
        seed_session(db, "s_c", "cli", parent_session_id="s_p")
        root, has_compression = _resolve_to_parent(persistence, "s_c")
        assert root == "s_gp"
        assert has_compression is True


class TestIsCompactedMessage:
    """Unit tests for the _is_compacted_message helper."""

    def test_active_message_returns_false(self, db, persistence):
        seed_session(db, "s1", "cli")
        mid = seed_messages(db, "s1", {"role": "user", "content": "hello"})[0]
        assert _is_compacted_message(persistence, mid) is False

    def test_compacted_message_returns_true(self, db, persistence):
        seed_session(db, "s1", "cli")
        mid = seed_messages(db, "s1", {"role": "user", "content": "archived content"})[0]
        db.messages.compact("s1", [NewMessage.from_live(m) for m in [
            {"role": "assistant", "content": "compacted summary"},
        ]])
        # mid is now active=0, compacted=1
        assert _is_compacted_message(persistence, mid) is True


class TestInPlaceCompactionDiscovery:
    """In-place compaction: archived turns on the SAME session_id must be
    discoverable from the current session."""

    def test_archived_content_discoverable_after_compaction(self, db, persistence):
        """The core regression: pre-compaction content on the current session
        must surface in discovery even though raw_sid == current_session_id."""
        seed_session(db, "s_compact", "cli")
        seed_messages(db, "s_compact", {
            "role": "user",
            "content": "The spectral phoenix only spawns during full moons",
        })
        seed_messages(db, "s_compact", {
            "role": "assistant",
            "content": "Spectral phoenix requires moonstone bait",
        })
        db.messages.compact("s_compact", [NewMessage.from_live(m) for m in [
            {"role": "user", "content": "Summary: spectral phoenix discussed"},
            {"role": "assistant", "content": "Acknowledged spectral phoenix info"},
        ]])

        result = json.loads(session_search(
            query="spectral phoenix", db=db, persistence=persistence, current_session_id="s_compact",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        # The hit should be from the same session (archived rows)
        hit = result["results"][0]
        assert hit["session_id"] == "s_compact"

    def test_live_content_still_filtered_on_current_session(self, db, persistence):
        """Non-compacted (active) content on the current session stays filtered."""
        seed_session(db, "s_live", "cli")
        seed_messages(db, "s_live", {"role": "user", "content": "crystal golem farming route"})
        result = json.loads(session_search(
            query="crystal golem", db=db, persistence=persistence, current_session_id="s_live",
        ))
        assert result["count"] == 0


class TestLegacyRotationDiscovery:
    """Legacy rotation: parent session ended with end_reason='compression',
    child session created. Parent's pre-compaction content must be discoverable
    from the child."""

    def test_compression_parent_discoverable_from_child(self, db, persistence):
        seed_session(db, "s_parent", "cli")
        seed_messages(db, "s_parent", {
            "role": "user",
            "content": "The void crystal mining requires diamond pickaxe",
        })
        seed_messages(db, "s_parent", {
            "role": "assistant",
            "content": "Void crystal found in the deep caverns",
        })
        db.sessions.end("s_parent", "compression")

        seed_session(db, "s_child", "cli", parent_session_id="s_parent")
        seed_messages(db, "s_child", {"role": "user", "content": "Continue void crystal work"})

        result = json.loads(session_search(
            query="void crystal", db=db, persistence=persistence, current_session_id="s_child",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_parent" in sids


class TestDelegationExclusion:
    """Delegation children (delegate_task) must STAY excluded — their content
    is still visible to the parent agent. parent_session_id is set but the
    parent does NOT have end_reason='compression'."""

    def test_delegation_parent_excluded_from_child(self, db, persistence):
        """Child can see its own content but parent's live content stays
        excluded (it's in context via delegation)."""
        seed_session(db, "s_parent", "cli")
        seed_messages(db, "s_parent", {
            "role": "user",
            "content": "nebula deployment infrastructure setup",
        })
        seed_messages(db, "s_parent", {
            "role": "assistant",
            "content": "Nebula deployment configured successfully",
        })

        seed_session(db, "s_child", "cli", parent_session_id="s_parent")
        seed_messages(db, "s_child", {
            "role": "user",
            "content": "delegated nebula deployment subtask",
        })

        result = json.loads(session_search(
            query="nebula deployment", db=db, persistence=persistence, current_session_id="s_child",
        ))
        assert result["count"] == 0


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
        seed_session(db, "s_both", "cli")
        # Long normal opening — exercises the 1200-char bookend cap.
        seed_messages(db, "s_both", {
            "role": "user",
            "content": "Kick off the obsidian gateway migration. " + "o" * 5000,
        })
        seed_messages(db, "s_both", {
            "role": "assistant",
            "content": "Starting the obsidian gateway migration plan.",
        })
        # Padding so the anchored window doesn't swallow the bookends.
        for i in range(10):
            seed_messages(db, "s_both", {"role": "user", "content": f"migration step {i}"})
            seed_messages(db, "s_both", {"role": "assistant", "content": f"migration step {i} done"})
        # The FTS match target — will be archived by compaction below.
        seed_messages(db, "s_both", {
            "role": "user",
            "content": "the obsidian gateway needs a quartz keystone to activate",
        })
        seed_messages(db, "s_both", {
            "role": "assistant",
            "content": "Noted: quartz keystone required for the obsidian gateway.",
        })
        for i in range(5):
            seed_messages(db, "s_both", {"role": "user", "content": f"wrap-up {i}"})
            seed_messages(db, "s_both", {"role": "assistant", "content": f"wrapped {i}"})
        # Compact in place: everything above becomes active=0/compacted=1 and
        # the handoff summary is inserted as the new live tail.
        db.messages.compact("s_both", [NewMessage.from_live(m) for m in [
            {"role": "user",
             "content": "[CONTEXT COMPACTION — REFERENCE ONLY] "
                        "Earlier turns were compacted into this summary. " + "s" * 50000},
            {"role": "assistant", "content": "Continuing after compaction."},
        ]])

    def test_archived_hit_surfaces_with_bounded_summary_free_bookends(self, db, persistence):
        self._seed_compacted_session(db)

        result = json.loads(session_search(
            query="quartz keystone", db=db, persistence=persistence, current_session_id="s_both",
        ))

        # Layer 1 — discovery scope: the archived (active=0, compacted=1)
        # content on the CURRENT session must surface.
        assert result["success"] is True
        assert result["count"] >= 1
        entry = result["results"][0]
        assert entry["session_id"] == "s_both"

        # Layer 2a — summary exclusion: the compaction handoff row sits at the
        # session tail (freshly inserted by messages.compact), exactly where
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

    def test_compacted_messages_still_surface_alongside_rewind(self, db, persistence):
        """On the same session: compacted rows surface, rewind rows don't."""
        seed_session(db, "s_mixed", "cli")
        # Message that will be compacted
        seed_messages(db, "s_mixed", {
            "role": "user",
            "content": "compaction archived content beta",
        })
        db.messages.compact("s_mixed", [NewMessage.from_live(m) for m in [
            {"role": "assistant", "content": "Summary of beta"},
        ]])
        # Now add a post-compaction message and rewind it
        mid2 = seed_messages(db, "s_mixed", {
            "role": "user",
            "content": "rewound content gamma",
        })[0]
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
            (mid2,),
        ))

        # Compacted content should be discoverable
        result_compact = json.loads(session_search(
            query="compaction archived content beta", db=db, persistence=persistence,
            current_session_id="s_mixed",
        ))
        assert result_compact["count"] >= 1

        # Rewound content should NOT be discoverable. The OR-relaxed zero-result
        # retry may legitimately surface OTHER active rows sharing the common term
        # "content"; the invariant is that the rewound row itself never comes back.
        result_rewind = json.loads(session_search(
            query="rewound content gamma", db=db, persistence=persistence,
            current_session_id="s_mixed",
        ))
        for entry in result_rewind["results"]:
            assert entry["match_message_id"] != mid2
            assert "gamma" not in (entry.get("snippet") or "").lower()


class TestLegacyContinuationPlusDelegation:
    """Regression: a delegation child created under a compression continuation
    must stay excluded — its content is still live to the parent agent.
    Only the compression-ended ancestor's content should surface."""

    def test_compression_parent_surfaces_but_delegate_child_excluded(self, db, persistence):
        """Setup: grandparent (compression) → parent (compression) → child
        (active, current session). A delegation grandchild is created under
        the parent. Searching from the child should find grandparent/parent
        content but NOT the delegation grandchild's content."""
        # Grandparent: compression-ended, has searchable content
        seed_session(db, "s_gp", "cli")
        seed_messages(db, "s_gp", {
            "role": "user",
            "content": "grandparent cosmic anomaly research data",
        })
        db.sessions.end("s_gp", "compression")

        # Parent: compression-ended continuation
        seed_session(db, "s_p", "cli", parent_session_id="s_gp")
        seed_messages(db, "s_p", {
            "role": "user",
            "content": "parent cosmic anomaly follow-up notes",
        })
        db.sessions.end("s_p", "compression")

        # Current session: active child
        seed_session(db, "s_current", "cli", parent_session_id="s_p")

        # Delegation child under s_p (not compression-ended)
        seed_session(db, "s_delegate", "cli", parent_session_id="s_p")
        seed_messages(db, "s_delegate", {
            "role": "assistant",
            "content": "delegated cosmic anomaly subtask results",
        })

        result = json.loads(session_search(
            query="cosmic anomaly", db=db, persistence=persistence,
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
# Delegation children (live parent, no end_reason) must stay excluded.
# =========================================================================

def _seed_gateway_new_reset_chain(db, *, needle="ibuprofen night-dose protocol"):
    """A → B → C gateway /new chain. C is the empty current session."""
    seed_session(
        db,
        "s_aug12",
        "telegram",
        session_key="tg:user:1",
    )
    seed_messages(db, "s_aug12", {"role": "user", "content": "older unrelated chat"})
    db.sessions.end("s_aug12", "session_reset")

    seed_session(
        db,
        "s_night",
        "telegram",
        parent_session_id="s_aug12",
        session_key="tg:user:1",
        model_config={"_reset_from": "s_aug12"},
    )
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET title = ? WHERE id = ?",
        ("Night ibuprofen plan", "s_night"),
    ))
    seed_messages(db, "s_night", {"role": "user", "content": f"Remember the {needle}"})
    seed_messages(db, "s_night", {
        "role": "assistant",
        "content": f"Noted {needle} at 21:00",
    })
    db.sessions.end("s_night", "session_reset")

    seed_session(
        db,
        "s_today",
        "telegram",
        parent_session_id="s_night",
        session_key="tg:user:1",
        model_config={"_reset_from": "s_night"},
    )
    return needle


class TestNewResetLineageDiscovery:
    """After /new, yesterday's session must be searchable from the empty child."""

    def test_session_reset_parent_discoverable_from_child(self, db, persistence):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(
            query="ibuprofen", db=db, persistence=persistence, current_session_id="s_today",
        ))
        assert result["success"] is True
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_night" in sids
        blob = json.dumps(result["results"], ensure_ascii=False).lower()
        assert "ibuprofen" in blob

    def test_cli_new_session_parent_discoverable_from_child(self, db, persistence):
        seed_session(db, "s_cli_old", "cli")
        seed_messages(db, "s_cli_old", {
            "role": "user",
            "content": "quartz lantern wiring diagram from yesterday",
        })
        db.sessions.end("s_cli_old", "new_session")
        seed_session(
            db,
            "s_cli_new",
            "cli",
            parent_session_id="s_cli_old",
        )
        result = json.loads(session_search(
            query="quartz lantern", db=db, persistence=persistence, current_session_id="s_cli_new",
        ))
        assert result["count"] >= 1
        assert "s_cli_old" in [r["session_id"] for r in result["results"]]


    def test_branched_parent_still_excluded(self, db, persistence):
        """/branch verbatim-copies the transcript into the child, so the
        parent's content IS the branch child's live context — it must not
        surface as a same-lineage recall hit (unlike /new-reset parents)."""
        seed_session(db, "s_p", "cli")
        seed_messages(db, "s_p", {
            "role": "user",
            "content": "zephyr crystal cache design",
        })
        db.sessions.end("s_p", "branched")
        seed_session(
            db,
            "s_q",
            "cli",
            parent_session_id="s_p",
            model_config={"_branched_from": "s_p"},
        )
        # /branch copies history into the child
        seed_messages(db, "s_q", {
            "role": "user",
            "content": "zephyr crystal cache design",
        })
        result = json.loads(session_search(
            query="zephyr crystal", db=db, persistence=persistence, current_session_id="s_q",
        ))
        sids = [r["session_id"] for r in result.get("results", [])]
        assert "s_p" not in sids

    def test_title_match_reset_parent_not_dropped(self, db, persistence):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(
            query="Night ibuprofen plan", db=db, persistence=persistence, current_session_id="s_today",
        ))
        assert result["count"] >= 1
        sids = [r["session_id"] for r in result["results"]]
        assert "s_night" in sids

    def test_scroll_into_reset_parent_is_allowed(self, db, persistence):
        _seed_gateway_new_reset_chain(db)
        disc = json.loads(session_search(
            query="ibuprofen", db=db, persistence=persistence, current_session_id="s_today", limit=1,
        ))
        assert disc["count"] >= 1
        hit = disc["results"][0]
        scrolled = json.loads(session_search(
            session_id=hit["session_id"],
            around_message_id=hit["match_message_id"],
            db=db, persistence=persistence,
            current_session_id="s_today",
        ))
        assert scrolled["success"] is True
        assert scrolled["mode"] == "scroll"
        contents = " ".join(m.get("content") or "" for m in scrolled["messages"])
        assert "ibuprofen" in contents.lower()


class TestNewResetLineageBrowse:
    """Browse must list /new-reset conversations, not only cron/root rows."""

    def test_reset_parent_appears_in_browse(self, db, persistence):
        _seed_gateway_new_reset_chain(db)
        result = json.loads(session_search(db=db, persistence=persistence, current_session_id="s_today"))
        assert result["mode"] == "browse"
        sids = [r["session_id"] for r in result["results"]]
        assert "s_today" not in sids
        assert "s_night" in sids

    def test_browse_still_hides_live_delegation_child(self, db, persistence):
        seed_session(db, "s_main", "cli")
        seed_messages(db, "s_main", {"role": "user", "content": "parent work"})
        seed_session(
            db,
            "s_delegate",
            "cli",
            parent_session_id="s_main",
        )
        seed_messages(db, "s_delegate", {"role": "assistant", "content": "subagent work"})
        result = json.loads(session_search(db=db, persistence=persistence, current_session_id="s_other"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_delegate" not in sids
        assert "s_main" in sids

    def test_browse_lists_legacy_premarker_reset_child(self, db, persistence):
        """A pre-marker reset child (no _reset_from, admitted by the SQL
        same-key heuristic because its parent ended at a reset boundary on
        the same session_key) must not be re-hidden by a Python re-check.
        Regression guard for the follow-up to #85756."""
        seed_session(db, "s_old", "telegram", session_key="tg:legacy:1")
        seed_messages(db, "s_old", {"role": "user", "content": "legacy era chat"})
        db.sessions.end("s_old", "session_reset")
        # Legacy child: parent link + same session_key, NO _reset_from marker,
        # still live (end_reason=None).
        seed_session(
            db,
            "s_legacy_child",
            "telegram",
            parent_session_id="s_old",
            session_key="tg:legacy:1",
        )
        seed_messages(db, "s_legacy_child", {"role": "user", "content": "current era chat"})
        result = json.loads(session_search(db=db, persistence=persistence, current_session_id="s_other"))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_legacy_child" in sids


def _unix(year, month, day):
    return int(datetime(year, month, day, tzinfo=UTC).timestamp())


def _set_started(db, **started_at_by_sid):
    for sid, ts in started_at_by_sid.items():
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (ts, sid),
        ))


class TestDiscoveryTemporalNarrowing:
    @pytest.mark.parametrize("value, expected", [
        ("24h", 86400), ("1d", 86400), ("2w", 2 * 604800), ("7D", 7 * 86400), (" 3 d ", 3 * 86400),
        ("2026-06-01", _unix(2026, 6, 1)), ("2026-06-01T12:00:00Z", _unix(2026, 6, 1) + 43200),
        (None, None), ("", None), ("7x", ValueError), ("not-a-date", ValueError),
    ])
    def test_parse_bound(self, value, expected):
        """Relative durations are now-minus-N (checked as an offset); ISO is absolute UTC."""
        from tools.session_search_tool import _parse_iso_bound
        if expected is ValueError:
            with pytest.raises(ValueError):
                _parse_iso_bound(value)
        elif isinstance(value, str) and value.strip() and value.strip()[-1].lower() in "hdw":
            assert abs((int(time.time()) - _parse_iso_bound(value)) - expected) < 5
        else:
            assert _parse_iso_bound(value) == expected

    def test_iso_window_is_applied_in_sql_not_as_a_post_filter(self, db, persistence, monkeypatch):
        """A June hit must survive even when FTS rank would fill the scan with August rows, so the
        bound has to live in the WHERE clause rather than trim the already-truncated top-N. An
        invalid bound is a tool error, not a silent unbounded search."""
        monkeypatch.setattr("tools.session_search_tool._DISCOVER_SCAN_LIMIT", 2)
        for i in range(4):
            sid = f"s_aug_{i}"
            seed_session(db, sid, "cli")
            seed_messages(db, sid, {"role": "user", "content": "modpack modpack modpack modpack"})
            seed_messages(db, sid, {"role": "assistant", "content": "modpack details " * 20})
            _set_started(db, **{sid: _unix(2026, 8, 10)})
        seed_session(db, "s_june", "cli")
        seed_messages(db, "s_june", {"role": "user", "content": "modpack"})
        seed_messages(db, "s_june", {"role": "assistant", "content": "ok"})
        _set_started(db, s_june=_unix(2026, 6, 15))

        result = json.loads(session_search(
            query="modpack", limit=5, after="2026-06-01", before="2026-07-01", db=db, persistence=persistence))
        assert result["success"] is True
        assert [r["session_id"] for r in result["results"]] == ["s_june"]
        assert json.loads(session_search(query="modpack", after="not-a-date", db=db, persistence=persistence))["success"] is False

    def test_relative_bounds_reach_the_tool_through_the_inline_executor(self, db, persistence):
        """after="7d" keeps the last week, before="7d" keeps everything older — and both must
        survive the production dispatch (INLINE_TOOL_EXECUTORS), which maps schema args to kwargs
        explicitly and used to drop the new parameters on the floor."""
        from types import SimpleNamespace

        from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

        _seed_modpack_sessions(db)
        now = int(time.time())
        _set_started(db, s_newest=now - 2 * 86400, s_middle=now - 30 * 86400, s_oldest=now - 60 * 86400)
        agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, persistence=persistence, session_id="current")
        ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")

        def run(**args):
            out = json.loads(INLINE_TOOL_EXECUTORS["session_search"](
                agent, {"query": "modpack", "limit": 5, **args}, ctx))
            assert out["success"] is True
            return {r["session_id"] for r in out["results"]}

        assert run(after="7d") == {"s_newest"}
        assert run(before="7d") == {"s_middle", "s_oldest"}
        assert run(exclude_session_ids=["s_newest"]) == {"s_middle", "s_oldest"}


class TestDiscoverySessionExclusion:
    def test_exclude_session_ids_drops_whole_lineage(self, db, persistence):
        seed_session(db, "s_root", "cli")
        seed_messages(db, "s_root", {"role": "user", "content": "unique lineage token alpha"})
        seed_messages(db, "s_root", {"role": "assistant", "content": "noted unique lineage token alpha"})
        seed_session(db, "s_child", "cli", parent_session_id="s_root")
        seed_messages(db, "s_child", {"role": "user", "content": "follow-up unique lineage token alpha"})
        seed_messages(db, "s_child", {"role": "assistant", "content": "child unique lineage token alpha"})
        _set_started(db, s_root=_unix(2026, 6, 1), s_child=_unix(2026, 6, 2))

        found = json.loads(session_search(query="unique lineage token alpha", limit=5, db=db, persistence=persistence))
        assert {r["session_id"] for r in found["results"]} & {"s_root", "s_child"}
        excluded = json.loads(session_search(
            query="unique lineage token alpha", limit=5, exclude_session_ids=["s_child"], db=db, persistence=persistence))
        assert not {r["session_id"] for r in excluded["results"]} & {"s_root", "s_child"}


class TestBrowseDefaultLimit:
    def test_no_limit_browse_returns_ten_on_every_entry_point(self, db, tmp_path, monkeypatch):
        """Browse must not inherit discovery's default of 3 via the function, the registry
        handler, or the inline executor (the live path), each of which used to bake in 3."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # a call without a handle reads the home's store
        from types import SimpleNamespace

        from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
        from tools.registry import registry

        for i in range(12):
            seed_session(db, f"s_browse_{i:02d}", "cli")
            seed_messages(db, f"s_browse_{i:02d}", {"role": "user", "content": f"browse seed {i}"})
        agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, session_id="current")
        ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")

        for raw in (session_search(db=db), registry.get_entry("session_search").handler({}, db=db),
                    INLINE_TOOL_EXECUTORS["session_search"](agent, {}, ctx)):
            result = json.loads(raw)
            assert result["mode"] == "browse"
            assert len(result["results"]) == 10
        assert len(json.loads(session_search(db=db, limit=4))["results"]) == 4
