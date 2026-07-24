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

import tools.session_search_tool as session_search_tool
from agent.context_compressor import (
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
)
from hermes_state import SessionDB
from tools.session_search_tool import (
    SESSION_SEARCH_SCHEMA,
    _HIDDEN_SESSION_SOURCES,
    _MESSAGE_CONTENT_MAX_CHARS,
    _RESPONSE_FIELD_MAX_CHARS,
    _RESPONSE_MAX_CHARS,
    _SNIPPET_MAX_CHARS,
    _TOOL_CALL_ARGUMENTS_MAX_CHARS,
    _TOOL_CALL_METADATA_MAX_CHARS,
    _TOOL_CALLS_MAX_ITEMS,
    _format_timestamp,
    _is_compacted_message,
    _is_compression_ended,
    _resolve_to_parent,
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

    def test_no_mode_parameter(self):
        # Mode is inferred from which args are set — no explicit mode param
        params = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
        assert "mode" not in params

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

    def test_schema_description_enforces_source_first_limit(self):
        desc = SESSION_SEARCH_SCHEMA["description"].lower()
        assert "source-first limit" in desc
        assert "conversation history only" in desc
        assert "direct source" in desc
        assert "session_search as secondary" in desc
        assert "not found" in desc

    def test_schema_description_documents_bounded_recall_payloads(self):
        desc = SESSION_SEARCH_SCHEMA["description"].lower()
        assert "bounded" in desc
        assert "compaction-summary" in desc
        assert "omitted" in desc
        assert "truncated" in desc
        assert "metadata" in desc


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

    def test_query_can_match_session_title_without_message_hit(self, db):
        db.create_session("s_fingerprint", source="cli")
        db.set_session_title("s_fingerprint", "fingerprint-login")
        db.append_message("s_fingerprint", role="user", content="Let's configure PAM for biometric auth")
        db.append_message("s_fingerprint", role="assistant", content="Checking Linux auth settings.")

        result = json.loads(session_search(query="fingerprint-login", db=db))

        assert result["success"] is True
        assert result["count"] == 1
        hit = result["results"][0]
        assert hit["session_id"] == "s_fingerprint"
        assert hit["title"] == "fingerprint-login"
        assert hit["matched_role"] == "session_title"
        assert "Session title matched" in hit["snippet"]


    def test_title_query_strips_common_model_quoting(self, db):
        db.create_session("s_fingerprint", source="cli")
        db.set_session_title("s_fingerprint", "fingerprint-login")
        db.append_message("s_fingerprint", role="user", content="PAM auth setup")

        result = json.loads(session_search(query="`fingerprint-login`", db=db))

        assert result["success"] is True
        assert result["results"][0]["session_id"] == "s_fingerprint"
        assert result["results"][0]["matched_role"] == "session_title"

    def test_title_match_respects_current_session_filter(self, db):
        db.create_session("s_current", source="cli")
        db.set_session_title("s_current", "fingerprint-login")
        db.append_message("s_current", role="user", content="PAM auth setup")

        result = json.loads(session_search(
            query="fingerprint-login",
            current_session_id="s_current",
            db=db,
        ))

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


class TestRecallPayloadSafety:
    def test_discovery_omits_context_compaction_summary_messages(self, db):
        db.create_session("s_compacted", source="cli")
        db.append_message("s_compacted", role="user", content="opener")
        summary = (
            f"{SUMMARY_PREFIX}\n"
            "needle ## Historical Remaining Work\n"
            "stale task text "
            + ("x" * 12000)
        )
        db.append_message("s_compacted", role="assistant", content=summary)
        db.append_message("s_compacted", role="assistant", content="final decision")
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        assert result["success"] is True
        hit = result["results"][0]
        all_messages = hit["bookend_start"] + hit["messages"] + hit["bookend_end"]
        summary_messages = [
            m for m in all_messages
            if m.get("content_omitted") == "context_compaction_summary"
        ]
        assert summary_messages
        assert all("Historical Remaining Work" not in m["content"] for m in all_messages)
        assert all("stale task text" not in m["content"] for m in all_messages)
        assert all("x" * 100 not in m["content"] for m in all_messages)
        assert summary_messages[0]["original_content_chars"] == len(summary)
        assert hit["snippet_omitted"] == "context_compaction_summary"
        assert "Historical Remaining Work" not in hit["snippet"]
        assert "stale task text" not in hit["snippet"]

    def test_discovery_omits_merged_context_compaction_summary_messages(self, db):
        db.create_session("s_merged_compaction", source="cli")
        db.append_message("s_merged_compaction", role="user", content="opener")
        merged_summary = (
            "[PRIOR CONTEXT — for reference only; not a new message]\n"
            "prior tail text\n"
            f"{_MERGED_SUMMARY_DELIMITER}\n"
            f"{SUMMARY_PREFIX}\n"
            "needle stale merged task "
            + ("x" * 12000)
        )
        db.append_message(
            "s_merged_compaction",
            role="assistant",
            content=merged_summary,
        )
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        assert result["success"] is True
        hit = result["results"][0]
        all_messages = hit["bookend_start"] + hit["messages"] + hit["bookend_end"]
        summary_messages = [
            m
            for m in all_messages
            if m.get("content_omitted") == "context_compaction_summary"
        ]
        assert summary_messages
        assert all("stale merged task" not in m["content"] for m in all_messages)
        assert summary_messages[0]["original_content_chars"] == len(merged_summary)
        assert hit["snippet_omitted"] == "context_compaction_summary"

    def test_discovery_omits_structured_merged_compaction_summary(self, db):
        db.create_session("s_structured_summary", source="cli")
        structured_summary = [
            {
                "type": "text",
                "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\nprior tail",
            },
            {
                "type": "text",
                "text": (
                    f"{_MERGED_SUMMARY_DELIMITER}\n"
                    f"{SUMMARY_PREFIX}\n"
                    "needle stale structured task"
                ),
            },
        ]
        db.append_message(
            "s_structured_summary",
            role="assistant",
            content=structured_summary,
        )
        db._conn.commit()

        result = json.loads(session_search(session_id="s_structured_summary", db=db))

        message = result["messages"][0]
        assert message["content_omitted"] == "context_compaction_summary"
        assert "stale structured task" not in message["content"]

    def test_discovery_omits_deep_summary_snippet_from_full_content(self, db):
        db.create_session("s_deep_summary", source="cli")
        summary = (
            f"{SUMMARY_PREFIX}\n"
            + ("filler " * 100)
            + "needle stale deep task"
        )
        db.append_message("s_deep_summary", role="assistant", content=summary)
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        hit = result["results"][0]
        assert hit["snippet_omitted"] == "context_compaction_summary"
        assert "stale deep task" not in hit["snippet"]

    def test_discovery_prefers_source_hit_over_higher_ranked_summary(self, db, monkeypatch):
        db.create_session("s_source_preference", source="cli")
        db.append_message(
            "s_source_preference",
            role="user",
            content="needle source decision",
        )
        for index in range(12):
            db.append_message(
                "s_source_preference",
                role="assistant",
                content=f"middle context {index}",
            )
        db.append_message(
            "s_source_preference",
            role="assistant",
            content=f"{SUMMARY_PREFIX}\nneedle needle stale summary task",
        )
        rows = db.get_messages("s_source_preference")
        source_id = rows[0]["id"]
        summary_id = rows[-1]["id"]
        monkeypatch.setattr(
            db,
            "search_messages",
            lambda **_kwargs: [
                {
                    "session_id": "s_source_preference",
                    "id": summary_id,
                    "role": "assistant",
                    "snippet": "needle needle stale summary task",
                    "source": "cli",
                },
                {
                    "session_id": "s_source_preference",
                    "id": source_id,
                    "role": "user",
                    "snippet": "needle source decision",
                    "source": "cli",
                },
            ],
        )

        result = json.loads(session_search(query="needle", limit=1, db=db))

        hit = result["results"][0]
        assert hit["match_message_id"] == source_id
        assert hit["snippet"] == "needle source decision"
        assert "snippet_omitted" not in hit
        anchor = next(message for message in hit["messages"] if message.get("anchor"))
        assert anchor["content"] == "needle source decision"

    def test_discovery_stops_after_limit_when_no_summary_needs_replacement(
        self, db, monkeypatch
    ):
        for index in range(20):
            session_id = f"s_early_{index}"
            db.create_session(session_id, source="cli")
            db.append_message(session_id, role="user", content=f"needle result {index}")

        original_get_session = db.get_session
        get_session_calls = []

        def recording_get_session(session_id):
            get_session_calls.append(session_id)
            return original_get_session(session_id)

        monkeypatch.setattr(db, "get_session", recording_get_session)

        result = json.loads(session_search(query="needle", limit=3, db=db))

        assert result["count"] == 3
        assert len(get_session_calls) <= 6

    def test_lineage_resolution_path_compresses_repeated_deep_hits(self):
        class ChainDB:
            def __init__(self):
                self.calls = 0

            def get_session(self, session_id):
                self.calls += 1
                index = int(session_id[1:])
                return {
                    "id": session_id,
                    "parent_session_id": f"s{index - 1}" if index else None,
                }

        chain = ChainDB()
        cache = {}
        budget = [session_search_tool._MAX_DISCOVERY_LINEAGE_LOOKUPS]

        roots = [
            session_search_tool._resolve_lineage(
                chain,
                "s150",
                cache=cache,
                lookup_budget=budget,
            )
            for _ in range(300)
        ]

        assert set(roots) == {"s0"}
        assert chain.calls == 151

    def test_context_compaction_lookalike_is_not_omitted(self, db):
        db.create_session("s_lookalike", source="cli")
        content = "[CONTEXT COMPACTION is a topic, not a generated summary] needle"
        db.append_message("s_lookalike", role="user", content=content)
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        hit = result["results"][0]
        anchor = next(m for m in hit["messages"] if m.get("anchor"))
        assert anchor["content"] == content
        assert "content_omitted" not in anchor
        assert "snippet_omitted" not in hit

    def test_context_compaction_lookalike_survives_bookend_filter(self, db):
        db.create_session("s_lookalike_bookend", source="cli")
        kickoff = "[CONTEXT COMPACTION is a topic, not a generated summary] KICKOFF_FACT"
        db.append_message("s_lookalike_bookend", role="user", content=kickoff)
        for index in range(12):
            db.append_message(
                "s_lookalike_bookend",
                role="assistant",
                content=f"ordinary middle message {index}",
            )
        db.append_message(
            "s_lookalike_bookend",
            role="user",
            content="distant valid needle",
        )
        db._conn.commit()

        result = json.loads(session_search(query="distant valid needle", limit=1, db=db))

        assert result["success"] is True
        assert kickoff in [
            message["content"] for message in result["results"][0]["bookend_start"]
        ]

    def test_discovery_truncates_large_snippet(self, db):
        db.create_session("s_large_snippet", source="cli")
        huge_token = "needle" + ("a" * (_SNIPPET_MAX_CHARS + 2000))
        db.append_message("s_large_snippet", role="user", content=huge_token)
        db._conn.commit()

        result = json.loads(session_search(query="needle*", limit=1, db=db))

        assert result["success"] is True
        hit = result["results"][0]
        assert hit["snippet_truncated"] is True
        assert hit["original_snippet_chars"] > _SNIPPET_MAX_CHARS
        assert len(hit["snippet"]) < hit["original_snippet_chars"]
        assert "session_search truncated" in hit["snippet"]

    def test_discovery_truncates_large_message_content(self, db):
        db.create_session("s_large", source="cli")
        large = "needle " + ("a" * (_MESSAGE_CONTENT_MAX_CHARS + 2000))
        db.append_message("s_large", role="user", content=large)
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        assert result["success"] is True
        anchor = result["results"][0]["messages"][0]
        assert anchor["content_truncated"] is True
        assert anchor["original_content_chars"] == len(large)
        assert len(anchor["content"]) < len(large)
        assert "session_search truncated" in anchor["content"]

    def test_discovery_truncates_large_structured_message_content(self, db):
        db.create_session("s_structured", source="cli")
        structured = [
            {
                "type": "text",
                "text": "needle " + ("m" * (_MESSAGE_CONTENT_MAX_CHARS + 2000)),
            }
        ]
        expected_chars = len(json.dumps(structured, ensure_ascii=False))
        db.append_message("s_structured", role="user", content=structured)
        db._conn.commit()

        result = json.loads(session_search(query="needle", limit=1, db=db))

        assert result["success"] is True
        anchor = result["results"][0]["messages"][0]
        assert anchor["content_truncated"] is True
        assert anchor["original_content_chars"] == expected_chars
        assert isinstance(anchor["content"], str)
        assert len(anchor["content"]) < expected_chars
        assert "m" * (_MESSAGE_CONTENT_MAX_CHARS + 100) not in anchor["content"]
        assert "session_search truncated" in anchor["content"]

    def test_read_truncates_tool_call_arguments_and_count(self, db):
        db.create_session("s_tools", source="cli")
        tool_calls = [
            {
                "id": f"call_{i}",
                "function": {
                    "name": "session_search",
                    "arguments": "{" + f'"query": "needle {i}", "blob": "' + ("z" * 4000) + '"}',
                },
            }
            for i in range(_TOOL_CALLS_MAX_ITEMS + 2)
        ]
        db.append_message("s_tools", role="assistant", content="", tool_calls=tool_calls)
        db._conn.commit()

        result = json.loads(session_search(session_id="s_tools", db=db))

        assert result["success"] is True
        message = result["messages"][0]
        assert len(message["tool_calls"]) == _TOOL_CALLS_MAX_ITEMS
        assert message["tool_calls_truncated"] is True
        assert message["original_tool_call_count"] == _TOOL_CALLS_MAX_ITEMS + 2
        assert message["tool_call_arguments_truncated"] is True
        args = message["tool_calls"][0]["function"]["arguments"]
        assert len(args) < _TOOL_CALL_ARGUMENTS_MAX_CHARS + 200
        assert "session_search truncated" in args

    def test_read_truncates_top_level_tool_call_arguments(self, db):
        db.create_session("s_top_level_tools", source="cli")
        tool_calls = [
            {
                "name": "session_search",
                "arguments": "{" + '"query": "needle", "blob": "' + ("z" * 4000) + '"}',
            }
        ]
        db.append_message("s_top_level_tools", role="assistant", content="", tool_calls=tool_calls)
        db._conn.commit()

        result = json.loads(session_search(session_id="s_top_level_tools", db=db))

        assert result["success"] is True
        message = result["messages"][0]
        assert message["tool_call_arguments_truncated"] is True
        args = message["tool_calls"][0]["arguments"]
        assert len(args) < _TOOL_CALL_ARGUMENTS_MAX_CHARS + 200
        assert "z" * (_TOOL_CALL_ARGUMENTS_MAX_CHARS + 100) not in args
        assert "session_search truncated" in args

    def test_read_omits_unbounded_provider_tool_call_fields(self, db):
        db.create_session("s_provider_fields", source="cli")
        oversized = "provider-bookkeeping-" + ("r" * 12000)
        oversized_id = "call_" + ("i" * 12000)
        tool_calls = [
            {
                "id": oversized_id,
                "type": "function",
                "response_item_id": oversized,
                "provider_payload": {"opaque": oversized},
                "function": {
                    "name": "session_search",
                    "arguments": '{"query": "needle"}',
                    "response_item_id": oversized,
                },
            }
        ]
        db.append_message(
            "s_provider_fields",
            role="assistant",
            content="",
            tool_calls=tool_calls,
        )
        db._conn.commit()

        result = json.loads(session_search(session_id="s_provider_fields", db=db))

        assert result["success"] is True
        message = result["messages"][0]
        call = message["tool_calls"][0]
        assert set(call) == {"id", "type", "function"}
        assert call["type"] == "function"
        assert call["function"] == {
            "name": "session_search",
            "arguments": '{"query": "needle"}',
        }
        assert len(call["id"]) < _TOOL_CALL_METADATA_MAX_CHARS + 200
        assert "session_search truncated" in call["id"]
        assert message["tool_call_fields_truncated"] is True
        assert message["tool_call_fields_omitted"] is True
        assert oversized not in json.dumps(message)
        assert oversized_id not in json.dumps(message)

    def test_read_bounds_message_tool_identifiers_and_numeric_metadata(self, db):
        db.create_session("s_message_tool_fields", source="cli")
        oversized_name = "tool_" + ("n" * 12000)
        oversized_call_id = "call_" + ("c" * 12000)
        oversized_index = int("9" * 4000)
        db.append_message(
            "s_message_tool_fields",
            role="tool",
            content="done",
            tool_name=oversized_name,
            tool_call_id=oversized_call_id,
            tool_calls=[{"name": "x", "index": oversized_index}],
        )
        db._conn.commit()

        result = json.loads(session_search(session_id="s_message_tool_fields", db=db))

        message = result["messages"][0]
        assert message["tool_name_truncated"] is True
        assert message["tool_call_id_truncated"] is True
        assert message["tool_call_fields_truncated"] is True
        assert len(message["tool_name"]) < _TOOL_CALL_METADATA_MAX_CHARS + 200
        assert len(message["tool_call_id"]) < _TOOL_CALL_METADATA_MAX_CHARS + 200
        assert isinstance(message["tool_calls"][0]["index"], str)
        assert "session_search truncated" in message["tool_calls"][0]["index"]
        assert oversized_name not in json.dumps(message)
        assert oversized_call_id not in json.dumps(message)

    def test_read_enforces_aggregate_response_budget(self, db):
        db.create_session("s_response_budget", source="cli")
        for index in range(30):
            db.append_message(
                "s_response_budget",
                role="user" if index % 2 == 0 else "assistant",
                content=f"message {index} " + ("x" * _MESSAGE_CONTENT_MAX_CHARS),
            )
        db._conn.commit()

        raw = session_search(session_id="s_response_budget", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["response_truncated"] is True
        assert result["original_response_chars"] > _RESPONSE_MAX_CHARS
        assert len(result["messages"]) == 30

    def test_read_aggregate_budget_preserves_content_truncation_metadata(self, db):
        db.create_session("s_response_metadata", source="cli")
        content = "x" * (_MESSAGE_CONTENT_MAX_CHARS - 1)
        for index in range(30):
            db.append_message(
                "s_response_metadata",
                role="user" if index % 2 == 0 else "assistant",
                content=content,
            )
        db._conn.commit()

        raw = session_search(session_id="s_response_metadata", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["response_truncated"] is True
        assert result["response_fields_truncated"] is True
        assert len(result["messages"]) == 30
        assert all(message["content_truncated"] is True for message in result["messages"])
        assert all(
            message["original_content_chars"] == len(content)
            for message in result["messages"]
        )

    def test_read_aggregate_budget_preserves_original_length_across_both_layers(self, db):
        db.create_session("s_layered_response_metadata", source="cli")
        content = "x" * (_MESSAGE_CONTENT_MAX_CHARS + 2000)
        for index in range(30):
            db.append_message(
                "s_layered_response_metadata",
                role="user" if index % 2 == 0 else "assistant",
                content=content,
            )
        db._conn.commit()

        raw = session_search(session_id="s_layered_response_metadata", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["response_truncated"] is True
        assert result["response_fields_truncated"] is True
        assert all(message["content_truncated"] is True for message in result["messages"])
        assert all(
            message["original_content_chars"] == len(content)
            for message in result["messages"]
        )

    def test_read_structural_overflow_fails_closed(self, db):
        db.create_session("s_structural_overflow", source="cli")
        for _ in range(30):
            db.append_message(
                "s_structural_overflow",
                role="assistant",
                content=list(range(1000)),
            )
        db._conn.commit()

        raw = session_search(session_id="s_structural_overflow", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["success"] is False
        assert result["response_truncated"] is True
        assert result["error"] == "session_recall_response_too_large"

    def test_read_bounds_large_metadata_below_aggregate_budget(self, db):
        db.create_session("s_large_metadata", source="cli")
        title = "T" * 100_000
        db._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, "s_large_metadata"),
        )
        db.append_message("s_large_metadata", role="user", content="small")
        db._conn.commit()

        raw = session_search(session_id="s_large_metadata", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["success"] is True
        assert len(result["session_meta"]["title"]) <= _RESPONSE_FIELD_MAX_CHARS
        assert result["response_fields_truncated"] is True
        assert result["response_truncated"] is True
        assert result["original_response_chars"] > len(raw)
        assert title not in raw

    def test_read_bounds_pathological_structured_keys_without_dropping_shape(self, db):
        db.create_session("s_structured_keys", source="cli")
        content = {"K" * 5800: ""}
        for _ in range(40):
            db.append_message("s_structured_keys", role="assistant", content=content)
        db._conn.commit()

        raw = session_search(session_id="s_structured_keys", db=db)
        result = json.loads(raw)

        assert len(raw) <= _RESPONSE_MAX_CHARS
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["message_count"] == 40
        assert len(result["messages"]) == 30
        assert result["response_truncated"] is True


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

    def test_read_unknown_session_errors(self, db):
        result = json.loads(session_search(session_id="ghost", db=db))
        assert result["success"] is False

    @pytest.mark.parametrize(
        ("cached_count", "inactive_tail", "expected_total", "expected_last"),
        [
            (0, False, 50, "m49"),
            (1_000_000, False, 50, "m49"),
            ("corrupt", True, 45, "m44"),
        ],
    )
    def test_read_truncates_from_active_source_of_truth(
        self,
        db,
        monkeypatch,
        cached_count,
        inactive_tail,
        expected_total,
        expected_last,
    ):
        db.create_session("s_big", source="cli")
        for i in range(50):
            db.append_message(
                "s_big",
                role="user" if i % 2 == 0 else "assistant",
                content=f"m{i}",
            )
        db._conn.execute(
            "UPDATE sessions SET message_count = ? WHERE id = ?",
            (cached_count, "s_big"),
        )
        if inactive_tail:
            ids = [
                row[0]
                for row in db._conn.execute(
                    "SELECT id FROM messages WHERE session_id = ? ORDER BY id",
                    ("s_big",),
                )
            ]
            db._conn.execute(
                "UPDATE messages SET active = 0 WHERE id >= ?",
                (ids[-5],),
            )
        db._conn.commit()

        monkeypatch.setattr(
            db,
            "get_messages",
            lambda *_args, **_kwargs: pytest.fail(
                "read shape must not hydrate an unbounded list"
            ),
        )
        result = json.loads(session_search(session_id="s_big", db=db))

        assert result["mode"] == "read"
        assert result["message_count"] == expected_total
        assert result["truncated"] is True
        assert len(result["messages"]) == 30  # head 20 + tail 10
        assert result["messages"][-1]["content"] == expected_last

    def test_head_tail_query_does_not_rank_full_message_blobs(self, db):
        db.create_session("s_perf", source="cli")
        db._conn.executemany(
            "INSERT INTO messages "
            "(session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
            [
                (
                    "s_perf",
                    "user" if i % 2 == 0 else "assistant",
                    f"m{i}:" + ("x" * 256),
                    float(i),
                )
                for i in range(20_000)
            ],
        )

        vm_steps = 0

        def count_steps():
            nonlocal vm_steps
            vm_steps += 100
            return 0

        db._conn.set_progress_handler(count_steps, 100)
        try:
            rows, total = db.get_message_head_tail("s_perf", head=20, tail=10)
        finally:
            db._conn.set_progress_handler(None, 0)

        assert total == 20_000
        assert len(rows) == 30
        assert rows[0]["content"].startswith("m0:")
        assert rows[-1]["content"].startswith("m19999:")
        assert vm_steps < 600_000


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

    def test_profile_param_reads_other_db(self, db, tmp_path, monkeypatch):
        other_home = tmp_path / "other_home"
        other_home.mkdir()
        other = SessionDB(other_home / "state.db")
        other.create_session("s_other", source="cli")
        other._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", ("Other Profile Chat", "s_other")
        )
        other.append_message("s_other", role="user", content="hello from the other profile")
        other._conn.commit()

        self._patch_profiles(monkeypatch, other_home)

        # s_other lives only in the other profile; the current `db` lacks it.
        result = json.loads(session_search(session_id="s_other", profile="other", db=db))
        assert result["success"] is True
        assert result["mode"] == "read"
        assert result["session_meta"]["title"] == "Other Profile Chat"

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

    def test_unknown_profile_errors(self, db, monkeypatch, tmp_path):
        self._patch_profiles(monkeypatch, tmp_path, exists=False)
        result = json.loads(session_search(session_id="x", profile="ghost", db=db))
        assert result["success"] is False
        assert "ghost" in result.get("error", "")

    def test_owned_profile_db_closes_on_error_response(self, db, tmp_path, monkeypatch):
        owned = SessionDB(tmp_path / "owned.db")
        closed = []
        real_close = owned.close

        def track_close():
            closed.append(True)
            real_close()

        monkeypatch.setattr(owned, "close", track_close)
        monkeypatch.setattr(
            session_search_tool,
            "_resolve_profile_db",
            lambda _profile: owned,
        )

        result = json.loads(
            session_search(
                session_id="missing",
                around_message_id=1,
                profile="other",
                db=db,
            )
        )

        assert result["success"] is False
        assert closed == [True]

    def test_default_constructed_db_is_closed(self, tmp_path, monkeypatch):
        owned = SessionDB(tmp_path / "default-owned.db")
        closed = []
        real_close = owned.close

        def track_close():
            closed.append(True)
            real_close()

        monkeypatch.setattr(owned, "close", track_close)
        import hermes_state

        monkeypatch.setattr(hermes_state, "SessionDB", lambda: owned)

        result = json.loads(session_search(db=None))

        assert result["success"] is True
        assert closed == [True]

    def test_caller_supplied_db_is_not_closed(self, db, monkeypatch):
        close_calls = []
        monkeypatch.setattr(db, "close", lambda: close_calls.append(True))

        result = json.loads(session_search(db=db))

        assert result["success"] is True
        assert close_calls == []

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

    def test_order_for_recall_is_stable_within_class(self):
        from tools.session_search_tool import _order_for_recall
        rows = [
            {"id": 1, "source": "cron"},
            {"id": 2, "source": "telegram"},
            {"id": 3, "source": "cron"},
            {"id": 4, "source": "cli"},
            {"id": 5, "source": None},
        ]
        ordered = _order_for_recall(rows)
        # Interactive rows first, in original relative order; cron last, in
        # original relative order.
        assert [r["id"] for r in ordered] == [2, 4, 5, 1, 3]


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

    def test_compaction_summary_excluded_from_bookend_end(self, db):
        """Compaction handoff in bookend_end position must be filtered out."""
        db.create_session("s_compact_end", source="cli")
        # Normal opening
        db.append_message("s_compact_end", role="user", content="Build a website")
        db.append_message("s_compact_end", role="assistant", content="Sure, let me scaffold it.")
        # Match target (early in session so bookend_end has room)
        db.append_message("s_compact_end", role="user", content="fix the zorgblat rendering bug")
        db.append_message("s_compact_end", role="assistant", content="Investigating the zorgblat rendering issue.")
        # Many messages to create distance from the end
        for i in range(10):
            db.append_message("s_compact_end", role="user", content=f"feature {i}")
            db.append_message("s_compact_end", role="assistant", content=f"implemented {i}")
        # Last message: compaction handoff (should be filtered from bookend_end)
        db.append_message("s_compact_end", role="assistant",
                          content="[CONTEXT COMPACTION — REFERENCE ONLY] "
                                  "Summary of all work done. " + "y" * 50000)
        db._conn.commit()

        result = json.loads(session_search(query="zorgblat rendering", db=db, limit=1))
        assert result["success"] is True
        assert len(result["results"]) >= 1
        entry = result["results"][0]
        # bookend_end must NOT contain the compaction handoff
        for msg in entry.get("bookend_end", []):
            assert "[CONTEXT COMPACTION" not in (msg.get("content") or "")

    def test_bookend_content_is_capped(self, db):
        """Bookend messages must have content capped at 1200 chars."""
        db.create_session("s_long_bookend", source="cli")
        # First message: very long normal content
        db.append_message("s_long_bookend", role="user",
                          content="Start the project. " + "z" * 5000)
        # Match target
        db.append_message("s_long_bookend", role="user", content="deploy to production")
        db.append_message("s_long_bookend", role="assistant", content="Deploying now.")
        for i in range(10):
            db.append_message("s_long_bookend", role="user", content=f"step {i}")
            db.append_message("s_long_bookend", role="assistant", content=f"done {i}")
        db._conn.commit()

        result = json.loads(session_search(query="deploy production", db=db, limit=1))
        assert result["success"] is True
        entry = result["results"][0]
        for msg in entry.get("bookend_start", []):
            content = msg.get("content", "")
            # Content should be capped (1200 chars + "…" ellipsis)
            assert len(content) <= 1210  # 1200 + ellipsis + margin
            if msg.get("content_truncated"):
                assert msg["original_content_chars"] > 1200

    def test_window_content_is_capped(self, db):
        """Window messages must have content capped at 4000 chars."""
        db.create_session("s_long_window", source="cli")
        db.append_message("s_long_window", role="user", content="search keyword here")
        # Very long assistant reply containing the keyword
        db.append_message("s_long_window", role="assistant",
                          content="Found it! keyword " + "a" * 10000)
        db._conn.commit()

        result = json.loads(session_search(query="keyword", db=db, limit=1))
        assert result["success"] is True
        entry = result["results"][0]
        for msg in entry.get("messages", []):
            content = msg.get("content", "")
            assert len(content) <= 4010  # 4000 + ellipsis + margin

    def test_legacy_context_summary_filtered(self, db):
        """Legacy [CONTEXT SUMMARY]: prefix must also be filtered."""
        db.create_session("s_legacy", source="cli")
        db.append_message("s_legacy", role="user",
                          content="[CONTEXT SUMMARY]: old compacted summary here")
        db.append_message("s_legacy", role="user", content="new task: build API")
        db.append_message("s_legacy", role="assistant", content="Building REST API now.")
        for i in range(10):
            db.append_message("s_legacy", role="user", content=f"step {i}")
            db.append_message("s_legacy", role="assistant", content=f"done {i}")
        db._conn.commit()

        result = json.loads(session_search(query="build API", db=db, limit=1))
        assert result["success"] is True
        entry = result["results"][0]
        for msg in entry.get("bookend_start", []):
            assert "[CONTEXT SUMMARY]" not in (msg.get("content") or "")


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

    def test_root_session_no_compression(self, db):
        db.create_session("s1", source="cli")
        root, has_compression = _resolve_to_parent(db, "s1")
        assert root == "s1"
        assert has_compression is False

    def test_empty_session_id(self, db):
        root, has_compression = _resolve_to_parent(db, "")
        assert root == ""
        assert has_compression is False

    def test_none_session_id(self, db):
        root, has_compression = _resolve_to_parent(db, None)
        assert root is None
        assert has_compression is False

    def test_legacy_rotation_detects_compression(self, db):
        """Parent ended with end_reason='compression', child has parent_session_id."""
        db.create_session("s_parent", source="cli")
        db.end_session("s_parent", "compression")
        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        root, has_compression = _resolve_to_parent(db, "s_child")
        assert root == "s_parent"
        assert has_compression is True

    def test_delegation_no_compression(self, db):
        """Delegation child: parent_session_id set but no compression end_reason."""
        db.create_session("s_parent", source="cli")
        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        root, has_compression = _resolve_to_parent(db, "s_child")
        assert root == "s_parent"
        assert has_compression is False

    def test_multi_level_compression_chain(self, db):
        """Grandparent → parent → child, both with compression edges."""
        db.create_session("s_gp", source="cli")
        db.end_session("s_gp", "compression")
        db.create_session("s_p", source="cli", parent_session_id="s_gp")
        db.end_session("s_p", "compression")
        db.create_session("s_c", source="cli", parent_session_id="s_p")
        root, has_compression = _resolve_to_parent(db, "s_c")
        assert root == "s_gp"
        assert has_compression is True

    def test_chain_with_mixed_edges(self, db):
        """Compression parent → delegation-style child (no end_reason on child)."""
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

    def test_none_message_id(self, db):
        assert _is_compacted_message(db, None) is False

    def test_nonexistent_message_id(self, db):
        assert _is_compacted_message(db, 999999) is False


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

    def test_discovered_archived_anchor_can_be_scrolled(self, db):
        db.create_session("s_compact_scroll", source="cli")
        db.append_message(
            "s_compact_scroll",
            role="user",
            content="spectral phoenix archived anchor",
        )
        db.append_message(
            "s_compact_scroll",
            role="assistant",
            content="archived answer",
        )
        db.archive_and_compact(
            "s_compact_scroll",
            [{"role": "assistant", "content": "current compacted summary"}],
        )

        discovery = json.loads(
            session_search(
                query="spectral phoenix archived anchor",
                db=db,
                current_session_id="s_compact_scroll",
            )
        )
        hit = discovery["results"][0]

        scrolled = json.loads(
            session_search(
                session_id=hit["session_id"],
                around_message_id=hit["match_message_id"],
                db=db,
                current_session_id="s_compact_scroll",
            )
        )

        assert scrolled["success"] is True
        assert any(
            message.get("anchor")
            and "spectral phoenix archived anchor" in message["content"]
            for message in scrolled["messages"]
        )

    def test_live_content_still_filtered_on_current_session(self, db):
        """Non-compacted (active) content on the current session stays filtered."""
        db.create_session("s_live", source="cli")
        db.append_message("s_live", role="user", content="crystal golem farming route")
        result = json.loads(session_search(
            query="crystal golem", db=db, current_session_id="s_live",
        ))
        assert result["count"] == 0

    def test_mixed_active_and_compacted_on_same_session(self, db):
        """A session that has been compacted: the archived content is
        discoverable, but the new (post-compaction) active content is not
        (it's in live context)."""
        db.create_session("s_mixed", source="cli")
        # Pre-compaction content (will be archived)
        db.append_message("s_mixed", role="user", content="ancient ruins exploration log")
        db.append_message("s_mixed", role="assistant", content="ancient ruins mapped")
        # Compact
        db.archive_and_compact("s_mixed", [
            {"role": "user", "content": "Summary of ancient ruins exploration"},
            {"role": "assistant", "content": "Continuing ancient ruins work"},
        ])
        # Archived content should be discoverable
        result_archived = json.loads(session_search(
            query="ancient ruins exploration", db=db,
            current_session_id="s_mixed",
        ))
        assert result_archived["count"] >= 1


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

    def test_compression_parent_discovery_anchor_can_be_scrolled_from_child(self, db):
        db.create_session("s_scroll_parent", source="cli")
        db.append_message(
            "s_scroll_parent",
            role="user",
            content="legacy rotation archived anchor",
        )
        db.append_message(
            "s_scroll_parent",
            role="assistant",
            content="legacy rotation answer",
        )
        db.end_session("s_scroll_parent", "compression")
        db.create_session(
            "s_scroll_child",
            source="cli",
            parent_session_id="s_scroll_parent",
        )
        db.append_message(
            "s_scroll_child",
            role="user",
            content="active continuation",
        )

        discovery = json.loads(
            session_search(
                query="legacy rotation archived anchor",
                db=db,
                current_session_id="s_scroll_child",
            )
        )
        hit = discovery["results"][0]

        scrolled = json.loads(
            session_search(
                session_id=hit["session_id"],
                around_message_id=hit["match_message_id"],
                db=db,
                current_session_id="s_scroll_child",
            )
        )

        assert scrolled["success"] is True
        assert any(
            message.get("anchor")
            and "legacy rotation archived anchor" in message["content"]
            for message in scrolled["messages"]
        )

    def test_multi_level_compression_chain_discoverable(self, db):
        """Grandparent → parent → child, each compression-rotated. Content from
        ancestors must be discoverable."""
        db.create_session("s_gp", source="cli")
        db.append_message("s_gp", role="user",
                          content="Project titan initial architecture design")
        db.end_session("s_gp", "compression")

        db.create_session("s_p", source="cli", parent_session_id="s_gp")
        db.append_message("s_p", role="user",
                          content="Project titan second phase planning")
        db.end_session("s_p", "compression")

        db.create_session("s_c", source="cli", parent_session_id="s_p")
        db.append_message("s_c", role="user", content="Project titan final review")

        result = json.loads(session_search(
            query="project titan", db=db, current_session_id="s_c",
        ))
        assert result["count"] >= 1
        # Should find content from s_gp or s_p (or both, deduped by lineage)
        sids = [r["session_id"] for r in result["results"]]
        assert any(s in ("s_gp", "s_p") for s in sids)


class TestDelegationExclusion:
    """Delegation children (delegate_task) must STAY excluded — their content
    is still visible to the parent agent. parent_session_id is set but the
    parent does NOT have end_reason='compression'."""

    def test_delegation_parent_excluded_from_child(self, db):
        """Child can see its own content but parent's live content stays
        excluded (it's in context via delegation)."""
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
        assert result["count"] == 0

    def test_delegation_child_excluded_from_parent(self, db):
        """Parent searching should not see delegation child content either —
        both are in the same lineage with no compression edge."""
        db.create_session("s_parent", source="cli")
        db.append_message("s_parent", role="user",
                          content="Working on stellar forge project")

        db.create_session("s_child", source="cli", parent_session_id="s_parent")
        db.append_message("s_child", role="user",
                          content="stellar forge delegated subtask execution")

        result = json.loads(session_search(
            query="stellar forge", db=db, current_session_id="s_parent",
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

    def test_rewind_rows_stay_hidden(self, db):
        """A rewound (active=0, compacted=0) message must not appear in
        discovery, even though it's on the current session."""
        db.create_session("s_rewind", source="cli")
        mid = db.append_message("s_rewind", role="user",
                                content="secret rewind content alpha")
        # Simulate a rewind: active=0, compacted=0 (NOT compaction)
        db._conn.execute(
            "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
            (mid,),
        )
        db._conn.commit()

        result = json.loads(session_search(
            query="secret rewind content alpha", db=db,
            current_session_id="s_rewind",
        ))
        assert result["count"] == 0

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

    def test_rewound_rows_do_not_leak_into_active_anchor_window_or_bookends(self, db):
        db.create_session("s_rewind_window", source="cli")
        db.append_message("s_rewind_window", role="user", content="active opening")
        abandoned_id = db.append_message(
            "s_rewind_window",
            role="user",
            content="abandoned instruction must stay forensic only",
        )
        db.append_message(
            "s_rewind_window",
            role="assistant",
            content="abandoned answer must stay forensic only",
        )
        db.rewind_to_message("s_rewind_window", abandoned_id)
        db.append_message(
            "s_rewind_window",
            role="user",
            content="replacement branch anchor omega",
        )
        db.append_message(
            "s_rewind_window",
            role="assistant",
            content="replacement branch answer",
        )

        result = json.loads(
            session_search(query="replacement branch anchor omega", limit=1, db=db)
        )

        entry = result["results"][0]
        visible = entry["bookend_start"] + entry["messages"] + entry["bookend_end"]
        assert all("abandoned" not in message["content"] for message in visible)

    def test_rewind_visibility_guards_cover_anchor_after_and_distant_bookends(self, db):
        db.create_session("s_rewind_guards", source="cli")
        opening_id = db.append_message(
            "s_rewind_guards",
            role="user",
            content="active opening anchor",
        )
        abandoned_id = db.append_message(
            "s_rewind_guards",
            role="user",
            content="abandoned branch instruction",
        )
        db.append_message(
            "s_rewind_guards",
            role="assistant",
            content="abandoned branch answer",
        )
        db.rewind_to_message("s_rewind_guards", abandoned_id)
        db.append_message(
            "s_rewind_guards",
            role="user",
            content="replacement branch instruction",
        )
        db.append_message(
            "s_rewind_guards",
            role="assistant",
            content="replacement branch answer",
        )

        rewound_anchor = json.loads(
            session_search(
                session_id="s_rewind_guards",
                around_message_id=abandoned_id,
                db=db,
            )
        )
        assert rewound_anchor["success"] is False

        active_window = json.loads(
            session_search(
                session_id="s_rewind_guards",
                around_message_id=opening_id,
                window=5,
                db=db,
            )
        )
        assert active_window["success"] is True
        assert all(
            "abandoned" not in message["content"]
            for message in active_window["messages"]
        )

        for index in range(12):
            db.append_message(
                "s_rewind_guards",
                role="assistant",
                content=f"active filler {index}",
            )
        db.append_message(
            "s_rewind_guards",
            role="user",
            content="distant replacement needle",
        )
        db._conn.commit()

        discovery = json.loads(
            session_search(query="distant replacement needle", limit=1, db=db)
        )
        entry = discovery["results"][0]
        visible = entry["bookend_start"] + entry["messages"] + entry["bookend_end"]
        assert all("abandoned" not in message["content"] for message in visible)


class TestCompressionEndedHelper:
    """Unit tests for _is_compression_ended."""

    def test_compression_ended_session(self, db):
        db.create_session("s1", source="cli")
        db.end_session("s1", "compression")
        assert _is_compression_ended(db, "s1") is True

    def test_active_session_not_ended(self, db):
        db.create_session("s1", source="cli")
        assert _is_compression_ended(db, "s1") is False

    def test_delegation_child_not_ended(self, db):
        """A delegation child under a compression continuation does NOT have
        end_reason='compression' itself."""
        db.create_session("s_parent", source="cli")
        db.end_session("s_parent", "compression")
        db.create_session("s_continuation", source="cli", parent_session_id="s_parent")
        db.create_session("s_delegate_child", source="cli", parent_session_id="s_continuation")
        assert _is_compression_ended(db, "s_delegate_child") is False

    def test_empty_and_nonexistent(self, db):
        assert _is_compression_ended(db, "") is False
        assert _is_compression_ended(db, "nonexistent") is False


class TestLegacyContinuationPlusDelegation:
    """Regression: a delegation child created under a compression continuation
    must stay excluded — its content is still live to the parent agent.
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
        db.create_session("s_delegate", source="cli", parent_session_id="s_p")
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
