"""Tests for bounded lexical query expansion (port of openclaw/openclaw#121196).

Covers the keyword extractor in tools/query_expansion.py and the
supplemental OR probe wired into session_search's discovery shape.
"""
import json
import time

import pytest

from hermes_state import SessionDB
from tools import query_expansion as qe
from tools.session_search_tool import session_search


# =========================================================================
# Keyword extraction
# =========================================================================

class TestExtractKeywords:
    def test_conversational_query_keeps_meaningful_terms(self):
        kws = qe.extract_keywords("that thing we discussed about the API")
        assert kws == ["API"]

    def test_filler_only_query_yields_nothing(self):
        assert qe.extract_keywords("that thing we talked about yesterday") == []

    def test_identifiers_survive_edge_punct_stripping(self):
        kws = qe.extract_keywords("what did we decide about chat-send and app.config?")
        assert "chat-send" in kws
        assert "app.config" in kws

    def test_bounded_to_max_terms(self):
        query = " ".join(f"keyword{i}" for i in range(20))
        assert len(qe.extract_keywords(query)) == qe.MAX_EXPANSION_TERMS

    def test_dedupes_case_insensitively(self):
        assert qe.extract_keywords("Docker docker DOCKER networking") == [
            "Docker", "networking",
        ]

    def test_cjk_tokens_exempt_from_length_floor(self):
        kws = qe.extract_keywords("我们 讨论 的 那个 方案")
        # Two-char CJK tokens are full words and must survive.
        assert "讨论" in kws
        assert "方案" in kws

    def test_empty_query(self):
        assert qe.extract_keywords("") == []
        assert qe.extract_keywords(None) == []


class TestOperatorDetection:
    def test_quoted_phrase_detected(self):
        assert qe.has_fts_operators('"docker networking"')

    def test_boolean_operators_detected(self):
        assert qe.has_fts_operators("alpha OR beta")
        assert qe.has_fts_operators("python NOT java")
        assert qe.has_fts_operators("a AND b")

    def test_wildcard_detected(self):
        assert qe.has_fts_operators("deploy*")

    def test_lowercase_or_is_not_an_operator(self):
        # FTS5 operators are uppercase-only; "this or that" is plain prose.
        assert not qe.has_fts_operators("this or that")

    def test_plain_query_not_detected(self):
        assert not qe.has_fts_operators("auth refactor discussion")


class TestNoopGuard:
    def test_all_tokens_kept_is_noop(self):
        query = "docker networking bridge"
        kws = qe.extract_keywords(query)
        assert qe.expansion_is_noop(query, kws)

    def test_narrowed_query_is_not_noop(self):
        query = "that thing about docker networking"
        kws = qe.extract_keywords(query)
        assert not qe.expansion_is_noop(query, kws)


class TestBuildQuery:
    def test_joins_with_or(self):
        assert qe.build_expansion_query(["a", "b", "c"]) == "a OR b OR c"


# =========================================================================
# Discovery-shape integration
# =========================================================================

@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db):
    now = int(time.time())
    db.create_session("s_auth", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 5000, "OAuth token refresh", "s_auth"),
    )
    db.append_message("s_auth", role="user", content="Fix the OAuth refresh flow")
    db.append_message(
        "s_auth", role="assistant",
        content="API authentication uses short-lived OAuth tokens now.",
    )
    db.create_session("s_deploy", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 3000, "Deploy pipeline", "s_deploy"),
    )
    db.append_message("s_deploy", role="user", content="Set up the deploy pipeline")
    db.append_message(
        "s_deploy", role="assistant",
        content="Deployment strategy requires a maintenance window.",
    )
    db._conn.commit()


class TestDiscoveryExpansion:
    def test_conversational_query_finds_keyword_sessions(self, db):
        _seed(db)
        # Strict AND query matches nothing: "thing", "we", "discussed" never
        # co-occur with API in one message. Expansion must recover the hit.
        result = json.loads(
            session_search(query="that thing we discussed about the API", db=db)
        )
        assert result["success"] is True
        assert result["count"] >= 1
        assert any(r["session_id"] == "s_auth" for r in result["results"])
        assert result.get("expanded_terms") == ["API"]

    def test_strict_hits_rank_above_expanded_hits(self, db):
        _seed(db)
        # "deploy pipeline" strictly matches s_deploy; expansion may pull in
        # more, but the strict hit must stay first.
        result = json.loads(session_search(query="deploy pipeline", db=db, limit=3))
        assert result["results"][0]["session_id"] == "s_deploy"

    def test_operator_queries_skip_expansion(self, db):
        _seed(db)
        result = json.loads(
            session_search(query='"phrase that matches nothing at all"', db=db)
        )
        assert result["success"] is True
        assert "expanded_terms" not in result

    def test_no_expansion_marker_when_strict_pool_is_full(self, db):
        _seed(db)
        result = json.loads(session_search(query="OAuth", db=db, limit=1))
        assert result["count"] == 1
        assert "expanded_terms" not in result

    def test_filler_only_query_still_returns_cleanly(self, db):
        _seed(db)
        result = json.loads(
            session_search(query="that thing from yesterday", db=db)
        )
        assert result["success"] is True
