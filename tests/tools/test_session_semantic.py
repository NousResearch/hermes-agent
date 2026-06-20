"""Tests for hybrid semantic session search (#44075).

Layers covered:
  - rrf_merge / serialize_f32 — pure functions, no deps
  - message_embeddings schema — plain table + invalidation triggers,
    works without sqlite-vec
  - vector KNN + hybrid end-to-end — require the sqlite-vec extension;
    skipped when it is not installed or the sqlite3 build cannot load
    extensions

No live network anywhere: embedding calls are faked with deterministic
topic-axis vectors.
"""
import json
import re
import struct
import time

import pytest

from hermes_state import SessionDB
import tools.session_semantic as session_semantic
from tools.session_semantic import rrf_merge, serialize_f32
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _require_sqlite_vec(db):
    if not db.vector_search_available():
        pytest.skip("sqlite-vec extension not available")


# A tiny deterministic "embedding space": each axis counts vocabulary hits
# for one topic, so cosine distance clusters paraphrases without a network
# call. Axis 3 is the catch-all so no vector is all-zero.
_TOPIC_AXES = (
    {"smart", "home", "tuya", "bardi", "bulb", "bulbs", "lights"},
    {"hr", "talenta", "clock", "attendance", "payroll"},
    {"modpack", "minecraft", "neoforge", "quest"},
)


def fake_embed_texts(texts, cfg=None):
    out = []
    for text in texts:
        words = set(re.findall(r"\w+", text.lower()))
        vec = [float(len(words & axis)) for axis in _TOPIC_AXES] + [0.0]
        if not any(vec):
            vec[3] = 1.0
        out.append(serialize_f32(vec))
    return out


_ENABLED_CFG = {
    "semantic": True,
    "embedding_provider": "",
    "embedding_model": "fake-embed",
    "embedding_base_url": "",
    "embedding_api_key": "",
    "hybrid_weight_vector": 0.7,
    "hybrid_weight_bm25": 0.3,
    "min_similarity": 0.25,
    "index_batch_size": 96,
}


def _seed_smart_home_sessions(db):
    """One session about Tuya bulbs (no 'smart home' wording), one about HR."""
    now = int(time.time())
    db.create_session("s_bulbs", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 20000, "Bardi bulb setup", "s_bulbs"),
    )
    db.append_message("s_bulbs", role="user", content="Set up my Bardi Tuya bulbs")
    db.append_message("s_bulbs", role="assistant", content="Paired both Tuya bulbs and added schedules.")

    db.create_session("s_hr", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 10000, "Talenta automation", "s_hr"),
    )
    db.append_message("s_hr", role="user", content="Automate Talenta clock attendance")
    db.append_message("s_hr", role="assistant", content="Talenta clock automation scripted.")
    db._conn.commit()


# =========================================================================
# Pure functions
# =========================================================================

class TestSerializeF32:
    def test_roundtrip(self):
        vec = [0.25, -1.5, 3.0]
        blob = serialize_f32(vec)
        assert len(blob) == 12
        assert list(struct.unpack("<3f", blob)) == vec


class TestRRFMerge:
    def test_dedupes_and_tags_both(self):
        fts = [{"id": 1, "snippet": ">>>hit<<<"}, {"id": 2, "snippet": "b"}]
        vec = [{"id": 1, "snippet": "plain"}, {"id": 3, "snippet": "c"}]
        merged = rrf_merge(fts, vec, weight_bm25=0.5, weight_vector=0.5)
        by_id = {row["id"]: row for row in merged}
        assert set(by_id) == {1, 2, 3}
        assert by_id[1]["match_type"] == "both"
        assert by_id[2]["match_type"] == "keyword"
        assert by_id[3]["match_type"] == "semantic"
        # FTS row wins field content on dedupe — keeps the highlighted snippet
        assert by_id[1]["snippet"] == ">>>hit<<<"
        # A row in both lists outranks single-source rows
        assert merged[0]["id"] == 1

    def test_weights_bias_ranking(self):
        fts = [{"id": 10}]
        vec = [{"id": 20}]
        vec_heavy = rrf_merge(fts, vec, weight_bm25=0.1, weight_vector=0.9)
        assert vec_heavy[0]["id"] == 20
        bm25_heavy = rrf_merge(fts, vec, weight_bm25=0.9, weight_vector=0.1)
        assert bm25_heavy[0]["id"] == 10

    def test_empty_inputs(self):
        assert rrf_merge([], [], 0.3, 0.7) == []
        only_vec = rrf_merge([], [{"id": 5}], 0.3, 0.7)
        assert only_vec[0]["match_type"] == "semantic"


# =========================================================================
# Schema: plain table + triggers (no sqlite-vec needed)
# =========================================================================

class TestEmbeddingSchema:
    def test_upsert_and_count(self, db):
        db.create_session("s1", source="cli")
        mid = db.append_message("s1", role="user", content="hello world")
        written = db.upsert_message_embeddings(
            [(mid, "fake-embed", 4, serialize_f32([1, 0, 0, 0]))]
        )
        assert written == 1
        assert db.count_message_embeddings() == 1
        assert db.count_message_embeddings("fake-embed") == 1
        assert db.count_message_embeddings("other-model") == 0
        # Replace, not duplicate
        db.upsert_message_embeddings(
            [(mid, "fake-embed", 4, serialize_f32([0, 1, 0, 0]))]
        )
        assert db.count_message_embeddings() == 1

    def test_unembedded_selection(self, db):
        db.create_session("s1", source="cli")
        m1 = db.append_message("s1", role="user", content="first")
        db.append_message("s1", role="tool", content="tool output", tool_name="t")
        db.append_message("s1", role="user", content="   ")
        m4 = db.append_message("s1", role="assistant", content="second")

        pending = db.get_unembedded_messages("fake-embed", limit=10)
        # Tool role and whitespace-only content excluded; newest first
        assert [row["id"] for row in pending] == [m4, m1]

        db.upsert_message_embeddings([(m4, "fake-embed", 4, serialize_f32([1, 0, 0, 0]))])
        assert [r["id"] for r in db.get_unembedded_messages("fake-embed", limit=10)] == [m1]
        # A different model still sees both rows as pending
        assert len(db.get_unembedded_messages("other-model", limit=10)) == 2

    def test_delete_trigger_removes_embedding(self, db):
        db.create_session("s1", source="cli")
        mid = db.append_message("s1", role="user", content="ephemeral")
        db.upsert_message_embeddings([(mid, "fake-embed", 4, serialize_f32([1, 0, 0, 0]))])
        db._conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
        db._conn.commit()
        assert db.count_message_embeddings() == 0

    def test_content_update_invalidates_embedding(self, db):
        db.create_session("s1", source="cli")
        mid = db.append_message("s1", role="user", content="original")
        db.upsert_message_embeddings([(mid, "fake-embed", 4, serialize_f32([1, 0, 0, 0]))])
        db._conn.execute("UPDATE messages SET content = 'edited' WHERE id = ?", (mid,))
        db._conn.commit()
        assert db.count_message_embeddings() == 0
        # Non-content updates (e.g. rewind toggling active) keep the embedding
        mid2 = db.append_message("s1", role="user", content="kept")
        db.upsert_message_embeddings([(mid2, "fake-embed", 4, serialize_f32([1, 0, 0, 0]))])
        db._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (mid2,))
        db._conn.commit()
        assert db.count_message_embeddings() == 1


# =========================================================================
# Fallback gating — hybrid_search must return None (FTS-only behaviour)
# =========================================================================

class TestFallbackGating:
    def test_disabled_by_default(self, db):
        # Fresh HERMES_HOME (autouse fixture) → default config → semantic off
        assert session_semantic.hybrid_search(db, query="anything") is None

    def test_empty_query(self, db, monkeypatch):
        monkeypatch.setattr(session_semantic, "get_semantic_config", lambda: dict(_ENABLED_CFG))
        assert session_semantic.hybrid_search(db, query="   ") is None

    def test_sqlite_vec_missing(self, db, monkeypatch):
        monkeypatch.setattr(session_semantic, "get_semantic_config", lambda: dict(_ENABLED_CFG))
        monkeypatch.setattr(session_semantic, "_ensure_sqlite_vec", lambda: False)
        assert session_semantic.hybrid_search(db, query="smart home") is None

    def test_embedding_failure(self, db, monkeypatch):
        monkeypatch.setattr(session_semantic, "get_semantic_config", lambda: dict(_ENABLED_CFG))
        monkeypatch.setattr(session_semantic, "_ensure_sqlite_vec", lambda: True)
        monkeypatch.setattr(db.__class__, "vector_search_available", lambda self: True)
        monkeypatch.setattr(session_semantic, "embed_texts", lambda texts, cfg=None: None)
        assert session_semantic.hybrid_search(db, query="smart home") is None

    def test_discover_falls_back_to_fts(self, db):
        """With semantic off, the tool answers exactly like before (search_mode fts)."""
        _seed_smart_home_sessions(db)
        result = json.loads(session_search(query="Tuya", db=db))
        assert result["success"] is True
        assert result["search_mode"] == "fts"
        assert result["count"] == 1

    def test_index_pending_survives_readonly_db(self, monkeypatch):
        """Cross-profile DBs open read-only; indexing must not raise."""
        monkeypatch.setattr(session_semantic, "embed_texts", fake_embed_texts)
        import sqlite3

        class _RoDB:
            def get_unembedded_messages(self, model, limit=128, roles=("user", "assistant")):
                return [{"id": 1, "content": "tuya bulbs"}]

            def upsert_message_embeddings(self, rows):
                raise sqlite3.OperationalError("attempt to write a readonly database")

        assert session_semantic.index_pending(_RoDB(), dict(_ENABLED_CFG)) == 0


# =========================================================================
# Vector KNN + hybrid end-to-end (need sqlite-vec)
# =========================================================================

class TestVectorSearch:
    def test_knn_orders_by_cosine_distance(self, db):
        _require_sqlite_vec(db)
        _seed_smart_home_sessions(db)
        monkeypatch_rows = db.get_unembedded_messages("fake-embed", limit=50)
        blobs = fake_embed_texts([r["content"] for r in monkeypatch_rows])
        db.upsert_message_embeddings(
            [(r["id"], "fake-embed", 4, b) for r, b in zip(monkeypatch_rows, blobs)]
        )

        query_blob = fake_embed_texts(["what did we do about smart home lights"])[0]
        rows = db.search_messages_by_vector(query_blob, model="fake-embed", dim=4, limit=10)
        assert rows, "expected vector hits"
        assert rows[0]["session_id"] == "s_bulbs"
        assert rows[0]["distance"] < rows[-1]["distance"] or len(rows) == 1
        assert "snippet" in rows[0] and "content" not in rows[0]

        # Model/dim mismatch returns nothing instead of erroring
        assert db.search_messages_by_vector(query_blob, model="other", dim=4) == []
        assert db.search_messages_by_vector(serialize_f32([1.0]), model="fake-embed", dim=1) == []

    def test_role_and_source_filters(self, db):
        _require_sqlite_vec(db)
        _seed_smart_home_sessions(db)
        rows = db.get_unembedded_messages("fake-embed", limit=50)
        blobs = fake_embed_texts([r["content"] for r in rows])
        db.upsert_message_embeddings(
            [(r["id"], "fake-embed", 4, b) for r, b in zip(rows, blobs)]
        )
        query_blob = fake_embed_texts(["tuya bulbs"])[0]
        user_only = db.search_messages_by_vector(
            query_blob, model="fake-embed", dim=4, role_filter=["user"], limit=10
        )
        assert user_only and all(r["role"] == "user" for r in user_only)
        excluded = db.search_messages_by_vector(
            query_blob, model="fake-embed", dim=4, exclude_sources=["cli"], limit=10
        )
        assert excluded == []


class TestHybridEndToEnd:
    @pytest.fixture
    def semantic_db(self, db, monkeypatch):
        _require_sqlite_vec(db)
        _seed_smart_home_sessions(db)
        monkeypatch.setattr(session_semantic, "get_semantic_config", lambda: dict(_ENABLED_CFG))
        monkeypatch.setattr(session_semantic, "embed_texts", fake_embed_texts)
        return db

    def test_semantic_recall_without_keyword_overlap(self, semantic_db):
        """The motivating case from #44075: 'smart home' finds the Tuya
        bulbs session even though no message contains those words."""
        merged = session_semantic.hybrid_search(semantic_db, query="smart home")
        assert merged, "hybrid search returned no rows"
        assert merged[0]["session_id"] == "s_bulbs"
        assert merged[0]["match_type"] == "semantic"

    def test_keyword_and_semantic_agree(self, semantic_db):
        merged = session_semantic.hybrid_search(semantic_db, query="Tuya bulbs")
        assert merged[0]["session_id"] == "s_bulbs"
        assert merged[0]["match_type"] == "both"

    def test_opportunistic_indexing_runs_on_search(self, semantic_db):
        assert semantic_db.count_message_embeddings() == 0
        session_semantic.hybrid_search(semantic_db, query="smart home")
        assert semantic_db.count_message_embeddings("fake-embed") == 4

    def test_discover_tool_reports_hybrid(self, semantic_db):
        result = json.loads(session_search(query="smart home", db=semantic_db))
        assert result["success"] is True
        assert result["search_mode"] == "hybrid"
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s_bulbs"
        assert result["results"][0]["match_type"] == "semantic"

    def test_sort_newest_biases_time(self, semantic_db):
        # Both sessions hit (hr via keyword-ish vocab? no — use a query
        # touching both topics) and newest-first ordering applies.
        merged = session_semantic.hybrid_search(
            semantic_db, query="tuya bulbs talenta clock", sort="newest"
        )
        assert len(merged) >= 2
        timestamps = [row.get("timestamp") or 0 for row in merged]
        assert timestamps == sorted(timestamps, reverse=True)
